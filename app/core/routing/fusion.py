"""层② 打分与融合 —— 词面（免费）先行，语义（要花一次 embedding）**按需**升级。

惰性升级：本模块最重要的一条时序约定
------------------------------------
用户给的路由前提是"不能引入过度的时间开销"。原设计里第 2 关**无条件**算一次
query 向量，那等于把省下来的一次 LLM 调用换成一次 embedding 网络往返——
对"我还能休几天"这种词面判不出的句子值，对"报销流程怎么走"纯属白花。

所以升级条件是**门控说了算**，不是"到了第 2 关就升"::

    词面打分(免费) → 门控 → 通过？→ 结束，一次 embedding 都不花
                          → 灰区？→ 才升级到语义层

实现上没有单独的"要不要升级"函数——它就是 :func:`app.core.routing.gating.gate`
跑在**只有词面一路**的候选上。同一个门控函数被复用两次（词面后、融合后），
既少一份逻辑，也让"两层判定标准一致"成为构造上的事实而非口头约定。

融合用 RRF，但**门控不用 RRF 分**（一条必须说清的边界）
------------------------------------------------------
RRF 只看排名，跨 embedding 供应商、跨语料都稳定，这与本项目混合召回
（``docs/history/rag-architecture-benchmark.md``）的选型是同一套哲学——同一个项目里
两处融合，不应出现两套理由。

但它的分数**不能拿来做边际阈值**：RRF 分只由排名决定，rank1 与 rank2 的差
恒为 ``1/(k+1) ≈ 1.6%``，与"领先一大截"还是"咬得很紧"完全无关。拿它比阈值，
结果要么永远进灰区、要么永远不进。所以::

    融合分(RRF, 相对)  →  只用来**展示**与**并列裁决**
    地板 / 边际 / 排序  →  一律用**同层同量纲的绝对证据**
                          （词面用加权命中分，语义用原始余弦）

⚠️ **排序也必须用绝对证据**，这一点比地板更容易搞错。首版曾按融合分排序，
于是出现"融合分选出的 top 在语义上输给第二名"——算出的 gap 是负数，
边际判定永远不通过，本该直接判对的掉进灰区白花一次 LLM 调用。
排序与门控必须同量纲，是同一条原则的两个面。

这也是 D3"地板用绝对分"的姊妹条款：既然地板不能用归一化分，
边际与排序同样不能——凡是拿来做判断的，就必须是同一个量纲。
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app import config
from app.core.routing import catalog, signals, vocabulary
from app.utils.logger import logger


class SemanticUnavailable(RuntimeError):
    """语义路不可用（未配 / 超时 / 调用失败）。

    调用方**必须**捕获它并按 D5 降级（权重按词面重新归一继续走），
    而不是把整条路由判为失败——一次依赖抖动不该放大成路由全面降级。
    """


@dataclass(frozen=True)
class Scored:
    """单路的原始打分结果。``score`` 是**绝对证据**，不做任何归一化。"""

    name: str
    channel: str
    score: float
    #: 词面命中的关键词，用于预演接口与排障（"它凭什么得这个分"）。
    hits: Tuple[str, ...] = ()


@dataclass
class Candidate:
    """一路或多路融合后的候选。"""

    name: str
    channel: str
    lexical: float = 0.0
    semantic: float = 0.0
    has_semantic: bool = False
    #: 归一化 RRF 分 ∈ [0, 1]。1.0 = 在所有生效通道里都排第一。
    #: **只用于排序与展示，不进任何阈值比较**（见模块 docstring）。
    fused: float = 0.0
    rank_lexical: Optional[int] = None
    rank_semantic: Optional[int] = None
    hits: Tuple[str, ...] = ()

    @property
    def abs_score(self) -> float:
        """本候选的**绝对证据**。

        有语义分时用原始余弦，否则退回词面加权分。地板就压在这个值上——
        用归一化分做地板的话 top1 恒等于 1.0，地板将永远通过，等于没有地板（D3）。
        """
        return self.semantic if self.has_semantic else self.lexical

    @property
    def has_evidence(self) -> bool:
        """是否真的**参与过**打分（在任一路里拿到正分并因此获得排名）。

        注意区分"参与过但得分低"与"压根没参与"：前者是竞争者，
        后者不是。把它们混在一起，门控的"次优通道"会变成一个 0 分的空壳，
        边际比较随即失去意义。
        """
        return self.rank_lexical is not None or self.rank_semantic is not None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "channel": self.channel,
            "fused": round(self.fused, 4),
            "lexical": round(self.lexical, 4),
            "semantic": round(self.semantic, 4),
            "abs_score": round(self.abs_score, 4),
            "hits": list(self.hits),
        }


# ---------------------------------------------------------------------------
# 带超时的同步调用
# ---------------------------------------------------------------------------
def call_with_timeout(fn: Callable[[], object], timeout_ms: int) -> object:
    """在守护线程里执行 ``fn``；超过 ``timeout_ms`` 抛 :class:`TimeoutError`。

    为什么是线程而不是 ``signal.alarm`` / ``asyncio.wait_for``：这段代码同时被
    **同步链路**（LangGraph 节点）与**预演接口**（FastAPI 线程池里的同步端点）调用，
    既没有可靠的 event loop，也不能动进程级 signal。

    超时后那个线程会继续跑到 httpx 自己的超时为止（daemon 线程，不会阻止进程退出），
    但我们**立刻**降级——绝不为了一个慢依赖押上整轮请求的延迟。
    """
    holder: Dict[str, object] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            holder["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 —— 原样转交给调用方线程
            holder["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_run, daemon=True, name="route-io")
    worker.start()
    if not done.wait(max(timeout_ms, 0) / 1000.0):
        raise TimeoutError(f"等待 {timeout_ms}ms 仍未返回")
    error = holder.get("error")
    if error is not None:
        raise error
    return holder.get("value")


# ---------------------------------------------------------------------------
# 词面打分（零成本、无 IO）
# ---------------------------------------------------------------------------
def guard_fired(spec: catalog.IntentSpec, query: str, vocab: vocabulary.Vocabulary) -> bool:
    """该能力的 guard 是否命中（命中即清零）。

    guard 对**词面分与语义分同时生效**。只拦词面的话，「怎么申请邮箱扩容」
    仍可能靠语义相似度命中 ``employee_attr`` —— 一条 guard 拦一半，
    等于没拦，而且症状极隐蔽（词面看着是对的）。

    判据只描述**句式**（"这句话长什么样"），它否定哪个能力由 ``spec.guards`` 声明。
    句式中用到的领域实词来自 ``vocab``，本模块一个业务词都不认识。
    """
    return any(signals.GUARD_FUNCS[g](query, vocab) for g in spec.guards)


def score_lexical(
    query: str,
    specs: Optional[Sequence[catalog.IntentSpec]] = None,
    vocab: Optional[vocabulary.Vocabulary] = None,
) -> List[Scored]:
    """词面打分：``Σ len(命中关键词)``。

    **长词权重更高**是有意的：专名（"入职时间""假期余额"）比通用词
    （"部门""流程"）更能说明意图，而"张三"与"张伟"在向量空间几乎重合——
    专名靠词面远比靠向量准。这实现为"权重 = 词长"，不需要额外的权重表。

    ⚠️ "权重 = 词长"也正是它的**固有偏向**：一个意图穷举了更多宾语词
    （"年假" + "调休"）就会压过另一个只命中提问意图词（"区别"）的意图。
    这类漏判调权重治不好（没有可调参数），要靠句式 guard 去否定错的意图——
    但 guard 只该为**这一类**问题而加，不要为单个句子量身定做。
    """
    text = (query or "").strip().lower()
    vocab = vocab if vocab is not None else catalog.vocabulary()
    out: List[Scored] = []
    for spec in (specs if specs is not None else catalog.all_specs()):
        if guard_fired(spec, query, vocab):
            out.append(Scored(spec.name, spec.channel, 0.0, ()))
            continue
        hits = tuple(kw for kw in spec.keywords if kw.lower() in text)
        out.append(Scored(spec.name, spec.channel, float(sum(len(kw) for kw in hits)), hits))
    return out


# ---------------------------------------------------------------------------
# 语义打分（要花一次 embedding，故由门控决定是否调用）
# ---------------------------------------------------------------------------
_UTTERANCE_VECS: Dict[str, List[float]] = {}


def reset_utterance_cache() -> None:
    """清空 utterances 向量缓存（测试与"目录热更新"用）。"""
    _UTTERANCE_VECS.clear()


def _ensure_utterance_vectors(specs: Sequence[catalog.IntentSpec]) -> int:
    """补齐缺失的 utterances 向量，返回本次新算的条数。

    key 用**文本内容**而不是能力名:改一条 utterance 只会让它自己失效,
    不会误用旧向量。首次调用会一次性批量算完(API 模式下由
    ``CachedAPIEmbeddings`` 落到文档向量持久化缓存,进程重启也不再付费)。
    """
    missing = [u for s in specs for u in s.utterances if u not in _UTTERANCE_VECS]
    if not missing:
        return 0
    vectors = _get_embeddings().embed_documents(missing)
    for text, vec in zip(missing, vectors):
        _UTTERANCE_VECS[text] = vec
    return len(missing)


def _embed_query_raw(query: str) -> List[float]:
    """真正打一次 embedding。**不碰 request_ctx**，理由见 :func:`score_semantic`。"""
    vector = _get_embeddings().embed_query(query)
    if not isinstance(vector, list) or not vector:
        raise SemanticUnavailable("embedding 返回空向量")
    return vector


def _get_embeddings():
    from app.providers.embeddings import get_embeddings

    return get_embeddings()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。两个向量都已 L2 归一化，但仍做完整计算以防外部 provider 不归一。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _score_all(
    query: str,
    specs: Sequence[catalog.IntentSpec],
    query_vec: Sequence[float],
    vocab: vocabulary.Vocabulary,
) -> List[Scored]:
    out: List[Scored] = []
    for spec in specs:
        if guard_fired(spec, query, vocab):
            out.append(Scored(spec.name, spec.channel, 0.0))
            continue
        best = 0.0
        for text in spec.utterances:
            vec = _UTTERANCE_VECS.get(text)
            if vec:
                best = max(best, cosine(query_vec, vec))
        out.append(Scored(spec.name, spec.channel, best))
    return out


def _prepare_and_score(
    query: str,
    specs: Sequence[catalog.IntentSpec],
    cached_vec,
    vocab: vocabulary.Vocabulary,
):
    """在**工作线程**里跑：预热 utterances → 取得 query 向量 → 打分。

    返回 ``(scores, vector)``，向量交回调用方线程写进请求上下文。
    """
    _ensure_utterance_vectors(specs)
    vector = cached_vec if cached_vec else _embed_query_raw(query)
    return _score_all(query, specs, vector, vocab), vector


def score_semantic(
    query: str,
    specs: Optional[Sequence[catalog.IntentSpec]] = None,
    timeout_ms: Optional[int] = None,
    vocab: Optional[vocabulary.Vocabulary] = None,
) -> List[Scored]:
    """语义打分：query 向量与每个能力的 utterances 取 **max 余弦**。

    限时覆盖**整段**（utterances 预热 + query embedding），不只是 query 那一次请求。
    这一点是必须的：首次调用要批量算完全部 utterances 的向量，那可能是一次
    好几秒的开销；只给 query 限时，预热会绕过预算，把路由的承诺打穿（首版实测
    出现过 5.2s 的一轮请求）。

    ⚠️ 超时**不是白费**：``call_with_timeout`` 放弃的是**等待**，不是那个线程。
    它仍在后台把 utterances 算完并写进缓存，于是只有第一个请求降级、
    后续请求正常。这是刻意选用"守护线程 + 放弃等待"而不是
    "可取消任务"的原因——顺手把冷启动预热做掉了，且不必让启动自检去等 embedding。

    ⚠️ **query 向量的读与写必须留在调用方线程**，这是踩过两次的同一个坑
    （见 ``request_ctx`` 模块 docstring）：线程有**自己独立**的 context，
    在工作线程里 ``ContextVar.set`` 调用方读不到。首版把 ``set_query_vector``
    写在了工作线程里，症状是——D9 的复用完全失效、检索层又算了一遍向量，
    而**任何日志与断言都不会报错**，只是每一轮问答都悄悄多花一次 embedding。
    所以这里把"算"放线程、"存"回调用方，两件事分开。

    Raises:
        SemanticUnavailable: 任何一步失败/超时。调用方据此走 D5 降级，
            **不要**在这里吞掉异常——吞掉就无法区分"语义路不可用"与
            "这句话确实跟哪个能力都不像"，而这两件事的处置完全不同。
    """
    from app.core import request_ctx

    specs = specs if specs is not None else catalog.all_specs()
    vocab = vocab if vocab is not None else catalog.vocabulary()
    limit = config.ROUTE_EMBED_TIMEOUT_MS if timeout_ms is None else timeout_ms
    cached = request_ctx.get_query_vector(query)
    try:
        scores, vector = call_with_timeout(
            lambda: _prepare_and_score(query, specs, cached, vocab), limit
        )
    except SemanticUnavailable:
        raise
    except Exception as exc:
        # 不加 noqa：这里把原异常**链接**成 SemanticUnavailable 再抛，
        # BLE001 不认为这是"盲except"（它没有把异常吞掉）。
        raise SemanticUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if not cached and vector:
        request_ctx.set_query_vector(query, vector)
    return list(scores)


# ---------------------------------------------------------------------------
# 融合（RRF）
# ---------------------------------------------------------------------------
def _competition_ranks(scores: Sequence[Scored]) -> Dict[str, int]:
    """在**正分**候选上做竞争排名（``1, 1, 3, …``）：同分即同名次。

    为什么先滤掉 0 分：0 分不是"最后一名"，是"没参与"。若把 0 分也编上名次，
    它就会凭空得到一个正融合分，进而挤进候选表、把"次优通道"污染成一个空壳。
    这是门控最容易错的地方，也是最不容易看出来的地方。

    为什么不用 ``enumerate`` 直接编号：两条证据一模一样的候选拿到不同名次，
    会得到不同的融合分，读者排障时会看到"它凭什么赢"却找不到任何原因。
    同分同名次让 ``fused`` 保持可解释。
    """
    positive = sorted((s for s in scores if s.score > 0), key=lambda s: -s.score)
    ranks: Dict[str, int] = {}
    for idx, item in enumerate(positive):
        if idx and item.score == positive[idx - 1].score:
            ranks[item.name] = ranks[positive[idx - 1].name]
        else:
            ranks[item.name] = idx + 1
    return ranks


def fuse(
    lexical: Sequence[Scored],
    semantic: Optional[Sequence[Scored]] = None,
    *,
    k: Optional[int] = None,
    specs: Optional[Sequence[catalog.IntentSpec]] = None,
) -> List[Candidate]:
    """两路 RRF 融合，返回按 ``fused`` 降序的候选表。

    语义路不可用（``semantic is None``）时**按剩余信号重新归一权重**（D5）：
    不做重归一的话，一次 embedding 故障会把所有分数腰斩、全部掉进灰区——
    一次依赖故障被放大成路由全面降级。这是五个参考项目共同的盲区。
    """
    k = config.RRF_K if k is None else k
    specs = specs if specs is not None else catalog.all_specs()

    weights: Dict[str, float] = {"lexical": 1.0}
    if semantic is not None:
        weights["semantic"] = 1.0
    total_w = sum(weights.values()) or 1.0
    weights = {name: w / total_w for name, w in weights.items()}

    #: 理论满分 = 在所有生效通道里都排第一。归一化后 ``fused ∈ [0, 1]``，
    #: 且**与"参与了几路"无关**——单路与双路的 1.0 含义一致，可跨请求比较。
    max_fused = sum(weights.values()) / (k + 1)

    lex_ranks = _competition_ranks(lexical)
    sem_ranks = _competition_ranks(semantic) if semantic is not None else {}
    lex_map = {s.name: s for s in lexical}
    sem_map = {s.name: s for s in (semantic or ())}

    candidates: List[Candidate] = []
    for spec in specs:
        raw = 0.0
        if "lexical" in weights and spec.name in lex_ranks:
            raw += weights["lexical"] / (k + lex_ranks[spec.name])
        if "semantic" in weights and spec.name in sem_ranks:
            raw += weights["semantic"] / (k + sem_ranks[spec.name])
        lex_item = lex_map.get(spec.name)
        sem_item = sem_map.get(spec.name)
        candidates.append(
            Candidate(
                name=spec.name,
                channel=spec.channel,
                lexical=lex_item.score if lex_item else 0.0,
                semantic=sem_item.score if sem_item else 0.0,
                has_semantic=semantic is not None,
                fused=(raw / max_fused) if max_fused else 0.0,
                rank_lexical=lex_ranks.get(spec.name),
                rank_semantic=sem_ranks.get(spec.name),
                hits=lex_item.hits if lex_item else (),
            )
        )
    candidates.sort(key=lambda c: (-c.fused, c.name))
    return candidates


def soft_warn(message: str) -> None:
    """把"被吞掉的可降级故障"留痕，绝不静默。

    这是 ``request_ctx.add_soft_warning`` 的薄封装：路由层的降级必须能被看见，
    否则一次 embedding 服务故障在日志里与"这句话确实跟谁都不像"完全同形。
    """
    from app.core import request_ctx

    request_ctx.add_soft_warning(f"路由：{message}")
    logger.warning("路由降级：%s", message)
