"""RAG 五层架构 · L3 检索优化层。

职责边界：把用户问题变成「高相关度的上下文片段集合」。
    查询改写（可选） → 多路召回 → RRF 融合 → 去重 → 阈值决策 → 软回退

为什么用 RRF（Reciprocal Rank Fusion）而不是加权求和分数：
    不同召回路的分数根本不在同一量纲上——向量余弦相似度落在 0~1（且不同
    embedding 供应商的分布差异极大，bge-m3 常在 0.05~0.15，而 OpenAI 常在
    0.3~0.9），词面匹配分是我们自定义的启发式值。直接加权求和需要归一化，
    而归一化参数会随语料漂移，极不稳定。

    RRF 只使用「排名」，完全绕开量纲问题：
        score(d) = Σ_i w_i / (k + rank_i(d))
    其中 k=60 是 SIGIR 2009 原论文给出的经验常数，对 k 不敏感，跨数据集稳健，
    已成为 Elasticsearch / OpenSearch / Azure AI Search 的默认融合算法。

多路召回的互补性（这是融合有效的根本前提）：
    - dense（向量）：擅长语义改写与同义表述，"怎么请假"能命中"休假申请流程"；
    - lexical（词面）：擅长专有名词与制度条款的精确字符，能救回向量在小语料
      上把无关段落排到前面的情况（例如"年假"必须命中"年休假"章节）。
"""
import fnmatch
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app import config
from app.core.request_ctx import get_query_vector, set_query_vector
from app.db.vector_db import get_vector_store
from app.rag.lexical import chunk_key, get_lexical_index
from app.utils.embedding import get_embeddings
from app.utils.logger import logger, preview

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------
RRF_K: int = getattr(config, "RRF_K", 60)                 # RRF 平滑常数
DENSE_WEIGHT: float = getattr(config, "DENSE_WEIGHT", 0.7)   # 向量路权重
LEXICAL_WEIGHT: float = getattr(config, "LEXICAL_WEIGHT", 0.3)  # 词面路权重
QUERY_REWRITE_ENABLED: bool = getattr(config, "QUERY_REWRITE_ENABLED", False)

# 中文检索词面匹配：汉字按「字符集合包含」判定（兼容「年假」查询命中「年休假」片段）
_CJK_CHAR = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[a-z0-9]{2,}")

# 词面分低于此值视为「未命中」，不参与词面路排名
_LEXICAL_FLOOR = 0.01


# ---------------------------------------------------------------------------
# 词面召回打分
# ---------------------------------------------------------------------------
def _cjk_runs(text: str) -> List[str]:
    """提取连续中文串，如「年假有多少天」→ ['年假', '有多少天']。"""
    return [run for run in re.split(r"[^一-鿿]+", text) if len(run) >= 2]


# 中文高频虚词：不计入词面匹配。
# 这是区分度的关键——若不剔除，「年假有多少天」的「有/多/少」会命中任何
# 含「有多少人」的片段，「的/了/是」更是几乎出现在每个汉语句子里，
# 会让所有片段的词面分一起冲到 1.0，词面路彻底失去排序能力。
_CJK_STOP = set(
    "的了是有着和在就都而我你他她它们这那吗呢吧啊很太最更也很还再又只才不没无"
    "给让向往把被对从到以为及其或与个们多少怎如何什么可以请问谢谢"
)


def lexical_score(query: str, content: str) -> float:
    """词面匹配分（0~1）：查询实义字符覆盖度 + 连续子串精确奖励。

    三点设计，每一点都对应一个真实踩过的坑：

    1. **过滤虚词**：中文没有空格分词，直接取字符集会把「的/了/是/有/多/少」
       这类高频字卷进来，导致几乎所有片段都拿到满分词面分，排序完全失效。
    2. **查询覆盖度**而非「匹配数量」：用 |q∩c| / |q| 衡量查询被文档覆盖的
       比例。只看匹配绝对数量的话，长查询天然占便宜，且无关片段只要凑够
       几个常见字就能拿满分。
    3. **连续子串奖励**：文档里原样写着「年假」应当严格优于只有字符重合的
       「年休假」，故给 +0.3 区分度（对 ≥2 字的连续中文串判定）。
    """
    query = query or ""
    content = content or ""
    # 中文路与拉丁路分别计算后取较强者：
    # 中英混合查询（如「VPN 连不上」「报销流程 2024」）里，真正的判别性关键词
    # 往往是英文/数字那半个查询。若检测到一个分支就 return，另一路的信号会被
    # 整个丢弃——「VPN 连不上」的中文部分匹配不到「连接失败」，英文部分明明
    # 命中了却拿不到分，词面路会静默失效。
    cjk_score = _cjk_lexical_score(query, content)
    latin_score = _latin_lexical_score(query, content)
    return max(cjk_score, latin_score)


def _cjk_lexical_score(query: str, content: str) -> float:
    """中文词面分：实义字符覆盖度 + 连续子串精确奖励。"""
    q_raw = set(_CJK_CHAR.findall(query.lower()))
    if not q_raw:
        return 0.0
    q_chars = q_raw - _CJK_STOP
    # 过滤后不足 2 字（查询本身几乎全是虚词），回退到全集，避免分母过小放大噪声
    if len(q_chars) < 2:
        q_chars = q_raw

    c_chars = set(_CJK_CHAR.findall(content))
    matched = q_chars & c_chars
    if len(matched) < 2:
        return 0.0

    coverage = len(matched) / len(q_chars)
    c_lower = content.lower()
    bonus = 0.3 if any(run in c_lower for run in _cjk_runs(query.lower())) else 0.0
    return min(1.0, coverage + bonus)


def _latin_lexical_score(query: str, content: str) -> float:
    """拉丁词面分：查询中英文/数字关键词在片段中的覆盖比例。"""
    q_words = set(_LATIN.findall(query.lower()))
    if not q_words:
        return 0.0
    c_lower = (content or "").lower()
    hit = sum(1 for w in q_words if w in c_lower)
    return hit / len(q_words)


# ---------------------------------------------------------------------------
# 查询改写（可选）
# ---------------------------------------------------------------------------
def rewrite_query(query: str) -> List[str]:
    """生成查询改写变体，用于扩展召回。

    默认关闭：每次改写都要多一次大模型调用，即**多一次完整往返**（实测 1~2 秒）。
    本项目当前的目标是把首字延迟压到最低，这点召回提升不足以抵偿，故默认不开。
    仅在 config.QUERY_REWRITE_ENABLED 打开时生效。

    Returns:
        改写后的查询列表（不含原查询）；失败或未启用时返回空列表。
    """
    if not QUERY_REWRITE_ENABLED or not query.strip():
        return []
    try:
        from langchain_core.prompts import ChatPromptTemplate

        from app.core.llm_factory import get_chat_model
        from app.core.prompts import get as get_prompt

        prompt = ChatPromptTemplate.from_template(get_prompt("query_rewrite"))
        raw = (prompt | get_chat_model()).invoke({"query": query}).content
        variants = [line.strip(" \t-·*") for line in raw.splitlines()]
        return [v for v in variants if v and v != query][:2]
    except Exception as exc:  # noqa: BLE001
        logger.warning("查询改写失败，回退为原查询：%s", exc)
        return []


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------
def rrf_fuse(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    weights: Optional[Sequence[float]] = None,
    k: int = RRF_K,
) -> Dict[str, float]:
    """对多路排名做 RRF 融合。

    Args:
        ranked_lists: 每路一个已排序列表，元素为 (doc_key, 该路原始分)。
                      列表必须按相关性降序排列，排名即下标 + 1。
        weights: 各路权重，缺省等权。
        k: 平滑常数，默认 60。

    Returns:
        {doc_key: rrf_score}，按分数降序排列的字典。
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused: Dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, (doc_key, _score) in enumerate(ranked, start=1):
            fused[doc_key] = fused.get(doc_key, 0.0) + weight / (k + rank)
    return dict(sorted(fused.items(), key=lambda item: -item[1]))


# ---------------------------------------------------------------------------
# 来源白名单过滤（按来源/部门做知识隔离）
# ---------------------------------------------------------------------------
def filter_by_allowed_sources(
    docs: List[Dict[str, Any]],
    allowed_sources: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """按来源白名单（fnmatch 模式）过滤检索结果。

    Args:
        allowed_sources: None 表示不过滤；空列表 ``[]`` 表示全部拒绝；
            非空列表为 fnmatch 模式（如 ``"hr/*"``、``"finance/report.xlsx"``），
            命中任一模式的片段才保留。

    Returns:
        过滤后的片段列表（新列表，不修改入参）。
    """
    if allowed_sources is None:
        return docs
    patterns = list(allowed_sources)
    if not patterns:
        return []
    out: List[Dict[str, Any]] = []
    for d in docs:
        src = d.get("source") or ""
        if any(fnmatch.fnmatch(src, p) for p in patterns):
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# 主检索入口
# ---------------------------------------------------------------------------
def _hit_key(hit) -> str:
    """向量命中的检索键：(source, chunk_index)。

    用位置而非内容做 key，比哈希内容更省内存，且天然区分「内容相同但来源不同」
    的片段。**必须复用 ``lexical.chunk_key``** —— 词面索引写入时用的是同一个
    函数，两处算法一旦分叉，同一片段就会被认成两条，RRF 融合随之退化成
    「两份互不相干的结果」。
    """
    return chunk_key(hit.metadata or {})


def _attach_parent_content(results: List[Dict[str, Any]]) -> None:
    """命中子块后回捞父块正文，供生成端消费更完整的上下文（T6-1）。

    只**新增** `parent_content` 字段，不改动 `content` / `score` / `fused` ——
    排序、阈值判定、去重、评测、前端展示全部沿用子块结果，因此该特性
    开启与否对检索层指标零影响，只影响喂给 LLM 的上下文完整度。

    回捞失败（父块缺失或存储损坏）时静默跳过，退回子块内容，
    等价于该特性未启用 —— 绝不因旁路存储问题打断检索主链路。
    """
    from app.rag import parent_store

    for item in results:
        pid = (item.get("metadata") or {}).get("parent_id") or ""
        if not pid:
            continue
        try:
            content = parent_store.get_content(pid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("父块回捞失败（parent_id=%s）：%s", pid, exc)
            continue
        if content and content != item.get("content"):
            item["parent_content"] = content


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    allowed_sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """执行 L3 完整检索流程：三路并发召回 → RRF 融合 → 阈值过滤 → 去重 → 软回退。

    三路召回（借鉴 dify 的 ThreadPoolExecutor 多路并发 + as_completed 早错早停）：
        - dense（线程1）：向量路，embed query 后 store.search；
        - lexical（线程2）：词面路，独立倒排索引 BM25 召回（不再依附向量候选集）；
        - rewrite（线程3）：查询改写变体（默认关闭，见 rewrite_query）。

    阈值过滤用 fused 分（而非向量 score）：
        词面路独立后，一个词面高分、向量低分的片段 fused 分应偏高，
        若仍用向量分过滤会把这类优质片段误杀——这是词面路独立的配套修复。

    Args:
        allowed_sources: 来源（文档 source 元数据）白名单。非空时只返回
            `source` 在该集合内的片段（按来源/部门做知识隔离），其余一律剔除；
            传空列表 `[]` 表示「全部拒绝」。默认 None 不做来源过滤。

    Returns:
        [{content, source, score, lexical, fused, fallback, ...}, ...]
        - score: 向量语义分（保留原始语义信号，供上层展示）
        - lexical: 词面匹配分
        - fused: RRF 融合分（最终排序与阈值过滤依据）
        - fallback: 是否来自软回退
    """
    top_k = top_k or config.SIMILARITY_TOP_K
    store = get_vector_store()
    if store.count() == 0:
        logger.warning("知识库为空，无法检索")
        return []

    embeddings = get_embeddings()
    threshold = config.effective_score_threshold()
    fallback_min = config.effective_fallback_min()
    candidate_k = min(max(top_k * 4, 30), store.count())

    # query 向量在主线程统一计算（而非 dense 子线程内）：这样它能通过 contextvars
    # 存进请求上下文，供后续「动态路由 prototype 打分」复用——同一个 query 只 embed 一次，
    # 省掉路由侧一次约 400ms 的重复 embedding 调用。
    query_vector = get_query_vector(query)
    if query_vector is None:
        query_vector = embeddings.embed_query(query)
        set_query_vector(query, query_vector)

    # ---------- 1. 三路并发召回 ----------
    # dense 路：向量检索 + 附带计算每条的词面分（供展示，不参与独立召回）
    def _dense_recall() -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Dict[str, Dict[str, Any]]]:
        raw_hits = store.search(query_vector, k=candidate_k)
        dense_pool: List[Tuple[str, float]] = []
        dense_lexical_pool: List[Tuple[str, float]] = []
        meta_by_key: Dict[str, Dict[str, Any]] = {}
        for hit in raw_hits:
            key = _hit_key(hit)
            dense_pool.append((key, float(hit.score)))
            lex = lexical_score(query, hit.content)
            if lex >= _LEXICAL_FLOOR:
                dense_lexical_pool.append((key, lex))
            meta_by_key[key] = {
                "content": hit.content,
                "source": hit.metadata.get("source", "unknown"),
                "score": round(float(hit.score), 4),
                "lexical": round(lex, 4),
                "chunk_index": hit.metadata.get("chunk_index"),
                # 透传完整 chunk 元数据（含 heading_path/chapter/section），
                # 供 L5 评测做章节命中（expect_section）判定，零额外检索成本。
                "metadata": dict(hit.metadata or {}),
            }
        dense_pool.sort(key=lambda item: -item[1])
        dense_lexical_pool.sort(key=lambda item: -item[1])
        return dense_pool, dense_lexical_pool, meta_by_key

    # lexical 路：独立倒排索引召回，返回 (doc_key, bm25分)
    def _lexical_recall() -> List[Tuple[str, float]]:
        index = get_lexical_index()
        return index.search(query, top_k=candidate_k)

    def _rewrite_recall() -> List[Tuple[str, float]]:
        """改写路（默认关闭）：对每个改写变体做向量召回，归并进词面分数池。"""
        variants = rewrite_query(query)
        if not variants:
            return []
        out: List[Tuple[str, float]] = []
        for variant in variants:
            try:
                vec = embeddings.embed_query(variant)
                for hit in store.search(vec, k=candidate_k):
                    out.append((_hit_key(hit), float(hit.score)))
            except Exception:  # noqa: BLE001
                continue
        out.sort(key=lambda item: -item[1])
        return out

    dense_pool: List[Tuple[str, float]] = []
    dense_lexical_pool: List[Tuple[str, float]] = []
    lexical_pool: List[Tuple[str, float]] = []
    rewrite_pool: List[Tuple[str, float]] = []
    meta_by_key: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_dense_recall): "dense",
            pool.submit(_lexical_recall): "lexical",
            pool.submit(_rewrite_recall): "rewrite",
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                # 单路失败不阻断整体：哪路挂了就缺哪路，其余照常融合
                logger.warning("召回路 %s 失败，已跳过：%s", name, exc)
                continue
            if name == "dense":
                dense_pool, dense_lexical_pool, meta_by_key = result
            elif name == "lexical":
                lexical_pool = result
            else:
                rewrite_pool = result

    # 词面路独立召回后，需要把向量路里算出的词面分与 BM25 路合并：
    # 词面分数池 = 向量候选集上的词面重打分（保留精确子串奖励）∪ 倒排索引 BM25 命中。
    # 两条词面子路取各自分数，统一进 RRF 按排名融合。
    lexical_merged: Dict[str, float] = {}
    for key, lex in dense_lexical_pool:
        lexical_merged[key] = max(lexical_merged.get(key, 0.0), lex)
    for key, bm25 in lexical_pool:
        # BM25 分无上界，与 0~1 的词面分不在同一量纲，但 RRF 只用排名，量纲无关。
        lexical_merged[key] = max(lexical_merged.get(key, 0.0), bm25)
    lexical_pool = sorted(lexical_merged.items(), key=lambda item: -item[1])

    # 词面路独立命中的片段，其内容/元数据不在 meta_by_key 里（它们没进向量候选集），
    # 从倒排索引补全。
    for key, _ in lexical_pool:
        if key not in meta_by_key:
            index = get_lexical_index()
            meta = index.doc_meta(key)
            meta_by_key[key] = {
                "content": index.doc_text(key),
                "source": meta.get("source", "unknown"),
                "score": 0.0,
                "lexical": round(lexical_merged[key], 4),
                "chunk_index": meta.get("chunk_index"),
                # 词面路命中的片段，其元数据取自词面索引（index_chunks 已同步写入
                # heading_path/chapter/section），保证与向量路一致可查章节。
                "metadata": dict(meta or {}),
            }

    # 改写路命中并入 dense 池（改写本质是语义召回，归 dense 侧）
    if rewrite_pool:
        dense_pool = sorted(dense_pool + rewrite_pool, key=lambda item: -item[1])

    # ---------- 2. RRF 融合 ----------
    fused = rrf_fuse([dense_pool, lexical_pool], weights=[DENSE_WEIGHT, LEXICAL_WEIGHT])

    logger.info(
        "L3 检索 | query=%s | dense=%d lexical=%d rewrite=%d | embedding=%s threshold=%s",
        preview(query, 50), len(dense_pool), len(lexical_pool), len(rewrite_pool),
        embeddings.__class__.__name__, threshold,
    )

    # ---------- 3. 阈值过滤（用 fused 分）----------
    ranked_keys: List[str] = list(fused.keys())
    if threshold is not None:
        # fused 是 RRF 分，无统一阈值尺度；这里沿用「向量分阈值」语义会误杀词面路。
        # 故仅在用户显式配置 SCORE_THRESHOLD 时按 fused 过滤：
        # 与旧逻辑的区别是过滤依据从 score 换成 fused，让词面高分片段也能通过。
        ranked_keys = [key for key in ranked_keys if fused[key] >= threshold]
        logger.info("阈值过滤后剩余 %d 条（threshold=%.2f）", len(ranked_keys), threshold)

    # ---------- 4. 去重（仅去完全相同内容，不做同源合并）----------
    # 同源合并会把同一文档的不同章节（年休假 / 发薪）误并为一，导致具体问答
    # 召回错误片段——这是之前踩过的坑，此处只去「内容完全一致」的重复项。
    seen_content: set = set()
    deduped: List[str] = []
    for key in ranked_keys:
        content_key = meta_by_key[key]["content"].strip()
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        deduped.append(key)

    # ---------- 5. 组装结果 ----------
    ordered_docs: List[Dict[str, Any]] = []
    for key in deduped:
        item = dict(meta_by_key[key])
        item["fused"] = round(fused[key], 6)
        item["fallback"] = False
        ordered_docs.append(item)
    # 来源白名单：剔除无权限访问的片段（按来源/部门做知识隔离）。
    ordered_docs = filter_by_allowed_sources(ordered_docs, allowed_sources)

    # 可选 rerank 精排（默认关闭）：cross-encoder 对 top-N 候选二次评分重排。
    # 只改顺序 + 附 rerank_score，不改 fused——置信度仍基于原始 RRF 分，保持跨模型可比。
    if config.RERANK_ENABLED:
        from app.rag.rerank import rerank_docs

        ordered_docs = rerank_docs(query, ordered_docs, top_n=config.RERANK_TOP_N)

    results = ordered_docs[:top_k]

    # ---------- 6. 软回退 ----------
    # 无结果时（阈值过滤过严），退回向量 Top 候选作为兜底，但保留抑制门槛。
    if not results and dense_pool:
        best_vec = max((s for _k, s in dense_pool), default=0.0)
        if best_vec < fallback_min:
            logger.info(
                "软回退已抑制：Top1 向量分 %.4f 低于最小相关度 %.2f，判定知识库无答案",
                best_vec, fallback_min,
            )
            return []
        logger.info("阈值过滤后无结果，触发软回退：top1_vec=%.4f", best_vec)
        for key, _vec in dense_pool[: min(3, len(dense_pool))]:
            if key in meta_by_key:
                item = dict(meta_by_key[key])
                # 必须写入**该片段真实的 RRF 融合分**，而不是 0.0。
                #
                # estimate_confidence 用 fused / fused_max 归一化得置信度，写 0 会让
                # 归一化结果恒为 0；紧接着的软回退降权是 `base *= 0.6`，
                # 而 0 × 0.6 仍然是 0 —— 降权逻辑形同虚设，置信度必然跌破
                # REFUSE_THRESHOLD，表现为「明明命中了 3 条却拒答」。
                #
                # 这些 key 都来自 dense_pool，已全量参与上面的 rrf_fuse，
                # 因此 fused 字典里一定有其真实分数，直接取用即可（无需重算）。
                item["fused"] = round(fused.get(key, 0.0), 6)
                item["fallback"] = True
                results.append(item)
        # 软回退结果同样受来源白名单约束。
        results = filter_by_allowed_sources(results, allowed_sources)

    # ---------- 7. 父子回捞（T6-1，默认关闭）----------
    if config.PARENT_CHUNK_ENABLED and results:
        _attach_parent_content(results)

    if results:
        logger.info(
            "L3 检索完成：返回 %d 条 | Top1 来源=%s vec=%.4f lex=%.2f fused=%.4f",
            len(results), results[0]["source"].split("/")[-1],
            results[0]["score"], results[0]["lexical"], results[0]["fused"],
        )
    return results


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """语义检索（供知识库管理面板调试使用）。"""
    return retrieve(query, top_k=top_k)


def retrieval_stats() -> Dict[str, Any]:
    """检索层配置快照（供 /stats 展示与 L5 调参归因）。"""
    return {
        "rrf_k": RRF_K,
        "dense_weight": DENSE_WEIGHT,
        "lexical_weight": LEXICAL_WEIGHT,
        "query_rewrite_enabled": QUERY_REWRITE_ENABLED,
        "top_k": config.SIMILARITY_TOP_K,
        "score_threshold": config.effective_score_threshold(),
        "fallback_min_score": config.effective_fallback_min(),
    }
