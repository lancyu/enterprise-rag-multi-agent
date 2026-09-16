"""漏斗编排 —— 把四层串起来，并保证"永不抛异常、每一格降级都有定义"。

主流程（惰性升级：每一步只在前一步**判不了**时才启动）
----------------------------------------------------
::

    ① 锚定（零成本）            命中 → 结束
    ②a 词面打分（零成本）        门控通过 → 结束，一次 embedding 都不花
    ②b 语义打砠（一次 embedding） 门控通过 → 结束        ← 只在 ②a 判不了时才启动
    ③  门控（随 ②a/②b 各跑一次）
    ④  灰区 LLM 仲裁（一次模型）  ← 只在 ②b 也判不了时才启动

**"判不了"的判定标准是同一个函数**（``gating.gate``），只是喂给它的候选
从"只有词面一路"变成"词面 + 语义两路"。这样"该不该升级"不需要另写一套逻辑，
两层判定标准不可能漂移。

时间预算是"**不许开始**"，并给出**可以证明的总上界**
----------------------------------------------------
每一步动手之前先算 ``剩余 = 截止时刻 - 现在``，剩余不足就不启动这一步；
启动仲裁时再把它的等待上限收窄到 ``min(自身上限, 剩余)``。
于是有一条能写进文档的硬保证：**路由总耗时 ≤ ``ROUTE_BUDGET_MS``**。

方向很重要：如果做成"超时后改走更快的兜底"，看起来等价、实则相反——
**层④ 是整条链路里最慢的一步**。用一个已经超时的预算去换一次更慢的模型调用，
等于把延迟再放大一截。所以预算只做减法：**它永远不会额外发起一次调用。**
（首版没有把剩余预算传下去，实测出现过 3.2s 的一轮——预算形同虚设。）

``degraded`` 与 ``gray_reason`` 是两个字段，不能混
--------------------------------------------------
灰区是"我拿不准"的**正确表达**，不是故障；降级是"本该做的事没做成"。
把两者并成一条，``true_degrade_rate`` 这个指标就彻底失去意义了
（``_archive`` 里的教训：失败兜底不得谎报状态）。

``source`` 取值域是闭集
-----------------------
``anchor``（层① 确定性）/ ``lexical``（词面直接通过）/ ``fused``（融合后通过）
/ ``arbitration``（灰区仲裁）/ ``fallback``（兜底）。
**如实记录是哪一层做的决定**，否则"路由为什么这么判"就永久失去了证据。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from app import config
from app.core.routing import anchors, arbitration, catalog, fusion, gating, vocabulary
from app.core.routing.catalog import (
    DEFAULT_SCENE,
    OUT_OF_SCOPE_ANSWER,
    SCENE_OUT_OF_SCOPE,
)
from app.core.routing.fusion import Candidate
from app.utils.logger import logger

#: ``source`` 闭集。前端徽章与评测都按它分支，**不要**塞入临时字符串。
SOURCE_ANCHOR = "anchor"
SOURCE_LEXICAL = "lexical"
SOURCE_FUSED = "fused"
SOURCE_ARBITRATION = "arbitration"
SOURCE_FALLBACK = "fallback"
SOURCES: Tuple[str, ...] = (
    SOURCE_ANCHOR, SOURCE_LEXICAL, SOURCE_FUSED, SOURCE_ARBITRATION, SOURCE_FALLBACK,
)

#: 预算耗尽的灰区原因。与 ``gating`` 的三个灰区原因并列——它同样**不是故障**，
#: 而是"这一次没能拿到足够证据"的一种如实说法（同时会把 ``degraded`` 置真）。
GRAY_BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class RoutingDecision:
    """一次路由的完整结论。``channel`` 是**唯一驱动图分支**的字段，其余都是给人看的。"""

    channel: str = DEFAULT_SCENE
    #: 命中的能力名（开集）。``None`` 表示"只判到通道级"（如显式标识符锚点）。
    capability: Optional[str] = None
    #: 由哪一层做出的决定，见模块 docstring 的闭集说明。
    source: str = SOURCE_FALLBACK
    #: 本轮原因的自然语言描述，进日志与响应，供人复盘。
    reason: str = ""
    #: 真降级（本该做的事没做成）。**与灰区严格区分。**
    degraded: bool = False
    #: 灰区原因（``no_candidate`` / ``low_floor`` / ``tight_margin`` / ``budget_exceeded``）。
    gray_reason: Optional[str] = None
    #: 完整候选得分表——没有它，"为什么进灰区"只能靠猜。预演接口直接返回它。
    candidates: Tuple[Candidate, ...] = ()
    #: 本次生效的阈值。**必须一并返回**：看到 ``low_floor`` 却不知道阈值是多少，
    #: 就无法区分"分低"与"阈值配错了"。
    floor: Optional[float] = None
    margin: Optional[float] = None
    #: top 与次优通道的绝对证据差。标定阈值时唯一有用的数。
    gap: Optional[float] = None
    elapsed_ms: int = 0
    error: Optional[str] = None
    #: 越界通道的对外话术（只有 ``out_of_scope`` 才有值）。
    out_of_scope_answer: Optional[str] = None
    #: 语义路是否**尝试过并失败**。与 ``degraded`` 分开，是为了让
    #: "embedding 挂了"这个具体事实在观测里一眼可见。
    semantic_attempted: bool = False

    def to_dict(self, *, explain: bool = False) -> dict:
        data = {
            "channel": self.channel,
            "capability": self.capability,
            "source": self.source,
            "reason": self.reason,
            "degraded": self.degraded,
            "gray_reason": self.gray_reason,
            "floor": None if self.floor is None else round(self.floor, 4),
            "margin": None if self.margin is None else round(self.margin, 4),
            "gap": None if self.gap is None else round(self.gap, 4),
            "elapsed_ms": self.elapsed_ms,
        }
        if explain:
            data["candidates"] = [c.to_dict() for c in self.candidates]
            # 只给预演接口用的两个字段：它们是"这次花了什么代价"的直接答案，
            # 生产响应体里不需要（生产看 route_decision 的 source/degraded 就够）。
            # 挂在 explain 下而不是无条件返回：`to_dict()` 也被 nodes.py 用来
            # 组装生产 state，改它的形状等于动生产契约。
            data["semantic_attempted"] = self.semantic_attempted
            data["out_of_scope_answer"] = self.out_of_scope_answer
        return data


def match_intent(
    query: str,
    *,
    model: Optional[Any] = None,
    specs: Optional[Sequence[catalog.IntentSpec]] = None,
    vocab: Optional[vocabulary.Vocabulary] = None,
    budget_ms: Optional[int] = None,
    embed_timeout_ms: Optional[int] = None,
) -> RoutingDecision:
    """把一句话判给五个通道之一。**永不抛异常。**

    参数里的 ``model`` / ``specs`` / ``vocab`` / ``budget_ms`` / ``embed_timeout_ms``
    都是可注入的：路由是"可判错但要可验证"的组件，测试必须能替换掉每一个外部依赖，
    否则只能测到"在某个特定环境下的行为"，那不是回归护栏。

    ``specs`` 与 ``vocab`` 一起构成一个**领域包**：能力目录 + 领域词表。
    换项目就是换这两样，本模块及以下（anchors / fusion / gating）一行不改。

    Returns:
        :class:`RoutingDecision`。任何一层的失败都被下一层接住，
        最后一层是"默认通道 + 如实标记降级"。
    """
    started = time.perf_counter()
    budget = config.ROUTE_BUDGET_MS if budget_ms is None else budget_ms
    deadline = started + max(budget, 0) / 1000.0

    def remaining_ms() -> float:
        """距截止时刻还剩多少毫秒。负数表示已超预算。"""
        return (deadline - time.perf_counter()) * 1000.0

    embed_limit = config.ROUTE_EMBED_TIMEOUT_MS if embed_timeout_ms is None else embed_timeout_ms
    text = (query or "").strip()

    if not text:
        return _decision(
            channel=DEFAULT_SCENE, source=SOURCE_FALLBACK, started=started,
            reason="空提问", degraded=True, gray_reason=gating.GRAY_NO_CANDIDATE,
        )

    # 领域包的两半都在这里落地，且**必须在下面任何判定之前**取好：
    # 层① 就要用 specs 去反查能力归属，不能等到 try 块里。
    specs = specs if specs is not None else catalog.all_specs()
    vocab = vocab if vocab is not None else catalog.vocabulary()

    # ── ① 确定性锚定：零成本、离线、逐条可单测 ────────────────────────────
    hit = anchors.match(text, vocab)
    if hit is not None:
        # 层① 只给 (通道, 锚点名) —— 能力名由**目录**反查，引擎不认识它。
        # 没有任何能力声明该锚点时（如"含显式标识符"）得到 None，这是正常情况：
        # 它只回答"要不要走 tool 通道"，具体能力交给 tool 通道内部处理（D8）。
        #
        # ⚠️ 传 specs 而不是让它查内置目录：注入了别的目录时，
        # 反查必须跟着换，否则会解析到**另一个领域的能力**（不报错，只是判错）。
        spec = catalog.spec_by_anchor(hit.anchor_name, specs) if hit.anchor_name else None
        capability = spec.name if spec else None
        logger.info("路由锚定：channel=%s capability=%s（%s）", hit.channel, capability, hit.reason)
        return _decision(
            channel=hit.channel, capability=capability, source=SOURCE_ANCHOR,
            started=started, reason=f"锚点：{hit.reason}",
        )

    try:
        # ── ②a 词面打分（免费）───────────────────────────────────────────
        lexical = fusion.score_lexical(text, specs, vocab)
        candidates = fusion.fuse(lexical, None, specs=specs)
        result = gating.gate(
            candidates,
            floor=config.ROUTE_LEXICAL_FLOOR,
            margin=config.ROUTE_LEXICAL_MARGIN,
        )
        if result.accepted:
            return _accepted(result, SOURCE_LEXICAL, started, "词面")

        # ── ②b 语义打分（惰性升级：只有 ②a 判不了才走到这里）──────────────
        degraded = False
        #: 语义路是否**真的尝试过**。与 ``degraded`` 是两个问题：
        #: 预算不足时会「没尝试且降级」，embedding 成功时会「尝试过且没降级」。
        #: 面板上"这次连 embedding 都没算"这句话，就是靠它说出来的。
        semantic_attempted = False
        if config.ROUTE_SEMANTIC_ENABLED:
            left = remaining_ms()
            if left <= 0:
                # 预算是"不许开始"：不启动这一步，并且**如实记为降级**——
                # 它确实是"本该做而没做"，与"灰区"不是一回事。
                degraded = True
                logger.info("路由：已用尽 %dms 预算，跳过语义层", budget)
            else:
                semantic_attempted = True
                try:
                    semantic = fusion.score_semantic(
                        text, specs, timeout_ms=_clamp_timeout(embed_limit, left), vocab=vocab
                    )
                    candidates = fusion.fuse(lexical, semantic, specs=specs)
                    result = gating.gate(
                        candidates,
                        floor=config.effective_route_semantic_floor(),
                        margin=config.effective_route_semantic_margin(),
                    )
                    if result.accepted:
                        return _accepted(
                            result, SOURCE_FUSED, started, "词面+语义融合",
                            semantic_attempted=True,
                        )
                except fusion.SemanticUnavailable as exc:
                    # D5：语义路不可用**不**判整条路由失败，退回词面结论继续走层④。
                    # 不这样做的话，一次 embedding 故障会让所有请求一起掉进灰区，
                    # 把依赖抖动放大成路由全面降级。
                    degraded = True
                    fusion.soft_warn(f"语义层不可用，按词面继续：{exc}")

        # ── ④ 灰区仲裁（唯一的模型调用点）────────────────────────────────
        gray_reason = result.reason
        if not config.ROUTE_ARBITRATION_ENABLED:
            return _decision(
                channel=DEFAULT_SCENE, source=SOURCE_FALLBACK, started=started,
                reason=f"灰区（{gray_reason}）且仲裁已关闭，走保守兜底",
                degraded=True, gray_reason=gray_reason, gate=result,
                semantic_attempted=semantic_attempted,
            )
        left = remaining_ms()
        if left <= 0:
            return _decision(
                channel=DEFAULT_SCENE, source=SOURCE_FALLBACK, started=started,
                reason=f"灰区（{gray_reason}）且已用尽 {budget}ms 预算，不启动仲裁",
                degraded=True, gray_reason=GRAY_BUDGET_EXCEEDED, gate=result,
                semantic_attempted=semantic_attempted,
            )

        picked = arbitration.choose(
            text, result.ranked, model=model,
            timeout_ms=_clamp_timeout(config.ROUTE_ARBITRATION_TIMEOUT_MS, left),
        )
        if picked is not None:
            spec = catalog.spec_by_name(picked)
            channel = spec.channel if spec else DEFAULT_SCENE
            logger.info("路由灰区仲裁：channel=%s capability=%s", channel, picked)
            return _decision(
                channel=channel, capability=picked, source=SOURCE_ARBITRATION,
                started=started, reason=f"灰区仲裁（{gray_reason}）→ {picked}",
                degraded=degraded, gray_reason=gray_reason, gate=result,
                semantic_attempted=semantic_attempted,
            )

        return _decision(
            channel=DEFAULT_SCENE, source=SOURCE_FALLBACK, started=started,
            reason=f"灰区（{gray_reason}）且仲裁未能给出结论，走保守兜底",
            degraded=True, gray_reason=gray_reason, gate=result,
            semantic_attempted=semantic_attempted,
        )
    except Exception as exc:  # noqa: BLE001
        # 路由**不能**是本轮失败的原因：它是一切路径的入口，入口抛异常
        # 等于整轮无回答。任何未预料的异常都在这里收束成一次保守兜底。
        logger.warning("路由漏斗异常，走保守兜底：%s", exc)
        return _decision(
            channel=DEFAULT_SCENE, source=SOURCE_FALLBACK, started=started,
            reason=f"路由异常兜底：{type(exc).__name__}", degraded=True,
            gray_reason=GRAY_BUDGET_EXCEEDED, error=str(exc),
        )


def _clamp_timeout(ceiling_ms: float, remaining_ms: float) -> int:
    """把一步的等待上限收窄到 ``min(自身上限, 剩余预算)``。

    至少留 1ms：若收窄成 0 或负数，``call_with_timeout`` 会**立刻**超时，
    表现为"这个配置项从来不起作用"——那是最难查的一类配置 bug。
    """
    return max(int(min(ceiling_ms, remaining_ms)), 1)


def _accepted(
    result: gating.GateResult,
    source: str,
    started: float,
    stage: str,
    *,
    semantic_attempted: bool = False,
) -> RoutingDecision:
    """把"通过门控"的结果转成决策，并把**它凭什么赢**写进 reason。"""
    top = result.top
    gap = "无竞争者" if result.gap is None else f"领先 {result.gap:.3f}"
    logger.info(
        "路由%s通过：channel=%s capability=%s（绝对分 %.3f，%s）",
        stage, top.channel, top.name, top.abs_score, gap,
    )
    return _decision(
        channel=top.channel, capability=top.name, source=source, started=started,
        reason=f"{stage}通过：{top.name}（{gap}）", gate=result,
        semantic_attempted=semantic_attempted,
    )


def _decision(
    *,
    channel: str,
    source: str,
    started: float,
    reason: str,
    capability: Optional[str] = None,
    degraded: bool = False,
    gray_reason: Optional[str] = None,
    gate: Optional[gating.GateResult] = None,
    error: Optional[str] = None,
    semantic_attempted: bool = False,
) -> RoutingDecision:
    """统一的决策构造：``elapsed_ms`` 与越界话术都在这里落定，避免各处漏填。"""
    if source not in SOURCES:
        # 兜底而不是抛异常：source 只影响观测，不该有能力阻断一次真实回答。
        logger.warning("未知的路由 source=%r，按 fallback 记录", source)
        source = SOURCE_FALLBACK
    return RoutingDecision(
        channel=channel,
        capability=capability,
        source=source,
        reason=reason,
        degraded=degraded,
        gray_reason=gray_reason,
        candidates=tuple(gate.ranked) if gate else (),
        floor=gate.floor if gate else None,
        margin=gate.margin if gate else None,
        gap=gate.gap if gate else None,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        error=error,
        out_of_scope_answer=OUT_OF_SCOPE_ANSWER if channel == SCENE_OUT_OF_SCOPE else None,
        semantic_attempted=semantic_attempted,
    )


def describe_catalog() -> List[dict]:
    """给预演接口用的目录快照：把"规则是怎么配的"也暴露出去。

    没有它，看到一次误判只能反推规则；有了它，可以直接对照"这条规则当时长什么样"。
    """
    return [
        {
            "name": s.name,
            "channel": s.channel,
            "description": s.description,
            "keywords": list(s.keywords),
            "utterances": list(s.utterances),
            "guards": list(s.guards),
            "anchors": list(s.anchors),
        }
        for s in catalog.all_specs()
    ]
