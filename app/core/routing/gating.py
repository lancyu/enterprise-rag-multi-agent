"""层③ 门控 —— 回答"够不够格"（地板）与"赢不赢得干脆"（边际）两个问题。

它**不调模型、不做 IO**，只有两个比较。正因为便宜，它在漏斗里被复用了两次：
词面打分之后一次（决定要不要花一次 embedding 升级到语义），
融合打分之后再一次（决定要不要花一次 LLM 调用升级到仲裁）。
同一个函数、同一套标准，两层判定不会漂移。

两道门的职责必须分开
--------------------
========  ==========================================  ==============================
门        问的问题                                     用哪个分
========  ==========================================  ==============================
地板      这个分**本身**算不算"识别出来了"               **绝对**证据
边际     第一名与**次优通道**差得够不够开                **绝对**证据之差
========  ==========================================  ==============================

⚠️ 两条都用绝对分，一条都不能用归一化分或 RRF 分：

- 地板用归一化分 → top1 恒为 1.0，地板永远通过，等于没有地板（D3）。
- 边际用 RRF 分 → RRF 只由**排名**决定，rank1 与 rank2 的差恒为
  ``1/(k+1) ≈ 1.6%``，与"领先一大截"还是"咬得很紧"完全无关。
  拿它比阈值，要么永远进灰区、要么永远不进——两个都不是"边际"想表达的意思。

为什么归并到**通道**层级（I2）
------------------------------
按**能力**判胶着是自找的灰区。``employee_attr`` 与 ``leave_balance`` 同属
``tool``，它们打平时**不该进灰区**：无论判给谁，都进同一个 function calling
循环，由模型读工具描述自愈（``tool_agent.py`` 的"不做收窄"）。
归并到通道后这类胶着自然消失。

⚠️ 但要说清它救的是**哪一类**：``simple_rag`` 与 ``complex_rag`` 是**两个不同
通道**（走不同的链、成本不同），它们打平**确实**要进灰区。归并规则是对的，
只是别指望它顺带解决跨通道的胶着——那正是层④ 存在的理由。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from app.core.routing.fusion import Candidate

#: 灰区原因（闭集）。**"灰区"不是"故障"**，它是"我不确定"的正确表达，
#: 与 ``degraded`` 是两个字段——把两者混用会让 ``true_degrade_rate`` 彻底失去意义。
GRAY_NO_CANDIDATE = "no_candidate"
GRAY_LOW_FLOOR = "low_floor"
GRAY_TIGHT_MARGIN = "tight_margin"


@dataclass(frozen=True)
class GateResult:
    """门控结论。无论通过与否，都把**过程数据**带出去（预演接口要用）。"""

    accepted: bool
    #: 通过的候选；被拒时是"最接近通过的那个"（排障时最想知道的就是它）。
    top: Optional[Candidate]
    #: 通道层归并后的候选，按**本层绝对证据**降序。长度 ≥ 2 才有"边际"可言。
    ranked: Tuple[Candidate, ...]
    #: 未通过的原因（``None`` 表示通过）。取值域见上面三个常量。
    reason: Optional[str]
    floor: float
    margin: float
    #: top 与次优通道的**绝对证据差**；只有一个通道时无竞争者，记 ``None``。
    #: 这个数是标定阈值时唯一有用的东西——没有它，"为什么进灰区"永远只能靠猜。
    gap: Optional[float]

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "floor": round(self.floor, 4),
            "margin": round(self.margin, 4),
            "gap": None if self.gap is None else round(self.gap, 4),
        }


def gate(candidates: Sequence[Candidate], *, floor: float, margin: float) -> GateResult:
    """在通道层级做地板 + 边际判定。命中即返回，**不抛异常**。

    Args:
        candidates: 融合后的候选表（全量，可含未参与的 0 分能力）。
        floor: 绝对证据的地板。调用方按**当前层**选择量纲：
            词面层用 ``config.ROUTE_LEXICAL_FLOOR``，
            语义层用 ``config.effective_route_semantic_floor()``。
        margin: 绝对证据的边际。同上，用对应层的值。

    排序口径：**按本层绝对证据 ``abs_score`` 降序，融合分只作并列裁决**。

    为什么排序不能按融合分：排序与门控必须用**同一个量纲**，否则会出现
    "融合分选出来的 top，在语义上却输给第二名"——此时算出的 gap 是负数，
    边际判定永远不通过，本该直接判对的一律掉进灰区。
    实测过这个坑：「公司有哪些部门」在词面只有 employee_attr 命中一个泛化词
    「部门」，但语义上 policy_single 因为收了这条 utterance 而遥遥领先；
    按融合分排序时 top 是 employee_attr（它在两路里都占名次），gap 为负 →
    白白多花一次 LLM 仲裁。按绝对证据排序则直接判对。
    """
    rated = [c for c in candidates if c.has_evidence]
    if not rated:
        return GateResult(False, None, (), GRAY_NO_CANDIDATE, floor, margin, None)

    # 按通道归并：候选已按 (绝对证据, 融合分, 名字) 降序，每通道首次出现即该通道最优。
    best_by_channel: dict = {}
    for cand in sorted(rated, key=lambda c: (-c.abs_score, -c.fused, c.name)):
        best_by_channel.setdefault(cand.channel, cand)
    ranked: Tuple[Candidate, ...] = tuple(best_by_channel.values())

    top = ranked[0]
    gap: Optional[float] = None
    if len(ranked) > 1:
        gap = top.abs_score - ranked[1].abs_score

    if top.abs_score < floor:
        return GateResult(False, top, ranked, GRAY_LOW_FLOOR, floor, margin, gap)
    if gap is not None and gap < margin:
        return GateResult(False, top, ranked, GRAY_TIGHT_MARGIN, floor, margin, gap)
    return GateResult(True, top, ranked, None, floor, margin, gap)
