"""三个子 Agent —— 闲聊 / 简单 RAG / 复杂 RAG。

放在同一个模块里的原因：它们的差别只在**怎么取证据**（不取 / 取一次 / 取多次），
产出形状完全一致（``AgentAnswer``）。拆成三个文件会得到三份几乎相同的
dataclass 与三份 trace 写法，而它们必须保持一致——分开写必然漂移。

第五个 Agent（工具）在 ``app/core/tool_agent.py``：它的产出形状与这里一致，
但那套逻辑（多轮 function calling、参数护栏、正文回捞）体量完全不同，
混在一起会把这个文件撑成一个什么都装的口袋。

统一产出形状 ``AgentAnswer``
============================
**子 Agent 只负责取回证据或给出直答，不负责组织最终话术。** 三个字段分别承载：

- ``text`` 非 None → **直答出口**：本轮不需要 L4（只有闲聊会这样）。
  刻意的语义：``text=""`` 与 ``text=None`` 不是一回事——前者是"说了但内容是空的"，
  后者是"没打算说话"，把两者混成一个真值判断会让空回答被当成正常直答。
- ``docs`` 非空 → **证据出口**：交 L4 受控生成，以保留引用编号、置信度、
  拒答与流式输出（这四项能力只在 L4 一处产生，见 ``app/graph/workflow_graph.py``）。
- ``error`` 非空 → 本轮**无法自动处理**，调用方转人工。

什么时候 ``error`` 才算 error：判据与全项目一致——「**能不能从其他来源得到答案**」。
检索失败可降级（L4 的"没找到"与真实语义一致，记 ``soft_warnings``），
所以这里的 ``error`` 只留给"这个 Agent 本身不可用"这类情况。

闲聊 Agent 为什么**不调模型**
=============================
用户给这个 Agent 的定位是「不访问知识库、不调用任何工具，避免闲聊误触发 RAG 或
FunctionCall 而浪费 token」。模板直出是对这句话的最强落实：0 次模型调用、
0 次检索、0 幻觉。三个理由按重要性排序：

1. **零幻觉**：模板里不出现任何业务事实，也就不可能编造事实。用一个模型调用去
   生成"你好呀，有什么可以帮你"是拿幻觉风险换措辞变化，不划算。
2. **离线一致**：未配 Key 时 Mock 模型对闲聊会返回一段与 Context 有关的文本
   （见 ``app/providers/llm.py::_mock_answer``）。若闲聊走模型，用户会看到
   「你好」被回以「未在知识库中检索到与您问题相关的内容」——荒唐且难以排查。
3. **可枚举**：寒暄的回复内容是有限的几类（问候 / 致谢 / 道别 / 身份询问），
   模板覆盖得住。

代价照实说：措辞固定、不会因人因时变化；"你是谁"的回答是一张能力清单。
这是可接受的——能力清单本就该固定，它同时是**对模型的约束**（写死的清单不会
被模型发挥出不存在的能力）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app import config
from app.core.llm_access import content_of, default_model
from app.core.prompts import render as render_prompt
from app.core.rag_engine import retrieve_knowledge_docs
from app.core.tracing import span
from app.utils.logger import logger


# =============================================================
# 统一产出形状
# =============================================================
@dataclass
class AgentAnswer:
    """子 Agent 的统一产出。字段语义见模块 docstring。"""

    #: 非 None 表示直答（答案已就绪，不进 L4）；None 表示"没打算说话"。
    text: Optional[str] = None
    docs: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    #: 复杂 RAG 拆出的子问题（简单 RAG / 闲聊恒为空）。透出来是为了让
    #: 「模型把问题拆成了什么」可见——拆歪了是静默错误的主要来源。
    sub_queries: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    degraded: bool = False
    error: Optional[str] = None

    @property
    def is_direct(self) -> bool:
        return self.text is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direct": self.is_direct,
            "degraded": self.degraded,
            "docs": len(self.docs),
            "sub_queries": list(self.sub_queries),
            "steps": list(self.steps),
        }


# =============================================================
# 一、闲聊 Agent（模板直出，零模型调用）
# =============================================================
_SMALLTALK_IDENTITY = (
    "我是企业内部智能助手，可以帮你查两类信息：\n"
    "① 公司制度与流程（年假、报销、考勤、权限申请等）；\n"
    "② 员工基础信息与假期余额。\n"
    "直接说你要查什么就行。"
)
_SMALLTALK_GREETING = (
    "你好！我是企业内部智能助手，可以帮你查公司制度流程、员工信息"
    "和假期余额。有什么想问的，直接说就行。"
)
_SMALLTALK_THANKS = "不客气。还有别的内容想查，随时告诉我。"
_SMALLTALK_BYE = "好的，有需要随时找我。再见！"
_SMALLTALK_DEFAULT = (
    "我在。如果你想查公司制度、员工信息或假期余额，"
    "把问题说清楚就行；其他问题我可能帮不上忙。"
)

# 模板挑选用的正则。**与路由 Agent 的兜底正则不是同一件事**：
# 那边是"要不要把这个句子划进闲聊"（必须严格，错判会把业务问题打发掉），
# 这边是"已经在闲聊了，挑哪句回复更像话"（可以宽松，挑错也只是措辞不同）。
# 两者目的不同，故不共用；共用反而会让其中一方被另一方的约束绑住。
#
# ⚠️ 所以这里**故意**是与 ``app/core/routing/anchors.py`` 不同的写法
# （子串匹配、不锚定、不共享长度上限）。那边有护栏按文本扫描钉住"只许一处定义"，
# 扫描的是各自的**短语表特征片段**，本组不在其中——别把它当重复实现顺手并过去。
_IDENTITY_RE = re.compile(
    r"你是谁|你是什么|你是做什么的|你叫什么|介绍一下你|你能做什么|能帮(我)?做什么|你会什么|你有什么功能",
    re.I,
)
_THANKS_RE = re.compile(r"谢谢|多谢|感谢|辛苦了|thanks|thank you|thx|3q", re.I)
_BYE_RE = re.compile(r"再见|拜拜|bye|goodbye|see you|先这样|回头聊|下次聊", re.I)
_GREETING_RE = re.compile(r"你好|您好|\bhi\b|\bhello\b|\bhey\b|哈喽|嗨", re.I)


def _pick_smalltalk_reply(query: str) -> tuple:
    """选模板，返回 ``(模板名, 回复文本)``。顺序即优先级。"""
    text = (query or "").strip()
    if _IDENTITY_RE.search(text):
        return "identity", _SMALLTALK_IDENTITY
    if _THANKS_RE.search(text):
        return "thanks", _SMALLTALK_THANKS
    if _BYE_RE.search(text):
        return "bye", _SMALLTALK_BYE
    if _GREETING_RE.search(text):
        return "greeting", _SMALLTALK_GREETING
    return "default", _SMALLTALK_DEFAULT


def run_smalltalk_agent(query: str) -> AgentAnswer:
    """寒暄 / 身份询问的直答。**不读知识库、不调工具、不调模型。**"""
    with span("smalltalk") as s:
        name, reply = _pick_smalltalk_reply(query)
        s.attrs["template"] = name
    logger.info("闲聊 Agent 模板直出：template=%s", name)
    return AgentAnswer(
        text=reply,
        steps=[{"node": "smalltalk", "kind": "template", "detail": name, "elapsed_ms": 0}],
    )


# =============================================================
# 二、简单 RAG Agent（单次检索）
# =============================================================
def run_simple_rag_agent(
    query: str,
    allowed_sources: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> AgentAnswer:
    """单文档即可回答的制度类问题：一次混合检索，取回证据交 L4。

    检索失败**不**升级为本轮失败（``error`` 保持为空）：L4 会据此给出
    「知识库中没有找到相关信息」——那是一个诚实且可行动的回答，与真实语义一致。
    但失败必须记进 ``soft_warnings``：检索服务抖动与"知识库确实没这条"
    在返回值上完全同形（都是空列表），不留痕就等于没发生。
    """
    answer = AgentAnswer()
    with span("simple_rag") as s:
        try:
            docs = retrieve_knowledge_docs(query, top_k=top_k, allowed_sources=allowed_sources)
        except Exception as exc:  # noqa: BLE001
            # 可降级故障：答案仍能给出（L4 的"没找到"），但质量已下降。
            logger.warning("简单 RAG 检索失败（已降级为无依据作答）：%s", exc)
            answer.soft_warnings.append(f"检索失败已降级：{type(exc).__name__}: {exc}")
            answer.degraded = True
            docs = []
        answer.docs = docs
        s.attrs["hits"] = len(docs)
        s.attrs["degraded"] = answer.degraded
    answer.steps.append(
        {"node": "simple_rag", "kind": "retrieve", "detail": f"命中 {len(answer.docs)} 条", "elapsed_ms": 0}
    )
    logger.info("简单 RAG Agent 完成：命中 %d 条", len(answer.docs))
    return answer


# =============================================================
# 三、复杂 RAG Agent（拆解 → 多次检索 → 合并去重）
# =============================================================
def decompose_query(query: str, model: Optional[Any] = None) -> List[str]:
    """把复杂问题拆成 2~4 个自足子问题。

    **返回值的下界是 ``[query]``**：任何失败（模型不可用、输出不是 JSON 数组、
    只拆出一个）都退化成"就用原问题检索一次"。理由与路由 Agent 的兜底一致——
    拆解是一次增益，不该成为本轮的失败点；退化成单次检索，答案质量回到简单路径，
    但**用户仍然拿得到答案**。

    离线（未配 Key）时直接返回 ``[query]``，不浪费一次必然无意义的模型调用。
    """
    text = (query or "").strip()
    if not text:
        return []
    if model is None and not config.USE_REAL_LLM:
        logger.info("复杂 RAG：离线模式，跳过拆解，直接用原问题检索")
        return [text]

    try:
        chat = model or default_model()
        reply = chat.invoke(render_prompt("decompose", user_query=text))
        sub_queries = _parse_subqueries(content_of(reply))
    except Exception as exc:  # noqa: BLE001
        logger.warning("问题拆解失败，退化为单次检索：%s", exc)
        return [text]

    if len(sub_queries) < 2:
        logger.info("问题拆解未产出多个子问题，退化为单次检索")
        return [text]

    limit = max(1, config.COMPLEX_RAG_MAX_SUBQUERIES)
    trimmed = sub_queries[:limit]
    if len(sub_queries) > limit:
        logger.info("问题拆解产出 %d 个子问题，按上限截到 %d 个", len(sub_queries), limit)
    logger.info("复杂 RAG 拆解：%s", " | ".join(trimmed))
    return trimmed


def _parse_subqueries(raw: str) -> List[str]:
    """从模型输出里提取子问题列表（去重、去空、保序）。"""
    if not raw:
        return []
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return []
    if not isinstance(payload, list):
        return []
    out: List[str] = []
    seen = set()
    for item in payload:
        sub = str(item or "").strip()
        if not sub or sub in seen:
            continue
        seen.add(sub)
        out.append(sub)
    return out


def run_complex_rag_agent(
    query: str,
    allowed_sources: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    model: Optional[Any] = None,
    max_docs: Optional[int] = None,
) -> AgentAnswer:
    """需要跨文档对比 / 综合推理的问题：拆解 → 多次检索 → 合并去重 → 交 L4。

    两个刻意的设计，都是为了防"拆歪了还不自知"：

    1. **原问题也在检索集合里**。模型拆出的子问题若偏离原意，只按子问题检索会
       把最相关的片段整段漏掉，而 L4 拿着一堆"相关但不对"的片段照样能自信地
       生成答案——这是静默错误。多花一次检索，换"用户原话一定被检索过"。
    2. **子问题会透出到 state 与日志**，不吞在内部。拆解质量只有可见才可评估。

    与简单 RAG 相同：单次检索失败只记 ``soft_warnings``，不升级为本轮失败。
    """
    answer = AgentAnswer()
    max_docs = max_docs or config.COMPLEX_RAG_MAX_DOCS

    with span("complex_rag") as s:
        sub_queries = decompose_query(query, model=model)
        # 原问题始终参与检索（理由见 docstring 第 1 条）。
        search_queries = _dedupe_keep_order([query, *sub_queries])
        answer.sub_queries = search_queries

        merged: Dict[tuple, Dict[str, Any]] = {}
        failures = 0
        for sub in search_queries:
            try:
                docs = retrieve_knowledge_docs(sub, top_k=top_k, allowed_sources=allowed_sources)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.warning("复杂 RAG 子查询检索失败（跳过）：%s | %s", sub[:40], exc)
                answer.soft_warnings.append(
                    f"子查询检索失败已跳过：{type(exc).__name__}: {exc}"
                )
                continue
            for doc in docs:
                key = (str(doc.get("source", "")), str(doc.get("content", "")))
                prev = merged.get(key)
                # 同一片段被多个子问题召回：取较高的融合分。RRF 融合分跨查询
                # 同量纲（都是 1/(k+rank) 的累加），直接比大小是合理的；
                # 而向量分 score 跨查询不可比，故不参与择优。
                if prev is None or float(doc.get("fused", 0) or 0) > float(prev.get("fused", 0) or 0):
                    merged[key] = doc

        if failures and failures == len(search_queries):
            # 全部子查询都失败 → 与简单 RAG 同款降级：交给 L4 诚实作答。
            answer.degraded = True

        docs = sorted(merged.values(), key=lambda d: float(d.get("fused", 0) or 0), reverse=True)
        if len(docs) > max_docs:
            logger.info("复杂 RAG 合并后 %d 条，按上限截到 %d 条", len(docs), max_docs)
            docs = docs[:max_docs]
        answer.docs = docs
        s.attrs["sub_queries"] = len(search_queries)
        s.attrs["hits"] = len(docs)
        s.attrs["failures"] = failures

    answer.steps.append(
        {
            "node": "complex_rag",
            "kind": "retrieve_multi",
            "detail": f"{len(search_queries)} 个子查询 → 去重后 {len(answer.docs)} 条",
            "elapsed_ms": 0,
        }
    )
    logger.info(
        "复杂 RAG Agent 完成：子查询 %d 个，去重合并 %d 条", len(search_queries), len(answer.docs)
    )
    return answer


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
