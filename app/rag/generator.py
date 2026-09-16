"""RAG 五层架构 · L4 生成控制层。

职责边界：把 L3 的片段变成「可溯源、可拒答、不编造」的答案。
    上下文组装（带引用编号） → 受控生成 → 引用抽取 → 置信度评估 → 优雅拒答

三个控制点，对应 RAG 最常见的三类事故：

1. **上下文组装**——给每个片段编号 [1][2][3]，并要求模型在结论后标注依据编号。
   没有编号制度时，模型会把多份文档的信息混着说，出错后无法定位是哪份文档的问题。

2. **幻觉抑制**——prompt 明确「只使用参考内容、不足则直说没有」。
   这是成本最低且最有效的幻觉治理手段，比事后校验更划算。

3. **优雅拒答**——当检索置信度过低时直接拒答，而不是硬凑无关片段。
   「不知道」远好过「一本正经地编」：前者可转人工，后者会 silently 误导决策。

置信度为什么用 RRF 融合分而不是原始向量分：
    向量分的绝对尺度随 embedding 供应商剧烈漂移（bge-m3 常在 0.05~0.15，
    OpenAI 常在 0.3~0.9），无法设定跨模型稳定的阈值。而 RRF 分数只取决于
    「排名」，其理论上限是一个常数 (w_dense + w_lex) / (k + 1)，
    用实测分除以上限即可得到跨模型可比的归一化置信度。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from app import config
from app.core.prompts import get as get_prompt
from app.rag.retriever import DENSE_WEIGHT, LEXICAL_WEIGHT, RRF_K
from app.utils.logger import logger

# 低于此置信度时拒答（宁可转人工，也不硬凑无关片段）
REFUSE_THRESHOLD: float = getattr(config, "REFUSE_THRESHOLD", 0.25)

ANSWER_PROMPT = ChatPromptTemplate.from_template(get_prompt("answer"))

# 答案中出现的引用标记，如 [1] 或 [1][3]
_CITATION_MARK = re.compile(r"\[(\d{1,2})\]")


@dataclass
class StreamStats:
    """流式生成统计（模型无关，供可观测性消费）。

    为什么单独统计而不只看 span 总耗时：
        一次 LLM 生成的时间 = 首 token 延迟（网络往返 + 排队 + 模型调度 + 首个
        token 计算）+ 后续逐 token 流式产出。两者慢的根因完全不同——前者是
        网络/供应商排队，后者是模型吐字速度或输出过长。只有分开记，换模型时
        才能直接定位「慢在哪」，而不是被一个总耗时黑盒误导。

    ttft_ms   首 token 延迟：从请求发出到第一个内容 chunk 返回的墙钟时间。
              注意：推理模型（如 kimi-k2.6 默认开启思考）在正文之前会先产出一段
              隐藏的 reasoning token，这段时间 ttft 会包含「思考」耗时——实测同一
              问句思考开启 11123ms、关闭 1137ms。故 ttft 异常高时先怀疑思考阶段，
              而不是网络。
    chunks    产出的内容 chunk 数（约等于流式 token 数）。
    chars     累计输出字符数（中文按字、英文按字符）。
    finish_reason 供应商返回的结束原因：``stop`` 正常结束 / ``length`` 撞上
              ``max_tokens`` 上限被**截断** / None 未收到终止帧。
    """

    ttft_ms: int = 0
    chunks: int = 0
    chars: int = 0
    finish_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        """是否因撞上 max_tokens 而被截断。

        为什么必须显式暴露：截断曾经是**完全静默**的——答案写到一半断掉，日志里却
        只有「L4 生成完成：N 字」，看不出异常。用户报「回答不完整」时无从定位。
        推理模型会放大该问题：思考 token 与正文共享 ``max_tokens``，思考吃掉一半
        预算，正文就更早撞顶（实测多次 ``finish_reason=length``）。
        """
        return self.finish_reason == "length"


@dataclass
class GenerateResult:
    """生成层输出：答案 + 溯源信息 + 质量信号。"""

    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    error: Optional[str] = None
    stats: Optional[StreamStats] = None


# ---------------------------------------------------------------------------
# 上下文组装
# ---------------------------------------------------------------------------
def build_context(docs: List[Dict[str, Any]], tool_result: Optional[str] = None) -> tuple[str, List[Dict[str, Any]]]:
    """把检索片段组装成带编号的上下文，并产出可溯源清单。

    Returns:
        (context_text, citations)
        citations 形如 [{"index": 1, "source": "...", "file_name": "...", "score": 0.12}]
    """
    blocks: List[str] = []
    citations: List[Dict[str, Any]] = []

    # 重排片段以缓解 lost-in-the-middle（仅改变喂给 LLM 的顺序，不动分数）。
    # 放在 build_context 而非 retrieve：检索分数顺序需保持原样供评测/前端展示。
    from app.rag.reorder import reorder_docs

    ordered_docs = reorder_docs(list(docs or []))

    for i, doc in enumerate(ordered_docs, start=1):
        source = str(doc.get("source", "unknown"))
        # 父子双层（T6-1）：命中子块后回捞的父块正文优先 —— 检索用细粒度子块保证准，
        # 生成用父块保证上下文完整。未启用该特性时 parent_content 不存在，
        # 行为与改造前完全一致。
        body = doc.get("parent_content") or doc.get("content", "")
        blocks.append(
            f"[{i}] 来源：{source.split('/')[-1]}\n{body}"
        )
        citations.append(
            {
                "index": i,
                "source": source,
                "file_name": source.split("/")[-1].split("\\")[-1],
                "score": doc.get("score", 0.0),
                "lexical": doc.get("lexical", 0.0),
                "fallback": bool(doc.get("fallback", False)),
            }
        )

    if tool_result:
        # ⚠️ 这里**必须用【】而不是 []**。
        #
        # 提示词第 2 条要求模型「用方括号标注依据的片段编号」（如 `年休假为 5 天[1]`）。
        # 早先用 `[业务查询结果]` 作标签时，模型把这个标签也当成了可沿用的引用语法，
        # 答案末尾会自动补一个 `[业务查询结果]` —— 实测工具链路与纯制度两个答案都出现了。
        # 用户看到的是一个不存在的引用编号样式，且它既无法被 `_CITATION_MARK`
        # 解析（那是 `\[(\d{1,2})\]`），也不会报错，属于静默的格式污染。
        #
        # 修法是让「业务数据段」用一套**与引用语法不相交**的括号：全角【】
        # 永远不会被误认成 `[n]`。根因不在模型的措辞理解，而在两种语义
        # 共用了同一种字符形式——所以要在产生处分开，而不是在提示词里反复叮嘱。
        blocks.append(f"【业务查询结果】\n{tool_result}")

    return "\n\n".join(blocks), citations


# ---------------------------------------------------------------------------
# 置信度
# ---------------------------------------------------------------------------
def estimate_confidence(docs: List[Dict[str, Any]], tool_result: Optional[str] = None) -> float:
    """评估「这次回答有据可依」的程度（0~1）。

    - 有工具结果：业务查询返回的通常是确定事实（员工信息、假期余额），置信度高；
    - 有检索片段：以 Top1 的 RRF 融合分归一化（除以理论上限）为基线；
    - 软回退片段降权，词面精确命中加权。
    """
    if not docs:
        # 无检索依据：只有工具结果才敢说话，否则置信度为 0（触发拒答）
        return 0.6 if tool_result else 0.0

    fused_max = (DENSE_WEIGHT + LEXICAL_WEIGHT) / (RRF_K + 1)
    top = docs[0]
    base = min(1.0, float(top.get("fused", 0.0)) / fused_max) if fused_max else 0.0

    if top.get("fallback"):
        base *= 0.6                                   # 软回退：证据本身就弱
    if float(top.get("lexical", 0.0)) >= 0.5:
        base = min(1.0, base + 0.1)                   # 词面精确命中：检索确实对上了
    if tool_result:
        base = min(1.0, base + 0.2)                   # 叠加确定性业务数据

    return round(base, 3)


# ---------------------------------------------------------------------------
# 引用抽取
# ---------------------------------------------------------------------------
def extract_cited_indexes(answer: str, citations: List[Dict[str, Any]]) -> List[int]:
    """从答案文本中抽取实际被引用到的片段编号（用于 L5 引用覆盖率评估）。"""
    valid = {c["index"] for c in citations}
    found = {int(m) for m in _CITATION_MARK.findall(answer or "")}
    return sorted(found & valid)


# ---------------------------------------------------------------------------
# 主生成入口（非流式 = 准备 + 一次性生成）
# ---------------------------------------------------------------------------
def _format_history(history: List[Dict[str, str]]) -> str:
    """把**已裁剪**的历史格式化为 prompt 片段。

    只做格式化，不做任何裁剪。窗口裁剪的唯一入口是
    ``app.memory.short_term.build_window``（轮数 + 字符预算 + 问答成对）。

    此处原为 ``history[-limit:]`` 且 ``limit`` 硬编码 6，等于在上游
    ``SHORT_TERM_WINDOW=6`` 轮（12 条）之上又砍一刀到 3 轮，
    使该配置名存实亡——无论把它调成 6 还是 20，实际都只有 3 轮生效。
    裁剪点必须唯一，否则参数会静默失效。

    格式化逻辑委托给 ``app.memory.short_term.format_history``：两份实现曾经
    并存且几乎逐字相同，合并后只剩一份，避免出现「改了这边忘了那边」。
    """
    from app.memory.short_term import format_history

    return format_history(history)


def prepare_generation(
    user_query: str,
    docs: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    tool_result: Optional[str] = None,
    memory_context: str = "",
    retrieval_confidence: Optional[float] = None,
):
    """生成前置：组装上下文 + 评估置信度 + 拒答判定。

    供非流式（generate_answer）与流式（SSE 端点）两条链路复用，
    保证「引用编号 / 拒答阈值 / prompt 约束」在两种输出方式下完全一致。

    调用契约（重要）：
        ``chat_history`` 必须是**已经裁剪过**的窗口，由调用方先经
        ``app.memory.build_short_term_window`` 处理。本层只负责格式化，
        不再做任何长度裁剪——两处裁剪会让 ``SHORT_TERM_WINDOW`` 这类
        配置静默失效（详见 ``_format_history`` 的说明）。

    Returns:
        dict:
            citations  可溯源清单
            confidence 置信度（0~1）
            refused    是否触发拒答
            refusal    拒答文案（refused=True 时非空）
            inputs     大模型 prompt 变量（refused=True 时为 None）

    Args:
        retrieval_confidence: 上游路由节点已算好的检索置信度。传入则直接复用，
            不再重算——**同一个事实每请求只应计算一次**，否则「路由认为 0.8、
            级联认为 0.0」这类自相矛盾会出现且看不出来。
            传 None 表示上游没算（例如直接调用生成层的测试），此时自行计算。
            注意用 `is not None` 判断：0.0 是合法且重要的取值（= 没有证据）。
    """
    context, citations = build_context(docs, tool_result)
    confidence = (
        retrieval_confidence
        if retrieval_confidence is not None
        else estimate_confidence(docs, tool_result)
    )

    if confidence < REFUSE_THRESHOLD and not tool_result:
        logger.info("生成前置拒答：置信度 %.3f 低于阈值 %.2f", confidence, REFUSE_THRESHOLD)
        return {
            "citations": citations,
            "confidence": confidence,
            "refused": True,
            "refusal": (
                "知识库中没有找到能够回答该问题的相关信息。\n"
                "您可以：换一种说法再问一次，或联系行政 / IT 服务台获取人工帮助。"
            ),
            "inputs": None,
        }

    inputs = {
        "context": context or "（知识库无相关内容）",
        "user_query": user_query,
        "chat_history": _format_history(chat_history or []),
        "memory_context": memory_context or "（无长期记忆）",
    }
    return {
        "citations": citations,
        "confidence": confidence,
        "refused": False,
        "refusal": None,
        "inputs": inputs,
    }


def stream_answer_tokens(inputs: Dict[str, Any], stats: Optional[StreamStats] = None):
    """逐 token 生成器：yield 模型输出的文本增量。

    Args:
        stats: 可选的统计收集器。传入后，本函数会记录首 token 延迟（ttft_ms）、
            chunk 数与累计字符数，供调用方挂到 span 观测（模型无关）。

    空流容错：
    - 若流式**连终止帧都没收到**（部分供应商对超频流式请求静默返回空流而非
      429），自动降级为 chain.invoke 整段生成，保证调用方永远不会拿到
      "零 token 且无错误"的空回答。

    （原先还有一层「限流重试包装器在流式下回退到 _generate」的容错，
    已随自造的限流包装层一起移除：模型现在直连 ChatOpenAI。）

    **不回退的例外**：收到了终止帧（finish_reason 非空）但正文为空——这通常是
    推理模型把 max_tokens 全耗在思考上，重试只会再付一整个思考周期。该情形直接
    返回空，由上层兜底，避免等待时间翻倍。

    本函数不吞最终异常：彻底失败时抛错，由 SSE 端点发送 error 事件。
    """
    from app.core.llm_factory import get_chat_model

    chain = ANSWER_PROMPT | get_chat_model()
    t0 = time.perf_counter()
    produced = False
    for chunk in chain.stream(inputs):
        # 结束原因只在终止帧出现，逐帧取最新非空值即可。
        reason = (getattr(chunk, "response_metadata", None) or {}).get("finish_reason")
        if reason and stats is not None:
            stats.finish_reason = reason
        piece = getattr(chunk, "content", None)
        if piece:
            if stats is not None:
                if stats.chunks == 0:
                    stats.ttft_ms = int((time.perf_counter() - t0) * 1000)
                stats.chunks += 1
                stats.chars += len(piece)
            produced = True
            yield piece

    if stats is not None and stats.truncated:
        # 截断必须留痕：此前答案写到一半断掉，日志里只有「L4 生成完成：N 字」，
        # 看不出任何异常，用户报「回答不完整」时无从定位。
        logger.warning(
            "生成被 max_tokens 截断：正文 %d 字后撞上上限（finish_reason=length）。"
            "请调大 LLM_MAX_TOKENS；若为推理模型，思考 token 与正文共享该预算，"
            "可考虑 LLM_DISABLE_THINKING=true。",
            stats.chars,
        )

    if not produced:
        # 推理模型在思考开启时，可能把 max_tokens 全部耗在 reasoning 上，正文为 0。
        # 这种「预算耗尽型空正文」再回退一次只会重付一整个思考周期，且结果大概率
        # 相同——实测一次这样的回退把等待从 28s 拉长到 57s。故直接返回，交由上层
        # 按空回答处理；只有「连终止帧都没收到」才视为真正的供应商空流并回退。
        if stats is not None and stats.finish_reason is not None:
            logger.warning(
                "流式未产出正文（finish_reason=%s），判定为预算耗尽，不再回退重试",
                stats.finish_reason,
            )
            return
        logger.warning("流式输出为空（疑似供应商空流），回退为一次性生成")
        result = chain.invoke(inputs)
        if result.content:
            if stats is not None:
                stats.chars += len(result.content)
            yield result.content


def generate_answer(
    user_query: str,
    docs: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    tool_result: Optional[str] = None,
    memory_context: str = "",
    retrieval_confidence: Optional[float] = None,
) -> GenerateResult:
    """执行受控生成。

    Args:
        memory_context: 长期记忆上下文（来自 app.memory），作为背景信息注入。
            它与「参考内容」的区别：参考内容是本轮检索到的硬证据，
            长期记忆是跨会话累积的背景认知，后者不得作为事实依据被引用。
        retrieval_confidence: 上游已算好的检索置信度；传入则复用，不再重算。

    注意：本函数不抛异常。所有生成失败都以 GenerateResult.error 返回，
    由上层（工作流节点）决定是否转人工——保证单次模型抖动不会击穿整个服务。
    """
    prepared = prepare_generation(
        user_query, docs, chat_history=chat_history,
        tool_result=tool_result, memory_context=memory_context,
        retrieval_confidence=retrieval_confidence,
    )
    citations, confidence = prepared["citations"], prepared["confidence"]

    if prepared["refused"]:
        return GenerateResult(
            answer=prepared["refusal"],
            citations=citations,
            confidence=confidence,
            refused=True,
        )

    try:
        stats = StreamStats()
        answer = "".join(stream_answer_tokens(prepared["inputs"], stats=stats))
    except Exception as exc:  # noqa: BLE001
        logger.exception("L4 生成失败")
        return GenerateResult(citations=citations, confidence=confidence, error=str(exc))

    cited = extract_cited_indexes(answer, citations)
    logger.info(
        "L4 生成完成：%d 字 | 置信度 %.3f | 引用 %d/%d 个片段 | ttft=%dms | finish=%s%s",
        len(answer), confidence, len(cited), len(citations), stats.ttft_ms,
        stats.finish_reason or "unknown",
        "（已截断）" if stats.truncated else "",
    )
    return GenerateResult(answer=answer, citations=citations, confidence=confidence, stats=stats)


def generator_stats() -> Dict[str, Any]:
    """生成层配置快照。"""
    return {"refuse_threshold": REFUSE_THRESHOLD}
