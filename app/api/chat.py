"""智能对话接口 —— 核心问答入口，支持多轮上下文、长期记忆与 SSE 流式输出。

两条链路的关系（读代码前先看这一段）
------------------------------------
``/chat/ask``（非流式）与 ``/chat/ask/stream``（SSE）**共用同一批节点、同一套
条件边、同一个生成前置**：

    共用：graph/nodes.py 的全部节点 + graph/workflow_graph.py 的装配函数
    分叉：非流式整体 invoke **完整图**；流式 invoke **前置子图**（证据取回即止），
          再由端点自己逐 token 推送生成结果

前置子图（``pre_generation_workflow``）与完整图由**同一个装配函数**编译而来，
唯一差别是"证据出口通向哪"（见 ``workflow_graph._wire``）。因此流式链路**不可能**
重抄业务逻辑——这正是被一个真实缺陷逼出来的约束：旧实现在流式端点里重抄了整条
前置链路（记忆 → 意图 → 检索/工具 → 动态路由），结果工具抛异常时，图会把答案换成
「已转接人工」，流式链路却照样去调模型生成，meta 里却报 ``need_human=true``——
同一次提问在两条链路上给出不一致的回答。
"""
import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from app.core.errors import public_detail
from app.core.llm_factory import get_llm_mode
from app.core.source_acl import resolve_allowed_sources
from app.core.tracing import begin_trace, end_trace, span
from app.graph.nodes import build_generation_inputs
from app.graph.state import create_initial_state
from app.graph.workflow_graph import enterprise_workflow, pre_generation_workflow
from app.memory import maybe_consolidate
from app.memory.chat_history import clear_history, get_history, save_message
from app.rag.evaluator import record_feedback
from app.rag.generator import (
    StreamStats,
    extract_cited_indexes,
    stream_answer_tokens,
)
from app.utils.doc_loader import file_name
from app.utils.logger import logger
from app.utils.validator import ChatRequest, sanitize_text

router = APIRouter(prefix="/chat", tags=["对话服务"])

#: 回传给客户端的「历史轮数」上限。
_HISTORY_ROUNDS_CAP = 10


def history_rounds(history: list) -> int:
    """本轮对话携带的历史轮数（含本轮）。

    ``history`` 是消息列表（一问一答各一条），故轮数 = ``len // 2 + 1``。
    封顶避免长会话把这个字段撑成一个无意义的数字。

    ⚠️ 抽成函数不是为了复用几行，而是因为**两条链路（流式 / 非流式）的响应体
    必须给出同一个数**：此前这个式子在两处各写了一遍，改一处就会让
    `/chat/ask` 与 `/chat/ask/stream` 的同一字段含义分叉，且不会有任何报错。
    """
    return min(len(history) // 2 + 1, _HISTORY_ROUNDS_CAP)


def collect_soft_warnings(result: dict) -> int:
    """读取本轮的可降级故障：写日志，并返回可对外的数量。

    为什么必须有一个「读」点
    ------------------------
    `GraphState.soft_warnings` 曾**只有写入点、没有任何读取点**（检索抖动
    被记进状态后就此消失）。"只写不读"等于没观测：这类故障不转人工、不丢答案，
    恰恰因此最容易静默累积——检索质量一路下滑到某天彻底失效才会被发现，
    而中间那些天的日志里什么都查不到。

    与 `error_msg` 的分工（两者互不影响，别混用）：
    - `error_msg` —— 致命错误，参与路由（`need_human` 决定是否转人工）；
    - `soft_warnings` —— 可降级故障，**只进日志与响应体**，绝不参与路由。

    为什么响应体只给数量而不是原文
    ------------------------------
    故障文本里带着异常原文，可能含内网地址、文件路径等内部细节。本服务对外的
    既定口径是**不回显异常原文**（见 500 处理器的 trace_id 与
    docs/history/project-assessment.md P0-1），所以这里只透出数量：客户端据此知道
    "本轮答案是在降级状态下产生的"即可，排障细节看日志 / trace_id。

    Returns:
        可降级故障条数（0 表示本轮一切正常）。
    """
    # 先按原值判空再字符串化：`str(None)` 会得到 "None" 这种"看起来有内容"的假条目
    raw = [
        str(w).strip()
        for w in (result.get("soft_warnings") or [])
        if w and str(w).strip()
    ]
    if raw:
        logger.warning(
            "本轮存在可降级故障 %d 项（未影响作答，仅记录）：%s",
            len(raw), " | ".join(raw),
        )
    return len(raw)


def _sources_view(docs) -> list:
    """检索片段对外视图（两条链路共用，保证字段一致）。"""
    return [
        {
            "content": d.get("content", ""),
            "source": file_name(str(d.get("source", ""))),
            "score": d.get("score", 0),
            "lexical": d.get("lexical", 0),
            "fused": d.get("fused", 0),
            "fallback": d.get("fallback", False),
        }
        for d in (docs or [])
    ]


def _meta_common(result: dict, answer: str) -> dict:
    """两条链路共用的元数据字段。

    抽出来是为了让 ``/chat/ask`` 与 ``/chat/ask/stream`` 的响应体**字段级一致**：
    此前两份字典各写一遍，漏字段与语义漂移都发生过（例如 ``intent_capability``
    一度只在非流式链路出现）。

    ``scene`` 与 ``intent`` 的分工见 ``app/graph/state.py`` 的模块 docstring：
    前者是"路由 Agent 判给谁"，后者是"答案实际怎么来的"。前端徽章用前者
    （它更贴近用户可理解的分类），排障用后者。
    """
    return {
        "code": 0,
        "answer": answer,
        # 路由 Agent 的场景判定（smalltalk/simple_rag/complex_rag/tool/out_of_scope）
        "scene": result.get("scene"),
        "scene_reason": result.get("scene_reason"),
        "scene_source": result.get("scene_source"),
        # 复杂 RAG 实际用于检索的查询（含原问题）：拆歪了是静默错误，故对前端可见。
        "sub_queries": result.get("sub_queries", []),
        "intent": result.get("intent_type"),
        "intent_capability": result.get("intent_capability"),
        "intent_source": result.get("intent_source"),
        # 档位字段恒为 default：动态路由（Flash/Pro 选型）已随「改用不限流
        # 模型 + function calling」删除，保留字段只为不让前端契约被打碎。
        "model_tier": result.get("model_tier") or "default",
        "route_decision": result.get("route_decision") or {},
        # 逐步的执行记录（工具调用 / 检索轮次），供前端展示各 Agent 做了什么。
        "agent_steps": result.get("agent_steps", []),
        "sources": _sources_view(result.get("retrieve_docs")),
        "tool_result": result.get("tool_result"),
        "need_human": result.get("need_human", False),
    }


class FeedbackRequest(BaseModel):
    """用户反馈：用于 L5 评估迭代。"""

    session_id: str = Field(..., max_length=64)
    question: str = Field(..., max_length=2000)
    answer: str = Field("", max_length=8000)
    rating: str = Field(..., description="good 或 bad")
    comment: str = Field("", max_length=1000)
    user_id: str = Field("default", max_length=64)


async def _load_history(session_id: str, req: ChatRequest):
    """载入多轮历史：服务端会话优先，其次使用请求体携带的历史。"""
    history = await get_history(session_id)
    if not history and req.chat_history:
        history = [
            {"role": m.get("role", "user"), "content": sanitize_text(str(m.get("content", "")))}
            for m in req.chat_history
            if isinstance(m, dict)
        ]
    return history


@router.post("/ask")
async def chat_ask(req: ChatRequest) -> dict:
    """核心智能对话接口：记忆加载 → Agent 决策（function calling）→ 受控生成。"""
    start = time.perf_counter()
    session_id = req.session_id or uuid.uuid4().hex[:16]
    user_id = req.user_id or "default"
    # 服务端来源白名单（ACL 配了就按 ACL 来，覆盖客户端传入的 allowed_sources，
    # 防止客户端自行放宽权限）。ACL 未配置时回落到客户端字段（如网关已注入）。
    allowed_sources = resolve_allowed_sources(user_id) or req.allowed_sources
    begin_trace()

    try:
        # 根 span 覆盖「收到请求 → 返回响应」的完整生命周期（历史加载、工作流、
        # 持久化、记忆整理都包在内），使根 span 的 duration 与接口 elapsed_ms 同口径。
        with span("chat_request"):
            history = await _load_history(session_id, req)

            # 图在线程中执行，避免阻塞事件循环。asyncio.to_thread 会拷贝提交线程的
            # contextvars，节点内 `with span(...)` 因此能正确挂到外层 chat_request
            # 根 span 之下（线程隔离不成问题）。
            state = create_initial_state(
                user_query=req.query, chat_history=history, user_id=user_id,
                session_id=session_id, allowed_sources=allowed_sources,
            )
            result = await asyncio.to_thread(enterprise_workflow.invoke, state)

            answer = result.get("answer") or "抱歉，未能生成有效回答，请稍后重试或转接人工客服。"

            await save_message(session_id, "user", req.query)
            await save_message(session_id, "assistant", answer)

            # 记忆整理：会话超阈值时把旧消息压缩归档。放到线程里执行，避免阻塞
            # 事件循环；未达阈值时内部直接返回，零开销。
            updated_history = history + [
                {"role": "user", "content": req.query},
                {"role": "assistant", "content": answer},
            ]
            consolidated = await asyncio.to_thread(
                maybe_consolidate, user_id, session_id, updated_history
            )
    except Exception:  # noqa: BLE001
        end_trace()
        logger.exception("对话接口异常")
        # 不回显异常原文：它可能含文件路径、内网地址、依赖版本等内部细节。
        # 对外文案由 public_detail 唯一构造（固定前缀 + trace_id）—— 用户报障时
        # 提供它即可在 logs/ 中精确定位，既保留可追溯性，又不泄露任何内部信息。
        raise HTTPException(
            status_code=500,
            detail=public_detail("对话服务异常，请稍后重试"),
        )

    span_tree = end_trace()
    elapsed = int((time.perf_counter() - start) * 1000)
    soft_warning_count = collect_soft_warnings(result)
    logger.info(
        "对话完成 session=%s user=%s 场景=%s 出口=%s(%s) 工具=%s 置信度=%.2f 耗时=%dms 命中=%d 记忆整理=%s",
        session_id, user_id, result.get("scene"), result.get("intent_type"),
        result.get("intent_source"), result.get("intent_capability") or "-",
        result.get("confidence", 0), elapsed, len(result.get("retrieve_docs", [])),
        "是" if consolidated else "否",
    )

    return {
        **_meta_common(result, answer),
        "session_id": session_id,
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", 0.0),
        "refused": result.get("refused", False),
        # 生成为何而止。客户端据此可判断"答案是不是被截断的"——
        # `truncated=True` 时调大 LLM_MAX_TOKENS 或关思考（推理模型两者共享预算）。
        "finish_reason": result.get("finish_reason"),
        "truncated": result.get("truncated", False),
        "trace": result.get("trace", []),
        "span_tree": span_tree,
        # 本轮可降级故障条数（>0 说明答案是在降级状态下产生的）：
        # 只给数量，不给异常原文——与「不向客户端回显异常」的既定口径一致。
        "soft_warning_count": soft_warning_count,
        "memory_consolidated": bool(consolidated),
        "elapsed_ms": elapsed,
        "llm_mode": get_llm_mode(),
        "history_rounds": history_rounds(history),
    }


@router.post("/ask/stream")
async def chat_ask_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式对话：与 /chat/ask 同一条业务链路，但答案逐 token 推送。

    事件协议（Server-Sent Events）：
        stage  前置阶段完成（Agent 决策 / 工具调用 / 命中片段数）
        token  生成增量 {"t": "..."}
        meta   最终元数据（来源 / 引用 / 置信度 / 耗时，与 /chat/ask 字段一致）
        error  服务端异常 {"detail": "..."}
        done   结束标记
    """
    session_id = req.session_id or uuid.uuid4().hex[:16]
    user_id = req.user_id or "default"
    allowed_sources = resolve_allowed_sources(user_id) or req.allowed_sources
    history = await _load_history(session_id, req)

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        start = time.perf_counter()
        begin_trace()
        try:
            with span("chat_request"):
                # 1) 前置节点（同步节点在线程中执行，不阻塞事件循环）。
                #    这里 invoke 的是**前置子图**，它与完整图由同一个装配函数
                #    编译而来（差别只有"证据出口通向哪"），因此节点、条件边全部
                #    复用——不需要在这里复刻任何分支判断。
                def _pre():
                    state = create_initial_state(
                        user_query=req.query, chat_history=history, user_id=user_id,
                        session_id=session_id, allowed_sources=allowed_sources,
                    )
                    return pre_generation_workflow.invoke(state)

                state = await asyncio.to_thread(_pre)
                yield sse("stage", {
                    # 路由 Agent 的场景判定与理由（本轮为什么走这条路）
                    "scene": state.get("scene"),
                    "scene_reason": state.get("scene_reason"),
                    "scene_source": state.get("scene_source"),
                    "sub_queries": state.get("sub_queries", []),
                    "intent": state.get("intent_type"),
                    # 实际执行成功的工具名（如 query_leave_balance）。
                    # 通道是闭集、对外稳定；工具名是开集，新增工具时前端与日志
                    # 能立刻看见它，不必改契约。
                    "intent_capability": state.get("intent_capability"),
                    "intent_source": state.get("intent_source"),
                    "model_tier": state.get("model_tier") or "default",
                    "route": state.get("route_decision") or {},
                    "agent_steps": state.get("agent_steps", []),
                    "hits": len(state.get("retrieve_docs", [])),
                    "tool_result": state.get("tool_result"),
                })

                # 2) 产出答案。三种情形：
                #    直答 / 人工兜底（答案已就绪，整段推送）；
                #    证据出口（拒答整段推送，否则逐 token 流式）。
                pieces: list = []
                truncated = False
                citations: list = []
                confidence, refused = 0.0, False

                if state.get("answer") is not None:
                    answer = state.get("answer") or ""
                    yield sse("token", {"t": answer})
                    if state.get("need_human"):
                        # 人工兜底：文案已由 human_fallback_node 写好（与非流式
                        # 链路同一条），直接整段推送。置信度记 0：本轮没有产出
                        # 任何依据。
                        citations, confidence, refused = [], 0.0, False
                    else:
                        # 直答出口（闲聊模板 / 越界话术 / 工具 Agent 的追问）：
                        # 答案已就绪，不进 L4。
                        citations = state.get("citations", [])
                        confidence = state.get("confidence", 1.0)
                        refused = state.get("refused", False)
                else:
                    inputs = build_generation_inputs(state)
                    prepared = inputs["prepared"]
                    citations = prepared["citations"]
                    confidence = prepared["confidence"]
                    refused = prepared["refused"]
                    if prepared["refused"]:
                        answer = prepared["refusal"]
                        yield sse("token", {"t": answer})
                    else:
                        # 生成是整条链路最耗时的阶段，必须包进 span，否则 span 树
                        # 会漏掉 LLM 生成（此前 span 树只有前置节点、总耗时几百 ms，
                        # 与实际回消息 10s+ 对不上，正是这个原因）。
                        # 同时把首 token 延迟（ttft_ms）与 chunk/字符数挂到 span，
                        # 换任何模型都能区分「网络/排队慢」与「吐字慢」，模型无关。
                        stats = StreamStats()
                        with span("generate_answer") as gen_span:
                            async for piece in iterate_in_threadpool(
                                stream_answer_tokens(prepared["inputs"], stats=stats)
                            ):
                                pieces.append(piece)
                                yield sse("token", {"t": piece})
                            gen_span.attrs["ttft_ms"] = stats.ttft_ms
                            gen_span.attrs["chunks"] = stats.chunks
                            gen_span.attrs["answer_chars"] = stats.chars
                            # 截断是被静默吞掉最久的缺陷：答案写到一半断掉，
                            # 日志与 span 都只记字数，看不出异常。这里把结束原因
                            # 一并留痕，便于「回答不完整」类问题直接定位。
                            gen_span.attrs["finish_reason"] = stats.finish_reason
                            gen_span.attrs["truncated"] = stats.truncated
                        truncated = stats.truncated
                        answer = "".join(pieces)

                # 兜底：任何情况下都不落库 / 不下发空回答
                if not answer.strip():
                    answer = "抱歉，本次回答生成失败，请稍后重试或联系管理员。"
                    yield sse("token", {"t": answer})

                # 3) 持久化 + 记忆整理（与非流式链路一致）
                await save_message(session_id, "user", req.query)
                await save_message(session_id, "assistant", answer)
                updated_history = history + [
                    {"role": "user", "content": req.query},
                    {"role": "assistant", "content": answer},
                ]
                consolidated = await asyncio.to_thread(
                    maybe_consolidate, user_id, session_id, updated_history
                )
        except asyncio.CancelledError:
            end_trace()
            raise
        except Exception:  # noqa: BLE001
            end_trace()
            logger.exception("流式对话异常")
            # 与非流式链路同一个构造。这里此前是 `f"对话服务异常：{exc}"` ——
            # 同一次故障，SSE 把文件路径/供应商返回体送到了浏览器，非流式却只给
            # trace_id。两条链路各写一遍文案，分歧就必然出现（P0-2 的原始证据）。
            yield sse("error", {"detail": public_detail("对话服务异常，请稍后重试")})
            return

        span_tree = end_trace()
        elapsed = int((time.perf_counter() - start) * 1000)
        soft_warning_count = collect_soft_warnings(state)
        logger.info(
            "流式对话完成 session=%s user=%s 场景=%s 出口=%s(%s) 工具=%s 耗时=%dms 命中=%d",
            session_id, user_id, state.get("scene"), state.get("intent_type"),
            state.get("intent_source"), state.get("intent_capability") or "-",
            elapsed, len(state.get("retrieve_docs", [])),
        )
        yield sse("meta", {
            **_meta_common(state, answer),
            "session_id": session_id,
            "citations": citations,
            "cited": extract_cited_indexes(answer, citations),
            "confidence": confidence,
            "refused": refused,
            # 答案是否因撞上 max_tokens 而被截断（finish_reason=length）。
            # 前端可据此提示「回答未写完」，运维可据此判断该调大 LLM_MAX_TOKENS。
            "truncated": truncated,
            "trace": state.get("trace", []),
            "span_tree": span_tree,
            "soft_warning_count": soft_warning_count,
            "memory_consolidated": bool(consolidated),
            "elapsed_ms": elapsed,
            "llm_mode": get_llm_mode(),
            "history_rounds": history_rounds(history),
        })
        yield sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/history/{session_id}")
async def chat_history(session_id: str) -> dict:
    """查询指定会话的历史消息。"""
    return {"code": 0, "session_id": session_id, "messages": await get_history(session_id)}


@router.delete("/history/{session_id}")
async def chat_clear(session_id: str) -> dict:
    """清空指定会话的历史消息。"""
    await clear_history(session_id)
    return {"code": 0, "message": f"会话 {session_id} 已清空"}


@router.post("/feedback")
async def chat_feedback(req: FeedbackRequest) -> dict:
    """提交用户反馈（好评/差评），供 L5 评估迭代消费。"""
    ok = record_feedback(
        session_id=req.session_id,
        question=req.question,
        answer=req.answer,
        rating=req.rating,
        comment=req.comment,
        extra={"user_id": req.user_id},
    )
    if not ok:
        raise HTTPException(status_code=400, detail="评分只能是 good 或 bad")
    return {"code": 0, "message": "反馈已记录"}
