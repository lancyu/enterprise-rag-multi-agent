"""LangGraph 业务节点 —— 每个节点职责单一、可独立迭代。

节点清单（按执行顺序）::

    0. memory_load     请求初始化（身份解析）+ 加载长期记忆
    1. router          **路由 Agent**：意图识别 + 边界管控（唯一入口）
    2. ├ smalltalk     闲聊 Agent：模板直答，不检索不调工具
       ├ simple_rag    简单 RAG Agent：单次检索
       ├ complex_rag   复杂 RAG Agent：拆解 → 多次检索 → 合并
       ├ tool          工具 Agent：function calling（见 app/core/tool_agent.py）
       └ out_of_scope  **越界拦截**：就地给出统一回复，不进任何子 Agent
    3. generate_answer L4 受控生成（引用 + 置信度 + 拒答）
    4. human_fallback  异常兜底

与上一版（单 Agent）的差别
--------------------------
上一版只有 ``memory_load → agent → generate_answer``：由**一个** Agent 在一次
function calling 决策里既决定"要不要检索"又决定"要不要查业务数据"。合并的问题
不是"慢"，而是**失败模式不可分**：模型不调工具时有三种互不相干的原因
（寒暄 / 参数没给全 / 忘了调），它们在日志里长得一模一样，只能靠人品。

拆成五个之后，每一类失败都有归属：寒暄走模板（不可能失败）、越界在入口拦下
（不消耗任何下游预算）、检索失败降级为"没找到"（诚实）、业务数据查不到转人工
（不能拿"知识库没找到"糊弄）。

两个出口（本文件最需要看懂的一处）
----------------------------------
子 Agent 产出两种状态，下游据此分流：

- **直答**（``intent_type == "direct"``）：答案已经就绪（闲聊模板 / 越界话术 /
  工具 Agent 的追问），**不进 L4**。没有检索片段时进 L4 会被拒答逻辑改写成
  「知识库中没有找到…」，把一句「你好」变成拒答。
- **证据**（``intent_type`` 为 ``knowledge`` / ``tool``）：子 Agent 只负责
  **取回证据**，组织成话交给 L4——片段编号、引用溯源、置信度拒答、流式输出
  全部保留在 L4 一处产生，不因架构变化而分裂成两份实现。

请求作用域为什么在 ``tool_node`` 里建
-------------------------------------
LangGraph 在**复制的 context** 里执行节点，在 A 节点里建的作用域随 A 返回即失效
（详见 ``app/core/request_ctx.py``）。工具在 LangChain 的 ``Runnable.invoke`` 里
执行，又在**另一个**副本里——所以作用域必须在"调用工具的同一个节点"里建立。
放在 ``memory_load_node`` 会得到"工具查到了、生成层拿到 0 条"的静默故障。
"""
import time
from typing import Any, Dict, List

# 只导入「建作用域」与「写授权事实」两类操作。读取证据的 ``get_retrieved_docs`` /
# ``get_tool_results`` 刻意**不在这里导入**：证据由子 Agent 从作用域读出后
# 挂在返回值上带出（见 ``ToolDecision.docs`` 的说明）。若这里再读一次作用域，
# 就等于又依赖「当前 context 是哪一个」——而跨上下文读不到证据正是踩过的坑。
from app.core.request_ctx import (
    reset_request_context,
    set_allowed_sources,
)
from app.core.router_agent import (
    SCENE_COMPLEX_RAG,
    SCENE_OUT_OF_SCOPE,
    SCENE_SIMPLE_RAG,
    SCENE_SMALLTALK,
    SCENE_TOOL,
    OUT_OF_SCOPE_ANSWER,
    route_query,
)
from app.core.sub_agents import (
    run_complex_rag_agent,
    run_simple_rag_agent,
    run_smalltalk_agent,
)
from app.core.tool_agent import run_tool_agent
from app.core.tracing import span
from app.graph.state import GraphState
from app.memory import get_memory_context, get_short_term_memory
from app.rag.generator import prepare_generation, stream_answer_tokens
from app.utils.logger import logger

#: 直答出口的 ``intent_type`` 取值（答案已就绪，不进 L4）。
EXIT_DIRECT = "direct"
#: 证据出口（检索片段类）。
EXIT_KNOWLEDGE = "knowledge"
#: 证据出口（业务工具类）。
EXIT_TOOL = "tool"


def _trace(state: GraphState, node: str, detail: str = "", elapsed_ms: int = 0) -> List[Dict[str, Any]]:
    """追加节点执行轨迹。"""
    trace = list(state.get("trace", []))
    trace.append({"node": node, "detail": detail, "elapsed_ms": elapsed_ms})
    return trace


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


# =============================================================
# 0. 请求初始化 + 记忆加载
# =============================================================
def memory_load_node(state: GraphState) -> GraphState:
    """图的入口节点：加载长期记忆与多轮历史。

    本节点永不失败：记忆读不到就按空记忆继续。

    设计为独立节点而非揉进 router：记忆读取涉及文件 I/O，独立成节点后可以在
    前端工作流面板上看到它的耗时，也便于在记忆系统出问题时单独降级。

    注意：请求级**作用域**的建立不在这里，而在 ``tool_node``——原因见模块 docstring。
    """
    new_state = dict(state)
    start = time.perf_counter()
    with span("memory_load") as s:
        try:
            context = get_memory_context(state.get("user_id", "default"))
        except Exception as exc:  # noqa: BLE001
            # 记忆不可用绝不能阻断主流程
            logger.warning("长期记忆加载失败，已跳过：%s", exc)
            context = ""
        new_state["memory_context"] = context
        s.attrs["memory_chars"] = len(context)

    new_state["trace"] = _trace(
        new_state, "memory_load",
        f"记忆 {len(new_state.get('memory_context', ''))} 字",
        _elapsed_ms(start),
    )
    return new_state


# =============================================================
# 1. 路由节点（入口层：意图识别 + 边界管控）
# =============================================================
def router_node(state: GraphState) -> GraphState:
    """判定本轮该由哪个 Agent 处理，并把越界问题在入口拦下。

    **本节点永不失败、也永不转人工。** 路由失败会走确定性规则兜底
    （``route_query`` 内部处理）：它是一次粗分类，转人工的代价（用户拿不到任何
    回答）远大于分错路的代价（多检索一次）。

    「幻觉管控集中在入口层」在这里落地：越界问题被判定后由
    ``out_of_scope_node`` 用**常量话术**回复，不经过任何模型——一个只被
    判别而不被生成的回答，不可能有幻觉。
    """
    new_state = dict(state)
    start = time.perf_counter()
    with span("router") as s:
        decision = route_query(
            query=state["user_query"],
            chat_history=state.get("chat_history", []),
            memory_context=state.get("memory_context", ""),
        )
        s.attrs["scene"] = decision.scene
        s.attrs["source"] = decision.source
        s.attrs["degraded"] = decision.degraded

        new_state["scene"] = decision.scene
        new_state["scene_reason"] = decision.reason
        new_state["scene_confidence"] = decision.confidence
        new_state["scene_source"] = decision.source
        # route_decision 是**本轮决策摘要**：路由 Agent 判定什么场景、
        # 后续环节又发生了哪些降级。两个写入点（本节点与 tool_node）共用它，
        # 因为对前端而言"这轮为什么走成这样"就该是一个对象。
        degradations: List[str] = list(new_state.get("route_decision", {}).get("degradations") or [])
        soft_warnings = list(new_state.get("soft_warnings") or [])
        if decision.degraded:
            degradations.append(f"路由降级：{decision.reason}")
            soft_warnings.append(f"路由降级（已按 {decision.scene} 处理）：{decision.reason}")
        new_state["soft_warnings"] = soft_warnings
        new_state["route_decision"] = {
            **decision.to_dict(),
            "direct": False,
            "used_tools": [],
            "recovered_calls": 0,
            "degradations": degradations,
        }
    new_state["trace"] = _trace(
        new_state, "router",
        f"{new_state['scene']}（{new_state['scene_source']}）",
        _elapsed_ms(start),
    )
    return new_state


# =============================================================
# 2a. 闲聊 Agent 节点
# =============================================================
def smalltalk_node(state: GraphState) -> GraphState:
    """寒暄 / 身份询问：模板直答，**不进 L4、不检索、不调工具、不调模型**。

    为什么不进 L4：L4 的输入是"参考内容 + 问题"，没有参考内容时会触发拒答逻辑，
    把「你好」改写成「知识库中没有找到相关信息」。
    """
    new_state = dict(state)
    start = time.perf_counter()
    answer = run_smalltalk_agent(state["user_query"])
    new_state["answer"] = answer.text or ""
    new_state["intent_type"] = EXIT_DIRECT
    new_state["intent_source"] = SCENE_SMALLTALK
    new_state["intent_capability"] = None
    new_state["agent_steps"] = list(answer.steps)
    new_state["citations"] = []
    # 直答没有任何检索依据，置信度按「确定」记 1.0：
    # 语义是「这句话不是知识问答，不适用证据置信度」，
    # 与「有强证据」区分开的是 intent_type=direct 这个标记。
    new_state["confidence"] = 1.0
    new_state["refused"] = False
    new_state["route_decision"] = {**new_state.get("route_decision", {}), "direct": True}
    new_state["trace"] = _trace(
        new_state, "smalltalk", f"{len(new_state['answer'])} 字（模板直出）", _elapsed_ms(start)
    )
    return new_state


# =============================================================
# 2b. 越界拦截节点
# =============================================================
def out_of_scope_node(state: GraphState) -> GraphState:
    """越界问题的统一对外话术。**答案是一个常量，不经过模型。**

    这是"把幻觉管控集中在入口层"最关键的一处：一个只被判别、不被生成的回答，
    不可能编造任何业务事实；而且它对每一类越界问题都给出同一份可审计的答复。

    ``need_human`` 刻意保持 False：本轮**已经**给出了完整、明确的回答
    （"这件事不归我管，该去哪儿"），它是一次成功的处理，不是"无法自动处理"。
    标成 True 会让前端显示"已转接人工"——那是假话。
    """
    new_state = dict(state)
    new_state["answer"] = OUT_OF_SCOPE_ANSWER
    new_state["intent_type"] = EXIT_DIRECT
    new_state["intent_source"] = SCENE_OUT_OF_SCOPE
    new_state["intent_capability"] = None
    new_state["citations"] = []
    new_state["confidence"] = 1.0
    new_state["refused"] = False
    new_state["route_decision"] = {**new_state.get("route_decision", {}), "direct": True}
    new_state["trace"] = _trace(new_state, "out_of_scope", "入口拦截，未进入任何子 Agent", 0)
    logger.info("越界问题已在入口拦截：%s", (state.get("user_query") or "")[:40])
    return new_state


# =============================================================
# 2c / 2d. 检索类 Agent 节点（简单 / 复杂 RAG）
# =============================================================
def _apply_retrieval(new_state: GraphState, answer, source: str) -> None:
    """把检索类 Agent 的产出写进状态（简单与复杂 RAG 共用）。

    ``soft_warnings`` 与 ``agent_steps`` 一律**追加**而不是覆盖：工具 Agent 降级
    改道简单 RAG 时，工具那一段的故障记录必须先于本段存在，覆盖掉它们等于
    把"这轮为什么降级"的唯一线索擦掉。
    """
    new_state["retrieve_docs"] = list(answer.docs)
    new_state["tool_result"] = None
    new_state["intent_type"] = EXIT_KNOWLEDGE
    new_state["intent_source"] = source
    new_state["intent_capability"] = None
    new_state["sub_queries"] = list(answer.sub_queries)
    new_state["soft_warnings"] = [*(new_state.get("soft_warnings") or []), *answer.soft_warnings]
    new_state["agent_steps"] = [*(new_state.get("agent_steps") or []), *answer.steps]


def simple_rag_node(state: GraphState) -> GraphState:
    """单文档即可回答的制度类问题：一次混合检索。"""
    new_state = dict(state)
    start = time.perf_counter()
    with span("simple_rag_node") as s:
        answer = run_simple_rag_agent(
            state["user_query"],
            allowed_sources=state.get("allowed_sources"),
        )
        _apply_retrieval(new_state, answer, SCENE_SIMPLE_RAG)
        s.attrs["hits"] = len(answer.docs)
        s.attrs["degraded"] = answer.degraded
    new_state["trace"] = _trace(
        new_state, "simple_rag", f"命中 {len(new_state['retrieve_docs'])} 条", _elapsed_ms(start)
    )
    return new_state


def complex_rag_node(state: GraphState) -> GraphState:
    """需要跨文档对比 / 综合推理的问题：拆解 → 多次检索 → 合并去重。"""
    new_state = dict(state)
    start = time.perf_counter()
    with span("complex_rag_node") as s:
        answer = run_complex_rag_agent(
            state["user_query"],
            allowed_sources=state.get("allowed_sources"),
        )
        _apply_retrieval(new_state, answer, SCENE_COMPLEX_RAG)
        s.attrs["sub_queries"] = len(answer.sub_queries)
        s.attrs["hits"] = len(answer.docs)
        s.attrs["degraded"] = answer.degraded
    new_state["trace"] = _trace(
        new_state, "complex_rag",
        f"{len(new_state['sub_queries'])} 个查询 → 去重后 {len(new_state['retrieve_docs'])} 条",
        _elapsed_ms(start),
    )
    return new_state


# =============================================================
# 2e. 工具 Agent 节点（function calling）
# =============================================================
def _collect_degradations(decision) -> List[str]:
    """把工具阶段的**可降级故障**收进 ``soft_warnings``（不参与路由）。

    三个来源，对应三类「答案仍能给出、但质量已下降」：

    1. ``decision.soft_warnings`` —— 工具**内部**吞掉的异常（当前 4 个工具刻意
       不吞基础设施异常、直接上抛，故这个来源通常为空；保留它是因为
       ``RequestScope.soft_warnings`` 是工具与调用方之间的通用通道，
       将来新增"检索型"工具时会用到）。
    2. ``step.status == "error"`` —— 工具执行抛异常（如 SQLite 不可用）。
    3. ``step.status == "rejected"`` —— 护栏拒绝了模型给出的参数。
       这不是系统故障，而是**模型侧的信号**；出现频繁说明提示词或模型选择需要调整。
       它同样让本轮少了一份证据，所以也必须可见。

    刻意**不**收 ``ok`` 的步骤：正常工具调用不是故障，收进来会让计数长期虚高，
    真正需要关注的抖动就被淹没了。
    """
    warnings: List[str] = list(decision.soft_warnings or [])
    for step in decision.steps:
        if step.status == "error":
            warnings.append(f"工具 {step.tool} 执行失败：{step.detail or '未知原因'}")
        elif step.status == "rejected":
            warnings.append(f"工具 {step.tool} 调用被护栏拒绝：{step.detail or '参数不合法'}")
    if decision.recovered_calls:
        # 正文回捞成功不算故障（用户拿到的是正确结果），但它是**换模型前必看**的
        # 格式稳定性信号：数值高说明该端点的 function calling 不可靠。
        warnings.append(f"模型以正文形式输出了 {decision.recovered_calls} 个工具调用（已回捞）")
    return warnings


def _failed_business_tools(decision) -> List[str]:
    """本轮**执行失败**的工具名。

    判据是「**能不能从其他来源得到答案**」，而不是「这次调用有没有报错」：

    - 工具执行抛异常（``error``）→ 转人工。用户问「李四的年假还剩几天」而
      SQLite 不可用时，我们**没有任何别的来源**能回答；硬让 L4 回一句
      「知识库中没有找到相关信息」是**答非所问**——用户会以为这位同事没有记录，
      而不是「系统暂时查不了」。
    - 参数不合 schema（``rejected``）→ **不**转人工，也不算失败。那是模型侧
      的问题，工具已把「哪个字段、哪里不对」回给模型，它可以在剩余轮次里
      自我纠正；纠正不了则由 L4 按无依据诚实作答。
    """
    return [step.tool for step in decision.steps if step.status == "error"]


def tool_node(state: GraphState) -> GraphState:
    """工具 Agent：模型自行决定「要不要调工具、调哪个、参数是什么」。

    本节点永不抛异常：决策失败（模型不可达、返回结构异常）记 ``error_msg``
    并置 ``need_human``，由 ``tool_route_edge`` 转人工兜底。

    ``degraded`` 的处理值得单独说明：模型不支持 function calling 时本节点**不自己
    降级检索**，只把 ``tool_degraded`` 写进状态，由 ``tool_route_edge`` 改道简单
    RAG。「工具路走不通该改走哪条」是一个**路由决策**，与"意图识别"同类；
    让它落在边上而不是节点里，拓扑才在代码里完整可见。
    """
    new_state = dict(state)
    start = time.perf_counter()
    # 工具调用前必须先在**本节点的 context** 里建好请求作用域：
    #   ① 来源白名单要能被工具读到（模型既看不到也改不了它）；
    #   ② 工具写入的证据要与本节点读到的**是同一个对象**。
    # ② 是本项目踩过的一个坑：LangChain 的 Runnable.invoke 在复制的 context 里
    #   执行工具，若作用域是在工具内部才第一次创建，它建在那个副本里，
    #   工具写进去的证据本节点读不到 —— 症状是「工具日志显示查询成功，
    #   生成层却拿到 0 条数据」。详见 app/core/request_ctx.py。
    reset_request_context()
    set_allowed_sources(state.get("allowed_sources"))

    with span("tool") as s:
        decision = run_tool_agent(
            query=state["user_query"],
            chat_history=state.get("chat_history", []),
            memory_context=state.get("memory_context", ""),
        )
        s.attrs["steps"] = len(decision.steps)
        s.attrs["tools"] = decision.tool_names()
        s.attrs["degraded"] = decision.degraded

        new_state["agent_steps"] = [step.to_dict() for step in decision.steps]
        new_state["soft_warnings"] = [
            *(new_state.get("soft_warnings") or []),
            *_collect_degradations(decision),
        ]
        degradations = list(new_state.get("route_decision", {}).get("degradations") or [])
        if decision.degraded:
            degradations.append("工具模型不支持 function calling，已改道简单 RAG")
        new_state["route_decision"] = {
            **new_state.get("route_decision", {}),
            "direct": decision.is_direct,
            "used_tools": list(decision.used_tools),
            "recovered_calls": decision.recovered_calls,
            "degraded": bool(new_state.get("route_decision", {}).get("degraded")) or decision.degraded,
            "degradations": degradations,
        }

        if decision.error:
            # 决策阶段失败 = 本轮确实无法自动处理，转人工。
            # 注意与「工具执行失败」的区别：后者可降级，不进 error_msg
            # （见 docs/history/project-assessment.md P0-1：可恢复故障写进 error_msg
            #  会导致条件边在答案生成之后覆盖掉已生成的有效答案）。
            logger.warning("工具 Agent 决策失败，转人工兜底：%s", decision.error)
            new_state["error_msg"] = decision.error
            new_state["need_human"] = True
            new_state["trace"] = _trace(
                # detail 会随响应体下发（前端面板直接显示），故只写**事实**。
                # 诊断串（decision.error）只进上一行的日志 —— 它可能含路径与
                # 内网地址，而"转人工"这件事用户在答案里本来就看得见。
                new_state, "tool", "决策失败，转人工兜底", _elapsed_ms(start)
            )
            return new_state

        if decision.degraded:
            # 什么都不写出口：改道由 tool_route_edge 决定，simple_rag_node 会写。
            # 这里若先把 intent_type 定成 direct/knowledge，等于替下游预判了结果。
            new_state["tool_degraded"] = True
            # 降级必须留痕：它不报错、不丢答案，只是答案质量会下降
            # （制度类问题仍能答，业务数据类问题则答不了），正是最容易静默累积的一类。
            new_state["soft_warnings"] = [
                *(new_state.get("soft_warnings") or []),
                "工具 Agent 不可用（模型不支持 function calling），已改道简单 RAG",
            ]
            logger.info("工具 Agent 不可用（模型不支持 function calling），改道简单 RAG")
            new_state["trace"] = _trace(new_state, "tool", "降级：改道简单 RAG", _elapsed_ms(start))
            return new_state

        # 证据由 run_tool_agent 显式带出（而不是这里再读一次作用域）：
        # 工具是函数，其产物不该要求调用方关心「当前 context 是哪一个」。
        # 见 ToolDecision.docs 的说明。
        new_state["retrieve_docs"] = list(decision.docs)
        business = list(decision.tool_results)
        new_state["tool_result"] = "\n\n".join(business) if business else None
        new_state["intent_capability"] = decision.tool_names()

        if decision.is_direct:
            # 直答出口：模型没调工具（或只提了不合法的调用）就直接说话了，
            # 典型场景是"必填参数缺失 → 主动向用户追问"。它自己的话就是答案。
            answer = (decision.direct_answer or "").strip()
            new_state["intent_type"] = EXIT_DIRECT
            new_state["intent_source"] = SCENE_TOOL
            if answer:
                new_state["answer"] = answer
                new_state["citations"] = []
                new_state["confidence"] = 1.0
                new_state["refused"] = False
                logger.info("工具 Agent 直答：%d 字（未取到业务数据）", len(answer))
            else:
                # 模型既没调工具也没说话：交给 L4，按「无依据」给出诚实回答。
                # 不在这里编一句友好话术——那会把一次模型异常伪装成正常回答。
                logger.warning("工具 Agent 未调用工具且未产出内容，转 L4 按无依据处理")
                new_state["intent_type"] = EXIT_KNOWLEDGE
        else:
            new_state["intent_type"] = EXIT_TOOL
            new_state["intent_source"] = SCENE_TOOL
            logger.info(
                "工具 Agent 收集证据完成：工具=%s 业务结果=%d 项",
                decision.tool_names(), len(business),
            )

        # 例外：**工具执行失败**要转人工（见 _failed_business_tools 的判据说明）。
        # 这是「可降级故障不写 error_msg」规则的唯一例外，且理由不同——它不是
        # 「本轮仍能作答」，而是「本轮**答不了**」。
        failed = _failed_business_tools(decision)
        if failed:
            logger.warning("业务工具执行失败，转人工兜底：%s", ",".join(failed))
            new_state["error_msg"] = f"业务工具不可用：{','.join(failed)}"
            new_state["need_human"] = True

    new_state["trace"] = _trace(
        new_state, "tool",
        f"{new_state.get('intent_type')} 工具={new_state.get('intent_capability') or '-'}",
        _elapsed_ms(start),
    )
    return new_state


# =============================================================
# 3. 答案生成节点（RAG L4）
# =============================================================
def build_generation_inputs(state: GraphState) -> Dict[str, Any]:
    """组装 L4 生成的输入：引用清单 + 置信度 + 拒答判定 + prompt 变量。

    **非流式节点与流式端点共用这一处。** 两份实现曾经并存（``/chat/ask`` 走
    节点、``/chat/ask/stream`` 在端点里重抄一遍前置链路），后果是同一状态在
    两条链路上给出不一致的回答——工具抛异常时图会换成「已转接人工」，
    流式链路却照样去调模型生成。裁剪点与拒答阈值必须只有一个产生点。

    ``retrieval_confidence`` 传 None 让 L4 自行计算：架构改造后已经**没有人**
    在生成之前算过它了（原先由 model_route 节点算，只为给档位选型用，
    顺带被复用于拒答）。现在它只有一个消费者，就地计算才是唯一真相。
    """
    window = get_short_term_memory(
        state.get("session_id", ""), state.get("chat_history", [])
    ).window
    prepared = prepare_generation(
        user_query=state["user_query"],
        docs=state.get("retrieve_docs", []),
        chat_history=window,
        tool_result=state.get("tool_result"),
        memory_context=state.get("memory_context", ""),
        retrieval_confidence=state.get("retrieval_confidence"),
    )
    return {"prepared": prepared, "window": window}


def generate_answer_node(state: GraphState) -> GraphState:
    """受控生成：引用溯源 + 置信度 + 优雅拒答。

    级联（Flash 生成不达标时升 Pro 重生成）已随档位机制一并删除：
    现在只有一个模型，「升档」没有落点。
    """
    new_state = dict(state)
    with span("generate_answer") as s:
        try:
            inputs = build_generation_inputs(state)
            prepared = inputs["prepared"]
            new_state["citations"] = prepared["citations"]
            new_state["confidence"] = prepared["confidence"]
            new_state["refused"] = prepared["refused"]
            new_state["retrieval_confidence"] = prepared["confidence"]

            if prepared["refused"]:
                new_state["answer"] = prepared["refusal"]
                logger.info("生成环节触发优雅拒答：置信度 %.3f", prepared["confidence"])
            else:
                from app.rag.generator import StreamStats

                stats = StreamStats()
                answer = "".join(stream_answer_tokens(prepared["inputs"], stats=stats))
                new_state["answer"] = answer
                # 把生成统计挂到 span（模型无关）：ttft_ms 区分「首 token 延迟
                # （网络/排队）vs 生成时长（吐字速度）」，换任何模型都能定位慢在哪。
                s.attrs["ttft_ms"] = stats.ttft_ms
                s.attrs["chunks"] = stats.chunks
                s.attrs["answer_chars"] = stats.chars
                # 截断必须可见且要透到响应体：被 max_tokens 拦腰截断的答案与
                # 正常写完的答案，在「只看字数」的口径下**完全一样**。
                s.attrs["finish_reason"] = stats.finish_reason or "unknown"
                s.attrs["truncated"] = stats.truncated
                new_state["finish_reason"] = stats.finish_reason
                new_state["truncated"] = stats.truncated
        except Exception as exc:  # noqa: BLE001
            logger.exception("答案生成异常")
            new_state["error_msg"] = str(exc)
            new_state["need_human"] = True
    new_state["trace"] = _trace(
        new_state, "generate_answer",
        f"生成 {len(new_state.get('answer') or '')} 字 · 置信度 {new_state.get('confidence', 0):.2f}",
        s.elapsed_ms(),
    )
    return new_state


# =============================================================
# 4. 人工兜底节点
# =============================================================
def human_fallback_node(state: GraphState) -> GraphState:
    new_state = dict(state)
    # 不对用户回显异常原文：它可能含文件路径、堆栈、内网地址等内部细节。
    # 异常详情只进日志，用户只看到可行动的提示。
    new_state["answer"] = (
        "当前问题暂时无法自动处理，已为您转接人工客服。\n"
        "您也可以稍后重试，或联系行政 / IT 服务台获取帮助。"
    )
    new_state["need_human"] = True
    new_state["intent_source"] = "human_fallback"
    # detail 会随响应体下发（前端面板直接显示），所以这里**不能**放 error_msg：
    # 它是内部诊断串（可含异常原文），此前被截前 60 字塞进 trace，等于给
    # "不回显异常"开了一条侧路 —— 用户看不到答案里的原文，却能在 trace 里看到。
    # 步骤名已经能说明是哪一段失败的（前一条 trace 就是它）。
    new_state["trace"] = _trace(new_state, "human_fallback", "已转人工兜底", 0)
    logger.warning("触发人工兜底：%s", state.get("error_msg"))
    return new_state
