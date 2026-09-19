"""工作流组装编译 —— 构建企业级 LangGraph 有向图。

拓扑结构（9 节点 / 3 条件边）::

    memory_load（身份解析 + 长期记忆）
      → router（路由 Agent：意图识别 + 边界管控）
           ├─ smalltalk ────────────────────────────→ END（模板直答）
           ├─ out_of_scope ─────────────────────────→ END（常量话术，不生成）
           ├─ simple_rag ─┐
           ├─ complex_rag ┤
           └─ tool ───────┤
                          │   ├─ 决策失败 ──────────→ human_fallback
                          │   ├─ 反问用户（有答案）──→ END
                          │   ├─ 不支持 function calling ──→ simple_rag（改道）
                          │   └─ 有证据 ─┐
                          └──────────────┴──→ generate_answer（引用 + 置信度 + 拒答）
                                                   ├─ 生成失败 → human_fallback
                                                   └─ 正常 ───→ END

为什么分成"路由 + 四个执行者"而不是一个大 Agent
----------------------------------------------
单 Agent 版本由**一个** Agent 在一次 function calling 决策里既决定"要不要检索"
又决定"要不要查业务数据"。合并的问题不是慢，而是**失败模式不可分**：模型不调
工具时有三种互不相干的原因（寒暄 / 参数没给全 / 忘了调），它们在日志里长得
一模一样。拆开之后每类失败都有归属，且边界规则只有一份（在入口）。

保留 generate_answer 的理由值得单独说明：子 Agent 只负责**取回证据**，不负责
组织成话。片段编号 ``[1][2]``、引用溯源、置信度拒答、流式输出都在 L4 一处产生——
让模型在工具循环里自由成文，等于把这四项能力从唯一的产生点拆成两处，
两处必然漂移，而漂移是静默的（答案看起来都很流利）。

两个编译产物：完整图与"生成之前"图
-----------------------------------
``/chat/ask`` 直接 invoke **完整图**；``/chat/ask/stream`` 要边生成边推送，不能
整体 invoke，只能走到"证据已取回、答案还没生成"为止，再由端点自己逐 token 推。

两条链路**必须**共用同一批节点与同一套分支，否则会漂移——历史缺陷正是如此：
旧实现在流式端点里重抄了整条前置链路，结果工具抛异常时图会把答案换成
「已转接人工」，流式链路却照样去调模型生成，同一次提问在两条链路上给出不一致的回答。

所以这里不重抄，而是**用同一个装配函数编译两次**：
``_wire(graph, generation_target=...)`` 的差别只有"证据出口通向哪"这一个参数。
节点函数、条件边函数、分支映射全部是同一份代码，漂移在结构上不可能发生。
"""
from langgraph.graph import END, StateGraph

from app.graph.edges import error_route_edge, scene_route_edge, tool_route_edge
from app.graph.nodes import (
    complex_rag_node,
    generate_answer_node,
    human_fallback_node,
    memory_load_node,
    out_of_scope_node,
    router_node,
    simple_rag_node,
    smalltalk_node,
    tool_node,
)
from app.graph.state import GraphState
from app.utils.logger import logger

#: 图节点名（供前端面板与拓扑校验使用，唯一来源）。
NODE_NAMES: tuple = (
    "memory_load",
    "router",
    "smalltalk",
    "out_of_scope",
    "simple_rag",
    "complex_rag",
    "tool",
    "generate_answer",
    "human_fallback",
)


def conditional_branch_count(compiled) -> int:
    """从**编译图**派生「有几个条件分支点」——不写死字面量。

    为什么不能直接数边：LangGraph 的 ``get_graph()`` 会把一条条件边按目标节点
    **展开**成多条（``router`` 那一条展开成 5 条），边长 5+4+2=11。按**出发节点**
    去重才是「有几个条件分支点」——本图是 3（``router`` / ``tool`` /
    ``generate_answer``），与 ``app/graph/edges.py`` 的「三条条件边」一致。

    为什么要派生而不是写常量：这个数字曾被写死为 4，而实际一直是 3，于是启动日志、
    本模块 docstring、前端横幅三处**一起**错，与 README / project-introduction 的
    「3 条条件边」自相矛盾（P0-5 记「同一仓库 5 种说法」）。数字只要还能被写死，
    就还会漂——这里改成从编译产物读，改图时不可能忘记同步。
    """
    edges = compiled.get_graph().edges
    return len({edge.source for edge in edges if getattr(edge, "conditional", False)})


def _wire(graph: StateGraph, *, generation_target: str) -> None:
    """装配生成之前的全部节点与分支；``generation_target`` 决定证据出口的去向。

    Args:
        graph: 待装配的 StateGraph。
        generation_target: 证据出口（检索类 Agent / 工具 Agent 取回证据后）通向
            的节点名。完整图传 ``"generate_answer"``；流式端点的前置图传 ``END``
            （生成由端点自己逐 token 完成）。
    """
    graph.add_node("memory_load", memory_load_node)
    graph.add_node("router", router_node)
    graph.add_node("smalltalk", smalltalk_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("simple_rag", simple_rag_node)
    graph.add_node("complex_rag", complex_rag_node)
    graph.add_node("tool", tool_node)
    graph.add_node("human_fallback", human_fallback_node)
    # 前置图里没有这个节点：生成由流式端点自己逐 token 完成（见模块 docstring）。
    if generation_target != END:
        graph.add_node("generate_answer", generate_answer_node)

    # 入口：先做请求初始化（身份解析）与长期记忆加载
    graph.set_entry_point("memory_load")
    graph.add_edge("memory_load", "router")

    # 路由 Agent 之后的五路分发：两个直答出口 + 三个证据出口
    graph.add_conditional_edges(
        "router",
        scene_route_edge,
        {
            "smalltalk": "smalltalk",
            "out_of_scope": "out_of_scope",
            "simple_rag": "simple_rag",
            "complex_rag": "complex_rag",
            "tool": "tool",
        },
    )
    # 直答出口：答案已就绪，直接结束（不进 L4，避免被拒答逻辑改写）
    graph.add_edge("smalltalk", END)
    graph.add_edge("out_of_scope", END)

    # 工具 Agent 之后的四分支。generate_answer 在"前置图"里退化为 END
    graph.add_conditional_edges(
        "tool",
        tool_route_edge,
        {
            "human_fallback": "human_fallback",
            "end": END,
            "simple_rag": "simple_rag",
            "generate_answer": generation_target,
        },
    )

    # 证据出口：检索类 Agent 一律进生成（或在前置图里结束）
    graph.add_edge("simple_rag", generation_target)
    graph.add_edge("complex_rag", generation_target)

    if generation_target != END:
        graph.add_conditional_edges(
            "generate_answer",
            error_route_edge,
            {"human_fallback": "human_fallback", "end": END},
        )

    graph.add_edge("human_fallback", END)


def build_workflow_graph():
    """构建并编译**完整**工作流（含受控生成）—— ``/chat/ask` 与非流式链路使用。"""
    graph = StateGraph(GraphState)
    _wire(graph, generation_target="generate_answer")
    compiled = graph.compile()
    logger.info(
        "LangGraph 工作流编译完成：%d 节点 / %d 条件边",
        len(NODE_NAMES), conditional_branch_count(compiled),
    )
    return compiled


def build_pre_generation_graph():
    """构建并编译**生成之前**的子图 —— 流式端点专用。

    与完整图的唯一差别是"证据出口通向 END 而不是 generate_answer"。
    节点与条件边全部复用（见 ``_wire``），因此两条链路的前置结果**结构上
    不可能不一致**；``tests/test_multi_agent.py`` 直接断言这一点。
    """
    graph = StateGraph(GraphState)
    _wire(graph, generation_target=END)
    compiled = graph.compile()
    logger.info("LangGraph 前置工作流编译完成（流式链路：证据取回后由端点自行生成）")
    return compiled


# 全局单例工作流
enterprise_workflow = build_workflow_graph()
#: 流式链路的前置子图（证据取回后停下，由端点逐 token 生成）
pre_generation_workflow = build_pre_generation_graph()


def get_mermaid() -> str:
    """工作流拓扑的 Mermaid 描述（供前端渲染）。"""
    return """flowchart TD
    M[memory_load 身份解析·记忆加载] --> R[router 路由 Agent 意图识别·边界管控]
    R -->|寒暄| S[smalltalk 闲聊·模板直答]
    R -->|越界| O[out_of_scope 统一拦截]
    R -->|单文档制度| S1[simple_rag 单次检索]
    R -->|多文档对比| C1[complex_rag 拆解·多次检索]
    R -->|结构化数据| T[tool 工具 Agent function calling]
    S --> E[END]
    O --> E
    S1 --> G[generate_answer 受控生成]
    C1 --> G
    T -->|反问用户| E
    T -->|决策失败| G2[human_fallback 人工兜底]
    T -->|不支持 function calling| S1
    T -->|取回证据| G
    G --> F{异常检测}
    F -->|异常| G2
    F -->|正常| E
    G2 --> E"""
