"""工作流组装编译 —— 构建企业级 LangGraph 有向图。

拓扑结构（10 节点 / 4 条件边）::

    memory_load（身份解析 + 长期记忆）
      → router（路由 Agent：意图识别 + 边界管控）
           ├─ smalltalk ──────────────────────────────→ END（模板直答）
           ├─ out_of_scope ───────────────────────────→ END（常量话术，不生成）
           ├─ simple_rag ─┐
           ├─ complex_rag ┤
           └─ tool ───────┴─┐   （tool 的三个旁路出口见下）
                            ↓
                     verifier（**按需**复核：有理由怀疑才判"这份证据答的是不是这个问题"）
                            ├─ 不符且还有重试预算 ──→ router（退回重判，图里唯一回边）
                            └─ 其余 ─┐
                                      ↓
                       generate_answer（引用 + 置信度 + 拒答）
                                 ├─ 生成失败 → human_fallback
                                 └─ 正常 ───→ END

    tool 的三个旁路出口（都在到达 verifier 之前离开主路）：
      决策失败 → human_fallback ／ 反问用户（已有答案）→ END ／
      不支持 function calling → simple_rag（改道，回到校验）

为什么分成"路由 + 四个执行者"而不是一个大 Agent
----------------------------------------------
单 Agent 版本由**一个** Agent 在一次 function calling 决策里既决定"要不要检索"
又决定"要不要查业务数据"。合并的问题不是慢，而是**失败模式不可分**：模型不调
工具时有三种互不相干的原因（寒暄 / 参数没给全 / 忘了调），它们在日志里长得
一模一样。拆开之后每类失败都有归属，且边界规则只有一份（在入口）。

``verifier`` 补的是**最后一类没有归属的失败**：证据取回了、结构完好、字段齐全，
语义上却答非所问。它同时是"下游可以否决上游"的出口。

**三个证据出口都汇到它**（``simple_rag`` / ``complex_rag`` / ``tool``）。触发则是
**按需**的：判据只看"这一轮有没有可疑迹象"，**不看"这是哪条链路"**。"走了工具
链路"这条判据属于**代理信号**（判据和风险不是同一件事），已删除；而这条边本身在
2026-09-24 经历过"**摘掉又接回**"两次方向相反的定案，完整过程与覆盖面的取舍写在
``app/graph/edges.py::tool_route_edge`` —— **别只看结论不看过程**（只看到"摘掉"
那一半，下次就会再摘一次）。

它是**修复手段，不是常规工序**：大多数轮次的证据是对的，每轮都请裁判等于把
"大多数时候不需要的服务"变成固定成本（实测 1.2~3.0 秒/轮，且 20 次采样零命中）。
所以它默认**按需出手**（``config.VERIFIER_MODE=auto``）——判据全部取自前置阶段
已经算出来的事实，见 ``app/core/verifier.should_verify``。``always`` 是随时可切回
的回退路径，**对三条链路都生效**：想连"调用成功、数据可用、但工具选错了"那一类
也兜住，就切它（代价是每轮工具请求都要多付一次复核）。

图里唯一的环（``verifier → router``）为什么是必要的，且为什么是安全的
---------------------------------------------------------------------
路由是单点入口，误判的代价由整条链路承担；而**发现误判所需的证据**只有下游
才有（"取回的东西答的不是这个问题"这件事，在入口处无法判断）。所以否决必须
从下游回到上游，环是这条信息的**唯一**通道。

安全性来自两条，缺一不可：

1. **判据只看事实**：``evidence_aligned is False``（校验真的跑了、真的判了不符），
   ``None``（没校验）不放行也不改道；
2. **有上界**：``config.ROUTE_RETRY_BUDGET`` 限制退回次数，超过就进生成。
   没有这条上界，LangGraph 会撞上递归上限并以异常收场——那不是"降级"，是整轮失败。

保留 generate_answer 的理由值得单独说明：子 Agent 只负责**取回证据**，不负责
组织成话。片段编号 ``[1][2]``、引用溯源、置信度拒答、流式输出都在 L4 一处产生——
让模型在工具循环里自由成文，等于把这四项能力从唯一的产生点拆成两处，
两处必然漂移，而漂移是静默的（答案看起来都很流利）。

``verifier`` 与 ``generate_answer`` 也是同一条理由：**裁判必须在生成之前**。
若把"证据对不对"塞进 generate_answer，生成完再判要么白生成一次、要么拒绝一份
已写好的答案（把可恢复的错误变成不可恢复的），且该节点会同时不可测。

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

from app.graph.edges import (
    error_route_edge,
    scene_route_edge,
    tool_route_edge,
    verifier_route_edge,
)
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
    verifier_node,
)
from app.graph.state import GraphState
from app.graph.topology import (
    compiled_edges as _compiled_edges,
    render_mermaid as _render_mermaid,
)
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
    "verifier",
    "generate_answer",
    "human_fallback",
)


def conditional_branch_count(compiled) -> int:
    """从**编译图**派生「有几个条件分支点」——不写死字面量。

    为什么不能直接数边：LangGraph 的 ``get_graph()`` 会把一条条件边按目标节点
    **展开**成多条（``router`` 那一条展开成 5 条），边长远大于分支点数。按**出发
    节点**去重才是「有几个条件分支点」——本图是 4（``router`` / ``tool`` /
    ``verifier`` / ``generate_answer``），与 ``app/graph/edges.py`` 的「四条条件边」
    一致。

    为什么要派生而不是写常量：这个数字曾被写死为 4，而当时实际一直是 3，于是启动
    日志、本模块 docstring、前端横幅三处**一起**错，与 README / project-introduction
    的「3 条条件边」自相矛盾（P0-5 记「同一仓库 5 种说法」）。数字只要还能被写死，
    就还会漂——这里改成从编译产物读，改图时不可能忘记同步。
    """
    edges = compiled.get_graph().edges
    return len({edge.source for edge in edges if getattr(edge, "conditional", False)})


def _wire(graph: StateGraph, *, generation_target: str) -> None:
    """装配生成之前的全部节点与分支；``generation_target`` 决定**生成之前那一段的终点**。

    Args:
        graph: 待装配的 StateGraph。
        generation_target: 证据链路走完后的去向节点名。完整图传
            ``"generate_answer"``；流式端点的前置图传 ``END``（生成由端点自己
            逐 token 完成）。

    它有**一个**消费方：``verifier_route_edge``（校验通过 / 校验未生效 / 预算用尽）。
    三个证据出口（``simple_rag`` / ``complex_rag`` / ``tool``）**全部先经过
    ``verifier``**，再由它决定"进生成"还是"退回路由重判"——所以工具链路不直接
    消费这个目标。这一点在 2026-09-24 被改错过一次（把 ``tool → verifier`` 摘掉，
    于是这里凭空多出第二个消费方），过程见 ``app/graph/edges.py::tool_route_edge``。

    三个出口**都装** ``verifier``，不是只在完整图里装：流式链路上取回的证据同样
    可能答非所问，而它的答案同样会被推给用户。差别只在于校验通过后是进
    ``generate_answer`` 还是直接结束（流式端点自己生成）。
    """
    graph.add_node("memory_load", memory_load_node)
    graph.add_node("router", router_node)
    graph.add_node("smalltalk", smalltalk_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("simple_rag", simple_rag_node)
    graph.add_node("complex_rag", complex_rag_node)
    graph.add_node("tool", tool_node)
    graph.add_node("verifier", verifier_node)
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

    # 工具 Agent 之后的四分支。取到证据那一路**进校验**（2026-09-24 二次定案：
    # 摘掉过一次又接回来），判据仍按需，理由见 app/graph/edges.py::tool_route_edge。
    graph.add_conditional_edges(
        "tool",
        tool_route_edge,
        {
            "human_fallback": "human_fallback",
            "end": END,
            "simple_rag": "simple_rag",
            "verifier": "verifier",
        },
    )

    # 三个证据出口（两个检索类 Agent + 工具 Agent）一律先进校验，
    # 由它决定进生成还是退回路由重判。
    graph.add_edge("simple_rag", "verifier")
    graph.add_edge("complex_rag", "verifier")

    # 图里唯一的一条回边：verifier → router（上界由 ROUTE_RETRY_BUDGET 兜住）
    graph.add_conditional_edges(
        "verifier",
        verifier_route_edge,
        {"router": "router", "generate_answer": generation_target},
    )

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

    与完整图的唯一差别是"证据就绪之后的去向"：这里通向 END（生成由端点自己逐
    token 完成），完整图通向 ``generate_answer``。节点与条件边全部复用
    （见 ``_wire``），因此两条链路的前置结果**结构上不可能不一致**；
    ``tests/test_multi_agent.py`` 直接断言这一点。
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
    """工作流拓扑的 Mermaid 描述（供前端渲染）。

    从**编译图的边集**派生，不是手写常量 —— 出边少了/多了会在这里表现为
    ``topology.ungrounded()`` 报错（由 ``tests/test_topology_rendering.py`` 守），
    而不是「前端图上悄悄少一条箭头」。呈现文案（别名 / 分支标签）在
    ``app/graph/topology.py``。
    """
    return _render_mermaid(_compiled_edges(enterprise_workflow))
