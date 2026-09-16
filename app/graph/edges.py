"""条件分支路由 —— 决定工作流的动态流转路径。

三条条件边，判据都**只看状态里的事实，不看任何猜测**：

1. ``scene_route_edge``：路由 Agent 之后的五路分发（smalltalk / simple_rag /
   complex_rag / tool / out_of_scope）。
2. ``tool_route_edge``：工具 Agent 之后的去向——决策失败转人工；已有答案（追问
   用户）直接结束；模型不支持 function calling 则**改道简单 RAG**；其余进受控生成。
3. ``error_route_edge``：生成之后的异常兜底。

为什么"改道简单 RAG"落在边上而不是节点里
----------------------------------------
工具 Agent 只如实报告"我干不了活"（``degraded=True``），**不自己决定改走哪条**。
「工具路走不通该改走哪条」是一个**路由决策**，与"意图识别"是同一类判断；
把它写在节点内部（在 ``tool_node`` 里直接调用检索）会让拓扑在代码里消失——
读 ``workflow_graph.py`` 的人看不到这条边存在，而它每天都在生效。
"""
from app.core.router_agent import SCENE_SIMPLE_RAG, SCENES
from app.graph.state import GraphState


def scene_route_edge(state: GraphState) -> str:
    """路由 Agent 判定之后的五路分发。

    判据只有 ``scene`` 一个字段，且它是**路由 Agent 写进去的**——这里不重新
    翻译一次。重复判断（"场景是 X 但也许该按 Y 处理"）会制造第二处可能与事实
    不符的地方，而两处不一致时没人知道该信哪个。

    ``scene`` 不在闭集里时兜底到 ``simple_rag``：正常链路上 ``route_query``
    已经校验过，但状态也可能由脚本 / 测试 / 未来的新入口直接构造。
    LangGraph 遇到条件边返回一个不在映射表里的值时抛 ``KeyError``——
    那个异常会被误读成"图配置坏了"，而真实原因是"状态里有个没见过的场景名"。
    """
    scene = state.get("scene") or SCENE_SIMPLE_RAG
    return scene if scene in SCENES else SCENE_SIMPLE_RAG


def tool_route_edge(state: GraphState) -> str:
    """工具 Agent 之后的四分支。

    顺序即优先级，每一步的理由：

    - ``need_human`` → 人工兜底。**只有工具 Agent 决策阶段本身失败**
      （模型不可达、返回结构异常）才会置位。工具执行失败不算——那由
      ``tool_node`` 单独判断并写 ``need_human``（判据见 ``_failed_business_tools``）。
    - ``answer is not None`` → 结束。这是**直答出口**：模型没取到业务数据就直接
      说话了，典型场景是"必填参数缺失 → 主动向用户追问"。
      用 ``is not None`` 而不是真值判断是刻意的：空字符串代表「模型说了一句话
      但内容是空的」，那是一个需要被下游按无依据处理的异常，不是一条答案；
      用真值判断会把它和「没有答案」混成一种情况。
    - ``tool_degraded`` → 改道简单 RAG。模型不支持 function calling 时，
      用户问的多半仍是制度类问题；多检索一次远好于转人工。
    - 其余 → 受控生成。走到了证据出口，答案必须带引用编号与置信度。
    """
    if state.get("need_human"):
        return "human_fallback"
    if state.get("answer") is not None:
        return "end"
    if state.get("tool_degraded"):
        return SCENE_SIMPLE_RAG
    return "generate_answer"


def error_route_edge(state: GraphState) -> str:
    """生成之后的异常分支：只有**确实无法自动处理**时才转人工。

    判定依据**只看 need_human**，不看 error_msg。

    历史缺陷（docs/history/project-assessment.md P0-1）
    ------------------------------------------
    此前写的是 `need_human or error_msg`。而检索节点在检索失败时会写
    `error_msg`，但生成节点随后仍能正常产出答案 —— 于是这个条件在
    **答案生成之后**触发，把已经生成好的有效答案覆盖成「已转接人工」。
    后果：一次网络抖动，用户就拿不到本可正常返回的答案。

    现在的约定
    ----------
    - `need_human=True`：真正无法自动处理，转人工。三个来源：
      ① 工具 Agent 决策失败（模型不可达）；② 生成失败；③ **业务工具执行失败**。
    - 可降级故障只记 soft_warnings，不参与路由 —— 宁可给出诚实的降级回答，
      也不要把用户推给人工。

    ③ 为什么算「无法自动处理」而不是「可降级」
    ------------------------------------------
    判据是「**能不能从其他来源得到答案**」，不是「这次调用有没有报错」：

    - **检索失败** → 可降级。L4 仍能给出「知识库中没有找到相关信息」，
      这与真实语义一致，是诚实且可行动的回答；
    - **业务工具失败** → 转人工。用户问「李四的年假还剩几天」而数据库挂掉时，
      我们**没有任何别的来源**能回答，硬让 L4 回一句「知识库中没有找到相关信息」
      是答非所问——用户会以为这位同事没有记录，而不是「系统暂时查不了」。
    """
    if state.get("need_human"):
        return "human_fallback"
    return "end"
