"""条件分支路由 —— 决定工作流的动态流转路径。

四条条件边，判据都**只看状态里的事实，不看任何猜测**：

1. ``scene_route_edge``：路由 Agent 之后的五路分发（smalltalk / simple_rag /
   complex_rag / tool / out_of_scope）。
2. ``tool_route_edge``：工具 Agent 之后的去向——决策失败转人工；已有答案（追问
   用户）直接结束；模型不支持 function calling 则**改道简单 RAG**；其余交给
   ``verifier`` 复核（**按需**：干净的轮次零代价跳过），由它决定进生成还是退回重判。
   判据里**没有**"走了工具链路"这一条，理由见该函数的 docstring（这是一处
   2026-09-24 两次定案、方向刚好相反的地方，别只看结论不看过程）。
3. ``verifier_route_edge``：证据校验之后的去向——证据与问题不符且**还有重试预算**
   则**退回路由重判**；否则进受控生成。这是图里唯一的一条回边。
4. ``error_route_edge``：生成之后的异常兜底。

为什么"改道简单 RAG"落在边上而不是节点里
----------------------------------------
工具 Agent 只如实报告"我干不了活"（``degraded=True``），**不自己决定改走哪条**。
「工具路走不通该改走哪条」是一个**路由决策**，与"意图识别"是同一类判断；
把它写在节点内部（在 ``tool_node`` 里直接调用检索）会让拓扑在代码里消失——
读 ``workflow_graph.py`` 的人看不到这条边存在，而它每天都在生效。
``verifier_route_edge`` 的"预算判断"出于同一条理由，见其 docstring。
"""
from app import config
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
    - 其余 → **证据出口**，先交给 ``verifier`` 复核（按需：干净的轮次零代价跳过），
      再由它决定进受控生成还是退回重判。理由见下一节。

    工具路为什么**要**接证据校验（2026-09-24 二次定案）
    --------------------------------------------------
    这里走过三段，整套记下来，因为它是一个**方向性错误**的标本：

    ① ``if tool_result:`` —— "走了工具链路就复核"。这是**代理信号**：判据讲的是
       "业务数据没有第二道防线"，而它描述的风险与要防的失败不是同一件事——工具结果
       是自己库里的一行结构化记录，压根不存在让检索偏掉的机制（片段被切碎 / Top1
       分数不达标 / 多路融合选错），**结构上不可能"硬凑"**。后果是**每轮工具请求
       都白付一次复核**：中位 1551ms，占整轮 17%~25%，而 54 次判定 MISMATCH = 0。

    ② 把这条判据删掉，**顺手把这条边也摘了** —— 过度治疗。删判据解决的是"每轮都付"，
       边解决的是"**出问题时有没有人复核**"，两件事被一起丢掉了。于是工具侧真正的
       可疑信号（调用被护栏拒绝 / 执行抛异常 / 正文回捞）虽然照旧写进 ``soft_warnings``
       + trace + 前端面板，却**再也没有复核环节**——留痕不等于兜底。

    ③ **本函数回到 ``"verifier"``**（2026-09-24 二次定案）。判据仍然是按需的，那条
       代理信号**没有**跟着回来，所以"每轮都付"不会重现：

       - **干净的工具轮**（调用正常、拿到数据、无降级痕迹）→ ``should_verify``
         在 ``soft_warnings`` / 软回退 / 路由摇摆上都不命中 → **零代价跳过，直接进 L4**
         —— 这正是"调完工具就按结果回答"该有的样子；
       - **出过状况的轮次** → 复核一次；判为不符且还有预算时由 ``verifier_route_edge``
         **退回路由重判**（图里唯一那条回边，有上界）—— 这就是兜底。

    两个条件必须同时成立，缺一个这条边就是**白接**的：

    - ``tool_result`` 要重新传进校验（见 ``app/core/verifier.py`` 的入参）：工具链路的
      ``docs`` 是空的，证据**全在** ``tool_result`` 里。不传，``should_verify`` 会在
      "没有证据可校验"那一条上短路，复核永远不会发生。
    - 判据里**不许**再出现"走了工具链路"：它一旦回来，边缘立刻退化回 ①。

    **覆盖面如实记下**：调用被拒 / 执行异常 / 正文回捞 / 路由摇摆这些轮次会被复核。
    **仍然兜不住的一类**："调用成功、数据可用，但工具本身选错了"（问部门却查了年假）
    —— 它**没有本地信号**，要抓到它只能每轮都判。要那种强度就把开关切到
    ``VERIFIER_MODE=always``（代价即 ① 里的那 1551ms/轮）。

    **留痕与兜底是两件事**（这一条是 ② 那步错误的根源）。工具侧的可疑信号
    （调用被护栏拒绝 / 执行抛异常 / 正文回捞）在 ``soft_warnings`` 里留痕，只说明
    "事后查得出来"；**它不等于这一轮做了补救**。两件事都要：
    留痕负责可观测，复核负责当场兜住。执行失败仍然另有出口——转人工
    （见 ``app/graph/nodes.py::_failed_business_tools``）。
    """
    if state.get("need_human"):
        return "human_fallback"
    if state.get("answer") is not None:
        return "end"
    if state.get("tool_degraded"):
        return SCENE_SIMPLE_RAG
    return "verifier"


def verifier_route_edge(state: GraphState) -> str:
    """证据校验之后的去向：受控生成 / **退回路由重判**。

    三条判据，顺序即优先级：

    - ``evidence_aligned is False`` 且**还有重试预算** → ``router``。
      这是图里**唯一的一条回边**，也是"下游可以否决上游"的落点：路由 Agent
      是单点入口，但从此不再一锤定音——取回证据后发现判错了路，可以把事实
      退回入口重判，而不是硬着头皮往下生成。没有它，误判的唯一表现就是
      "用户拿到一个自信但不对的答案"，而链路里没有任何一处会报错。
      判据写作 ``reroute_count <= ROUTE_RETRY_BUDGET``（计数由 ``verifier_node``
      在判定不符时 +1）：预算 1 时第 1 次不符可退回，第 2 次不符就走生成。
    - ``evidence_aligned is False`` 但预算用尽 → ``generate_answer``。
      预算用尽走的是**正常路径**，不是异常：两次判定都指向同一条路，说明问题
      多半本来就在证据侧（知识库里就是没有这一条）。此时 L4 会按无依据给出
      「知识库中没有找到相关信息」——诚实、可行动，远好于把用户推给人工。
    - 其余（``True`` / ``None``）→ ``generate_answer``。``None`` 是"没做过校验"，
      与"校验通过"一并放行：放行是安全方向（理由见 ``app/core/verifier.py``）。

    为什么预算判断在边上而不在节点里：与 ``tool_route_edge`` 把"改道简单 RAG"
    放在边上同一个理由——「退回还是继续」是一个**路由决策**。写在 ``verifier_node``
    内部会让这条回边在代码里消失，读 ``workflow_graph.py`` 的人看不到它存在，
    而它每天都在生效。

    ``config.ROUTE_RETRY_BUDGET`` 是这条环路的**唯一刹车**：verifier → router
    是图里唯一的环，没有上界就是死循环（LangGraph 的递归上限会以异常收场，
    那不是"降级"，是整轮失败）。
    """
    if state.get("evidence_aligned") is False:
        if int(state.get("reroute_count") or 0) <= config.ROUTE_RETRY_BUDGET:
            return "router"
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
