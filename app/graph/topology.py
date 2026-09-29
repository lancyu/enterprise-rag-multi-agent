"""工作流拓扑的**唯一呈现来源** —— 把「事实」与「呈现」分开。

为什么需要这个模块
------------------
2026-09-29 之前，同一个拓扑在本仓库被**手写三遍**：

1. ``app/graph/workflow_graph.py::get_mermaid()`` —— 手写的 mermaid 常量字符串；
2. ``app/api/workflow.py`` 的 ``NODE_LABELS`` / ``BRANCHES`` —— 手写的声明
   （那个文件的注释还写着「单一事实来源」，其实它是第三份副本）；
3. ``app/static/index.html`` —— 手画的拓扑 SVG。

面向读者的文档里还有第四、第五遍。守它们的是一套**用正则猜措辞**的测试护栏，
而 2026-09-24 那条 ``tool → verifier`` 边摘掉又接回时，三份副本真的分叉过。

本模块把两件事分开：

- **事实** = 编译图（``enterprise_workflow.get_graph()``）。唯一，且**由代码派生**。
- **呈现** = 下面两张表：节点别名与图例、每条边在图上写什么标签。

渲染函数只做一件事：**按事实（边集）遍历呈现表**。事实里出现、表里没有 →
``ungrounded()`` 报出来，**不静默跳过**。反过来，表里多出一条事实里没有的边，
同样报出来 —— 那正是「照着旧文档改代码」的来源。

为什么呈现表不写进 ``workflow_graph.py``
----------------------------------------
那张表是**文案**（别名、中文图例、分支标签），改它不需要重新编译图；而
``workflow_graph.py`` 是图的构造处。分开还有一个硬理由：本模块**零内部依赖**
（只 import 标准库），所以它可以被 ``workflow_graph`` 反向 import 而不会成环。
"""
from __future__ import annotations

from dataclasses import dataclass

#: LangGraph 编译产物里的伪节点名。
END_NODE = "__end__"
START_NODE = "__start__"

#: 图终点在 mermaid 里的别名。
END_ALIAS = "E"


@dataclass(frozen=True)
class NodeSpec:
    """一个节点怎么**呈现**（与它是什么无关）。"""

    #: mermaid 里的别名。必须唯一，且只能由字母数字下划线组成。
    alias: str
    #: mermaid 框里跟在节点名后面的短说明（渲染成 ``别名[节点名 短说明]``）。
    caption: str
    #: 给接口 / 面板用的中文名。``/workflow/status`` 的 ``nodes[].label`` 就是它。
    label: str


@dataclass(frozen=True)
class EdgeSpec:
    """一条边怎么**呈现**。"""

    #: mermaid 写在 ``-->|这里|`` 里的分支标签；``None`` 表示无条件边，不画标签。
    caption: str | None = None


#: 拓扑的绘制顺序 —— 只影响渲染出来的行序，不影响集合语义。
NODE_ORDER: tuple = (
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

NODE_PRESENTATION: dict = {
    "memory_load": NodeSpec("M", "身份解析·记忆加载", "请求初始化·身份解析·记忆加载"),
    "router": NodeSpec("R", "路由 Agent 意图识别·边界管控", "路由 Agent（意图识别·边界管控）"),
    "smalltalk": NodeSpec("S", "闲聊·模板直答", "闲聊 Agent（模板直答）"),
    "out_of_scope": NodeSpec("O", "统一拦截", "越界拦截（常量话术）"),
    "simple_rag": NodeSpec("S1", "单次检索", "简单 RAG Agent（单次检索）"),
    "complex_rag": NodeSpec("C1", "拆解·多次检索", "复杂 RAG Agent（拆解·多次检索）"),
    "tool": NodeSpec("T", "工具 Agent function calling", "工具 Agent（function calling）"),
    "verifier": NodeSpec("V", "证据校验·只判对齐", "证据校验（按需触发）"),
    "generate_answer": NodeSpec("G", "受控生成", "受控生成"),
    "human_fallback": NodeSpec("G2", "人工兜底", "人工兜底"),
}

#: 每条边的图上标签。键是编译图里的 ``(出发节点, 目标节点)``，**逐条对应**。
#: 这里只放「这条边在图上写什么」，不放「这条边存不存在」——后者由编译图说了算。
EDGE_PRESENTATION: dict = {
    ("memory_load", "router"): EdgeSpec(),
    ("router", "smalltalk"): EdgeSpec("寒暄"),
    ("router", "out_of_scope"): EdgeSpec("越界"),
    ("router", "simple_rag"): EdgeSpec("单文档制度"),
    ("router", "complex_rag"): EdgeSpec("多文档对比"),
    ("router", "tool"): EdgeSpec("结构化数据"),
    ("smalltalk", END_NODE): EdgeSpec(),
    ("out_of_scope", END_NODE): EdgeSpec(),
    ("simple_rag", "verifier"): EdgeSpec(),
    ("complex_rag", "verifier"): EdgeSpec(),
    ("tool", "verifier"): EdgeSpec("取回证据"),
    ("tool", "human_fallback"): EdgeSpec("决策失败"),
    ("tool", END_NODE): EdgeSpec("反问用户"),
    ("tool", "simple_rag"): EdgeSpec("不支持 function calling"),
    ("verifier", "generate_answer"): EdgeSpec("对齐"),
    ("verifier", "router"): EdgeSpec("不符且还有预算"),
    ("generate_answer", "human_fallback"): EdgeSpec("异常"),
    ("generate_answer", END_NODE): EdgeSpec("正常"),
    ("human_fallback", END_NODE): EdgeSpec(),
}

#: 条件分支节点的补充说明（``/workflow/status`` 的 ``branches[].note``）。
#: 这是**解释**，不是声明：哪条边存在由编译图说了算，这里只解释「为什么这么连」。
BRANCH_NOTES: dict = {
    "router": "五个场景各自独立；越界在入口拦下，不进任何子 Agent。二次判定（被退回时）也走这里",
    "tool": (
        "决策失败→人工兜底；反问用户→结束；不支持 function calling→改道简单 RAG；"
        "取回证据→证据校验（按需触发，出过状况才复核）"
    ),
    "verifier": (
        "证据与问题不符且未超重预算→退回路由重判（图里唯一回边）；其余→受控生成。"
        "由 simple_rag / complex_rag / tool 三个证据出口进入"
    ),
    "generate_answer": "生成阶段的运行时异常→人工兜底；正常→结束",
}


def compiled_nodes(graph) -> set:
    """编译图里的真实节点（滤掉 ``__start__`` / ``__end__`` 这类伪节点）。"""
    return {
        name
        for name in (getattr(graph, "nodes", {}) or {})
        if not name.startswith("__")
    }


def compiled_edges(graph) -> set:
    """编译图里的真实边 —— 去掉入口伪节点出发的那一条。

    ``__start__ → memory_load`` 不是拓扑声明，是 LangGraph 表达「入口是谁」的方式。
    它渲染成 mermaid 的首行，不跟其它边并列。
    """
    return {
        (edge.source, edge.target)
        for edge in graph.get_graph().edges
        if edge.source != START_NODE
    }


def conditional_sources(graph) -> set:
    """条件分支点集合（从编译产物派生，不是写死的）。"""
    return {
        edge.source
        for edge in graph.get_graph().edges
        if getattr(edge, "conditional", False) and not edge.source.startswith("__")
    }


def ungrounded(nodes: set, edges: set) -> list:
    """呈现表与编译图的**双向**接地检查，返回问题清单（空 = 一致）。

    - 事实里有、表里没有 → 渲染时会被跳过（mermaid 少一条箭头、面板少一条线），
      而**画面看起来仍然很正常**；
    - 表里有、事实里没有 → 渲染出一个不存在的流转，读者会照着它改代码。

    两个方向都要报。返回消息里带上具体符号，方便直接照着改。
    """
    problems = []

    for name in sorted(nodes - set(NODE_PRESENTATION)):
        problems.append(f"节点 `{name}` 在编译图里，但 NODE_PRESENTATION 里没有它（图上会缺一个框）")
    for name in sorted(set(NODE_PRESENTATION) - nodes):
        problems.append(f"NODE_PRESENTATION 里的 `{name}` 不在编译图里（图上会多一个框）")

    for src, dst in sorted(edges - set(EDGE_PRESENTATION)):
        problems.append(f"边 `{src} → {dst}` 在编译图里，但 EDGE_PRESENTATION 里没有它（图上会缺一条箭头）")
    for src, dst in sorted(set(EDGE_PRESENTATION) - edges):
        problems.append(f"EDGE_PRESENTATION 里的 `{src} → {dst}` 不在编译图里（图上会多一条箭头）")

    return problems


def _order_key(edge: tuple) -> tuple:
    """渲染顺序：按 :data:`NODE_ORDER` 排出发节点，再排目标节点。"""
    src, dst = edge
    head = NODE_ORDER.index(src) if src in NODE_ORDER else len(NODE_ORDER)
    tail = NODE_ORDER.index(dst) if dst in NODE_ORDER else len(NODE_ORDER)
    return (head, tail, dst)


def _alias(name: str) -> str:
    """节点名 → mermaid 别名。"""
    return END_ALIAS if name == END_NODE else NODE_PRESENTATION[name].alias


def _decl(name: str) -> str:
    """mermaid 里一个节点的**首次**声明：``别名[节点名 短说明]``。"""
    if name == END_NODE:
        return f"{END_ALIAS}[END]"
    spec = NODE_PRESENTATION[name]
    return f"{spec.alias}[{name} {spec.caption}]"


def render_mermaid(edges: set) -> str:
    """从**边集**派生 mermaid 源码。

    与手写版本的区别不是外观，是**来源**：手写版本里少画一条边不会报错，
    这里少一条边就少一个 ``EDGE_PRESENTATION`` 条目，:func:`ungrounded` 会当场报出来。
    """
    lines = ["flowchart TD"]
    seen: set = set()

    def token(name: str) -> str:
        """首次出现给 ``别名[标签]``，之后只给别名 —— 同一别名重复声明是可读性噪音。"""
        if name in seen:
            return _alias(name)
        seen.add(name)
        return _decl(name)

    for src, dst in sorted(edges, key=_order_key):
        caption = EDGE_PRESENTATION[(src, dst)].caption
        arrow = f"-->|{caption}|" if caption else "-->"
        lines.append(f"    {token(src)} {arrow} {token(dst)}")
    return "\n".join(lines)


def node_labels() -> dict:
    """``/workflow/status`` 的 ``nodes[].label`` —— 从呈现表派生。"""
    return {name: NODE_PRESENTATION[name].label for name in NODE_ORDER}


def branch_declarations(edges: set, sources: set) -> list:
    """``/workflow/status`` 的 ``branches`` —— 条件分支**从编译产物派生**。

    ``routes`` 不再手写：它就是编译图里该出发节点的所有出边（``__end__`` 写作
    ``"END"``）。手写时代价是明确的 —— 加一条分支要记得同时改这里，
    而忘了改的表现是「接口返回的拓扑少一条路径」，界面上看不出来。
    """
    ordered = sorted(
        sources,
        key=lambda n: NODE_ORDER.index(n) if n in NODE_ORDER else len(NODE_ORDER),
    )
    return [
        {
            "from": src,
            "type": "conditional",
            "routes": [
                "END" if dst == END_NODE else dst
                for node, dst in sorted(edges, key=_order_key)
                if node == src
            ],
            "note": BRANCH_NOTES.get(src, ""),
        }
        for src in ordered
    ]
