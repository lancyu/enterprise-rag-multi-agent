"""工作流的「拓扑声明」必须与编译产物一致，否则不许通过。

背景（P0-5 缺陷 C：门禁只守「行号对得上」，不守「话是对的」）
--------------------------------------------------------------
行号校验器把上千条声明守到全绿的同时，仓库里却长期写着**条件边有 4 条**
（实际一直是 3 条），而 `app/graph/edges.py` 自述「三条条件边」——
同一仓库 5 种说法。行号校验器发现不了它，因为**它从不比较「声明」与「编译产物」**。

P0-5 统一了口径（以代码为准：9 节点 / 3 条件边），但那一轮只改到文档层，
下面三处仍写着 4，于是 README 正文说「3 条条件边」、配图里却印着「4 条件分支」：

1. `app/graph/workflow_graph.py` 的模块 docstring；
2. 同文件的启动日志（写死字面量）；
3. `app/static/index.html` 的工作流面板横幅。

本组用例把这三处钉死在**编译图**上。与 `test_chunk_keys.py` / `test_meta_align.py`
同源：都是**同一语义在两处各定义一遍**——这里的一处是"文档/界面说的"，
另一处是"图里真的有的"。
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.api.workflow import BRANCHES  # noqa: E402
from app.graph import workflow_graph as wg  # noqa: E402

#: 「N 节点 / M 条件边」——口径以代码为准，故措辞固定为「条件边」而非「条件分支」。
_TOPOLOGY_RE = re.compile(r"(\d+)\s*节点\s*/\s*(\d+)\s*条件边")


def _actual_branch_sources() -> set:
    """编译图里真正的条件分支点（出发节点）。"""
    return {
        edge.source
        for edge in wg.enterprise_workflow.get_graph().edges
        if getattr(edge, "conditional", False)
    }


def _parse_or_fail(text: str, where: str) -> re.Match:
    """按 `_TOPOLOGY_RE` 抽取声明；抽不到就**失败**。

    宁可报「判据失配」也不许静默通过——判据扫不到东西时变绿是假绿，
    这正是本组用例要防的那类问题（见 `test_layering.py` 的判据自检）。
    """
    match = _TOPOLOGY_RE.search(text)
    assert match, (
        f"{where} 里找不到「N 节点 / M 条件边」的声明——判据失配。"
        f"若确实改了措辞，请同步更新本文件的 _TOPOLOGY_RE，而不是删掉这条断言。"
    )
    return match


# ---------------------------------------------------------------------------
# 1. 界面横幅
# ---------------------------------------------------------------------------
def test_panel_banner_matches_compiled_graph():
    """面板横幅的「N 节点 / M 条件边」必须等于编译图实际。"""
    html = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    match = _parse_or_fail(html, "app/static/index.html 的工作流面板横幅")

    assert int(match.group(1)) == len(wg.NODE_NAMES), (
        f"横幅写 {match.group(1)} 个节点，编译图实际 {len(wg.NODE_NAMES)} 个"
    )
    assert int(match.group(2)) == len(_actual_branch_sources()), (
        f"横幅写 {match.group(2)} 条条件边，编译图实际 {len(_actual_branch_sources())} 条"
        f"（{sorted(_actual_branch_sources())}）"
    )


# ---------------------------------------------------------------------------
# 2. 模块 docstring
# ---------------------------------------------------------------------------
def test_module_docstring_matches_compiled_graph():
    """模块 docstring 顶部的拓扑声明同样不许漂。"""
    match = _parse_or_fail(
        wg.__doc__ or "", "app/graph/workflow_graph.py 的模块 docstring"
    )
    assert int(match.group(1)) == len(wg.NODE_NAMES)
    assert int(match.group(2)) == len(_actual_branch_sources())


# ---------------------------------------------------------------------------
# 3. 拓扑声明（api 层的单一事实来源）
# ---------------------------------------------------------------------------
def test_declared_branches_cover_exactly_the_compiled_branch_points():
    """`BRANCHES` 的出发节点集合必须与编译图的条件分支点**完全相等**。

    少一条 = 面板漏画一条边；多一条 = 面板画了一条不存在的边。
    两者都是「声明与产物分叉」，而分叉是静默的（界面照样渲染）。
    """
    declared = {b["from"] for b in BRANCHES}
    assert declared == _actual_branch_sources(), (
        f"声明 {sorted(declared)} != 实际 {sorted(_actual_branch_sources())}"
    )


# ---------------------------------------------------------------------------
# 4. 反向验证：数字必须派生，不能写死
# ---------------------------------------------------------------------------
def test_startup_log_derives_the_count_instead_of_hardcoding():
    """启动日志的条件边数必须来自 `conditional_branch_count()`。

    把 `logger.info(... conditional_branch_count(compiled))` 换回字面量，
    本用例必须变红——写死字面量正是当初漂成 4 条的成因。
    """
    src = inspect.getsource(wg.build_workflow_graph)
    assert "conditional_branch_count(" in src, (
        "启动日志又写死条件边数了；请改回 conditional_branch_count(compiled)"
    )


def test_branch_count_helper_is_derived_not_constant():
    """`conditional_branch_count` 本身必须真的去读图，而不是返回常量。

    用临时拼出来的图反证：分支点数变了，返回值必须跟着变。

    两个从 LangGraph 里试出来的约束，都影响这个函数的写法：

    - **同一出发节点挂不了两条同名条件边**（`Branch with name ... already
      exists`），所以夹具里每个分支点用不同的出发节点；
    - **`get_graph()` 只画从入口可达的边**，不可达的分支点根本不会出现在
      边表里，故夹具把各节点串成一条链。
    """
    from langgraph.graph import END, StateGraph

    def _build(n_branch_points: int):
        graph = StateGraph(dict)
        names = [f"s{i}" for i in range(n_branch_points)]
        for name in names:
            graph.add_node(name, lambda s: s)
        graph.add_node("sink", lambda s: s)
        graph.set_entry_point(names[0])
        for idx, name in enumerate(names):
            nxt = names[idx + 1] if idx + 1 < n_branch_points else "sink"
            graph.add_conditional_edges(name, lambda s, nxt=nxt: nxt, {"next": nxt, "end": END})
        graph.add_edge("sink", END)
        return graph.compile()

    assert wg.conditional_branch_count(_build(1)) == 1
    assert wg.conditional_branch_count(_build(3)) == 3
