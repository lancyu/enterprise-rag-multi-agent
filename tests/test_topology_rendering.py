"""拓扑的每一种呈现都必须等于编译图 —— 三类呈现、一套判据。

背景：为什么不是「散文正则」
----------------------------
同一个拓扑曾经被**手写三遍**：``get_mermaid()`` 的常量字符串、
``api/workflow.py`` 的 ``NODE_LABELS`` / ``BRANCHES``、``static/index.html`` 的手画 SVG。
守它们的曾是一套**用正则猜措辞**的判据（3 条规则 / 7 条命中正则 / 15 条探针）。

它判的是**措辞**而不是**事实**。2026-09-29 实测，三句语义等价、意思相反的错话
**全部漏过**：

    「工具 Agent 拿到的数据不参与校验。」
    「工具 Agent 的输出不做证据核对。」
    「function calling 的结果无需复核即可进 L4。」

同一批判据还**假报**过 —— 它必须给自家文档的历史复盘开白名单
（"把这条判据删掉，顺手把 tool → verifier 这条边也摘了"）。

**判据要判事实，不要判事实的代理。** 这跟 ``verifier`` 触发判据里撤掉的那三条代理
信号（``scene_source != router:local``、``if tool_result:`` …）是同一个错误。

结构改完之后的判据
------------------
事实 = 编译图（唯一）；文案 = ``app/graph/topology.py`` 的呈现表。于是：

1. **呈现表接地** —— 表里少一个节点/一条边，图上就少一个框/一条箭头，而画面
   看起来仍然很正常；
2. **mermaid 箭头集 == 编译图边集**（双向）—— 派生的好处不是"不会错"，是
   "错了会被解析出来"：渲染器跳过一条边，这里立刻红；
3. **面板 ``data-edge`` 集合 == 编译图边集**（双向），且每条箭头的**起止点几何上
   确实落在它声明的两个方框上** —— 文案改了箭头没改，纯文本扫描抓不到它；
4. **面板方框覆盖全部节点**；
5. **文档里以代码 span 写的边引用**（`` `tool → verifier` ``）必须是真的边 ——
   这是可机械判定的那部分散文，判**符号**而不是措辞。

判据自检（每类都要）见各自用例里的 ``assert ... "判据失配"``：**判据扫不到东西时
变绿是假绿**，与 ``tests/test_layering.py`` 同一条纪律。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.graph.topology import (  # noqa: E402
    END_NODE,
    NODE_PRESENTATION,
    compiled_edges,
    compiled_nodes,
    render_mermaid,
    ungrounded,
)
from app.graph.workflow_graph import enterprise_workflow  # noqa: E402

#: 扫描面。**五份缺一不可**：2026-09-29 那次漂移里 ``docs/README.md`` 是唯一被漏掉的
#: 一份，也正是它让「只改了三份」的自查看起来像是通过了。
SCANNED_FILES: tuple = (
    "README.md",
    "docs/README.md",
    "docs/multi-agent-architecture.md",
    "docs/project-introduction.md",
    "app/static/index.html",
)

#: 面板拓扑图。``index.html`` 里还有若干图标 SVG，靠这个 viewBox 认出来。
PANEL_VIEWBOX = "0 0 1000 430"

_EDGES = compiled_edges(enterprise_workflow)
_NODES = compiled_nodes(enterprise_workflow)


def _read_scanned() -> dict:
    return {rel: (REPO_ROOT / rel).read_text(encoding="utf-8") for rel in SCANNED_FILES}


# ---------------------------------------------------------------------------
# 第一类：呈现表接地（少一个条目 = 图上少一个框 / 一条箭头，且不报错）
# ---------------------------------------------------------------------------
def test_presentation_table_is_grounded():
    """``topology`` 的呈现表与编译图必须**双向**一致。"""
    assert not ungrounded(_NODES, _EDGES)


def test_every_compiled_node_has_a_unique_mermaid_alias():
    """别名必须唯一 —— 两个节点共用一个别名，`get_mermaid()` 会把它们画成同一个框。"""
    aliases = [spec.alias for spec in NODE_PRESENTATION.values()]
    assert len(aliases) == len(set(aliases)), f"别名重复：{sorted(aliases)}"


# ---------------------------------------------------------------------------
# 第二类：mermaid 的箭头集
# ---------------------------------------------------------------------------
_ALIAS_DEF_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*\[([^\]\[]*)\]")
_ARROW_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*(?:\[[^\]]*\])?)\s*-->(?:\|[^|]*\|)?\s*"
    r"([A-Za-z][A-Za-z0-9_]*(?:\[[^\]]*\])?)"
)


def _bare(token: str) -> str:
    return token.split("[", 1)[0]


def _mermaid_edges(text: str) -> set:
    """把 ``render_mermaid()`` 的产出解析成 ``(出发节点, 目标节点)`` 集合。

    mermaid 里写的是**别名**（``M`` / ``V`` / ``G2``），别名在 ``别名[标签]`` 里定义，
    标签的**第一个词**是真实节点名 —— 这是 ``render_mermaid()`` 的书写约定。
    """
    labels = {m.group(1): m.group(2).strip() for m in _ALIAS_DEF_RE.finditer(text)}
    #: 别名 → 节点名；END 映射回 ``__end__``。
    mapping: dict = {}
    for alias, label in labels.items():
        head = label.split()[0] if label.split() else alias
        if head == "END":
            mapping[alias] = END_NODE
        elif head in _NODES:
            mapping[alias] = head

    edges: set = set()
    for m in _ARROW_RE.finditer(text):
        src, dst = mapping.get(_bare(m.group(1))), mapping.get(_bare(m.group(2)))
        assert src and dst, f"箭头 {m.group(0)!r} 里出现了没定义过的别名（判据失配）"
        edges.add((src, dst))
    return edges


def test_mermaid_parser_is_not_silently_empty():
    """判据自检：解析不出边就是**判据失配**，不许静默变绿。"""
    assert len(_mermaid_edges(render_mermaid(_EDGES))) >= 10, (
        "mermaid 解析器一条边都没解析出来——改的是解析器还是 render_mermaid 的写法？"
    )


def test_mermaid_draws_exactly_the_compiled_edges():
    """双向断言。

    - 画了图里没有的边 → 读者以为存在一条不存在的流转；
    - 漏画图里有的边 → 读者以为某个节点是孤立的。
    """
    declared = _mermaid_edges(render_mermaid(_EDGES))
    assert not declared - _EDGES, f"mermaid 画了编译图里不存在的边：{sorted(declared - _EDGES)}"
    assert not _EDGES - declared, f"mermaid 漏画了编译图里存在的边：{sorted(_EDGES - declared)}"


# ---------------------------------------------------------------------------
# 第三类：面板 SVG —— 声明 + 几何
# ---------------------------------------------------------------------------
def _panel_svg() -> str:
    html = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    start = html.index(f'<svg viewBox="{PANEL_VIEWBOX}"')
    return html[start : html.index("</svg>", start)]


def _rect_distance(point, box) -> float:
    """点到矩形框的距离（落在框内为 0）。"""
    px, py = point
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0, px - x1)
    dy = max(y0 - py, 0, py - y1)
    return float((dx * dx + dy * dy) ** 0.5)


def _panel_boxes(svg: str) -> dict:
    """面板里 ``{节点名: (x0, y0, x1, y1)}``。

    约定：每个节点是一个 ``<rect>``，其后紧跟的第一条 ``<text>节点名</text>`` 就是它的
    标题；``END`` 是一个 ``<circle>``（没有矩形框）。
    """
    boxes: dict = {}
    for m in re.finditer(r'<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"', svg):
        x, y, w, h = (int(g) for g in m.groups())
        found = re.search(r">([A-Za-z_]+)</text>", svg[m.end() : m.end() + 400])
        if found and found.group(1) in _NODES:
            boxes[found.group(1)] = (x, y, x + w, y + h)

    circle = re.search(r'<circle cx="(\d+)" cy="(\d+)" r="(\d+)"', svg)
    assert circle, "面板 SVG 里找不到 END 的圆形节点"
    cx, cy, r = (int(g) for g in circle.groups())
    boxes[END_NODE] = (cx - r, cy - r, cx + r, cy + r)
    return boxes


def _panel_arrows(svg: str) -> list:
    """面板拓扑箭头 ``[(声明边, d), ...]`` —— 只认带 ``data-edge`` 的 ``<path>``。"""
    return [
        (tuple(m.group(1).split(">")), m.group(2))
        for m in re.finditer(r'<path data-edge="([^"]+)" d="([^"]+)"', svg)
    ]


def _path_points(d: str) -> tuple:
    """path 的起点与终点（本图只用 M / C / L，末两个数就是终点）。"""
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
    assert len(nums) >= 4, f"path 里读不出起止坐标：{d!r}"
    return (nums[0], nums[1]), (nums[-2], nums[-1])


def test_panel_boxes_cover_every_compiled_node():
    """每个节点都要在图上有一个框，且不许出现编译图里没有的框。

    ``__end__`` 是图的终点（不是一个节点），面板把它画成圆形 —— 单独排除。
    """
    boxes = set(_panel_boxes(_panel_svg()))
    assert not _NODES - boxes, f"面板缺这些节点的方框：{sorted(_NODES - boxes)}"
    assert not boxes - _NODES - {END_NODE}, (
        f"面板有编译图里不存在的方框：{sorted(boxes - _NODES - {END_NODE})}"
    )


def test_panel_declares_exactly_the_compiled_edges():
    """每条箭头用 ``data-edge`` 声明它画的是哪条边，声明集合必须等于编译图。

    2026-09-29 本条**立刻抓到一处真缺陷**：面板只画了 18 条箭头，编译图有 19 条 ——
    少的正是 ``tool → END``（"反问用户 · 已有答案"）。它此前无人发现，因为旧判据
    只校验**一条**箭头的几何，而正则表只看文字、从不看 SVG。
    """
    declared = {edge for edge, _ in _panel_arrows(_panel_svg())}
    assert not declared - _EDGES, f"面板画了编译图里不存在的边：{sorted(declared - _EDGES)}"
    assert not _EDGES - declared, f"面板漏画了编译图里存在的边：{sorted(_EDGES - declared)}"


def test_panel_arrow_endpoints_land_on_the_declared_boxes():
    """几何判据：每条箭头的起点离**它声明的出发框**最近，终点离**声明的目标框**最近。

    这条抓的是「文案改了、箭头没改」—— 2026-09-24 那次漂移里 ``tool`` 的箭头就
    指向过 ``generate_answer``。箭头方向不写在任何一句话里，纯文本扫描抓不到。

    **双向都判**：只看终点的话，把整条箭头挪到别处（比如从 ``router`` 直连
    ``verifier``）同样能通过 —— 那条边在拓扑里并不存在。
    """
    svg = _panel_svg()
    boxes = _panel_boxes(svg)
    arrows = _panel_arrows(svg)
    assert arrows, "面板里一条带 data-edge 的箭头都没有——判据失配，别让它静默变绿"

    problems = []
    for (src, dst), d in arrows:
        start, end = _path_points(d)
        near_start = min(boxes, key=lambda n: _rect_distance(start, boxes[n]))
        near_end = min(boxes, key=lambda n: _rect_distance(end, boxes[n]))
        if near_start != src:
            problems.append(
                f"`{src} → {dst}` 的起点 {start} 离 `{near_start}` "
                f"({_rect_distance(start, boxes[near_start]):.0f}px) 比离 `{src}` "
                f"({_rect_distance(start, boxes[src]):.0f}px) 更近"
            )
        if near_end != dst:
            problems.append(
                f"`{src} → {dst}` 的终点 {end} 离 `{near_end}` "
                f"({_rect_distance(end, boxes[near_end]):.0f}px) 比离 `{dst}` "
                f"({_rect_distance(end, boxes[dst]):.0f}px) 更近"
            )
    assert not problems, "面板箭头的落点与它声明的边不一致：\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# 第四类：文档里可机械判定的那部分散文 —— 边引用（代码 span 形式）
# ---------------------------------------------------------------------------
#: `` `A → B` ``：A / B 都是真实节点名（或 END）。**判符号，不判措辞。**
_EDGE_REF_RE = re.compile(
    r"`(" + "|".join(sorted(_NODES, key=len, reverse=True)) + r")\s*→\s*(END|"
    + "|".join(sorted(_NODES, key=len, reverse=True))
    + r")`"
)


def test_document_edge_references_are_real():
    """文档里以代码 span 写的边引用必须是真的边。

    这条替换掉了旧的「散文正则规则表」：旧表枚举 7 种**措辞**，换个说法就漏
    （2026-09-29 实测 3/3 漏过）；这里判的是**符号**，文档只要用 `` `A → B` `` 这种
    写法点名一条边，无论前后文怎么写都会被查到。
    """
    problems = []
    for rel, text in _read_scanned().items():
        for match in _EDGE_REF_RE.finditer(text):
            src, dst = match.group(1), match.group(2)
            edge = (src, END_NODE if dst == "END" else dst)
            if edge not in _EDGES:
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{rel}:{line} 引用了不存在的边 `{src} → {dst}`")
    assert not problems, "文档点名了编译图里不存在的边：\n  " + "\n  ".join(problems)


def test_the_edge_reference_scan_is_not_silently_empty():
    """判据自检：这条正则必须真的扫得到东西。

    它扫不到东西时是**假绿** —— 而「扫不到」最常见的原因是有人把文档里的
    `` `A → B` `` 改成了自然语言（或反过来，正则被写错）。两种情况都必须响。
    """
    hits = sum(len(_EDGE_REF_RE.findall(text)) for text in _read_scanned().values())
    assert hits >= 5, (
        f"五份文件里只扫到 {hits} 处代码 span 形式的边引用（预期 ≥5）。"
        f"要么文档不再用 `A → B` 这种可判定的写法（那请换一种可判定的写法），"
        f"要么正则写错了——判据扫不到东西时变绿是假绿。"
    )


# ---------------------------------------------------------------------------
# 第五类：扫描面本身，以及「N 节点 / M 条件边」在其余文档里的口径
# ---------------------------------------------------------------------------
def test_scan_surface_includes_every_file_the_drift_touched():
    """扫描面必须含有那五份 —— 尤其是当初被漏掉的 ``docs/README.md``。

    这条防的是「护栏自己被缩小」：把一份文件从 ``SCANNED_FILES`` 里删掉，
    它上面的所有断言都会安静地少查一份，而测试仍然是绿的。
    """
    for rel in SCANNED_FILES:
        assert (REPO_ROOT / rel).is_file(), f"扫描面里的 {rel} 不存在"
    assert "docs/README.md" in SCANNED_FILES, (
        "docs/README.md 是 2026-09-29 漏掉的那一份，它必须在扫描面里"
    )


def test_counts_in_the_remaining_documents_are_current():
    """把「N 节点 / M 条件边」的口径补齐到其余文档。

    只认「条件边」这个说法，**不认「条件分支」** —— 后者在 ``docs/README.md`` 里
    正是用来描述**历史口径**的（"8 节点 / 2 条件分支"），不该被算作现行声明。
    """
    pattern = re.compile(r"(\d+)\s*节点\s*/\s*(\d+)\s*条件边")
    conditional_sources = {
        edge.source
        for edge in enterprise_workflow.get_graph().edges
        if getattr(edge, "conditional", False) and not edge.source.startswith("__")
    }
    expected = (len(_NODES), len(conditional_sources))

    checked = 0
    for rel, text in _read_scanned().items():
        for match in pattern.finditer(text):
            checked += 1
            line = text.count("\n", 0, match.start()) + 1
            assert (int(match.group(1)), int(match.group(2))) == expected, (
                f"{rel}:{line} 写「{match.group(0)}」，"
                f"编译图实际是 {expected[0]} 节点 / {expected[1]} 条件边"
            )
    assert checked, "一条「N 节点 / M 条件边」都没扫到——判据失配，别让它静默变绿"
