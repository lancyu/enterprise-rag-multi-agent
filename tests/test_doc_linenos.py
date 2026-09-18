#!/usr/bin/env python3
"""文档行号校验器自己的回归测试（P1-7）—— 门禁不能没有护栏。

守什么
------
`scripts/verify_doc_linenos.py` 是**唯一**守着几百条行号声明的机制，但在此之前
`tests/` 下没有**任何**一个用例 import 或调用它（全仓只有 `fix_doc_linenos.py` 引用）。
它自己那十三类声明的验证，历来靠**写进 `/tmp/` 的一次性变异脚本**，跑完即弃——
第 11、12、13 类都是这么验的。

代价已经付过一次：给第 13 类加「找不到定位标记就报错」这条守卫时，前置条件漏写，
立刻产生误报（拿另一份文档去跑才暴露）。**能守住别人的机制，自己无人守。**

所以本文件做三件事：

1. **把门禁搬进 pytest**：真文档必须全绿（等于把 `python scripts/verify_doc_linenos.py`
   从"每次记得手跑"变成"跑 pytest 就跑了"）；
2. **逐类证明校验器还有牙齿**：给文档副本施加一处变异，对应的那一类必须报出来。
   只测"现在绿不绿"是不够的——一条**恒真**的校验器和没有校验器一样糟；
3. **证明它克制**：某些改动**必须保持绿色**（中文量词后的数字不是行号），
   否则它会把正确的文档报成错误，用久了就会被人加 `# noqa` 关掉。

两个设计要点（都来自项目已踩过的坑）
------------------------------------
① **变异施加在副本上**：真文档一字不动。变异写错时最坏情况只是副本坏了；
② **按标记子串定位，且载体找不到时 fail 而不是 skip**：skip 会让用例在文档结构
   变化后静默失效——那正是本项目的「护栏恒真」坑。见
   `test_mutation_helper_refuses_to_run_when_its_anchor_is_gone`。
"""
from __future__ import annotations

import functools
import os
import pathlib
import re
import subprocess
import sys
from typing import Callable, List

import pytest

from verify_doc_linenos import DEFAULT_DOC, verify

ROOT = pathlib.Path(__file__).resolve().parents[1]
_DOC = ROOT / DEFAULT_DOC


# ---------------------------------------------------------------------------
# 变异机制
# ---------------------------------------------------------------------------
def _bump(anchor: str, nth: int = 0, delta: int = 1) -> Callable[[str], str]:
    """返回一个变异：把**唯一**一处 `anchor` 所在行里的第 `nth` 个数字加 `delta`。

    `nth` 数的是**整行**里的第几个数字（含锚点自身里的数字，如标题里的 `3.2`），
    这样锚点写成什么样都不影响计数。

    锚点必须唯一 —— 否则改到的可能是另一处（变异位置错了，用例的结论也就没意义）。
    """

    def _apply(text: str) -> str:
        hits = [m.start() for m in re.finditer(re.escape(anchor), text)]
        assert len(hits) == 1, (
            f"锚点 {anchor!r} 在文档里出现 {len(hits)} 次（要求恰好 1 次）。\n"
            "载体变了：请更新本用例的锚点，**不要**把它改成 skip —— "
            "skip 会让这条护栏在文档结构变化后静默失效。"
        )
        pos = hits[0]
        line_start = text.rfind("\n", 0, pos) + 1
        line_end = text.find("\n", pos)
        end = len(text) if line_end == -1 else line_end
        line = text[line_start:end]
        nums = list(re.finditer(r"\d+", line))
        assert len(nums) > nth, (
            f"锚点 {anchor!r} 所在行只有 {len(nums)} 个数字，取不到第 {nth} 个：{line!r}"
        )
        target = nums[nth]
        return (
            text[: line_start + target.start()]
            + str(int(target.group()) + delta)
            + text[line_start + target.end() :]
        )

    return _apply


def _replace(old: str, new: str) -> Callable[[str], str]:
    """返回一个变异：把唯一一处 `old` 换成 `new`（不是数字也能改）。"""

    def _apply(text: str) -> str:
        assert text.count(old) == 1, (
            f"锚点 {old!r} 出现 {text.count(old)} 次（要求恰好 1 次）；载体会变，请更新锚点"
        )
        return text.replace(old, new)

    return _apply


@pytest.fixture
def doc_copy(tmp_path):
    """把真文档复制一份并施加变异，返回副本路径。**真文档一字不动。**"""
    original = _DOC.read_text(encoding="utf-8")

    def _make(*ops: Callable[[str], str]) -> pathlib.Path:
        text = original
        for op in ops:
            text = op(text)
        copy = tmp_path / "doc-under-test.md"
        copy.write_text(text, encoding="utf-8")
        return copy

    return _make


@pytest.fixture
def problems(doc_copy):
    """施加变异 → 跑校验器 → 返回问题清单。"""

    def _run(*ops: Callable[[str], str]) -> List[str]:
        return verify(str(doc_copy(*ops)))

    return _run


def _assert_reported(items: List[str], needle: str) -> None:
    assert items, "变异之后校验器一个问题都没报 —— 这一类声明已经没人守着了"
    assert any(needle in item for item in items), (
        f"报出来的问题里没有 {needle!r}：\n  " + "\n  ".join(items)
    )


@pytest.fixture
def mini_doc(tmp_path):
    """写一份**最小可校验文档**（只含「松散单元格 + 符号行号」这一种写法）。

    作用是把单条判据**隔离**出来：真文档里想让某一个数字落在两种判据的边界上，
    得先凑巧存在那样一行（本项目查过，没有）。这里直接造出来。
    """

    def _make(row: str) -> List[str]:
        text = (
            "#### 📍 其余模块\n\n"
            "| 文件 | 关键行号 | 要点 |\n"
            "|---|---|---|\n"
            f"{row}\n"
        )
        path = tmp_path / "mini.md"
        path.write_text(text, encoding="utf-8")
        return verify(str(path))

    return _make


@pytest.fixture(scope="module", autouse=True)
def _cache_source_index():
    """把源码索引缓存到本模块。

    `verify()` 每次调用都会重新 AST 解析 `app/ + tests/ + scripts/`（实测 ~0.5s），
    而本文件要跑十几次变异。缓存只活在本测试进程内，不改生产代码，
    退出时恢复原函数，不跨模块泄漏。
    """
    import verify_doc_linenos as v

    original = (v.collect_symbols, v.collect_any_files)
    v.collect_symbols = functools.lru_cache(maxsize=None)(v.collect_symbols)
    v.collect_any_files = functools.lru_cache(maxsize=None)(v.collect_any_files)
    try:
        yield
    finally:
        v.collect_symbols, v.collect_any_files = original


# ---------------------------------------------------------------------------
# 1. 门禁本身（P1-1：把校验器纳入 pytest，四道本地门禁变三道）
# ---------------------------------------------------------------------------
def test_the_real_document_is_fully_consistent():
    """真文档必须全绿。这是把门禁 ④ 搬进 pytest 的那一条。"""
    items = verify(str(_DOC))
    assert items == [], (
        "文档里有行号声明与源码不一致：\n  " + "\n  ".join(items)
        + "\n改完代码后跑 `python scripts/fix_doc_linenos.py --write` 回填。"
    )


def test_the_cli_exits_zero_on_the_real_document():
    """命令行入口本身也要能跑通（退出码 0 才是门禁语义）。"""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_doc_linenos.py"), str(_DOC)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert proc.returncode == 0, f"退出码 {proc.returncode}：\n{proc.stdout[-2000:]}"
    assert "全部与源码 AST 一致" in proc.stdout, proc.stdout[-1000:]


def test_the_cli_exits_nonzero_on_a_broken_document(doc_copy):
    """反向：文档坏了必须给出非 0 退出码，否则它永远"绿"。"""
    broken = doc_copy(_bump("| `lifespan` | ", 0))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_doc_linenos.py"), str(broken)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert proc.returncode == 1, f"退出码 {proc.returncode}，stdout：\n{proc.stdout[-1000:]}"


# ---------------------------------------------------------------------------
# 2. 每一类都得有牙齿（十三类逐类反向验证）
# ---------------------------------------------------------------------------
def test_class1_symbol_row_drift_is_reported(problems):
    """第 1 类：表格里的符号行号。"""
    _assert_reported(problems(_bump("| `lifespan` | ", 0)), "app/main.py::lifespan")


def test_class2_file_total_range_drift_is_reported(problems):
    """第 2 类：``| `app/x.py` | **1-N** |`` 形式的文件总行数。"""
    _assert_reported(problems(_bump("| `app/main.py` | **1-")), "app/main.py 文档写")


def test_class3_appendix_bare_number_drift_is_reported(problems):
    """第 3 类：附录「完整文件索引」里的两列裸数字表。"""
    _assert_reported(problems(_bump("| `config.py` | ")), "app/config.py 文档写")


def test_class4_subtotal_drift_is_reported(problems):
    """第 4 类：章节小计 ``### 3.2 `app/api/` — …（N 行）``。"""
    _assert_reported(problems(_bump("### 3.2 `app/api/` — HTTP 接口层（", 2)), "app/api/ 文档写")


def test_class5_module_head_range_drift_is_reported(problems):
    """第 5 类：小标题 ``#### 📍 `app/config.py`（1-N）`` 括号里的范围。"""
    _assert_reported(problems(_bump("#### 📍 `app/config.py`（1-")), "app/config.py 文档写")


def test_class6_loose_cell_symbol_drift_is_reported(problems):
    """第 6 类：松散单元格 ``| `prompts.py` | `PROMPTS` 24-221 / … |``。

    **写成区间的必须精确**：原先只判「被包含」，于是 ``PROMPTS`` 24-186
    （真实跨度 24-221，少写 35 行）一路绿灯 —— 包含判据对「跨度变长」完全不敏感。
    本轮把区间收紧为精确比对后，全文档只多报出这一处，**零误报**。
    """
    _assert_reported(problems(_bump("| `prompts.py` | `PROMPTS` ", 1)), "PROMPTS")


def test_class6_single_number_is_only_a_starting_point(mini_doc):
    """反向约束：单数字（``| `prompts.py` | `PROMPTS` 24 |``）只声明**起点**。

    ``PROMPTS`` 是多行赋值（真实跨度 24-221），所以「24」本身不等于它的完整跨度。
    如果判据一刀切成「必须精确」，这类正确的写法会被整片报成错误 ——
    这正是收紧时**只收紧区间、不动单数字**的原因。
    """
    assert mini_doc("| `prompts.py` | `PROMPTS` 24 |") == []


def test_class6_a_range_in_the_same_cell_must_still_be_exact(mini_doc):
    """同一个单元格里换成区间写法时，同一个数字就必须精确了。"""
    items = mini_doc("| `prompts.py` | `PROMPTS` 24-186 |")
    assert any("PROMPTS" in item for item in items), items


def test_class7_prose_reference_out_of_range_is_reported(problems):
    """第 7 类（越界）：散文引用 ``app/x.py:A-B`` 超出文件总行数。"""
    _assert_reported(problems(_bump("`app/rag/__init__.py:1-", 1)), "越界")


def test_class10_prose_reference_must_hit_a_symbol_exactly(problems):
    """第 10 类（精确性）：改完代码后旧范围**仍在界内**，却已指向别的函数。

    这是最隐蔽的一类 —— 越界是硬错、会当场报错，不精确是静默漂移。
    实测一次重构后本文档 24 条散文引用里有 9 条这样漂移。
    """
    _assert_reported(problems(_bump("`app/core/tracing.py:131-", 2)), "未精确命中任何符号")


def test_class8_generic_file_line_count_drift_is_reported(problems):
    """第 8 类：通用文件行数（架构图里那种**没加反引号**的写法也要覆盖）。"""
    _assert_reported(problems(_bump("全局配置：app/config.py（")), "app/config.py 文档写")


def test_class9_region_table_out_of_range_is_reported(problems):
    """第 9 类：区域行号表（裸区间）。改到界外必须报出来。"""
    _assert_reported(problems(_bump("| 1-57 | 模块 docstring", 0, delta=-1)), "越界")


def test_class11_context_inherited_symbol_drift_is_reported(problems):
    """第 11 类：**不带文件名**的符号引用（靠 ``#### 📍`` 小标题继承上下文）。"""
    _assert_reported(problems(_bump("| `CHANNELS` | 70 | 包内别名")), "CHANNELS")


def test_class12_parenthesised_range_drift_is_reported(problems):
    """第 12 类：括号式区间引用 `` `GraphState`（A-B）``。"""
    _assert_reported(problems(_bump("改 `GraphState`（")), "GraphState")


def test_class13_config_partition_drift_is_reported(problems):
    """第 13 类：`app/config.py` 的「配置分区（按行号）」表。"""
    _assert_reported(problems(_bump("| 58-65 | 项目路径 |", 1)), "配置分区表第")


def test_class13_reports_when_its_own_marker_disappears(problems):
    """第 13 类的「别退化成空转」守卫。

    本文有 ``#### 📍 `app/config.py` `` 小节，却找不到分区表的定位标记时**必须报错**：
    否则改掉那行小标题，本类就悄悄变成空转，而门禁照样是绿的（护栏恒真）。
    这正是 2026-09-17 补这一类时踩到的那个前置条件。
    """
    _assert_reported(problems(_replace("配置分区（按行号）：", "配置分区：")), "定位标记")


# ---------------------------------------------------------------------------
# 3. 它也必须克制（该绿的要绿）
# ---------------------------------------------------------------------------
def test_a_number_followed_by_a_chinese_quantifier_is_not_a_line_number(problems):
    """数字后面跟中文量词时，它是个**计数**，不是行号 —— 不能报。

    ``| `CHANNELS` | 70 次 | 包内别名 |`` 与 ``| `CHANNELS` | 70 | 包内别名 |``
    的区别只在一个字，但前者改数字不该触发任何校验。
    """
    items = problems(_replace("| `CHANNELS` | 70 | 包内别名", "| `CHANNELS` | 70 次 | 包内别名"))
    assert items == [], "中文量词后的数字被当成了行号（误报）：\n  " + "\n  ".join(items)


def test_mutation_helper_refuses_to_run_when_its_anchor_is_gone():
    """变异机制的**载体找不到时必须 fail**，不能 skip。

    skip 会让本文件所有用例在文档结构变化后一起静默失效 ——
    那和「护栏恒真」是同一类问题，只是更隐蔽。
    """
    with pytest.raises(AssertionError, match="出现 0 次"):
        _bump("这个锚点在文档里绝对不存在")(_DOC.read_text(encoding="utf-8"))
