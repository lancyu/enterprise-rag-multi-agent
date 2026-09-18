"""配置层的契约：**每个语义只许有一处实现**。

为什么单独守着它
----------------
配置是最容易"抄一份"的地方——抄一份不需要理解上下文，而且抄错了**不报错**：
同一个词在不同配置上得到相反的结果，只表现为"某个功能在 A 机器上是开的、
在 B 机器上是关的"，没有任何日志指向配置解析本身。

本文件守两条：
1. **布尔环境变量只经 :func:`app.config._env_bool` 解析**（曾经的漂移见下）；
2. **文档内容长度上限的两处语义是"报错"与"截断"，不是同一个东西**（见末尾一节）。

全部为纯单元判定，不联网、不调模型、不读写真实文件。
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest

from app import config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PY = _REPO_ROOT / "app" / "config.py"

#: 内联写法曾经长这样。它与 ``_env_bool`` 的判定集合**不同**——少了 ``"none"``——
#: 于是 ``MEMORY_ENABLED=none``（走 _env_bool → 假）与 ``CHUNK_CONTEXT_HEADER=none``
#: （走内联 → 真）会得到**相反**的结果。这里按文本扫，是为了它一旦被写回来就变红。
_INLINE_BOOL_IDIOM = '("0", "false", "no", "off")'


# ---------------------------------------------------------------------------
# 1. 布尔解析只许有一处
# ---------------------------------------------------------------------------
def test_no_inline_boolean_parsing_anywhere():
    """整个 ``app/`` 里都不许再出现内联的布尔判定集合。

    ⚠️ 反向验证：在 ``app/config.py`` 里随便找一个布尔配置，把
    ``_env_bool("X", False)`` 改回 ``os.getenv("X", "false").strip().lower() not in (...)``，
    本条必须变红。
    """
    offenders = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in (_REPO_ROOT / "app").rglob("*.py")
        if _INLINE_BOOL_IDIOM in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        f"这些文件里还有内联的布尔解析：{offenders} —— 它的判定集合与 _env_bool "
        "不一致（少一个 \"none\"），同一个词在不同配置上会得到相反的结果。"
    )


def _bool_config_assignments() -> List[ast.AnnAssign]:
    """``app/config.py`` 里所有 ``X: bool = ...`` 的模块级赋值。"""
    tree = ast.parse(_CONFIG_PY.read_text(encoding="utf-8"))
    out: List[ast.AnnAssign] = []
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        ann = node.annotation
        name = ann.id if isinstance(ann, ast.Name) else getattr(ann, "value", None)
        if name == "bool" and node.value is not None:
            out.append(node)
    return out


def _mentions_env(node: ast.AST) -> bool:
    """这段表达式是否从环境变量取过值。

    按 ``_env*`` 前缀认自家的小工具（``_env`` / ``_env_bool`` / ``_env_list`` …），
    而不是逐个枚举名字：以后新增一个 ``_env_int`` 忘了加进来，
    本判据就会把它当成"派生值"静默放过——那正是护栏最坏的失效方式。
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == "getenv":
            return True
        if isinstance(sub, ast.Name) and sub.id.startswith("_env"):
            return True
    return False


def test_every_env_derived_bool_goes_through__env_bool():
    """凡是从环境变量取来的布尔值，一律经 ``_env_bool``——AST 判据，不靠人眼。

    刻意用 AST 而不是正则：正则分辨不出 ``os.environ["X"]`` 与别的写法，
    也容易被注释里的示例骗过去。

    例外是**派生值**（不读环境变量、由别的配置算出来的），
    例如 ``USE_REAL_LLM = bool(LLM_API_KEY)``——它没有"判定集合"可言。
    判据因此写成"只要不碰环境变量就不管"，而不是留一份手写白名单：
    白名单会过期，而过期的方式是**静默放过**。
    """
    offenders = []
    checked = 0
    for node in _bool_config_assignments():
        if not _mentions_env(node.value):
            continue  # 派生值：不读环境变量，无判定集合
        checked += 1
        value = node.value
        is_env_bool = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_env_bool"
        )
        if not is_env_bool:
            offenders.append(f"{node.target.id}（第 {node.lineno} 行）：{ast.unparse(value)}")

    assert checked >= 15, (
        f"只扫到 {checked} 个读环境变量的布尔配置，太少 —— "
        "要么配置被搬走了，要么本判据失效了（护栏恒真是最坏的失效方式）"
    )
    assert offenders == [], (
        "这些布尔配置绕过了 _env_bool：\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("raw", ["none", "None", "NONE", " none ", "0", "off", "no", "false", "FALSE"])
def test_falsy_words_are_false(monkeypatch, raw):
    """``none`` 与其余否定词一律为假。

    为什么 ``"none"`` 归假，而不是"没给值"：对布尔开关来说，"给了个空词"
    最危险的解读是**开**——一个手写错的 ``=none`` 会静默开启某个本不该开的功能；
    反向解读最多是功能没开，看得见。**取错误方向不对称的那一侧。**
    """
    monkeypatch.setenv("_PROBE_BOOL", raw)
    assert config._env_bool("_PROBE_BOOL", True) is False


def test_absent_or_blank_falls_back_to_default(monkeypatch):
    """未设 / 空串 / 纯空白 → 用默认值，**不是** False。

    这条与上一条是两件事：``""`` 是"没填"，``"none"`` 是"填了一个否定的词"。
    合并二者会让"没配就是关"蔓延到所有默认打开的开关上。
    """
    for raw in (None, "", "   "):
        if raw is None:
            monkeypatch.delenv("_PROBE_BOOL", raising=False)
        else:
            monkeypatch.setenv("_PROBE_BOOL", raw)
        assert config._env_bool("_PROBE_BOOL", True) is True
        assert config._env_bool("_PROBE_BOOL", False) is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "y", "随便什么非空值"])
def test_anything_else_is_true(monkeypatch, raw):
    """非空即真——包括我们不认识的值。这是刻意放宽的：宁可多开，也不要静默关掉。"""
    monkeypatch.setenv("_PROBE_BOOL", raw)
    assert config._env_bool("_PROBE_BOOL", False) is True


# ---------------------------------------------------------------------------
# 2. 文档长度上限：**数字一处，语义两处（且两处都必须显式告知）**
# ---------------------------------------------------------------------------
#: 这条上限的**数值**曾经在两处各写了一遍：validator 里硬编码 50_000，
#: config 里 ``MAX_DOC_CONTENT_CHARS`` 默认 "50000"。改一处忘一处时，
#: 两个上传入口会对同一份文档给出不同结果，且都不报错。
#:
#: 判据用 **AST 找整数字面量**而不是正则扫文本：正则会被注释和 docstring 骗到——
#: 本文件第一版就是正则版，它把"说明自己曾经硬编码过 50_000"的那句注释也当成了违规。
#: 护栏误报的下场是被关掉，所以这里必须只认**真正会被执行的字面量**。
_DOC_LIMIT_VALUE = 50_000


def test_doc_length_limit_number_is_defined_once():
    """``50_000`` 这个数字只许有一个来源：``config.MAX_DOC_CONTENT_CHARS``。"""
    from app.utils import validator

    assert validator.MAX_DOC_CONTENT == config.MAX_DOC_CONTENT_CHARS


def test_validator_does_not_hardcode_the_limit_again():
    """``app/utils/validator.py`` 里不许再出现硬编码的 50_000 字面量。

    ⚠️ 反向验证：把 ``MAX_DOC_CONTENT = config.MAX_DOC_CONTENT_CHARS`` 改回
    ``MAX_DOC_CONTENT = 50_000``，本条必须变红。
    """
    path = _REPO_ROOT / "app" / "utils" / "validator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == _DOC_LIMIT_VALUE
    ]
    assert offenders == [], (
        f"validator.py 第 {offenders} 行又出现了硬编码的长度上限 —— "
        "它与 config.MAX_DOC_CONTENT_CHARS 是两份数字，改一处忘一处时"
        "两个上传入口会对同一份文档给出不同结果。"
    )


def test_two_upload_paths_handle_over_limit_differently_and_say_so():
    """超限策略在两处**故意不同**，但两处都必须把结果显式告知调用方。

    这条守的不是"行为一致"（那会把它们强行统一到错的一侧），而是
    "差异是有意的、且看得见"：
      - JSON 接口（``POST /knowledge/upload``）：拒收 → 有 422/400 错误；
      - 文件接口（``POST /knowledge/upload/file``）：截断 → 响应体里有 ``truncated``。
    """
    upload = (_REPO_ROOT / "app" / "api" / "knowledge.py").read_text(encoding="utf-8")
    assert '"truncated": truncated' in upload, "文件上传超限截断后必须回 truncated，否则用户以为整篇都入库了"
    assert "MAX_DOC_CONTENT_CHARS" in upload, "文件上传的上限也必须取自 config，不能另写一个数"
    assert "truncated = len(content) > max_chars" in upload, "截断判据不见了——本用例的载体已失效，请同步更新"
