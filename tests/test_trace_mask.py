#!/usr/bin/env python3
"""trace 落盘前的脱敏层（P0-4）。

守什么
------
``span(name, **attrs)`` 与 ``span.attrs[k] = v`` 都是**开放字段**。今天没人往里写
正文，所以 ``logs/trace.jsonl`` 里没有正文 —— 但那是约定，不是机制。
约定不会在有人写下 ``span("retrieve", query=query)`` 的那天生效，
而那一天不会有任何东西报警。

所以本文件守三件事：
1. **真的造一份含凭据的 trace，走完整链路（``begin_trace`` → ``span`` → ``end_trace``），
   到 ``trace.jsonl`` 里搜那些凭据** —— 搜到 = 泄漏；
2. **脱敏发生在 ``_persist`` 之前**，且落盘与下发**是同一份数据**（口径唯一）；
3. **默认档不改变任何既有字段** —— 语料从 AST 派生 + 本地真实 trace 全量过一遍，
   逐字比对。脱敏如果把现有可观测性打坏了，它上线第一天就会被关掉，
   然后永远不再是护栏。

为什么"关掉开关"也要有用例
--------------------------
只测"开着时不泄漏"是不够的：把开关写成一个恒真的装饰（配置项读了但没用，
或默认值写死成 True 而没人能关）也能让那些用例全绿。所以另有一条用例**把开关关掉，
断言凭据真的原样出现在文件里** —— 它证明这个开关确实在起作用。
"""
from __future__ import annotations

import ast
import json
import pathlib
from typing import Dict, Iterator, List, Set

import pytest

from app import config
from app.core import trace_mask
from app.core.tracing import begin_trace, end_trace, span

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APP_DIR = _REPO_ROOT / "app"

#: 探针：四类必须被替换掉的东西 + 一段必须被截断的正文。
_FAKE_KEY = "sk-live-9Qm2Xr7TvBnKp1Ld8Wz"            # sk- 前缀密钥
_FAKE_EMAIL = "zhangsan@internal-corp.example"       # 邮箱
_FAKE_PHONE = "13901234567"                          # 手机号
_FAKE_ID = "11010119900307123X"                      # 身份证
_LONG_BODY = "年度调薪政策的适用范围是" + "全体正式员工" * 40   # 远超默认上限


def _emit_trace(**attrs) -> None:
    """走真实链路发出一次 trace：``begin_trace`` → ``span`` → ``end_trace``。"""
    begin_trace("trace-mask-test")
    with span("mask_probe") as node:
        node.attrs.update(attrs)
    end_trace(persist=True)


def _read_trace_jsonl(log_dir: pathlib.Path) -> str:
    path = log_dir / "trace.jsonl"
    assert path.exists(), "end_trace(persist=True) 没有写出 trace.jsonl"
    return path.read_text(encoding="utf-8")


def _capture_warnings(monkeypatch) -> List[str]:
    """把 logger 的 WARNING 收集起来（本项目 logger 是普通对象，直接替方法）。"""
    captured: List[str] = []

    def _record(msg, *args, **kwargs):
        captured.append(msg % args if args else msg)

    monkeypatch.setattr("app.utils.logger.logger.warning", _record)
    return captured


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """把日志目录指到 tmp，并复位「每进程一次」的告警标记（避免用例间顺序依赖）。"""
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    trace_mask.reset_warnings()
    yield


# ---------------------------------------------------------------------------
# 1. 行为：凭据与长正文不得落到磁盘
# ---------------------------------------------------------------------------
def test_fake_credentials_never_reach_trace_jsonl(tmp_path) -> None:
    """假密钥 / 邮箱 / 手机号 / 身份证 / 长正文，一个都不许在文件里出现。

    ⚠️ 反向验证：把 ``end_trace`` 里的 ``mask_tree(...)`` 换成裸的
    ``[r.to_dict(base_start) for r in roots]``，本条必须变红。
    """
    _emit_trace(
        api_used=_FAKE_KEY,
        email=_FAKE_EMAIL,
        phone=_FAKE_PHONE,
        badge=_FAKE_ID,
        snippet=_LONG_BODY,
    )
    raw = _read_trace_jsonl(tmp_path)

    for secret in (_FAKE_KEY, _FAKE_EMAIL, _FAKE_PHONE, _FAKE_ID):
        assert secret not in raw, f"trace.jsonl 里出现了原始凭据：{secret}"
    assert _LONG_BODY not in raw, "trace.jsonl 里落进了完整正文"
    # 长正文必须留下「本来有多长」这条诊断信息，否则脱敏把可观测性一起砍了
    assert f"<共 {len(_LONG_BODY)} 字>" in raw
    for placeholder in ("<REDACTED>", "<EMAIL>", "<PHONE>", "<ID>"):
        assert placeholder in raw, f"没有产生占位符 {placeholder}，规则可能整条失效"


def test_long_snippet_keeps_a_readable_head(tmp_path) -> None:
    """默认档（64）下，超长文本保留前 64 字 —— 这是有意的取舍，不是漏打码。

    理由：调试时要能看出「这是哪一段正文」。要彻底不留原文就设
    ``TRACE_MASK_MAX_CHARS=0``（见 ``test_strict_limit_leaves_only_lengths``）。
    """
    body = "A" * 100 + "SECRET-TAIL"
    _emit_trace(snippet=body)
    raw = _read_trace_jsonl(tmp_path)

    assert "SECRET-TAIL" not in raw, "尾部没有被截掉，正文仍然完整落盘"
    assert f"{'A' * 64}…<共 {len(body)} 字>" in raw, "没有按「前 N 字…<共 M 字>」截断"


def test_strict_limit_leaves_only_lengths(tmp_path, monkeypatch) -> None:
    """``TRACE_MASK_MAX_CHARS=0`` = 严格档：任何字符串只留长度，不留一个字原文。"""
    monkeypatch.setattr(config, "TRACE_MASK_MAX_CHARS", 0)
    body, enum = "短正文也算正文", "policy"
    _emit_trace(snippet=body, scene=enum)
    raw = _read_trace_jsonl(tmp_path)

    assert body not in raw
    assert enum not in raw, "严格档下连枚举值也不该留原文（这正是该档的定义）"
    assert f"<文本 {len(body)} 字>" in raw and f"<文本 {len(enum)} 字>" in raw


# ---------------------------------------------------------------------------
# 2. 行为：开关不是装饰
# ---------------------------------------------------------------------------
def test_switching_it_off_really_leaks(tmp_path, monkeypatch) -> None:
    """关掉开关 → 凭据**原样**落盘。

    这条用例的存在意义是「证伪」：如果哪天开关被写成恒真（配置读了但没用、
    或默认值把用户的选择覆盖掉），它会立刻变红。

    ⚠️ 反向验证：把 ``config.TRACE_MASK_ENABLED`` 的默认值改成 ``True`` 且
    在 ``trace_mask.is_enabled`` 里写死 ``return True``，本条必须变红。
    """
    monkeypatch.setattr(config, "TRACE_MASK_ENABLED", False)
    warnings = _capture_warnings(monkeypatch)

    _emit_trace(email=_FAKE_EMAIL)
    raw = _read_trace_jsonl(tmp_path)

    assert _FAKE_EMAIL in raw, "开关关掉了却还是脱敏了 —— 这个开关是装饰品"
    # 「原样落盘」是一种运行姿态，必须留痕；否则没人知道日志里全是原文
    assert any("TRACE_MASK_ENABLED=false" in w for w in warnings)


def test_disabled_warning_is_emitted_once_not_per_request(monkeypatch) -> None:
    """告警每个进程只打一次 —— 每条请求都刷会把日志淹掉，反而看不见。"""
    monkeypatch.setattr(config, "TRACE_MASK_ENABLED", False)
    warnings = _capture_warnings(monkeypatch)

    for _ in range(5):
        trace_mask.mask_tree({"a": "b"})

    assert len(warnings) == 1, f"告警打了 {len(warnings)} 次，应当只打一次"


def test_bad_limit_falls_back_to_the_stricter_side(tmp_path, monkeypatch) -> None:
    """非法的负数上限按 **0（严格档）** 处理：配错了应当多脱敏，而不是少脱敏。"""
    monkeypatch.setattr(config, "TRACE_MASK_MAX_CHARS", -5)
    warnings = _capture_warnings(monkeypatch)

    body = "这段正文不该被保留"
    _emit_trace(snippet=body)
    raw = _read_trace_jsonl(tmp_path)

    assert body not in raw
    assert f"<文本 {len(body)} 字>" in raw
    assert any("TRACE_MASK_MAX_CHARS" in w for w in warnings)


# ---------------------------------------------------------------------------
# 3. 行为：脱敏必须在落盘之前，且落盘与下发口径唯一
# ---------------------------------------------------------------------------
def test_persist_receives_already_masked_data(monkeypatch) -> None:
    """把 ``_persist`` 截下来，断言它拿到的**入参**已经脱敏。

    这是「脱敏在 ``_persist`` 之前」的判据里最强的一条：顺序本身测不到，
    但「谁在什么时候看到什么」测得到。若有人把 mask 挪进 ``_persist`` 内部，
    这条会拿到未脱敏的入参而变红。

    ⚠️ 反向验证：把 ``end_trace`` 里的 ``mask_tree(...)`` 去掉，本条必须变红。
    """
    seen: List[str] = []
    monkeypatch.setattr(
        "app.core.tracing._persist",
        lambda tree: seen.append(json.dumps(tree, ensure_ascii=False)),
    )

    _emit_trace(email=_FAKE_EMAIL, api_used=_FAKE_KEY)

    assert seen, "用例没有真的走到 _persist，判据是空的"
    assert _FAKE_EMAIL not in seen[0] and _FAKE_KEY not in seen[0]


def test_response_and_disk_carry_the_same_bytes(tmp_path) -> None:
    """返回值（会挂进 ``span_tree`` 下发）与落盘内容必须**逐字相同**。

    两处各脱各的必然分叉 —— 一个典型的结局是「日志干净、响应体里是原文」，
    而且只有在生产环境看抓包日志时才会发现。
    """
    begin_trace("trace-mask-test")
    with span("mask_probe") as node:
        node.attrs["email"] = _FAKE_EMAIL
    returned = end_trace(persist=True)

    record = json.loads(_read_trace_jsonl(tmp_path).strip().splitlines()[-1])
    assert returned == record["spans"]
    assert _FAKE_EMAIL not in json.dumps(record, ensure_ascii=False)


def test_masking_runs_before_truncation_not_after() -> None:
    """顺序不能反：**先脱敏、后截断**。

    先截断会把一个邮箱切成两半，后半段不再匹配邮箱规则，于是
    「截断反而让敏感串活下来」—— 一个只在长文本上复现的漏洞。
    上限设成 20，让邮箱横跨截断点。

    ⚠️ 反向验证：把 ``mask_text`` 里的截断挪到 ``for pattern`` 循环之前，本条必须变红。
    """
    text = "prefix prefix aaaa@example.com"
    assert len(text) > 20

    out = trace_mask.mask_text(text, 20)

    assert "@" not in out, f"截断发生在脱敏之前，邮箱被切碎后残留在结果里：{out!r}"
    assert "example.com" not in out


# ---------------------------------------------------------------------------
# 4. 自检：规则表不能有空转的条目
# ---------------------------------------------------------------------------
def test_every_pattern_actually_fires() -> None:
    """每条形态规则都要能在自己的样本上真的发生替换。

    一条写错的正则**静默地什么也不做** —— 脱敏于是变成装饰，
    而所有"搜不到密钥"的用例仍然全绿（因为本来就没往 trace 里写密钥）。
    样本表与规则表用 ``set`` 对齐，加规则而不加样本会直接变红。
    """
    samples: Dict[str, str] = {
        (
            r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
            r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9A-Za-z])"
        ): "证件 " + _FAKE_ID,
        r"(?<![0-9A-Za-z])1[3-9]\d{9}(?![0-9A-Za-z])": "手机 " + _FAKE_PHONE,
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+\b": "邮箱 " + _FAKE_EMAIL,
        r"\bsk-[A-Za-z0-9\-_]{12,}\b": "key=" + _FAKE_KEY,
        r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]{8,}=*": "Authorization: Bearer abcdefghijklmnop",
        r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password)\b\s*[=:]\s*\S+": (
            "api_key=abcdef123456"
        ),
    }
    declared = {pattern.pattern for pattern, _ in trace_mask._PATTERNS}
    assert declared == set(samples), (
        "规则表与样本表脱节："
        f"缺样本 {declared - set(samples)}，多余样本 {set(samples) - declared}"
    )
    assert declared, "规则表是空的 —— 脱敏什么都没做"

    for pattern, sample in samples.items():
        masked = trace_mask.mask_text(sample, trace_mask._DEFAULT_MAX_CHARS)
        assert masked != sample, f"这条规则没有生效（正则写错了？）：{pattern}"


def test_trace_id_shaped_like_a_phone_number_is_left_alone() -> None:
    """``trace_id`` 是 16 位十六进制串，其中可能**恰好**藏着一段手机号形态。

    实测踩到过：``b8c7b17477754875`` 从下标 4 起是 ``17477754875`` ——
    11 位、1 开头、第二位 7，完全符合手机号规则。判据若用 ``(?<!\\d)``，
    这个正常的 trace_id 会被整段替换成 ``<PHONE>``，脱敏自己成了故障源。

    正确的判据是**字母数字边界**：被更长的字母数字串包住 ⇒ 它是标识符的一部分，
    不是独立号码。
    """
    for trace_id in ("b8c7b17477754875", "343c1ecf28444377", "a1b2c3d4e5f60718"):
        assert trace_mask.mask_text(trace_id, trace_mask._DEFAULT_MAX_CHARS) == trace_id

    # 但真正独立的号码仍然要被替换（边界收紧不能把功能一起收紧掉）
    assert trace_mask.mask_text("手机 " + _FAKE_PHONE, 64) == "手机 <PHONE>"


def test_sensitive_key_denylist_actually_redacts() -> None:
    """键名命中凭据表时**整值**抹掉，且不看长度（凭据可以很短）。"""
    payload = {"authorization": "x", "api_key": "y", "my_access_token": "z", "scene": "policy"}
    out = trace_mask.mask_value(payload, trace_mask._DEFAULT_MAX_CHARS)

    assert out == {
        "authorization": "<REDACTED>",
        "api_key": "<REDACTED>",
        "my_access_token": "<REDACTED>",
        "scene": "policy",
    }


# ---------------------------------------------------------------------------
# 5. 派生自代码的「零误伤」回归
# ---------------------------------------------------------------------------
def _iter_span_call_names(tree: ast.AST) -> Iterator[str]:
    """所有 ``span("name", ...)`` 的 name 字面量 —— 它会成为记录里的 ``name`` 字段。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "span" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                yield first.value


def _iter_attr_assignments(tree: ast.AST) -> Iterator[ast.Assign]:
    """所有 ``X.attrs[...] = ...`` 形式的赋值节点。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Attribute)
            and t.value.attr == "attrs"
            for t in node.targets
        ):
            yield node


def _iter_attr_keys(tree: ast.AST) -> Iterator[str]:
    for node in _iter_attr_assignments(tree):
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                yield target.slice.value


def _iter_attr_value_literals(tree: ast.AST) -> Iterator[str]:
    """``X.attrs[...] = <expr>`` 里出现的字符串常量 —— 枚举值的实际候选。"""
    for node in _iter_attr_assignments(tree):
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                yield sub.value


def _strings_in(obj) -> Iterator[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str):
                yield key
            yield from _strings_in(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _strings_in(item)
    elif isinstance(obj, str):
        yield obj


def _current_trace_strings() -> List[str]:
    """本仓库**当前**会写进 trace 的字符串，以及本地真实记录里出现过的全部字符串。

    为什么要从 AST 派生而不是手抄一份：手抄的清单会漂移，而漂移的方向恰好是
    「新加的字段没被这条用例覆盖」。派生保证新字段自动进入语料。
    """
    corpus: Set[str] = set()
    scanned = 0
    for path in sorted(_APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scanned += 1
        corpus.update(_iter_span_call_names(tree))
        corpus.update(_iter_attr_value_literals(tree))

    # 自检：扫描面塌成 0 时，下面的断言会毫无意义地变绿 —— 这是最坏的失效方式
    assert scanned > 20, f"只扫到 {scanned} 个文件，AST 扫描面塌了"
    assert len(corpus) >= 10, f"只抽到 {len(corpus)} 个字符串字面量，派生逻辑失效了"

    # 本地若有真实 trace.jsonl（logs/ 被 .gitignore 排除，别的机器上可能没有），
    # 把它出现过的字符串值也纳入语料 —— 这是最强的零误伤证据。
    real = config.BASE_DIR / "logs" / "trace.jsonl"
    if real.exists():
        found = 0
        for line in real.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for value in _strings_in(json.loads(line)):
                corpus.add(value)
                found += 1
        assert found > 0, "真实 trace.jsonl 存在却一条字符串都没抽到"

    return sorted(corpus)


def test_masking_does_not_disturb_any_current_field() -> None:
    """默认档下，**现有全部 trace 字段逐字节不变**。

    脱敏如果把现有可观测性打坏了，上线第一天就会被关掉，然后永远不再是护栏。
    所以"零误伤"不是附加要求，是它能不能活下去的前提。
    """
    corpus = _current_trace_strings()
    changed = [v for v in corpus if trace_mask.mask_text(v, trace_mask._DEFAULT_MAX_CHARS) != v]
    assert changed == [], (
        f"默认档动了 {len(changed)} 个既有字段，例如 {changed[:5]} —— "
        "要么把上限调高，要么承认这些字段本来就不该进 trace"
    )


def test_denylist_does_not_hit_any_current_attr_key() -> None:
    """凭据键名表不得命中现有任何 attr 键名（``token`` 取裸子串，最容易误伤）。"""
    keys: Set[str] = set()
    for path in sorted(_APP_DIR.rglob("*.py")):
        keys.update(_iter_attr_keys(ast.parse(path.read_text(encoding="utf-8"))))
    assert keys, "没抽到任何 attrs 键名，派生逻辑失效了"

    offenders = sorted(k for k in keys if trace_mask._is_sensitive_key(k))
    assert offenders == [], f"这些现有 attr 键名被误判成凭据：{offenders}"


# ---------------------------------------------------------------------------
# 6. 结构：没有第二条通往磁盘的路
# ---------------------------------------------------------------------------
def _python_string_literals(tree: ast.AST) -> Iterator[str]:
    """模块里所有**代码中**的字符串字面量 —— 排除 docstring。

    排除 docstring 是必须的，否则这条结构护栏会被「在注释性文字里提到
    ``trace.jsonl``」判红：``trace_mask.py`` 的模块 docstring 就引用了这个文件名，
    而它显然不是「另一条写入路径」。
    """
    docstrings: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.value


def test_trace_jsonl_has_exactly_one_writer() -> None:
    """``trace.jsonl`` 只有 ``tracing.py`` 一处写入点（看代码，不看注释）。

    只要有第二处直接 ``open(..., "a")`` 写这个文件，它就能绕过脱敏，
    而且不会有任何用例变红 —— 因为用例只检查"已知的那条路干净"。

    判据取**非 docstring 的字符串字面量**：文档里提到这个文件名很常见，
    那不代表多了一条写入路径。
    """
    writers = []
    for path in sorted(_APP_DIR.rglob("*.py")):
        literals = _python_string_literals(ast.parse(path.read_text(encoding="utf-8")))
        if any("trace.jsonl" in literal for literal in literals):
            writers.append(path.relative_to(_REPO_ROOT).as_posix())

    assert writers == ["app/core/tracing.py"], f"trace.jsonl 的写入点不唯一：{writers}"


def test_persist_is_only_reachable_through_end_trace() -> None:
    """``_persist`` 只能被 ``end_trace`` 调用 —— 它就是"脱敏之后"这个前置条件的边界。

    脱敏放在 ``end_trace`` 里，那么任何绕过 ``end_trace`` 直接调 ``_persist``
    的代码都等于绕过脱敏。用 AST 找调用点，而不是靠肉眼看。
    """
    path = _APP_DIR / "core" / "tracing.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    callers: Set[str] = set()
    for func in tree.body:
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_persist"
            ):
                callers.add(func.name)

    assert callers == {"end_trace"}, f"_persist 的调用点不止 end_trace：{sorted(callers)}"


def test_mask_module_is_the_only_place_patterns_live() -> None:
    """形态规则与键名表只在 ``trace_mask.py`` 里定义一份。

    "同一语义在多处各定义一遍"是本项目反复出现的缺陷族：抄一份不需要理解上下文，
    抄错了不报错。这里用「``<REDACTED>`` 这个标记只许有一个出处」来守 —— 任何
    别处自己拼一套脱敏，都会引入它自己的标记。
    """
    holders = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _APP_DIR.rglob("*.py")
        if "<REDACTED>" in path.read_text(encoding="utf-8")
    ]
    assert holders == ["app/core/trace_mask.py"], f"脱敏标记出现在了多处：{holders}"
