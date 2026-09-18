#!/usr/bin/env python3
"""对外故障描述的唯一实现 —— 异常原文不得出现在任何响应里（P0-2）。

守什么
------
本服务有多条外发链路：``/chat/ask``（非流式）、``/chat/ask/stream``（SSE）、
``/workflow/execute``、``/evaluate/*``、``/memory/*``、``/dify/retrieval``
以及 ``app/main.py`` 的全局 500 兜底。它们曾经**各自决定**「异常怎么对外说」：

- ``/chat/ask`` 早就脱敏（只给 trace_id）；
- ``/chat/ask/stream`` 直接回显 ``str(exc)``，把文件路径、内网地址、
  供应商返回体一起送到浏览器；
- ``/workflow/execute`` 把图内部的 ``error_msg`` 原样透出（而它来自 ``str(exc)``）；
- ``/evaluate/*``、``/memory/*`` 各写一遍 ``f"...失败：{exc}"``。

同一个故障，用户在两条链路上看到的东西不一样 —— 没有任何一处会报错，
只在事故复盘时才会被发现。这正是"同一语义在多处各定义一遍"的老毛病。

现在规则只有一条：**响应体里的故障描述必须由 ``public_detail()`` 生成**
（固定前缀 + trace_id），异常原文只进 ``logs/``。

为什么这样守才不是恒真
----------------------
不去检查"代码长什么样"，而是**真的制造一次异常**，异常消息里塞一个独一无二的
探针串，然后在响应体里搜它：搜得到 = 泄漏。这样无论实现怎么重构，
只要重新开始回显异常就会变红 —— 这是本仓库验证护栏的既定手法
（把修复退回、用例必须变红）。

探针串故意长得不像任何真实内容：用 ``"boom"`` 这类词会因为日志模板、
依赖报错文本里本来就可能有而变成误报，误报的护栏最终一定会被关掉。
"""
import ast
import json
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: 探针串：唯一的泄漏标记。出现即泄漏。
_SENTINEL = "SENTINEL-7f3a1c-out-of-band"

#: 探针异常附带的三类典型内部细节（路径 / 内网地址 / 堆栈标记）。
_LEAKY_BITS = ("/etc/passwd", "10.0.0.7", "Traceback")


class _ProbeError(RuntimeError):
    """只为本次测试而生的异常类型，便于断言"类名也不该外泄"。"""


def _boom() -> None:
    raise _ProbeError(f"{_SENTINEL} at {' / '.join(_LEAKY_BITS)}")


def _assert_clean(body: str, where: str) -> None:
    """响应体里不许出现异常原文、异常类名、堆栈或路径。"""
    assert _SENTINEL not in body, f"{where} 泄漏了异常原文：{body[:400]}"
    for bit in _LEAKY_BITS:
        assert bit not in body, f"{where} 泄漏了内部细节 {bit!r}：{body[:400]}"
    assert _ProbeError.__name__ not in body, f"{where} 泄漏了异常类名：{body[:400]}"


def _patch_history(monkeypatch, chat_mod) -> None:
    """把会话历史摘掉：本用例不关心记忆，也不该连 Redis。"""
    async def _empty(session_id):
        return []

    monkeypatch.setattr(chat_mod, "get_history", _empty)


def _sse_data_blocks(text: str, event: str) -> list:
    """从 SSE 文本里取出指定事件的数据体。"""
    out = []
    for block in text.split("\n\n"):
        if block.startswith(f"event: {event}"):
            line = [ln for ln in block.splitlines() if ln.startswith("data: ")]
            out.append(json.loads(line[0][len("data: "):]))
    return out


# ---------------------------------------------------------------------------
# 行为护栏：真的让每条链路失败一次，再搜探针串
# ---------------------------------------------------------------------------
def test_sse_error_event_carries_no_exception_text(monkeypatch):
    """SSE 的 error 事件只给 trace_id —— 这是 P0-2 的原始证据点。

    退回验证：把 ``public_detail(...)`` 换回 ``f"对话服务异常：{exc}"``，本用例变红。
    """
    from fastapi.testclient import TestClient

    import app.api.chat as chat_mod
    from app.main import app

    class _ExplodingWorkflow:
        def invoke(self, state):
            _boom()

    _patch_history(monkeypatch, chat_mod)
    monkeypatch.setattr(chat_mod, "pre_generation_workflow", _ExplodingWorkflow())

    client = TestClient(app)
    resp = client.post(
        "/chat/ask/stream",
        json={"query": "年假有多少天", "session_id": "s-err", "user_id": "u1"},
    )
    assert resp.status_code == 200

    errors = _sse_data_blocks(resp.text, "error")
    assert errors, f"没走到异常分支，用例失去意义：{resp.text[:400]}"
    detail = errors[0]["detail"]
    _assert_clean(json.dumps(errors, ensure_ascii=False), "SSE error 事件")
    # 脱敏不等于什么都不给：必须留下可被客服检索的凭证
    assert "trace_id" in detail, f"error 事件既没原文也没凭证：{detail!r}"


def test_non_stream_500_carries_no_exception_text(monkeypatch):
    """非流式的 500 detail —— 与 SSE 必须同口径（此前只有它是对的）。"""
    from fastapi.testclient import TestClient

    import app.api.chat as chat_mod
    from app.main import app

    class _ExplodingWorkflow:
        def invoke(self, state):
            _boom()

    _patch_history(monkeypatch, chat_mod)
    monkeypatch.setattr(chat_mod, "enterprise_workflow", _ExplodingWorkflow())

    client = TestClient(app)
    resp = client.post(
        "/chat/ask",
        json={"query": "年假有多少天", "session_id": "s-err", "user_id": "u1"},
    )
    assert resp.status_code == 500
    _assert_clean(resp.text, "/chat/ask 500")
    assert "trace_id" in resp.json()["detail"]


def test_workflow_execute_carries_no_exception_text(monkeypatch):
    """/workflow/execute 的两条通道：异常分支的 detail，以及内部 error_msg 透出。"""
    from fastapi.testclient import TestClient

    import app.api.workflow as wf_mod
    from app.main import app

    class _ExplodingWorkflow:
        def invoke(self, state):
            _boom()

    monkeypatch.setattr(wf_mod, "enterprise_workflow", _ExplodingWorkflow())

    client = TestClient(app)
    resp = client.post("/workflow/execute", json={"query": "年假有多少天"})
    assert resp.status_code == 500
    _assert_clean(resp.text, "/workflow/execute 500")


def test_generate_failure_records_the_fault_but_keeps_the_trace_clean(monkeypatch):
    """生成节点失败时必须**留痕**，但留痕不能变成外发原文。

    这条守的是**绕过异常分支的走私通道**：接口层一个 except 都没进，
    异常原文却已经躺在响应体的 ``trace[i].detail`` 里了 —— 而 trace 的 detail
    正是前端面板直接显示的那一栏。

    ⚠️ 这里**不**要求 ``error_msg`` 本身干净：它是图内部的诊断串，按设计可以
    含原文（``tests/test_tool_agent.py`` 就钉着"工具决策失败时 error_msg 里能看到
    原因"这条），原文只进 ``logger.warning``。要做的是**别把它复制进会被下发的
    字段**——也就是本用例的 trace，以及 /workflow/execute 的 error_msg 字段
    （见 ``test_graph_internal_failure_never_reaches_the_client``）。
    """
    from app.graph import nodes

    monkeypatch.setattr(
        nodes, "build_generation_inputs",
        lambda state: {
            "prepared": {"citations": [], "confidence": 0.9, "refused": False, "inputs": {}},
            "window": [],
        },
    )

    def _explode(*args, **kwargs):
        _boom()

    monkeypatch.setattr(nodes, "stream_answer_tokens", _explode)

    out = nodes.generate_answer_node({"user_query": "年假有多少天", "session_id": "s"})
    assert out.get("need_human") is True, "生成失败应转人工兜底"
    assert out.get("error_msg"), "转人工了却没有留下任何故障记录，等于把故障藏起来"
    assert "human_fallback" not in out, "本节点不该自己转人工，那是 human_fallback_node 的职责"

    _assert_clean(json.dumps(out.get("trace") or [], ensure_ascii=False, default=str), "generate_answer 的 trace")


@pytest.mark.parametrize(
    "path, body",
    [
        ("/chat/ask", {"query": "年假有多少天", "session_id": "s1"}),
        ("/chat/ask/stream", {"query": "年假有多少天", "session_id": "s2"}),
        ("/workflow/execute", {"query": "年假有多少天"}),
    ],
)
def test_graph_internal_failure_never_reaches_the_client(monkeypatch, path, body):
    """图内部生成失败（200 响应）时，``trace`` / ``route_decision`` 也不能走私原文。

    与上面几条的区别：这里异常被节点自己吞掉了，HTTP 状态是 200，
    所以"看状态码"完全看不出异常 —— 只能靠搜内容。
    """
    from fastapi.testclient import TestClient

    import app.api.chat as chat_mod
    from app.graph import nodes
    from app.main import app

    _patch_history(monkeypatch, chat_mod)
    monkeypatch.setattr(
        nodes, "build_generation_inputs",
        lambda state: {
            "prepared": {"citations": [], "confidence": 0.9, "refused": False, "inputs": {}},
            "window": [],
        },
    )

    def _explode(*args, **kwargs):
        _boom()

    monkeypatch.setattr(nodes, "stream_answer_tokens", _explode)

    client = TestClient(app)
    resp = client.post(path, json=body)
    _assert_clean(resp.text, f"{path} 200 响应体")


def test_tool_decision_failure_keeps_trace_detail_free_of_diagnostics(monkeypatch):
    """工具 Agent 决策失败转人工时，``trace.detail`` 不能是 ``decision.error``。

    ``decision.error`` 按设计是**内部诊断串**（可含异常原文，见 tool_agent.py），
    它此前被截前 60 字塞进 ``trace.detail`` —— 而 trace 随 /chat/ask 的响应体外发。
    于是"答案里不显示异常"这条规则被一条侧路绕开了：用户看不到答案里的原文，
    却能在面板的 trace 栏里看到。

    注意这里替换的是**节点函数读的模块级引用**（``nodes.route_query`` /
    ``nodes.run_tool_agent``）而不是节点本身：已编译的图里存的是节点函数的引用，
    换节点函数没用。这是"接线要能被测试"的前提，也是本用例能成立的原因。
    """
    from fastapi.testclient import TestClient

    import app.api.chat as chat_mod
    from app.core.router_agent import SCENE_TOOL, RouteDecision
    from app.core.tool_agent import ToolDecision
    from app.graph import nodes
    from app.main import app

    _patch_history(monkeypatch, chat_mod)
    monkeypatch.setattr(
        nodes, "route_query",
        lambda **kw: RouteDecision(scene=SCENE_TOOL, reason="测试强制走工具路", confidence=1.0),
    )
    monkeypatch.setattr(
        nodes, "run_tool_agent",
        lambda **kw: ToolDecision(error=f"{_SENTINEL} at {' / '.join(_LEAKY_BITS)}"),
    )

    client = TestClient(app)
    resp = client.post("/chat/ask", json={"query": "帮我查一下工单状态", "session_id": "s4"})
    assert resp.status_code == 200
    assert resp.json()["need_human"] is True, f"决策失败应转人工：{resp.text[:400]}"
    _assert_clean(resp.text, "/chat/ask（工具决策失败）")


# ---------------------------------------------------------------------------
# 结构护栏：边界层不接触异常对象
# ---------------------------------------------------------------------------
#: 允许在边界层绑定异常变量的例外。每加一条都要写明"为什么非绑不可"。
#: 只减不增 —— 这张表一旦长起来，`public_detail` 就不再是唯一口径了。
_EXCEPTION_BINDING_ALLOWLIST = {
    ("app/main.py", "exc"): (
        "限流中间件要取 AppError 自带的对外文案（exc.message），"
        "那是异常**设计好要外发**的字段，不是 str(exc)"
    ),
}


def _boundary_modules() -> list:
    """边界层 = 所有会产生外发响应的地方。"""
    mods = sorted((_REPO_ROOT / "app" / "api").glob("*.py"))
    mods.append(_REPO_ROOT / "app" / "main.py")
    return mods


def test_boundary_layer_does_not_touch_exception_objects():
    """``app/api/`` 与 ``app/main.py`` 里的 ``except`` 不许把异常绑到名字上。

    为什么用"结构"而不是"搜字符串"当判据：``f"...{exc}"`` 与
    ``f"...{type(exc).__name__}"`` 在文本上毫无共同点，但都是同一种错误的
    不同写法。反过来，**只要异常对象到了边界层，就一定会有人顺手把它拼进响应里**
    —— 这是本仓库已经发生过的事（见文件头）。堵掉"拿到对象"这一步，
    比穷举"怎么拼"可靠。

    唯一合法用法 ``logger.exception(...)`` 不需要变量，所以这条约束没有代价。
    """
    violations = []
    seen = 0
    for path in _boundary_modules():
        rel = str(path.relative_to(_REPO_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not node.name:
                continue
            seen += 1
            if (rel, node.name) not in _EXCEPTION_BINDING_ALLOWLIST:
                violations.append(f"{rel}:{node.lineno} as {node.name}")

    assert not violations, (
        "边界层不许绑定异常对象（改用 logger.exception 落盘，对外用 public_detail 构造文案）：\n  "
        + "\n  ".join(violations)
    )
    # 判据自检：扫描面塌成 0 时，本用例会毫无意义地变绿 —— 那是最坏的失效方式。
    assert seen == len(_EXCEPTION_BINDING_ALLOWLIST), (
        f"扫描到 {seen} 处异常绑定，与豁免表（{len(_EXCEPTION_BINDING_ALLOWLIST)} 条）不符；"
        "若确实新增了豁免，请同步更新 _EXCEPTION_BINDING_ALLOWLIST 并写明理由"
    )


def test_public_detail_format_is_defined_in_exactly_one_place():
    """``（trace_id: …）`` 这个格式串只许在 ``app/core/errors.py`` 里出现一次。

    退回到"各处自己拼 trace_id"时本用例变红 —— 那正是 P0-2 之前的状态
    （``/chat/ask`` 与 ``/chat/ask/stream`` 各拼一遍，其中一份还把 exc 也拼了进去）。
    """
    fingerprint = "（trace_id: {"
    hits = []
    for path in sorted((_REPO_ROOT / "app").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        count = text.count(fingerprint)
        if count:
            hits.append((str(path.relative_to(_REPO_ROOT)), count))

    assert hits == [("app/core/errors.py", 1)], (
        f"trace_id 格式串的分布不对，应为 app/core/errors.py 一处：{hits}"
    )


def test_boundary_layer_does_not_read_the_trace_id_directly():
    """边界层不得直接读 ``get_trace_id`` —— 对外凭证一律经 ``trace_ref()``。

    这条比"格式串只有一处"更难绕开：格式串可以各写各的（``trace_id: x`` /
    ``[trace=x]`` / 单独一个字段），但**取值来源**只有一个。
    退回验证：把 ``app/main.py`` 的 ``trace_ref()`` 换回 ``get_trace_id() or "-"``
    （两者输出完全一样，行为上看不出任何差别）时，本用例变红 ——
    这正是"只靠行为测不出来、必须靠结构守"的那一类。
    """
    offenders = []
    for path in _boundary_modules():
        rel = str(path.relative_to(_REPO_ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_trace_id"
            ):
                offenders.append(f"{rel}:{node.lineno}")

    assert not offenders, (
        "边界层直接读了 get_trace_id，请改用 app.core.errors.trace_ref：\n  "
        + "\n  ".join(offenders)
    )
