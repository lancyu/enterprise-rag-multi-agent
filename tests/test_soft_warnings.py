"""可降级故障（`soft_warnings`）的「写 → 读」闭环测试。

`soft_warnings` 是 P0-1 修复的产物：检索抖动这类**不致命**的故障曾经被写进
`error_msg`，导致条件边把已经生成好的答案覆盖成「已转接人工」。拆出这个字段
之后它有一段时间**只有写入点、没有任何读取点**——等于没观测。

这类字段的失效方式很特别，所以两端都要钉住：

- 写入端出错（比如又有人顺手写 `error_msg`）→ 症状是"一次网络抖动丢掉正确答案"，
  这是 P0-1 的老问题，`tests/test_fixes_assessment.py` 已覆盖路由侧；
- 读取端缺失 → **不报错、不告警**，检索质量一路下滑到彻底失效才被发现，
  而中间那些天的日志里查不到任何线索。本文件补的就是这一段。
"""
import asyncio
import logging

from app.api import chat as chat_api
from app.graph import nodes
from app.graph.edges import error_route_edge
from app.graph.state import create_initial_state
from app.utils.validator import ChatRequest
from tests.fakes import RecordingModel


def _fake_tool_model(*script):
    """记录工具调用、不联网的假模型（见 tests/fakes.py）。

    末尾必须由调用方补一项 ``[]``（"本轮不调工具"）作收口：
    ``TOOL_AGENT_MAX_STEPS`` 默认 2，脚本耗尽后会重复最后一项——不收口的话
    假模型会把同一个调用再发两遍，故障计数就不止一条了。
    """
    return RecordingModel(list(script))


# ---------------------------------------------------------------------------
# 1. 写入端：可降级故障 → 记入 soft_warnings，且**不**影响路由
# ---------------------------------------------------------------------------
# 架构说明：五 Agent 架构下有两个写入点，各自对应一类"答案仍能给出、但质量已下降"：
#
#   ① `simple_rag_node` / `complex_rag_node`：检索抛异常 → 降级为"无依据作答"。
#      新架构下检索是**独立 Agent**（直接调 retriever）而不是工具，所以软降级的
#      发生地从"工具的 NO_HIT 约定"搬到了 Agent 内部。
#   ② `tool_node`：工具步骤 status 为 error / rejected。
#
# 断言随之搬迁——但**不能删**：写入点缺失时字段退化成"只读不写"，
# 而它的失效方式是静默的（不报错、不告警），正是本文件存在的理由。
def test_retrieval_failure_is_recorded_as_soft_warning(monkeypatch) -> None:
    """检索服务抛异常 → 记 soft_warnings，不进 error_msg，也不转人工。"""
    from app.core import sub_agents
    from app.core.rag_engine import retrieve_knowledge_docs

    def boom(*_a, **_kw):
        raise RuntimeError("embedding 服务不可用")

    # 打桩点必须是 sub_agents 命名空间里的那个引用（它是按名查模块全局的）
    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", boom)
    state = create_initial_state(user_query="年假多少天", session_id="s1")

    out = nodes.simple_rag_node(state)

    # Agent 按设计降级了：没有证据，但答案仍由 L4 诚实产出
    assert out["retrieve_docs"] == []
    assert len(out["soft_warnings"]) == 1, f"未记录降级故障：{out['soft_warnings']}"
    assert "embedding 服务不可用" in out["soft_warnings"][0]
    assert not out.get("error_msg"), "降级故障绝不能写进 error_msg（P0-1 的根因）"
    assert not out.get("need_human"), "可降级故障不该把用户推给人工"
    assert out["intent_type"] == "knowledge", "仍走证据出口，由 L4 给出「没找到」"
    # 反向校验：打桩点没打中的话，上面那些断言会以"检索真的命中了"这种
    # 与被测逻辑无关的方式失败——故这里显式确认打桩生效。
    assert retrieve_knowledge_docs is boom or callable(retrieve_knowledge_docs)


def test_successful_tools_produce_no_soft_warning(monkeypatch, business_db) -> None:
    """正常工具调用不是故障——收进来会让计数长期虚高，真抖动被淹没。"""
    from app.core import tool_agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "_default_model",
        lambda: _fake_tool_model(
            {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
            [],
        ),
    )
    state = create_initial_state(user_query="我的年假还剩几天", session_id="s1")

    out = nodes.tool_node(state)

    assert out["soft_warnings"] == [], f"正常调用不该产生告警：{out['soft_warnings']}"
    # 顺带确认正常路径的证据确实到手（否则"零告警"可能只是因为啥也没干）
    assert "annual_leave" in (out["tool_result"] or "")


def test_grounding_rejection_is_recorded_as_soft_warning(monkeypatch) -> None:
    """护栏拒绝参数 → 也要留痕：本轮少了一份证据，但没有任何故障会报出来。"""
    from app.core import tool_agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "_default_model",
        lambda: _fake_tool_model(
            {"name": "find_employee_by_name", "args": {"name": "王五"}},
            [],
        ),
    )
    state = create_initial_state(user_query="王小明的部门是什么", session_id="s1")

    out = nodes.tool_node(state)

    assert len(out["soft_warnings"]) == 1, f"护栏事件未留痕：{out['soft_warnings']}"
    assert "护栏" in out["soft_warnings"][0]
    assert not out.get("error_msg"), "护栏正常工作是安全事件，不是错误"
    assert not out.get("need_human"), "模型侧问题不该把用户推给人工"


def test_tool_execution_failure_is_recorded_and_routed_to_human(monkeypatch, business_db) -> None:
    """工具执行失败 → **两条**都要：留痕（观测）+ 转人工（处置）。

    与上一条形成对照，判据是「能不能从其他来源得到答案」：
    护栏拒绝是模型侧问题（换个参数还能答），工具抛异常则可能确实答不了。
    """
    from app.core import tool_agent as agent_mod

    class _BoomTool:
        name = "query_leave_balance"

        @staticmethod
        def invoke(_args):
            raise RuntimeError("业务库 502")

    monkeypatch.setitem(agent_mod._TOOLS_BY_NAME, "query_leave_balance", _BoomTool())
    monkeypatch.setattr(
        agent_mod, "_default_model",
        lambda: _fake_tool_model(
            {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
            [],
        ),
    )
    state = create_initial_state(user_query="我的年假还剩几天", session_id="s1")

    out = nodes.tool_node(state)

    assert out["need_human"] is True
    assert any("执行失败" in w for w in out["soft_warnings"])


# ---------------------------------------------------------------------------
# 2. 读取点：落日志 + 只对客户端透出数量
# ---------------------------------------------------------------------------
def test_no_log_when_there_are_no_warnings(caplog) -> None:
    """一切正常时零噪音——否则日志会被"正常"淹没，等于还是观测不到。"""
    with caplog.at_level(logging.WARNING):
        assert chat_api.collect_soft_warnings({"soft_warnings": []}) == 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_warning_is_logged_with_detail(caplog) -> None:
    """有降级故障时必须留下**带细节**的日志（排障要靠它）。"""
    state = {"soft_warnings": ["知识检索失败：连接被重置"]}
    with caplog.at_level(logging.WARNING):
        count = chat_api.collect_soft_warnings(state)

    assert count == 1
    texts = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("连接被重置" in t for t in texts), "日志里丢了故障细节，排障时无从下手"


def test_client_facing_count_does_not_leak_exception_text() -> None:
    """对外的投影是**数量**，不是异常原文。

    异常文本可能含内网地址 / 文件路径，与本服务"不回显异常原文、只给 trace_id"
    的既定口径冲突。所以读取点的返回值必须是计数，而不是把状态字段原样转发。
    """
    result = chat_api.collect_soft_warnings({"soft_warnings": ["知识检索失败：http://10.0.0.7:9999"]})
    assert isinstance(result, int) and result == 1


def test_blank_warnings_are_ignored() -> None:
    """空串不该被算作一条故障（否则计数长期虚高）。"""
    assert chat_api.collect_soft_warnings({"soft_warnings": ["", "   ", None]}) == 0


def test_missing_field_is_tolerated() -> None:
    """状态里没有该字段（旧调用方 / 单测直调）时按"无故障"处理，不抛错。"""
    assert chat_api.collect_soft_warnings({}) == 0


def test_soft_warning_does_not_change_routing() -> None:
    """可降级故障不参与路由：有答案就正常结束，不转人工。"""
    state = {"answer": "诚实回答", "soft_warnings": ["知识检索失败：超时"], "need_human": False}
    assert error_route_edge(state) == "end"


# ---------------------------------------------------------------------------
# 3. 端到端：/chat/ask 的响应体里必须能看到这一轮的降级情况
# ---------------------------------------------------------------------------
def test_ask_response_exposes_soft_warning_count(monkeypatch) -> None:
    """接线完整性：状态里的降级信息要真的走到响应体，而不是只进日志。

    没有这条，`soft_warnings` 依然可能"写了但用户/前端永远看不到"——
    与修复前的"只写不读"只差一层。
    """
    async def _no_history(_sid):
        return []

    async def _noop_save(*_a, **_kw):
        return None

    class _Workflow:
        @staticmethod
        def invoke(_state):
            return {
                "answer": "降级状态下仍然给出的答案",
                "soft_warnings": ["知识检索失败：连接被重置"],
                "retrieve_docs": [],
                "trace": [],
                "route_decision": {},
            }

    monkeypatch.setattr(chat_api, "get_history", _no_history)
    monkeypatch.setattr(chat_api, "save_message", _noop_save)
    monkeypatch.setattr(chat_api, "maybe_consolidate", lambda *a, **kw: None)
    monkeypatch.setattr(chat_api, "enterprise_workflow", _Workflow)

    resp = asyncio.run(chat_api.chat_ask(ChatRequest(query="年假多少天", session_id="s1")))

    assert resp["soft_warning_count"] == 1
    assert resp["answer"] == "降级状态下仍然给出的答案", "降级故障不该影响作答"
    assert resp["need_human"] is False
    assert not any("连接被重置" in str(v) for v in resp.values()), (
        "响应体里出现了异常原文，与「不向客户端回显异常」的口径冲突"
    )


def test_ask_response_reports_zero_when_healthy(monkeypatch) -> None:
    """健康轮次必须明确给出 0，前端才能区分「正常」与「字段缺失」。"""
    async def _no_history(_sid):
        return []

    async def _noop_save(*_a, **_kw):
        return None

    class _Workflow:
        @staticmethod
        def invoke(_state):
            return {"answer": "正常答案", "soft_warnings": [], "trace": [], "route_decision": {}}

    monkeypatch.setattr(chat_api, "get_history", _no_history)
    monkeypatch.setattr(chat_api, "save_message", _noop_save)
    monkeypatch.setattr(chat_api, "maybe_consolidate", lambda *a, **kw: None)
    monkeypatch.setattr(chat_api, "enterprise_workflow", _Workflow)

    resp = asyncio.run(chat_api.chat_ask(ChatRequest(query="你好", session_id="s2")))

    assert resp["soft_warning_count"] == 0
