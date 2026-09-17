"""工具 Agent（function calling）的离线回归测试。

本文件取代原 ``tests/test_tool_calling.py``，并在五 Agent 架构下重写：

- 旧文件守的是「模型**抽参**」——路由已由规则判好，模型只填参数；
- 本文件守的是「模型**决策**」——调不调工具、调哪个、参数是什么，全在一处。

被守的东西少了一层（不再有路由判定），但多出了三块：**两个出口的分工**、
**文本形式工具调用的回捞**、**证据跨越工具边界不丢**。

（曾经还有第四块「身份对账（参数是主张、会话身份是事实）」。服务端已无登录态，
这条设计随工单工具一起去掉了，取而代之的是「工号只能来自用户原句或工具返回值」
——见 ``GROUNDED_ARGS`` 的说明。）

测试一律注入假模型（``tests/fakes.py``），**绝不打真实模型配额**——这是本项目的
测试纪律：真实模型的端到端验证属于 ``artifacts/`` 下的诊断脚本，它验证的是
「当前这个模型行不行」；而单测验证的是「代码逻辑对不对」，两者不能混。

Mock 与真实模型的边界
----------------------
``MockChatModel`` 刻意不实现 ``bind_tools``。所以本文件的每个用例都必须注入假模型，
**不能**依赖默认模型——后者在 CI 下会走降级分支，测出来的不是 function calling。
"""
from __future__ import annotations

import contextvars

import pytest
from langchain_core.tools import tool

from app.core import request_ctx
from app.core.tool_agent import (
    AGENT_TOOLS,
    GROUNDED_ARGS,
    execute_tool_calls,
    parse_text_tool_calls,
    run_tool_agent,
)
from tests.fakes import RecordingModel


@pytest.fixture(autouse=True)
def _fresh_scope():
    """每个用例一个干净的请求作用域。

    作用域是 contextvars 承载的**进程内**状态，不清理会让上一个用例的证据渗进
    下一个用例——表现为「单独跑通过、整套跑失败」的顺序依赖。
    """
    request_ctx.reset_request_context()
    yield
    request_ctx.reset_request_context()


class _StubTool:
    """替换候选集条目的假工具：可控返回、可控抛错。"""

    def __init__(self, name: str, result: str = "ok", error: Exception | None = None) -> None:
        self.name = name
        self._result = result
        self._error = error

    def invoke(self, _args):
        if self._error is not None:
            raise self._error
        return self._result


# ===========================================================================
# 1. 两个出口：直答 vs 证据
# ===========================================================================
def test_direct_exit_when_model_calls_no_tool():
    """模型不调工具 → 直答出口：它自己的话就是答案，不走 L4。

    这是"必填参数缺失 → 主动向用户追问"的路径（规格第 3 条）。它**必须**
    绕过 L4：L4 的职责是「基于证据回答」，没有证据时会触发拒答，
    把一句「请问你叫什么名字」变成「知识库中没有找到相关信息」。

    注意追问在这里是**正常行为**而不是失败：服务端没有登录态，「我的年假还剩
    几天」里的"我"是谁，除了问用户没有别的来源。见 ``tool_agent`` 的模块说明。
    """
    decision = run_tool_agent("我的年假还剩几天", model=RecordingModel(["请问你叫什么名字？我帮你查。"]))
    answer = "请问你叫什么名字？我帮你查。"

    assert decision.is_direct
    assert decision.direct_answer == answer
    assert decision.used_tools == []
    assert not decision.attempted


def test_evidence_exit_when_model_calls_business_tool(business_db):
    """调用业务工具 → 证据出口：结果进证据集，答案交给 L4 生成。

    脚本第二项是 ``[]``（明确表达"本轮不调工具"）：``TOOL_AGENT_MAX_STEPS`` 默认 2，
    脚本耗尽后会重复最后一项——不显式收口的话，假模型会把同一个调用再发两遍，
    测出来的步骤数就不是我们想断言的东西了。
    """
    model = RecordingModel([
        {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
        [],
    ])

    decision = run_tool_agent("我的年假还剩几天", model=model)

    assert not decision.is_direct, "调了工具就必须走证据出口（要带业务数据）"
    assert decision.attempted
    assert decision.used_tools == ["query_leave_balance"]
    # 结果由 run_tool_agent 显式带出，调用方无需再读作用域
    assert len(decision.tool_results) == 1
    assert "annual_leave" in decision.tool_results[0]
    assert decision.docs == [], "工具 Agent 不产检索片段，只产确定性事实"


def test_not_found_is_a_successful_call_not_an_error(business_db):
    """查无此人**不是**系统故障：步骤状态仍是 ``ok``，结果是 ``ok:false`` 的信封。

    这个区分决定了后续处置：``error`` 会触发转人工，``ok`` 却让 L4 能如实
    回答"没有这位员工"。把两者混起来，会得到"用户只是打错了工号，却被转接到
    人工客服"这种误伤（反向的那个方向更糟，见下一个用例）。
    """
    model = RecordingModel([
        {"name": "query_leave_balance", "args": {"employee_id": "E9999"}},
        [],
    ])

    decision = run_tool_agent("我的年假还剩几天", model=model)

    assert decision.steps[0].status == "ok"
    assert '"ok": false' in decision.tool_results[0]
    assert not decision.error


# ===========================================================================
# 2. 候选集与幻觉护栏
# ===========================================================================
def test_all_tools_are_offered_without_narrowing():
    """候选集必须原样交给模型，**不做收窄**。

    旧实现按路由判出的能力名只给一个工具，等于把「模型读工具描述后自愈」
    这条路堵死：用户问"我的年假"而模型需要先 find 再 query 时，收窄后它没有
    这个机会。候选集本身就是白名单，不需要再收一次。
    """
    model = RecordingModel([[]])
    run_tool_agent("随便问问", model=model)

    assert model.bound_tools == list(AGENT_TOOLS)
    assert len(AGENT_TOOLS) == 3


@pytest.mark.parametrize(
    "tool_name,args,query,executed",
    [
        # 姓名是自由文本，模型最容易"顺手编"
        ("find_employee_by_name", {"name": "张三"}, "张三在哪个部门", True),
        ("find_employee_by_name", {"name": "王五"}, "王小明的部门", False),  # 幻觉
        ("find_employee_by_name", {"name": ""}, "张三在哪个部门", False),     # 空值
    ],
)
def test_grounding_guard(tool_name, args, query, executed):
    """受护栏保护的参数必须能在用户原句里定位，否则**拒绝执行**。

    这是 function calling 最危险的失败模式，也是最便宜的护栏：模型不是在
    "抄"原句，而是在"生成"参数——它可能生成一个库里恰好存在、但用户没说过
    的姓名。用户问「王小明的部门」却拿到「王五」的资料，而且
    **看不出来哪里错了**。静默错答比拒答严重得多。

    「值必须出现在原句里」不能消除全部错误（句中有多个候选时仍可能选错），
    但把"凭空捏造"这一类彻底挡掉，成本是零。
    """
    steps = [step for step, _ in execute_tool_calls([{"name": tool_name, "args": args}], query)]

    assert (steps[0].status == "ok") is executed, (
        f"{query!r} 抽 {args}：期望 {'执行' if executed else '拒绝'}，实际 {steps[0].status}"
    )


def test_employee_id_is_deliberately_not_grounded():
    """``employee_id`` **刻意不登记**在落地校验里 —— 但它并非无人把关。

    工号几乎不会出现在用户原句里：「张三的年假还剩几天」里只有姓名，
    工号要等 ``find_employee_by_name`` 返回（也就是链式调用的第二轮）。
    要求工号出现在原句里，会把整条链式调用拦死在第二轮。

    那"模型硬填一个工号"怎么办？把关点前移到**姓名**：正常路径下工号要先经
    ``find_employee_by_name`` 换取，而它的 ``name`` 是受护栏保护的。剩下的
    残余风险（模型直接猜一个工号去调 ``query_leave_balance``）由提示词的
    「缺参数就追问、禁止编造」兜底 —— 这是**有意的取舍**，不是漏检：
    在没有登录态的前提下，把工号也登记成"必须出现在原句里"会连链式调用一起废掉。
    """
    assert "employee_id" not in GROUNDED_ARGS.get("query_leave_balance", ())
    assert "employee_id" not in GROUNDED_ARGS.get("query_employee_info", ())


def test_tool_outside_candidates_is_rejected():
    """模型可能编出候选集外的工具名——宁可不答，也不乱调。"""
    step = execute_tool_calls([{"name": "delete_everything", "args": {}}], "张三在哪个部门")[0][0]

    assert step.status == "rejected"
    assert "候选集" in step.detail


def test_rejected_call_still_returns_a_tool_message():
    """被拒绝的调用**仍要**回一条 ToolMessage。

    OpenAI 兼容协议要求每条 tool 消息与一次 assistant 的 tool_call 一一配对，
    少一条下一次 invoke 会被接口以 400 直接拒绝。**护栏是业务决定，
    对话结构是协议要求，两者不能混。**
    """
    from langchain_core.messages import ToolMessage

    results = execute_tool_calls(
        [{"name": "find_employee_by_name", "args": {"name": "王五"}, "id": "c1"}], "王小明的部门"
    )

    assert results[0][0].status == "rejected"
    message = results[0][1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "c1", "tool_call_id 必须原样回填，否则协议不合法"


def test_tool_exception_is_contained(monkeypatch):
    """工具抛异常不得让整轮对话失败——按「无证据」给出诚实回答即可。

    工具自身约定"基础设施故障向上抛"（见 ``tools/sqlite_tools.py`` 第 3 条），
    这里验证的是 ``execute_tool_calls`` 再兜的那一层：**一次工具抖动不该让整轮
    对话失败**，上游会按「无证据」或转人工处置。
    """
    from app.core import tool_agent as agent_mod

    monkeypatch.setitem(
        agent_mod._TOOLS_BY_NAME,
        "query_employee_info",
        _StubTool("query_employee_info", error=RuntimeError("业务库查询失败")),
    )
    step, message = execute_tool_calls(
        [{"name": "query_employee_info", "args": {"employee_id": "E1001"}}], "E1001 是哪个部门的"
    )[0]

    assert step.status == "error"
    assert "RuntimeError" in str(message.content), "要回一条说明性消息，保证对话结构合法"
    # 异常不得升级为「整轮失败」，也不得被当成业务结果喂给 L4
    assert request_ctx.get_tool_results() == []


def test_schema_violation_is_rejected_not_error(business_db):
    """参数不合 schema（``pattern`` 不匹配）→ ``rejected``，**不是** ``error``。

    这条是被真实误伤逼出来的：用户只是把工号打错一位（漏了 ``E`` 前缀），
    pydantic 抛 ``ValidationError`` 被 ``except Exception`` 记成 error，而业务
    工具的 error 会触发转人工——用户被告知"已转接人工"，而问题只是格式。
    判据是**责任在谁**：参数不合 schema 是模型侧问题，模型有剩余轮次可以自纠。
    """
    step, message = execute_tool_calls(
        [{"name": "query_employee_info", "args": {"employee_id": "1001"}}], "1001 谁啊"
    )[0]

    assert step.status == "rejected"
    # 说明要压成"字段名 + 人话"，不能把 pydantic 的几十行原文塞给模型
    assert "employee_id" in str(message.content)
    assert "Traceback" not in str(message.content)


# ===========================================================================
# 3. 文本形式工具调用的回捞
# ===========================================================================
# 为什么这段必须有：实测 volcengine 的 doubao 端点**间歇性**不把工具调用放进
# tool_calls，而是按模型自己的对话模板渲染进正文。不回捞的后果是静默且严重的：
# tool_calls 为空会被判定为「模型认为无需工具」，把这段 JSON 原样当答案返回，
# **而且不报错**。详见 app/core/tool_agent.py 的说明。
@pytest.mark.parametrize(
    "text",
    [
        '{"name": "query_leave_balance", "parameters": {"employee_id": "E1001"}}',
        '{"name": "query_leave_balance", "arguments": {"employee_id": "E1001"}}',
        '<|FunctionCallBegin|>[{"name": "query_leave_balance", "parameters": {"employee_id": "E1001"}}]<|FunctionCallEnd|>',
        '<tool_call>{"name": "query_leave_balance", "arguments": {"employee_id": "E1001"}}</tool_call>',
        '好的，我来查一下：{"name": "query_leave_balance", "parameters": {"employee_id": "E1001"}}',
    ],
)
def test_text_form_tool_calls_are_recovered(text):
    """裸 JSON / 包装格式 / 夹在正文里的调用，都要能捞出来。"""
    calls = parse_text_tool_calls(text)

    assert len(calls) == 1, f"未回捞：{text!r}"
    assert calls[0]["name"] == "query_leave_balance"
    assert calls[0]["args"] == {"employee_id": "E1001"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "你好！我是企业智能助手。",                              # 正常直答，不该误捞
        '{"name": "not_a_real_tool", "parameters": {}}',         # 候选集外
        '{"name": "query_leave_balance", "parameters": "oops"}',  # 参数不是 dict
        "关于「年假」的规定如下：**5 天**[1]",                     # 正常答案含标点
    ],
)
def test_text_recovery_does_not_overreach(text):
    """宁可漏捞，也不可误捞——误捞会把正常答案变成工具调用。"""
    assert parse_text_tool_calls(text) == []


def test_recovered_calls_are_counted_and_executed(business_db):
    """回捞到的调用要真的执行，且计数透出（换模型前必看的指标）。"""
    model = RecordingModel([
        '{"name": "query_leave_balance", "parameters": {"employee_id": "E1001"}}',
        [],
    ])

    decision = run_tool_agent("我的年假还剩几天", model=model)

    assert decision.recovered_calls == 1
    assert decision.used_tools == ["query_leave_balance"], "回捞到的调用必须真的执行"
    assert decision.tool_results, "结果要带出去，否则 L4 会答「知识库里没有」"
    assert not decision.is_direct, "回捞成功就不能走直答出口（否则会把 JSON 当答案）"


def test_text_recovery_does_not_bypass_grounding():
    """回捞路径与真实 tool_call 走**同一套**护栏，不从文本路径放松任何一条。"""
    calls = parse_text_tool_calls(
        '{"name": "find_employee_by_name", "parameters": {"name": "王五"}}'
    )

    assert calls, "结构上应当能捞出来"
    step = execute_tool_calls(calls, "王小明的部门")[0][0]
    assert step.status == "rejected", "回捞路径也必须过落地校验"


# ===========================================================================
# 4. 降级：模型不支持 function calling
# ===========================================================================
def test_model_without_bind_tools_reports_degraded_and_does_not_act():
    """不实现 ``bind_tools`` → 如实报告 ``degraded``，**且什么都不做**。

    MockChatModel 在离线环境下的行为就长这样。注意这里与旧实现的差别：
    旧实现会自己"无条件检索一次"兜底，新架构下**不这么做**——
    "工具路走不通该改走哪条"是**路由决策**，由 ``graph/edges.py::tool_route_edge``
    决定改道简单 RAG。本模块只关心"怎么用工具"，不关心"用不了工具时怎么办"。
    """
    model = RecordingModel([[]], bind_error=NotImplementedError())

    decision = run_tool_agent("我的年假还剩几天", model=model)

    assert decision.degraded
    assert decision.error is None, "降级不是错误，不该转人工"
    assert decision.used_tools == [], "不得自作主张地检索"
    assert decision.docs == []
    assert not decision.attempted


def test_bind_tools_failure_other_than_notimplemented_also_degrades():
    """绑定工具时的任何异常都不得让整轮失败。"""
    model = RecordingModel([[]], bind_error=RuntimeError("接口不支持 tools 字段"))

    decision = run_tool_agent("我的年假还剩几天", model=model)

    assert decision.degraded
    assert decision.error is None


def test_model_exception_is_reported_as_error():
    """模型调用本身失败 → error 非空，调用方据此转人工（唯一该转人工的情形）。"""

    class _Broken(RecordingModel):
        def bind_tools(self, _tools):
            class _B:
                def invoke(self, _messages, **_kw):
                    raise RuntimeError("模型服务不可达")

            return _B()

    decision = run_tool_agent("我的年假还剩几天", model=_Broken([[]]))

    assert decision.error and "模型服务不可达" in decision.error


# ===========================================================================
# 5. 证据跨工具边界不丢（本项目最隐蔽的一个坑）
# ===========================================================================
def test_scope_mutation_survives_tool_boundary():
    """工具内就地修改作用域对象 → 调用方**必须**看得到。

    这是 ``RequestScope`` 存在的全部理由。LangChain 的 ``Runnable.invoke``
    在 ``contextvars.copy_context()`` 的副本里执行工具，复制的映射里携带的
    仍是**同一个对象引用**，所以就地修改两边可见。
    """

    @tool
    def probe(x: str) -> str:
        """把一条业务结果写进作用域。"""
        request_ctx.add_tool_result(x)
        return "ok"

    request_ctx.reset_request_context()
    probe.invoke({"x": "doc-1"})

    assert request_ctx.get_tool_results() == ["doc-1"]


def test_raw_contextvar_set_inside_tool_is_lost():
    """**反向验证**：若退回「工具内 set 独立 ContextVar」，写入就会丢失。

    没有这一条，上面那个用例只能说明"这次碰巧过了"。护栏的价值等于它变红的
    能力——这条测试证明**这个坑真实存在**，而不是我们多此一举地设计了一个对象。

    实测现象：工具内 ``set`` 之后，调用方 ``get`` 到的仍是旧值，
    **且不报任何错**。症状是「工具日志显示查询成功，生成层却拿到 0 条数据」，
    排查时极易误判成工具本身的问题。
    """
    marker: contextvars.ContextVar = contextvars.ContextVar("probe_marker", default=None)

    @tool
    def probe(x: str) -> str:
        """在工具内 set 一个独立 ContextVar。"""
        marker.set(x)
        return "ok"

    marker.set("outer")
    probe.invoke({"x": "inner"})

    assert marker.get() == "outer", "如果这里变成了 'inner'，说明依赖 copy_context 的实现变了"


def test_run_tool_agent_carries_evidence_out_of_the_scope(business_db):
    """``run_tool_agent`` 必须把证据挂在返回值上，而不是让调用方再读一次作用域。

    这是修复「跨 context 读不到证据」时加的第三道保险：调用方与
    「当前 context 是哪一个」彻底解耦。
    """
    model = RecordingModel([
        {"name": "find_employee_by_name", "args": {"name": "张三"}},
        [],
    ])

    decision = run_tool_agent("张三在哪个部门", model=model)

    assert len(decision.tool_results) == 1
    # 与作用域内容一致（同一份，不是两次读取碰巧相等）
    assert decision.tool_results == request_ctx.get_tool_results()


def test_parallel_calls_both_execute(business_db):
    """同一轮并行发多个调用：两条都要执行并各自回一条 ToolMessage。"""
    model = RecordingModel([
        [
            {"name": "find_employee_by_name", "args": {"name": "张三"}},
            {"name": "query_employee_info", "args": {"employee_id": "E1001"}},
        ],
        [],
    ])

    decision = run_tool_agent("张三的部门，顺便看看 E1001 的岗位", model=model)

    assert len(decision.steps) == 2
    assert decision.used_tools == ["find_employee_by_name", "query_employee_info"]
    assert len(decision.tool_results) == 2


def test_chained_calls_across_rounds(business_db):
    """链式调用：先 find 拿工号，再用工号查详情 —— 规格要求的第二种情形。

    这条也顺带钉住 ``TOOL_AGENT_MAX_STEPS`` 默认值至少为 2：
    设为 1 时第二轮不会发生，模型拿到工号却没法用它。
    """
    model = RecordingModel([
        {"name": "find_employee_by_name", "args": {"name": "张三"}},
        {"name": "query_employee_info", "args": {"employee_id": "E1001"}},
        [],
    ])

    decision = run_tool_agent("张三的岗位是什么", model=model)

    assert decision.used_tools == ["find_employee_by_name", "query_employee_info"]
    assert len(decision.steps) == 2
    assert "高级工程师" in decision.tool_results[-1]


# ===========================================================================
# 6. 图接线：tool_node 的出口与失败处置
# ===========================================================================
def _patch_model(monkeypatch, model):
    from app.core import tool_agent as agent_mod

    monkeypatch.setattr(agent_mod, "_default_model", lambda: model)


def test_tool_node_direct_sets_answer_and_skips_generation(monkeypatch):
    """直答出口：answer 已就绪，置信度 1.0、无引用、不拒答。"""
    from app.graph import nodes
    from app.graph.state import create_initial_state

    _patch_model(monkeypatch, RecordingModel(["请问你叫什么名字？我帮你查。"]))
    out = nodes.tool_node(
        create_initial_state(user_query="我的年假还剩几天", session_id="s1")
    )

    assert out["intent_type"] == "direct"
    assert out["intent_source"] == "tool"
    assert out["answer"] == "请问你叫什么名字？我帮你查。"
    assert out["confidence"] == 1.0
    assert out["citations"] == []
    assert out["refused"] is False
    assert not out.get("need_human")


def test_tool_node_evidence_exit_does_not_set_answer(monkeypatch, business_db):
    """证据出口：answer 必须留给 L4，节点自己不得写答案。"""
    from app.graph import nodes
    from app.graph.state import create_initial_state

    _patch_model(
        monkeypatch,
        RecordingModel([
            {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
            [],
        ]),
    )
    # 工号直接来自用户原话（服务端没有登录态，这是唯一合法的来源）
    state = create_initial_state(user_query="查一下 E1001 的年假还剩几天", session_id="s1")
    out = nodes.tool_node(state)

    assert out["intent_type"] == "tool"
    assert out.get("answer") is None, "证据出口的答案必须由 L4 产出"
    assert "annual_leave" in (out["tool_result"] or "")
    assert out["intent_capability"] == "query_leave_balance"


def test_tool_node_routes_to_human_on_decision_failure(monkeypatch):
    """决策阶段失败 → need_human（模型不可达，本轮确实无法自动处理）。"""
    from app.graph import nodes
    from app.graph.state import create_initial_state

    class _Broken:
        def bind_tools(self, _tools):
            class _B:
                def invoke(self, _messages, **_kw):
                    raise RuntimeError("模型服务不可达")

            return _B()

    _patch_model(monkeypatch, _Broken())
    out = nodes.tool_node(create_initial_state(user_query="我的年假还剩几天", session_id="s1"))

    assert out["need_human"] is True
    assert "模型服务不可达" in out["error_msg"]


def test_tool_node_marks_degraded_instead_of_choosing_a_fallback(monkeypatch):
    """模型不支持 function calling → 只写 ``tool_degraded``，**不**自己选降级路径。

    改道由 ``graph/edges.py::tool_route_edge`` 决定。节点替下游预判结果，会让
    「这轮为什么走了简单 RAG」这条边在代码里消失——而它每天都在生效。
    """
    from app.graph import nodes
    from app.graph.state import create_initial_state

    _patch_model(monkeypatch, RecordingModel([[]], bind_error=NotImplementedError()))
    out = nodes.tool_node(create_initial_state(user_query="我的年假还剩几天", session_id="s1"))

    assert out["tool_degraded"] is True
    assert out.get("intent_type") is None, "不得替下游把出口定死"
    assert not out.get("need_human"), "降级不是失败，不该转人工"
    assert any("function calling" in w for w in out["soft_warnings"])


def test_tool_node_routes_to_human_on_tool_execution_failure(monkeypatch, business_db):
    """**工具执行失败** → 转人工：用户问的是系统里的确定事实，没有别的来源可答。

    与「检索失败只记 soft_warnings」形成对照，判据是
    「**能不能从其他来源得到答案**」而不是「这次调用有没有报错」。
    用户问「E1001 在哪个部门」而数据库挂了时，让 L4 回一句
    「知识库中没有找到相关信息」是答非所问——用户会以为这位同事不存在。
    """
    from app.core import tool_agent as agent_mod
    from app.graph import nodes
    from app.graph.state import create_initial_state

    monkeypatch.setitem(
        agent_mod._TOOLS_BY_NAME,
        "query_employee_info",
        _StubTool("query_employee_info", error=RuntimeError("业务库 502")),
    )
    _patch_model(
        monkeypatch,
        RecordingModel([
            {"name": "query_employee_info", "args": {"employee_id": "E1001"}},
            [],
        ]),
    )
    state = create_initial_state(user_query="E1001 是哪个部门的", session_id="s1")
    out = nodes.tool_node(state)

    assert out["need_human"] is True, "工具执行失败必须转人工，不得让 L4 答非所问"
    assert "query_employee_info" in out["error_msg"]
    # 故障同时要留痕（观测通道）
    assert any("执行失败" in w for w in out["soft_warnings"])


def test_tool_node_keeps_soft_warning_only_for_rejected_args(monkeypatch, business_db):
    """对照：参数被护栏拒绝**不**转人工——那是模型侧问题，不是系统故障。"""
    from app.graph import nodes
    from app.graph.state import create_initial_state

    _patch_model(
        monkeypatch,
        RecordingModel([
            {"name": "find_employee_by_name", "args": {"name": "王五"}},
            [],
        ]),
    )
    state = create_initial_state(user_query="王小明的部门", session_id="s1")
    out = nodes.tool_node(state)

    assert not out.get("need_human"), "护栏拒绝不该把用户推给人工"
    assert any("护栏拒绝" in w for w in out["soft_warnings"]), "但必须留痕"


# ===========================================================================
# 7. 不变式
# ===========================================================================
def test_grounded_args_only_reference_existing_tools():
    """护栏登记的工具名必须真实存在——写错工具名等于护栏根本没生效。"""
    names = {t.name for t in AGENT_TOOLS}

    for tool_name in GROUNDED_ARGS:
        assert tool_name in names, f"GROUNDED_ARGS 登记了不存在的工具：{tool_name}"


def test_all_tools_are_read_only_sqlite_queries():
    """3 个工具全部来自 SQLite 只读模块，且没有任何一个是检索类工具。

    这是"工具职责单一"的可执行形式：检索一旦混进来，模型就会在
    "查制度"与"查员工资料"之间做一次没有必要的选择。
    """
    from app.tools import sqlite_tools

    for tool_obj in AGENT_TOOLS:
        assert tool_obj.name in sqlite_tools.TOOLS_BY_NAME
