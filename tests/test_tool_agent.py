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
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app import config
from app.core import request_ctx
from app.core.tool_agent import (
    AGENT_TOOLS,
    GROUNDED_ARGS,
    PLAN_DONE,
    PLAN_MORE,
    execute_tool_calls,
    parse_text_tool_calls,
    run_tool_agent,
)
from app.core.tool_agent import _parse_plan, _payload_is_usable, _strip_plan_token
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


# ---------------------------------------------------------------------------
# 1b. 收口轮的正文没有消费者（延迟优化留下的观测点）
# ---------------------------------------------------------------------------
def _walk_spans(nodes):
    """把 span 树摊平成字典列表（含嵌套子节点）。"""
    for node in nodes:
        yield node
        yield from _walk_spans(node.get("children") or [])


def _run_with_trace(query, model):
    """跑一次工具 Agent，并取回它这一趟的 span 树。

    ``persist=False`` 是刻意的：测试不能往 ``logs/trace.jsonl`` 里写记录——
    那个文件是离线定量分析的输入，混进测试请求会污染统计（本项目已因此
    把 928 条 span 里的 860 条空转误当成真实流量读过一次）。
    """
    from app.core.tracing import begin_trace, end_trace

    begin_trace("tool-agent-discard-check")
    decision = run_tool_agent(query, model=model)
    return decision, list(_walk_spans(end_trace(persist=False)))


def test_closing_round_answer_is_discarded_not_reused(business_db):
    """已取到证据后，收口轮写下的正文**不被采用**——它是一段纯浪费的生成。

    这一轮唯一的有效产出是「没有更多工具调用」，而它已经由 ``tool_calls``
    为空表达出来了；正文没有任何消费者（最终答案必须由 L4 受控生成，
    引用编号与置信度只在那一层产生）。

    代价是实打实的。真实埋点里的这条链路::

        09-15 20:08  tool 阶段 5096.6ms = 2499.7（1 次调用）+ 2571.0（0 次调用）
        09-17 20:34  tool 阶段 7748.1ms = 3633.3（1 次调用）+ 4105.1（0 次调用）

    收口轮占了工具阶段的一半，而 54 条真实链路里有 46 条正是这个形状。

    ⚠️ 09-17 复测顺带推翻了一个假设：**正文长度不是成本**。第二轮
    ``discarded_chars`` 只有 52 字（≈0.5s 吐字），剩下约 3.6s 是这一轮的
    模型往返本身。所以「让模型别写」最多省 0.5s，且省不掉这一轮——
    真正该问的是**这一轮该不该发生**。

    守的是**丢弃关系**而不是某个具体长度：正文要进 ``discarded_chars``
    供离线核对该指令有没有生效（09-17 核对结果：**没生效**），但**绝不能**
    进 ``direct_answer``——后者是直答出口，会被调用方直接当成答案返回。
    """
    wasted = "张三的剩余年假为 12 天，调休 3 天。"
    model = RecordingModel([
        {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
        wasted,
    ])

    decision, spans = _run_with_trace("张三的年假还剩几天", model)

    assert decision.used_tools == ["query_leave_balance"]
    assert decision.direct_answer is None, "已取到证据时正文不得被当成答案"

    closing = [s for s in spans if s["name"] == "agent_step_1"]
    assert len(closing) == 1, "第二轮必须真实发生——链式调用还要靠它再决策一次"
    assert closing[0]["attrs"]["tool_calls"] == 0
    assert closing[0]["attrs"]["discarded_chars"] == len(wasted)


def test_agent_prompt_tells_the_model_not_to_write_the_discarded_answer():
    """提示词必须明确要求：已取到数据且无需再调工具时，不要再撰写回答。

    这条守的是**文案，不是效果**。2026-09-17 实测该指令**未生效**（收口轮
    仍然写下 52 字再被丢弃），而且**正文长度本来就不是成本**——那一轮约 4s
    里只有约 0.5s 是吐字，其余是模型往返本身。所以它省不下时间。

    保留它，是因为"生成一段注定被丢弃的正文"在语义上仍然是错的，且它没有
    行为载体：删掉这句话不会让任何别的测试变红。但**别再指望它省时间**，
    也别再往提示词里加同义句——真正的浪费是这一轮该不该发生，见
    `test_closing_round_answer_is_discarded_not_reused` 的实测记录。
    """
    from app.core.tool_agent import SYSTEM_PROMPT

    assert "不要再写回答" in SYSTEM_PROMPT
    assert "会被直接丢弃" in SYSTEM_PROMPT


def test_agent_prompt_asks_for_sufficiency_before_calling_again():
    """提示词必须要求模型**每一轮先核对"手上的结果够不够"**，够了就不再调用工具。

    这条修的是一个真缺陷，而不是加一句泛泛的"要聪明一点"。改动前提示词第 2 条写的是

        用户问「张三的部门」，可以先用 find_employee_by_name 拿到工号，
        再用 query_employee_info 查详情。

    而 ``find_employee_by_name`` 的返回**本来就含部门**（见 ``tools/sqlite_tools.py``
    的 ``_ok({"ambiguous": ..., "candidates": rows})``，rows 就是工号/姓名/部门三列）。
    也就是说：**提示词自己教模型在这一问上多查一次**。文档里"三个工具"的描述与
    这条示例互相矛盾，模型自然选更稳的那条路——再查一次。

    代价有三层：多打一次库、上下文里多塞一份内容重叠的 JSON（这份 JSON 还要被
    L4 再 prefill 一遍）、以及把"链式调用"的样例教成了"每问都两跳"。

    ⚠️ 这条**不承诺更快**。实测第 2 轮无论发调用还是收口都要付一次模型往返
    （收口那轮因为要生成一段被丢弃的正文，中位反而更贵），所以修它买到的是
    正确性与 token，不是延迟。别拿它当性能优化的证据。

    ⚠️ 反向验证：把第 2 条改回"信息不够就去查"、或把 "张三的部门" 那条示例加回来，
    本条必须变红。
    """
    from app.core.tool_agent import SYSTEM_PROMPT

    assert "够不够回答" in SYSTEM_PROMPT, "缺少「先核对够不够」这条判据"
    assert "不必" in SYSTEM_PROMPT and "query_employee_info" in SYSTEM_PROMPT
    # 那条教模型多查一次的示例不能再出现——它和工具自身的返回值矛盾。
    assert "再用 query_employee_info 查详情" not in SYSTEM_PROMPT


def test_agent_prompt_tells_the_model_how_to_attribute_a_dead_end():
    """查不到时要**分清是哪一种**，不能一律含糊地说"没查到"。

    三种情形对用户的意义完全不同：调用选错了工具或参数（该重试）、问题对了但
    库里没有（该如实说没有）、这个问题本来就不归工具 Agent 管（制度类，该由
    检索回答）。把它们混成一句"没查到"，用户无从判断该不该换个说法再问。

    "不要拿别的工具硬凑"这一句是**闸门**：没有它，模型可能在制度类问题上用
    ``query_employee_info`` 硬凑出一段看起来像答案的话——那正是最坏的一类失败
    （结构完整、语义答非所问）。

    ⚠️ 反向验证：把第 4 条与"不要拿手边的工具硬凑"删掉，本条必须变红。
    """
    from app.core.tool_agent import SYSTEM_PROMPT

    assert "先分清是哪一种" in SYSTEM_PROMPT
    assert "不要重复同一个无效调用" in SYSTEM_PROMPT
    assert "三个工具都答不了" in SYSTEM_PROMPT
    assert "不要拿手边的工具硬凑" in SYSTEM_PROMPT


# ===========================================================================
# 1c. 计划词：用模型自己声明的计划省掉一整轮决策
#
# 背景见 `app/core/tool_agent.py` 里 `PLAN_DONE` 那一节。一句话：两轮里有一轮
# 并不是"链式调用的第二跳"，而只是模型回一句「够了」——而它在那句话里**已经把
# 答案写出来了**，随后被 L4 覆盖重写。让模型在出发前用一个词声明打算，
# 声明"够"且本轮数据干净时，后面那轮决策**整个省掉**。
#
# 本地不自己算"够不够"：从工具结果反推"够不够回答用户问的那件事"需要语义比对，
# 写出来必定是又一份与模型抢活的规则。模型知道自己打算查一步还是两步——
# 这个判断**不需要看到结果**，所以它才成立。
# ===========================================================================
class _PlannedModel:
    """可**携带正文**的假模型：一轮 = ``(正文, 工具调用列表)``。

    为什么不用 ``tests/fakes.py`` 的 ``RecordingModel``：它在"发出工具调用"的那
    一轮 ``content`` 恒为空串（``_RecordingBinding.invoke`` 里写死），而计划词机制
    的**全部输入**恰恰是「正文里带一个词、同时发出 ``tool_calls``」这个形状。
    改 ``RecordingModel`` 会波及所有依赖正文的既有用例——它是共享替身，动它等于
    同时改掉十几个用例的前提。故在此另起一个**同契约**的最小替身：同样只暴露
    ``bind_tools`` / ``calls``，同样"脚本耗尽后重复最后一项"。

    这不是替身放宽了约束：真实链路里 ``bound.invoke`` 返回的 ``AIMessage``
    本来就同时带 ``content`` 与 ``tool_calls``（见 ``content_of`` 的说明），
    ``RecordingModel`` 只是把它简化掉了。
    """

    def __init__(self, script: list) -> None:
        self.script = list(script) if script else [("", [])]
        self.calls: list = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _PlannedBinding(self)


class _PlannedBinding:
    def __init__(self, model: _PlannedModel) -> None:
        self._model = model

    def invoke(self, messages, **_kwargs):
        model = self._model
        model.calls.append(list(messages))
        index = min(len(model.calls) - 1, len(model.script) - 1)
        content, calls = model.script[index]
        return AIMessage(
            content=content,
            tool_calls=[
                {
                    "name": c["name"],
                    "args": c.get("args") or {},
                    "id": c.get("id") or f"call_{index}_{i}",
                    "type": "tool_call",
                }
                for i, c in enumerate(calls)
            ],
        )


def test_plan_done_with_usable_data_skips_the_next_decision_round(business_db):
    """声明「这次够了」且结果干净 → **第 2 轮整个不发生**。

    收益的唯一可观测形式就是"模型少被调用一次"：省掉的那一轮是一次真实往返
    （实测 1.8~2.9s），且它写下的正文本来就要被丢弃。所以这里断言 ``model.calls``
    的**次数**而不是耗时——离线测试里的耗时是假的，调用次数不是。

    ⚠️ 反向验证：去掉 ``_decide`` 末尾那个 ``if`` 里任意一道守卫，本条必须变红
    （第三道由 ``test_plan_done_on_the_last_round_skips_nothing`` 单独守）。
    """
    model = _PlannedModel([
        ("PLAN_DONE", [{"name": "query_leave_balance", "args": {"employee_id": "E1001"}}]),
    ])

    decision = run_tool_agent("E1001 的年假还剩几天", model=model)

    assert decision.plan_shortcut is True
    assert len(model.calls) == 1, "省掉的正是第 2 轮模型往返"
    assert decision.used_tools == ["query_leave_balance"], "省的是决策轮，不是证据"
    assert not decision.is_direct, "取到了证据 → 仍走证据出口，成文权在 L4"


def test_plan_more_still_consumes_the_next_round(business_db):
    """声明「还要看结果才知道下一步」→ 照旧走下一轮。

    这条同时是**链式调用的保护**：姓名 → 工号的第二跳必须看见第一跳的结果，
    计划词机制不能把它省掉。真实模型在「张三的年假还剩几天」「李四是什么时候
    入职的」上都声明 ``PLAN_MORE``，与链式调用的实际需要一致。
    """
    model = _PlannedModel([
        ("PLAN_MORE", [{"name": "find_employee_by_name", "args": {"name": "张三"}}]),
        ("", []),
    ])

    decision = run_tool_agent("张三的年假还剩几天", model=model)

    assert decision.plan_shortcut is False
    assert len(model.calls) == 2
    assert decision.steps[0].tool == "find_employee_by_name"


def test_plan_done_is_overridden_when_the_tool_returns_ambiguous_candidates(business_db):
    """声明 ``PLAN_DONE`` 但工具返回的是**候选人列表** → 不信它，照旧走下一轮。

    这是实测里唯一一次判断偏差，也正是第四道守卫存在的理由：真实模型对
    「王五的部门是什么」也吐了 ``PLAN_DONE``（重名时它自己也拿不准该说哪一位），
    而工具返回的是两位王五。此时必须让模型看见候选人，才有机会请用户确认
    （提示词第 3 条：重名时必须确认，不要自己挑一个）。

    ⚠️ 反向验证：把 ``and all(step.usable for step, _ in executed)`` 去掉，本条必须
    变红——而且它变红的**方式**很具体：模型看不到候选人，用户会拿到一个被替选中
    的部门，且从回答里看不出选过。
    """
    model = _PlannedModel([
        ("PLAN_DONE", [{"name": "find_employee_by_name", "args": {"name": "王五"}}]),
        ("", []),
    ])

    decision = run_tool_agent("王五的部门是什么", model=model)

    assert decision.plan_shortcut is False
    assert len(model.calls) == 2, "必须让模型看到两位王五"
    assert decision.steps[0].status == "ok", "工具跑通了"
    assert decision.steps[0].usable is False, "但拿回来的是候选人列表，不能直接作答"


def test_plan_done_is_overridden_when_the_tool_reports_not_found(business_db):
    """声明 ``PLAN_DONE`` 但查无此人（``ok:false``）→ 不信它。

    与重名那条是同一判据的两个分支：``status`` 都是 ``ok``（工具**跑通了**），
    却都不是能拿去作答的数据。少了这一分支，「E9999」这种打错的工号会静默变成
    "资料不足"，而模型本来还有一次机会换工具或改参数（提示词第 4 条）。
    """
    model = _PlannedModel([
        ("PLAN_DONE", [{"name": "query_leave_balance", "args": {"employee_id": "E9999"}}]),
        ("", []),
    ])

    decision = run_tool_agent("E9999 的年假还剩几天", model=model)

    assert decision.plan_shortcut is False
    assert len(model.calls) == 2
    assert decision.steps[0].status == "ok"
    assert decision.steps[0].usable is False


def test_plan_shortcut_can_be_switched_off(monkeypatch, business_db):
    """``TOOL_AGENT_PLAN_SHORTCUT=false`` → 与这个机制不存在时**完全一致**。

    这条守的是"退路现成"：机制会缩小一处覆盖面（模型自认为一次够、实际不够时
    就没有机会补查），所以它必须能一键回到旧行为。没有这条测试，那个开关就只是
    配置里的一行注释——没人知道它到底还接不接着。
    """
    monkeypatch.setattr(config, "TOOL_AGENT_PLAN_SHORTCUT", False)
    model = _PlannedModel([
        ("PLAN_DONE", [{"name": "query_leave_balance", "args": {"employee_id": "E1001"}}]),
    ])

    decision = run_tool_agent("E1001 的年假还剩几天", model=model)

    assert decision.plan_shortcut is False
    assert len(model.calls) == 2, "关掉开关就是旧行为：第 2 轮照常发生"


def test_plan_done_on_the_last_round_skips_nothing(business_db):
    """``max_steps=1`` 时第 0 轮就是最后一轮——本来就该退出，"省"无从谈起。

    第三道守卫（``round_index < steps_limit - 1``）守的是这个边界。它看起来像废话，
    去掉也不会让正例变红（正例里 ``round_index=0``、``steps_limit=2``），但会让
    ``PLAN_DONE`` 在**单轮配置**下照旧写一条"已省"的日志与埋点——观测数据会说
    "省了一轮"，而实际上一次都没少调。**埋点错比没有埋点更糟**：它会把之后所有
    关于收益的推断一起带偏。
    """
    model = _PlannedModel([
        ("PLAN_DONE", [{"name": "query_leave_balance", "args": {"employee_id": "E1001"}}]),
    ])

    decision = run_tool_agent("E1001 的年假还剩几天", model=model, max_steps=1)

    assert decision.plan_shortcut is False, "没有可省的轮次时不得记成已省"
    assert len(model.calls) == 1


def test_unrecognized_or_absent_plan_falls_back_to_the_old_two_rounds(business_db):
    """正文里没有可识别的计划词 → 与这个机制不存在时行为一致（**默认保守**）。

    真实模型偶尔会在调工具的那一轮写一句解释（「我这就帮你查。」）。认不出来就按
    "没声明"处理，绝不能反过来"猜它大概是想说够了吧"——两条错误方向的代价不对称：
    猜错的方向是用户拿到"资料不足"，保守的方向只是白花一次往返。
    """
    model = _PlannedModel([
        ("我这就帮你查。", [{"name": "query_leave_balance", "args": {"employee_id": "E1001"}}]),
        ("", []),
    ])

    decision = run_tool_agent("E1001 的年假还剩几天", model=model)

    assert decision.plan_shortcut is False
    assert len(model.calls) == 2


def test_plan_shortcut_is_observable_in_the_span_and_the_payload(business_db):
    """省与没省都必须可观测，否则"省了多少、省错几次"全靠猜。

    ``plan`` 记在**当轮 span** 上（模型到底声明了什么），``plan_shortcut`` 记在
    ``ToolDecision`` 与 ``tool`` span 上（最后到底省没省）。两个值分开记是必要的：
    只有 ``plan=PLAN_DONE`` 而**没省**的那些请求，正是被第四道守卫救回的那一类，
    它们的条数决定了"模型判断准不准"。
    """
    model = _PlannedModel([
        ("PLAN_DONE", [{"name": "query_leave_balance", "args": {"employee_id": "E1001"}}]),
    ])

    decision, spans = _run_with_trace("E1001 的年假还剩几天", model)

    first = [s for s in spans if s["name"] == "agent_step_0"]
    assert len(first) == 1, "第 1 轮必然发生"
    assert first[0]["attrs"]["plan"] == PLAN_DONE
    assert decision.to_dict()["plan_shortcut"] is True


def test_plan_word_never_leaks_into_a_direct_answer():
    """直答出口的正文要过一遍剥离——计划词是**给机器看的**，不该给用户看。

    约定上模型只在调工具时写计划词，但约定不是保证：这条路径下若原样透出，
    用户看到的第一行就是 ``PLAN_DONE``。剥离放在**出口处**而不是"相信模型不会写"，
    与 ``GROUNDED_ARGS`` 是同一个思路——把关卡放在能拦住的位置，不放在源头。
    """
    model = _PlannedModel([("PLAN_DONE\n请问你的工号是多少？我来查。", [])])

    decision = run_tool_agent("我的年假还剩几天", model=model)

    assert decision.is_direct
    assert decision.direct_answer == "请问你的工号是多少？我来查。"
    assert decision.plan_shortcut is False, "没有工具调用就没有「这一次够不够」这个问题"


def test_a_lone_plan_word_is_not_silently_turned_into_an_empty_answer():
    """整段正文**只有**计划词时保留原文——这是**有意的偏离**，别把它改成空串。

    "正确"的做法似乎是剥干净（返回空串），但那会让两件完全不同的事在观测上同形：
    ①模型什么都没说（模型故障 / 端点异常）；②模型只写了一个计划词（约定被违反）。
    前者该报警，后者只是一行怪字符串。空串会让调用方以为直答出口被正常履行了，
    用户却拿到一片空白——**静默失败比可见的异常难查得多**。
    """
    model = _PlannedModel([("PLAN_DONE", [])])

    decision = run_tool_agent("我的年假还剩几天", model=model)

    assert decision.is_direct
    assert decision.direct_answer == "PLAN_DONE"


@pytest.mark.parametrize(
    "content,expected",
    [
        ("PLAN_DONE", PLAN_DONE),
        ("PLAN_MORE", PLAN_MORE),
        ("  PLAN_DONE  ", PLAN_DONE),                # 前后空白
        ("**PLAN_DONE**", PLAN_DONE),                # 模型爱加粗
        ("`PLAN_MORE`", PLAN_MORE),                  # 或者写成行内代码
        ("plan_done", PLAN_DONE),                    # 大小写不敏感
        ("PLAN_DONE\n我这就去查。", PLAN_DONE),       # 只看第一个非空行
        ("\n\nPLAN_MORE\n先查工号。", PLAN_MORE),
        ("PLAN_DONE。", PLAN_DONE),                  # 尾随标点：startswith 的取舍
        ("我这就去查。", None),                       # 没有声明
        ("计划：PLAN_DONE", None),                    # 不做模糊匹配
        ("", None),
        (None, None),
    ],
)
def test_parse_plan_recognizes_only_the_declared_words(content, expected):
    """计划词的识别范围必须**窄且可预测**：认不出来就按没声明处理。

    刻意不做模糊匹配（不找子串、不看近义词），因为两条错误方向的代价不对称：
    认错（把没声明的当成已声明）会让模型失去补查机会，用户拿到"资料不足"；
    漏认只是白花一次往返。所以这里宁愿漏认。

    注意 ``PLAN_DONE。`` 那一条：识别用的是 ``startswith``，为的是容忍尾随标点，
    代价是 ``PLAN_DONEX`` 也会被认成 ``PLAN_DONE``。这是**有意的宽松**——
    真实模型不会写出那种串，而尾随句号很常见。
    """
    assert _parse_plan(content) == expected


@pytest.mark.parametrize(
    "content,expected",
    [
        ("PLAN_DONE\n我这就去查。", "我这就去查。"),
        ("PLAN_MORE\n\n先查工号，再查年假。", "先查工号，再查年假。"),
        ("PLAN_DONE\n第一行\n第二行", "第一行\n第二行"),
        ("没有计划词。", "没有计划词。"),
        ("", ""),
        (None, ""),
        ("PLAN_DONE", "PLAN_DONE"),  # 孤儿计划词：保留原文，见上一条用例
    ],
)
def test_strip_plan_token_removes_only_the_token_and_its_line(content, expected):
    """剥离只吃掉"计划词那一行"，正文其余部分必须原样保留。

    它作用在**直答出口**上，那里返回的是用户会直接读到的文本。多剥一行或少剥
    一行都很难在别处发现——直答出口没有下游消费者会再校验一次。
    """
    assert _strip_plan_token(content) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"ok": true, "data": {"employee_id": "E1001", "annual_leave": 5.0}}', True),
        ('{"ok": true, "data": {"ambiguous": false, "candidates": [{"employee_id": "E1001"}]}}', True),
        ('{"ok": true, "data": {"ambiguous": true, "candidates": [{}, {}]}}', False),
        ('{"ok": false, "error": "not_found", "message": "未找到该员工"}', False),
        ('{"ok": false, "error": "invalid_argument"}', False),
        ("这不是 JSON", False),
        ("[1, 2]", False),   # 顶层不是对象
        ("", False),
        (None, False),
    ],
)
def test_payload_usability_requires_ok_and_no_ambiguity(raw, expected):
    """"可用"= ``ok:true`` **且**不是重名歧义——两条都要满足。

    解析不出来一律按**不可用**处理。两个方向的错误代价不对称：说"不可用"只是多花
    一次往返；说"可用"却可能让用户拿到一个被替选中的答案，而且看不出来选过。
    """
    assert _payload_is_usable(raw) is expected


def test_not_found_still_hands_the_answer_back_to_l4(business_db):
    """``usable=False`` **不能**拿来判断走哪个出口——这两个判据必须分开。

    ``used_tools`` 的判据是 ``status == "ok"``（工具**跑通了**）。若顺手改成
    ``usable``，查无此人时就会落进直答出口：模型说什么就是什么。它完全可能把
    ``not_found`` 说成「该员工年假剩余 0 天」——而"查不到这个人"与"余额是 0"
    是两件事（提示词第 4 条专门要求分清）。成文权必须在 L4。
    """
    model = _PlannedModel([
        ("", [{"name": "query_leave_balance", "args": {"employee_id": "E9999"}}]),
        ("", []),
    ])

    decision = run_tool_agent("E9999 的年假还剩几天", model=model)

    assert not decision.is_direct, "查无此人仍走证据出口"
    assert decision.used_tools == ["query_leave_balance"]
    assert "not_found" in decision.tool_results[0]


def test_agent_prompt_declares_the_two_plan_words_and_when_not_to_write_them():
    """提示词是计划词约定的**唯一**来源，且必须说清"什么时候不写"。

    "不调用工具时不要写这个词"这半句不是修辞：直答出口的正文会原样给用户，
    少了这半句，模型可能在追问用户的句子上也带一个 ``PLAN_DONE``。出口处固然还有
    ``_strip_plan_token`` 兜底，但那是护栏、不是可以省掉约定的借口。

    两个词在**语义上**必须各有定义，不只是出现名字——光有 ``PLAN_DONE`` 这个词
    而没有"拿到结果后我就能回答用户了"这句，模型没有任何依据去判断该写哪个。
    换言之本条守的是**约定可被理解**，不是"字符串出现过"。

    ⚠️ 反向验证：把 ``prompts.py`` 里整段（``【发出工具调用时……`` 到那个右括号
    结尾的示例行）删掉，本条必须变红。只删段落标题**不足以**让它变红——
    这几条断言用的是段内的字样，不是标题。
    """
    from app.core.tool_agent import SYSTEM_PROMPT

    assert PLAN_DONE in SYSTEM_PROMPT
    assert PLAN_MORE in SYSTEM_PROMPT
    # 两个词的语义定义：模型据此判断该写哪一个。
    assert "这次调用拿到结果后，我就能回答用户了" in SYSTEM_PROMPT
    assert "这次只是中间一步" in SYSTEM_PROMPT
    # 约定里必须包含"什么时候不写"，否则追问用户时也会带上这个词。
    assert "不调用工具时不要写这个词" in SYSTEM_PROMPT
    assert "正文里除了这一个词之外" in SYSTEM_PROMPT


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

    monkeypatch.setattr(agent_mod, "default_model", lambda: model)


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
