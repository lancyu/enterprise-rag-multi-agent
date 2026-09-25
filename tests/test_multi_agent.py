"""五 Agent 协作架构的回归测试 —— 路由分发、边界拦截、各 Agent 的职责边界。

为什么单独一个文件
------------------
单 Agent 架构下的测试都在守「某一次 function calling 决策的产物」（候选集、
护栏、回捞……），见 ``tests/test_tool_agent.py``。那些用例换成五个 Agent 后
**一条都不能少**，但它们守不住新架构特有的东西——**"谁来做决定"这件事本身**。
本文件专门守这一类主张，它们全都是"不会被报错提醒"的那种失效：

| 主张 | 失效后的症状 | 本文件中的守卫 |
|---|---|---|
| 边界规则只有一份（在入口） | 五个子 Agent 各写五份，某次只改了四份 → 越界问题被静默放行 | ``test_boundary_rule_lives_in_exactly_one_prompt`` |
| 越界回复是常量、不经过模型 | 越界话术开始编造事实，且每次措辞不同、不可审计 | ``test_out_of_scope_answer_is_defined_once`` |
| 闲聊不检索、不调工具、不调模型 | 一句「你好」触发检索与 function calling，或在离线时被回成「知识库没找到」 | ``test_smalltalk_*`` |
| 场景（该谁干）与出口（答案怎么来的）是两个字段 | 合并后路由被迫"预判执行结果"，工具 Agent 的追问被当成拒答 | ``test_scene_and_intent_type_answer_different_questions`` |
| 降级决策落在边上、拓扑完整可见 | 改道逻辑藏进节点，读拓扑的人看不到它每天都在生效 | ``test_degraded_rerouting_is_visible_in_the_topology`` |
| 两个编译产物共用同一批节点 | 流式端点重抄一遍前置链路 → 两条链路对同一状态给出不一致回答 | ``test_full_and_pre_generation_graphs_share_every_node`` |
| ``user_id`` 真的进到图里 | 漏传 → 所有人的长期记忆互相串台，接口却照样返回 200 | ``test_workflow_execute_passes_user_id_into_the_graph`` |

全部为纯单元 / 进程内测试：不联网、不调真实模型、不读 ``data/enterprise.db``。
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app import config
from app.core.prompts import PROMPTS
from app.core.router_agent import (
    DEFAULT_SCENE,
    OUT_OF_SCOPE_ANSWER,
    SCENE_COMPLEX_RAG,
    SCENE_OUT_OF_SCOPE,
    SCENE_SIMPLE_RAG,
    SCENE_SMALLTALK,
    SCENE_TOOL,
    SCENES,
    RouteDecision,
    route_query,
)
from app.core.verifier import (
    TRIGGER_DEGRADED,
    TRIGGER_TIGHT_ROUTE,
    TRIGGER_WEAK_EVIDENCE,
)
from app.graph.edges import scene_route_edge, tool_route_edge
from app.graph.state import create_initial_state
from app.graph.workflow_graph import (
    NODE_NAMES,
    enterprise_workflow,
    pre_generation_workflow,
)
from tests.fakes import RecordingModel

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: 越界话术的**唯一产生点**。这句话术是 ``out_of_scope`` 通道的对外契约，
#: 所以它和通道闭集、图节点映射一起住在 ``app/core/routing/catalog.py``
#: （「意图即数据」的那张表）；``app/core/router_agent.py`` 只做 re-export，
#: ``from ... import`` 不含文本，故不会出现在下面的扫描结果里。
#:
#: 提成具名常量的好处：搬家是**一次有意的、可评审的**改动，而不是把路径
#: 散在断言里、让"改路径"和"改不变量"看起来像同一件事。
_OUT_OF_SCOPE_OWNER = "app/core/routing/catalog.py"


class _TextModel:
    """只回纯文本的假模型（路由分类 / 问题拆解用）。

    工具调用请用 ``tests/fakes.py::RecordingModel``——它才有 ``bind_tools``。
    这里不能用它，是因为 ``route_query`` / ``decompose_query`` 都是**直接
    ``invoke`` 一句话拿一段文本**，不经过工具绑定（``RecordingModel`` 因此没有
    ``invoke``，误用会报 ``AttributeError``，这个报错与真实故障无关）。
    """

    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list = []

    def invoke(self, messages, **_kwargs):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.reply)


def _route_edges(source: str, graph=None) -> set:
    """某个节点在编译产物里的全部出边目标（含条件边的每一个分支）。"""
    graph = graph or enterprise_workflow
    return {e.target for e in graph.get_graph().edges if e.source == source}


def _trace_nodes(state) -> list:
    return [step["node"] for step in state.get("trace", [])]


# ===========================================================================
# 一、路由 Agent：唯一入口，且**永不失败**
# ===========================================================================
@pytest.mark.parametrize("scene", SCENES)
def test_router_accepts_every_scene_in_the_closed_set(scene):
    """闭集里的每个场景都要能被解析出来，并且只有越界场景带话术。"""
    reply = json.dumps({"route": scene, "reason": "测试", "confidence": 0.9})
    decision = route_query("随便问一句", model=_TextModel(reply))

    assert decision.scene == scene
    assert decision.source == "router"
    assert decision.degraded is False
    assert (decision.out_of_scope_answer is not None) == (scene == SCENE_OUT_OF_SCOPE), (
        "越界话术只该挂在越界场景上——其它场景带上它，会被 out_of_scope_node 之外的地方误用"
    )


def test_router_rejects_scene_outside_the_closed_set():
    """模型编一个不存在的场景名 → 必须在这里挡住，不能漏进条件边。

    漏进去的后果不是"分类不准"，而是 LangGraph 抛 ``KeyError``——那个异常会被
    误读成"图配置坏了"，而真实原因是模型输出越界。
    """
    decision = route_query("年假多少天", model=_TextModel('{"route": "weather", "reason": "瞎编"}'))

    assert decision.scene in SCENES
    assert decision.degraded is True
    assert decision.source == "router:fallback"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "我不知道该选哪个",
        '{"route": "simple_rag"',          # 截断的 JSON
        '["simple_rag"]',                   # 数组而非对象
        '{"route": "weather"}',             # 合法 JSON，但场景名不在闭集里
    ],
)
def test_router_falls_back_when_output_is_unparseable(raw):
    """解析不出来 → 走确定性规则，而不是把异常抛给调用方。"""
    decision = route_query("年假多少天", model=_TextModel(raw))
    assert decision.scene in SCENES
    assert decision.source == "router:fallback"


def test_router_parses_json_wrapped_in_markdown():
    """模型偶尔把 JSON 包在代码块/解释里——尽力提取，别浪费一次分类。"""
    raw = '好的，我的判断是：\n```json\n{"route": "complex_rag", "reason": "需对比", "confidence": 0.8}\n```'
    decision = route_query("对比年假和调休", model=_TextModel(raw))

    assert decision.scene == SCENE_COMPLEX_RAG
    assert decision.source == "router"


def test_router_never_raises_on_model_failure():
    """模型不可达 → 兜底，不抛。**路由不能是本轮失败的原因**：它是所有路径的入口。"""
    decision = route_query("我的年假还剩几天", model=_TextModel(error=RuntimeError("连接被重置")))

    assert decision.scene in SCENES
    assert decision.degraded is True
    assert decision.error and "连接被重置" in decision.error


def test_router_fails_open_on_empty_query():
    """空提问也不抛——它可能来自前端误触发。"""
    assert route_query("   ").scene in SCENES


@pytest.mark.parametrize(
    "query, expected",
    [
        ("你好", SCENE_SMALLTALK),
        ("谢谢", SCENE_SMALLTALK),
        ("再见", SCENE_SMALLTALK),
        ("你是谁", SCENE_SMALLTALK),
        # 下面三条是**代价不对称**那一侧：宁可漏判寒暄（多检索一次），
        # 也不能把业务问题当闲聊打发掉（用户会以为这问题白问了）。
        ("我的年假还剩几天", SCENE_SIMPLE_RAG),
        ("好像这个制度不太清楚", SCENE_SIMPLE_RAG),
        ("明天天气怎么样", SCENE_SIMPLE_RAG),
    ],
)
def test_router_fallback_is_conservative(query, expected):
    """确定性兜底只认「明确是寒暄」，其余一律按制度查询处理。

    ``明天天气怎么样`` 落在 ``simple_rag`` 而不是 ``out_of_scope`` 是**刻意的**：
    误拦一个真业务问题 = 用户彻底拿不到答案，代价不可逆；漏拦则由 L4 用
    「知识库中没有相关信息」收尾，用户还能换个说法再问。
    """
    from app.core.router_agent import _fallback_route

    assert _fallback_route(query, "测试").scene == expected


def test_fallback_never_guesses_tool_or_out_of_scope():
    """兜底的默认场景必须落在"多检索一次"这一侧——这是一条结构性约束。

    它保证的是：将来有人给 ``DEFAULT_SCENE`` 换值时，会在这里被拦住并被迫
    想清楚"分错路的代价"（见 ``router_agent`` 模块 docstring）。
    """
    assert DEFAULT_SCENE not in (SCENE_TOOL, SCENE_OUT_OF_SCOPE)


def test_router_node_records_degradation_in_state(monkeypatch):
    """路由降级必须留痕：它不报错、不丢答案，只是分类可能不准。"""
    from app.graph import nodes

    monkeypatch.setattr(nodes, "route_query",
                        lambda **kw: RouteDecision(scene=SCENE_SIMPLE_RAG, reason="测试",
                                                   source="router:fallback", degraded=True))
    out = nodes.router_node(create_initial_state(user_query="年假多少天", session_id="s"))

    assert out["scene"] == SCENE_SIMPLE_RAG
    assert out["scene_source"] == "router:fallback"
    assert out["soft_warnings"], "路由降级不进 soft_warnings 就等于没发生"
    assert out["route_decision"]["degradations"]


def test_router_node_carries_the_gray_reason_into_state(monkeypatch):
    """漏斗"为什么没判"必须落到 state —— 证据校验读的就是它。

    ``router_node`` 是这条信息的**唯一入口**：漏写一步，``verifier_node`` 就只
    读得到空串，"路由摇摆"这一类再也触发不了复核 —— 而且**不会有任何报错**，
    面板上只会显示"证据无异常"。

    ``None`` 归一化成空串（本地快通道判了 / 拿不到这个信息）：两者混用会让
    "没有这个信息"与"漏斗看过、没问题"在观测上同形。

    ⚠️ 反向验证：删掉 ``router_node`` 里的
    ``new_state["route_gray_reason"] = decision.gray_reason or ""``，本条必须变红。
    """
    from app.graph import nodes

    monkeypatch.setattr(nodes, "route_query",
                        lambda **kw: RouteDecision(scene=SCENE_SIMPLE_RAG, reason="测试",
                                                   source="router", gray_reason="tight_margin"))
    out = nodes.router_node(create_initial_state(user_query="年假多少天", session_id="s"))
    assert out["route_gray_reason"] == "tight_margin"
    assert out["route_decision"]["gray_reason"] == "tight_margin"

    # 本地快通道判得了 → 漏斗的 gray_reason 是 None → state 里落到空串
    monkeypatch.setattr(nodes, "route_query",
                        lambda **kw: RouteDecision(scene=SCENE_SIMPLE_RAG, reason="测试",
                                                   source="router:local"))
    local = nodes.router_node(create_initial_state(user_query="年假多少天", session_id="s"))
    assert local["route_gray_reason"] == ""


# ---------------------------------------------------------------------------
# 本地快通道：零模型判定
#
# 它守的是一条**不会有人报错**的主张：路由这道闸门对句式固定的提问不该收费。
# 失效的表现不是崩溃，而是延迟悄悄退回 2.5 秒——没有任何日志会变红。
# ---------------------------------------------------------------------------
def test_local_fast_path_answers_without_touching_the_model(monkeypatch):
    """命中快通道时，一次模型调用都不该发生——这正是它存在的全部理由。"""
    from app.core import router_agent

    monkeypatch.setattr(config, "USE_REAL_LLM", True)

    def _must_not_be_called():
        raise AssertionError("本地快通道命中时不得调用路由模型")

    monkeypatch.setattr(router_agent, "default_model", _must_not_be_called)

    decision = route_query("你好")

    assert decision.scene == SCENE_SMALLTALK
    assert decision.source == router_agent.SOURCE_LOCAL
    assert decision.degraded is False


def test_local_fast_path_leaves_the_gray_zone_to_the_model(monkeypatch):
    """漏斗判不了时必须**交回模型**，不能自己拍一个默认场景。

    否则灰区请求会集体掉进 ``DEFAULT_SCENE``——那是一次静默的质量退化，
    而延迟指标反而会变得很好看。
    """
    from app.core import router_agent

    monkeypatch.setattr(config, "USE_REAL_LLM", True)
    stub = _TextModel(
        json.dumps({"route": SCENE_COMPLEX_RAG, "reason": "测试", "confidence": 0.9})
    )
    monkeypatch.setattr(router_agent, "default_model", lambda: stub)

    decision = route_query("请帮我分析一下当前国际形势对我们部门明年预算的影响")

    assert decision.scene == SCENE_COMPLEX_RAG
    assert decision.source == "router"
    assert stub.calls, "漏斗判不了时应当落到模型上"


def test_gray_reason_survives_the_model_path(monkeypatch):
    """漏斗"为什么没判"必须活到路由结论里 —— 它是证据校验的触发判据之一。

    这条断言看着像在测一个排障字段，其实守的是**通路**：``gray_reason`` 从漏斗
    出发，穿过 ``_local_route`` → ``route_query`` 的模型分支 → ``RouteDecision``
    → ``router_node`` → ``GraphState``。中间任何一处漏传，下游就只剩"漏斗没判"
    这个无差别的大桶，只能一律按可疑处理（实测那会让 85% 的正常提问被拉去复核）。

    为什么不能只看最终效果：``gray_reason`` 丢失的默认表现是"什么都不触发"，
    与"这一轮确实没问题"完全同形 —— 静默、不报错、指标还更好看。

    ⚠️ 反向验证：删掉 ``route_query`` 模型分支里的 ``gray_reason=gray_reason``，
    本条必须变红。
    """
    from app.core import router_agent

    monkeypatch.setattr(config, "USE_REAL_LLM", True)
    stub = _TextModel(
        json.dumps({"route": SCENE_COMPLEX_RAG, "reason": "测试", "confidence": 0.9})
    )
    monkeypatch.setattr(router_agent, "default_model", lambda: stub)

    decision = route_query("请帮我分析一下当前国际形势对我们部门明年预算的影响")

    assert decision.source == "router", "这条问句应当落到模型上"
    assert decision.gray_reason in {
        "no_candidate", "low_floor", "tight_margin", "budget_exceeded",
    }, f"漏斗的灰区原因在模型分支上被丢掉了：{decision.gray_reason!r}"
    assert decision.to_dict()["gray_reason"] == decision.gray_reason


def test_injecting_a_model_bypasses_the_local_fast_path():
    """注入 ``model`` 的语义是"这次路由交给它判"，快通道不得抢答。

    测试正是靠注入假模型来隔离外部依赖。若快通道在注入时也插一脚，
    注入的模型就永远轮不到——那不是隔离，是把被测行为掩蔽掉：
    上面那些守"模型判成什么样"的用例会在无声中变成空转。
    """
    stub = _TextModel(
        json.dumps({"route": SCENE_TOOL, "reason": "测试", "confidence": 0.9})
    )

    decision = route_query("你好", model=stub)

    assert decision.scene == SCENE_TOOL
    assert decision.source == "router"
    assert stub.calls


def test_local_fast_path_is_off_when_there_is_no_real_model(monkeypatch):
    """离线（无 Key）走的是 Mock 模型 + 确定性兜底，那是一条刻意设计的降级路径。

    快通道只挂在真实链路上：本次改动只为降延迟，不该顺手改离线行为。
    """
    from app.core import router_agent

    monkeypatch.setattr(config, "USE_REAL_LLM", False)

    decision = route_query("你好")

    assert decision.scene == SCENE_SMALLTALK, "离线仍应由确定性兜底认下寒暄"
    assert decision.source == "router:fallback"
    assert decision.source != router_agent.SOURCE_LOCAL, "快通道不得挂在离线链路上"


# ===========================================================================
# 二、场景（该谁干）与出口（答案怎么来的）是两个问题
# ===========================================================================
@pytest.mark.parametrize(
    "scene, node_name, expected_intent",
    [
        (SCENE_SMALLTALK, "smalltalk", "direct"),
        (SCENE_OUT_OF_SCOPE, "out_of_scope", "direct"),
        (SCENE_SIMPLE_RAG, "simple_rag", "knowledge"),
        (SCENE_COMPLEX_RAG, "complex_rag", "knowledge"),
        (SCENE_TOOL, "tool", "tool"),
    ],
)
def test_scene_and_intent_type_answer_different_questions(monkeypatch, scene, node_name, expected_intent):
    """``scene`` 由入口判定，``intent_type`` 由"实际发生了什么"决定。

    合并成一个字段会得到"路由要预判执行结果"的悖论。最清楚的证据是工具 Agent：
    它取到证据交 L4 时是 ``scene=tool / intent=tool``，但**同一个 Agent**
    在参数不全而反问用户时仍然是 ``scene=tool``，``intent`` 却是 ``direct``。
    """
    from app.graph import nodes
    from app.core import sub_agents

    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs",
                        lambda *a, **kw: [{"source": "s", "content": "c", "fused": 0.01}])
    monkeypatch.setattr(nodes, "route_query",
                        lambda **kw: RouteDecision(scene=scene, reason="测试", confidence=1.0))

    state = create_initial_state(user_query="随便问一句", session_id="s")
    if scene == SCENE_TOOL:
        monkeypatch.setattr(nodes, "run_tool_agent", lambda **kw: _no_tool_decision())

    out = getattr(nodes, f"{node_name}_node")(state)

    assert out["intent_type"] == expected_intent
    assert out["intent_source"] == scene, "intent_source 要回答「这是哪个 Agent 产出的」"


def _no_tool_decision():
    from app.core.tool_agent import ToolDecision

    return ToolDecision()


def test_tool_agent_switches_intent_type_within_the_same_scene(monkeypatch, business_db):
    """同一个 ``scene=tool`` 下，``intent_type`` 会随实际结果变化——这就是两字段的理由。"""
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    state = create_initial_state(user_query="我的年假还剩几天", user_id="u1", session_id="s")

    # ① 反问用户：拿不到证据 → 直答出口
    monkeypatch.setattr(agent_mod, "default_model",
                        lambda: RecordingModel(["请问你叫什么名字？我帮你查。", []]))
    direct = nodes.tool_node(state)
    assert direct["intent_type"] == "direct"

    # ② 取到证据：交 L4 → 证据出口
    monkeypatch.setattr(agent_mod, "default_model",
                        lambda: RecordingModel([
                            {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
                            [],
                        ]))
    evidence = nodes.tool_node(state)
    assert evidence["intent_type"] == "tool"


# ===========================================================================
# 三、边界管控集中在入口层（幻觉管控的唯一来源）
# ===========================================================================
def test_boundary_rule_lives_in_exactly_one_prompt():
    """越界规则只允许出现在 ``router`` 提示词里。

    分散写法的失效方式是**静默的**：五份判断互相漂移，某次只改了四份，
    第五份就悄悄放行了本该拦掉的问题，而没有任何一处能看出这件事。
    """
    owners = [name for name, text in PROMPTS.items() if SCENE_OUT_OF_SCOPE in text]
    assert owners == ["router"], f"越界规则出现在了多份提示词里：{owners}"


def test_out_of_scope_answer_is_defined_once():
    """越界话术必须**只有一个产生点**（常量），否则它就不再是可审计的。

    按**首句**而不是整个常量去扫源码：常量在源码里是多行相邻字面量拼接的，
    整段文本并不以单个字面量的形态存在（``"a" "b"`` 拼接发生在编译期）。
    首句是一个完整的单行字面量，足以定位"话术被复制到了哪里"。

    断言的是 ``hits == [_OUT_OF_SCOPE_OWNER]`` 而不是 ``len(hits) == 1``：
    **"恰好一个"与"恰好是那一个"是两件事**。前者在话术被整体搬到别的模块时
    会照样通过，而那时"谁是对外契约的持有者"这件事就已经悄悄变了。
    """
    first_line = OUT_OF_SCOPE_ANSWER.splitlines()[0]
    assert len(first_line) > 10, "首句太短会误命中无关代码"

    hits = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in (_REPO_ROOT / "app").rglob("*.py")
        if first_line in path.read_text(encoding="utf-8")
    ]
    assert hits == [_OUT_OF_SCOPE_OWNER], f"越界话术被复制到了多处：{hits}"


def test_out_of_scope_node_writes_the_constant_and_ends_the_turn():
    """越界回复是一个常量，不经过任何模型——只被判别、不被生成的回答不可能有幻觉。"""
    from app.graph import nodes

    state = create_initial_state(user_query="明天天气怎么样", session_id="s")
    out = nodes.out_of_scope_node(state)

    assert out["answer"] == OUT_OF_SCOPE_ANSWER
    assert out["intent_type"] == "direct"
    assert out["intent_source"] == SCENE_OUT_OF_SCOPE
    assert out["retrieve_docs"] == [], "越界问题不该消耗检索预算"
    assert out["tool_result"] is None
    # 本轮**已经**给出了完整明确的回答，它不是"无法自动处理"。
    # 标成 True 会让前端显示"已转接人工"——那是假话。
    assert out["need_human"] is False
    assert out["refused"] is False


def test_out_of_scope_is_intercepted_before_any_sub_agent_runs(monkeypatch):
    """端到端：越界问题在**花掉检索 / 工具 / L4 预算之前**就被拦下。"""
    from app.core import sub_agents
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    def boom(*_a, **_kw):
        raise AssertionError("越界问题不该进入任何子 Agent")

    monkeypatch.setattr(nodes, "route_query",
                        lambda **kw: RouteDecision(scene=SCENE_OUT_OF_SCOPE, reason="越界",
                                                   confidence=1.0,
                                                   out_of_scope_answer=OUT_OF_SCOPE_ANSWER))
    monkeypatch.setattr(nodes, "run_simple_rag_agent", boom)
    monkeypatch.setattr(nodes, "run_complex_rag_agent", boom)
    monkeypatch.setattr(nodes, "run_smalltalk_agent", boom)
    monkeypatch.setattr(nodes, "run_tool_agent", boom)
    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", boom)
    monkeypatch.setattr(agent_mod, "default_model", boom)

    out = enterprise_workflow.invoke(
        create_initial_state(user_query="明天天气怎么样", user_id="u1", session_id="s")
    )

    assert out["answer"] == OUT_OF_SCOPE_ANSWER
    assert out["intent_source"] == SCENE_OUT_OF_SCOPE
    assert "generate_answer" not in _trace_nodes(out), "越界问题不该进 L4"
    assert out["need_human"] is False


# ===========================================================================
# 四、闲聊 Agent：不检索、不调工具、不调模型
# ===========================================================================
def test_smalltalk_never_touches_retrieval_or_tools(monkeypatch):
    """闲聊 Agent 的全部依赖只有一张模板表——任何外部调用都是不该发生的。

    这是「避免闲聊误触发 RAG 或 FunctionCall 而浪费 token」的结构性守卫：
    不是"我们不去调"，而是"把检索和工具都换成会抛异常的桩，链路照样走完"。
    """
    from app.core import sub_agents
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    def boom(*_a, **_kw):
        raise AssertionError("闲聊不该触发检索 / 工具 / 模型")

    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", boom)
    monkeypatch.setattr(agent_mod, "run_tool_agent", boom)
    monkeypatch.setattr(agent_mod, "default_model", boom)
    monkeypatch.setattr("app.providers.llm.get_chat_model", boom)

    out = nodes.smalltalk_node(create_initial_state(user_query="你好", session_id="s"))

    assert out["answer"], "闲聊必须给出回答"
    assert out["intent_type"] == "direct"
    assert out["intent_source"] == SCENE_SMALLTALK
    assert out["soft_warnings"] == [], "闲聊没有可降级的东西"
    assert out["retrieve_docs"] == []


def test_smalltalk_prompt_is_opt_in_and_covers_only_safe_templates():
    """闲聊的模型路径**必须是可选的**，且只覆盖纯客套三类。

    ⚠️ 这条以前写的是 ``assert SCENE_SMALLTALK not in PROMPTS``——把「闲聊没有
    提示词」当作"不调模型"的结构性证据。需求变化后（允许在低风险场景下行一次
    轻量模型调用，见 ``config.SMALLTALK_LLM_ENABLED``）那个断言不再成立，
    但它想守的**性质**仍然成立。所以这里改成直接守性质，而不是把断言删掉——
    删掉就等于"这个仓库不再有人在看闲聊会不会偷偷调模型"。

    守三条：
      1. 提示词真实存在（开了可选路径却查无提示词，说明配置与实现脱节）；
      2. 允许走模型的模板**恰好**是问候 / 致谢 / 道别——纯客套，不含业务事实；
      3. 身份询问与兜底话术**不在其中**（见下一条测试的证明）。
    """
    from app.core.sub_agents import _SMALLTALK_LLM_TEMPLATES

    assert "smalltalk" in PROMPTS, "开启了可选模型路径，提示词就必须真实存在"
    assert _SMALLTALK_LLM_TEMPLATES == {"greeting", "thanks", "bye"}
    assert "identity" not in _SMALLTALK_LLM_TEMPLATES
    assert "default" not in _SMALLTALK_LLM_TEMPLATES


class _NamedModel:
    """返回固定文本的假模型：用来识别"这次回答到底是不是模型写的"。"""

    def __init__(self, text: str):
        self.text = text

    def invoke(self, _prompt):
        return AIMessage(content=self.text)


def test_smalltalk_identity_never_goes_to_the_model(monkeypatch):
    """即使开关打开，「你是谁」也**不**走模型——那句话是能力清单，不是措辞。

    用"模型会吐一个可识别的句子"来判，而不是断 ``steps[0]["kind"]``：
    ``_llm_smalltalk`` 失败时会回落模板，于是**任何**基于"有没有抛异常 / kind 是
    什么"的断言都会在"真的误调了模型"时照样通过——那是一条没有牙齿的护栏。

    ⚠️ 反向验证：把 ``identity`` 加进 ``_SMALLTALK_LLM_TEMPLATES``，本条必须变红。
    """
    from app.core import sub_agents

    monkeypatch.setattr(config, "SMALLTALK_LLM_ENABLED", True)
    monkeypatch.setattr(config, "USE_REAL_LLM", True)
    monkeypatch.setattr(sub_agents, "default_model", lambda: _NamedModel("模型编的回复"))

    answer = sub_agents.run_smalltalk_agent("你是谁")

    assert "模型编的回复" not in answer.text, "身份询问被送去模型改写了"
    assert "企业内部智能助手" in answer.text
    assert answer.steps[0]["kind"] == "template"


def test_smalltalk_low_risk_templates_do_use_the_model_when_enabled(monkeypatch):
    """开关打开后，纯客套三类**确实**走了模型——否则前一条只是"开关没接上"。

    两条测试必须成对：只证明"身份不走模型"，一个把开关整条拆掉的实现也能通过；
    只证明"问候走了模型"，一个把身份也一起送进去的实现同样能通过。
    """
    from app.core import sub_agents

    monkeypatch.setattr(config, "SMALLTALK_LLM_ENABLED", True)
    monkeypatch.setattr(config, "USE_REAL_LLM", True)
    monkeypatch.setattr(sub_agents, "default_model", lambda: _NamedModel("模型写的问候"))

    answer = sub_agents.run_smalltalk_agent("你好")

    assert answer.text == "模型写的问候"
    assert answer.steps[0]["kind"] == "llm"


def test_smalltalk_falls_back_to_template_when_the_model_is_unusable(monkeypatch):
    """模型不可用 / 输出不合法 → 回落模板。闲聊**永远**答得出来。

    这是开关能被接受的前提：模板直出永远可用，模型只是让措辞不那么固定；
    把一次可选增益接成必需品，等于给闲聊路径引入一个它本来没有的失败点。
    """
    from app.core import sub_agents

    monkeypatch.setattr(config, "SMALLTALK_LLM_ENABLED", True)
    monkeypatch.setattr(config, "USE_REAL_LLM", True)

    def boom():
        raise RuntimeError("模型不可达")

    monkeypatch.setattr(sub_agents, "default_model", boom)
    assert "你好" in sub_agents.run_smalltalk_agent("你好").text

    # 输出超长（模型开始"发挥"）同样回落——长度是"跑出既定定位"的代用信号。
    monkeypatch.setattr(sub_agents, "default_model", lambda: _NamedModel("啰" * 200))
    assert "你好" in sub_agents.run_smalltalk_agent("你好").text


def test_smalltalk_does_not_call_the_model_even_offline(monkeypatch):
    """离线（Mock 模型）时闲聊更不能走模型。

    Mock 模型对任何输入都回一段与 Context 有关的文本，其中包含
    「未在知识库中检索到…」。若闲聊走模型，用户会看到「你好」被回以
    「未在知识库中检索到与您问题相关的内容」——荒唐且难以排查。
    """
    from app.graph import nodes

    monkeypatch.setattr(config, "USE_REAL_LLM", False)
    out = nodes.smalltalk_node(create_initial_state(user_query="你好", session_id="s"))

    assert "未在知识库" not in out["answer"]
    assert "知识库" not in out["answer"]


@pytest.mark.parametrize(
    "query, marker",
    [
        ("你好", "你好"),
        ("谢谢你", "不客气"),
        ("再见", "再见"),
        ("你是谁", "企业内部智能助手"),
    ],
)
def test_smalltalk_templates_cover_the_four_kinds(query, marker):
    """四类寒暄各有一张模板；模板内容固定 = 不含任何业务事实 = 不可能编造。"""
    from app.core.sub_agents import run_smalltalk_agent

    answer = run_smalltalk_agent(query)
    assert answer.is_direct
    assert marker in answer.text
    assert answer.docs == [] and answer.tool_results == []


def test_smalltalk_leaves_the_graph_without_entering_l4(monkeypatch):
    """闲聊是一条直答出口：进 L4 会被拒答逻辑改写成「知识库中没有找到」。

    「没进 L4」用两件事一起证明：节点轨迹里没有 ``generate_answer``，
    且本轮没有产生任何引用与置信度（那两项只由 L4 写入）。
    """
    from app.graph import nodes

    monkeypatch.setattr(nodes, "route_query",
                        lambda **kw: RouteDecision(scene=SCENE_SMALLTALK, reason="寒暄",
                                                   confidence=1.0))

    out = enterprise_workflow.invoke(
        create_initial_state(user_query="你好", user_id="u1", session_id="s")
    )

    assert out["intent_type"] == "direct"
    assert out["answer"]
    assert _trace_nodes(out) == ["memory_load", "router", "smalltalk"]
    assert out["citations"] == []
    assert out["need_human"] is False


# ===========================================================================
# 五、简单 vs 复杂 RAG：差别只在"取一次"还是"取多次"
# ===========================================================================
@pytest.fixture
def _spy_retrieval(monkeypatch):
    """记录每次检索的查询词，返回一份可去重的假片段。"""
    from app.core import sub_agents

    seen: list = []

    def fake(query, top_k=None, allowed_sources=None):
        seen.append(query)
        return [{"source": f"{query}.txt", "content": f"关于 {query} 的内容", "fused": 0.01}]

    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", fake)
    return seen


def test_simple_rag_retrieves_exactly_once(_spy_retrieval):
    from app.core.sub_agents import run_simple_rag_agent

    answer = run_simple_rag_agent("年假有多少天")

    assert _spy_retrieval == ["年假有多少天"]
    assert len(answer.docs) == 1
    assert answer.sub_queries == [], "简单 RAG 不拆解"


def test_complex_rag_always_searches_the_original_query(monkeypatch, _spy_retrieval):
    """原问题**必须**参与检索——这是"拆歪了还不自知"的防线。

    只按子问题检索时，若模型把它拆偏，最相关的片段会整段漏掉，而 L4 拿着
    一堆"相关但不对"的片段照样能自信地生成答案：这是静默错误。
    """
    from app.core import sub_agents

    monkeypatch.setattr(sub_agents, "default_model",
                        lambda: _TextModel('["年假的天数规定", "调休的天数规定"]'))
    answer = sub_agents.run_complex_rag_agent("对比年假和调休的区别")

    assert _spy_retrieval[0] == "对比年假和调休的区别", "原问题必须排在检索集合第一位"
    assert set(_spy_retrieval) == {"对比年假和调休的区别", "年假的天数规定", "调休的天数规定"}
    assert answer.sub_queries == _spy_retrieval


def test_complex_rag_dedupes_and_keeps_the_higher_score(monkeypatch):
    """同一片段被多个子问题召回 → 只留一条，且取较高的融合分。

    RRF 融合分跨查询同量纲（都是 ``1/(k+rank)`` 的累加），可以直接比大小；
    向量 ``score`` 跨查询不可比，故不参与择优。
    """
    from app.core import sub_agents

    monkeypatch.setattr(sub_agents, "default_model",
                        lambda: _TextModel('["A", "B"]'))

    def fake(query, top_k=None, allowed_sources=None):
        same = {"source": "hr.txt", "content": "同一段内容", "fused": 0.01 if query == "A" else 0.03}
        return [same, {"source": f"{query}.txt", "content": query, "fused": 0.005}]

    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", fake)
    answer = sub_agents.run_complex_rag_agent("原问题")

    deduped = [d for d in answer.docs if d["content"] == "同一段内容"]
    assert len(deduped) == 1, "同一 (source, content) 只能出现一次"
    assert deduped[0]["fused"] == 0.03
    assert answer.docs[0]["content"] == "同一段内容", "合并后按融合分降序"


@pytest.mark.parametrize(
    "raw",
    ["", "我拆不出来", "不是数组", '["只有一个"]', '{"a": 1}', "```\n[]\n```"],
)
def test_decompose_falls_back_to_the_original_query(raw):
    """拆解失败一律退化为「就用原问题检索一次」——拆解是增益，不是本轮的失败点。"""
    from app.core.sub_agents import decompose_query

    assert decompose_query("对比年假和调休的区别", model=_TextModel(raw)) == ["对比年假和调休的区别"]


def test_decompose_respects_the_subquery_cap(monkeypatch):
    """子问题数量受配置约束——它直接决定检索次数（= 延迟与成本）。"""
    from app.core import sub_agents

    monkeypatch.setattr(config, "COMPLEX_RAG_MAX_SUBQUERIES", 2)
    raw = json.dumps(["一", "二", "三", "四", "五"])

    assert sub_agents.decompose_query("问题", model=_TextModel(raw)) == ["一", "二"]


def test_complex_rag_caps_merged_docs(monkeypatch):
    """合并后条数受配置约束——它决定 L4 上下文的长度。"""
    from app.core import sub_agents

    monkeypatch.setattr(sub_agents, "default_model", lambda: _TextModel('["A", "B"]'))
    monkeypatch.setattr(
        sub_agents, "retrieve_knowledge_docs",
        lambda q, top_k=None, allowed_sources=None: [
            {"source": f"{q}-{i}", "content": f"{q}-{i}", "fused": 0.01} for i in range(5)
        ],
    )

    answer = sub_agents.run_complex_rag_agent("原问题", max_docs=3)
    assert len(answer.docs) == 3


def test_single_subquery_failure_is_a_soft_warning_not_a_failure(monkeypatch):
    """个别子查询失败 → 跳过并留痕；其余子查询仍然给出答案。"""
    from app.core import sub_agents

    monkeypatch.setattr(sub_agents, "default_model", lambda: _TextModel('["A", "B"]'))

    def fake(query, top_k=None, allowed_sources=None):
        if query == "A":
            raise RuntimeError("向量库抖动")
        return [{"source": f"{query}.txt", "content": query, "fused": 0.01}]

    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", fake)
    answer = sub_agents.run_complex_rag_agent("原问题")

    assert answer.docs, "剩下那一路仍然要有证据"
    assert answer.soft_warnings
    assert answer.degraded is False, "还有来源能答，不该记成整体降级"


def test_all_subqueries_failing_is_degraded_not_fatal(monkeypatch):
    """全部子查询失败 → 与简单 RAG 同款降级：交 L4 诚实作答，**不抛**。"""
    from app.core import sub_agents

    monkeypatch.setattr(sub_agents, "default_model", lambda: _TextModel('["A", "B"]'))

    def boom(*_a, **_kw):
        raise RuntimeError("向量库连接超时")

    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", boom)
    answer = sub_agents.run_complex_rag_agent("原问题")

    assert answer.docs == []
    assert answer.degraded is True
    assert answer.error is None, "可降级故障不是 error（error 会转人工）"
    assert answer.soft_warnings


# ===========================================================================
# 六、工具 Agent：追问出口、以及"降级决策落在边上"
# ===========================================================================
def test_tool_agent_asks_back_instead_of_fabricating(monkeypatch, business_db):
    """必填参数缺失 → 模型向用户追问，走**直答出口**，不进 L4。

    进 L4 的话，这句"请提供姓名或工号"会被拒答逻辑改写成「知识库中没有找到
    相关信息」——用户看到的是系统答不了，而他实际只缺一个称呼。
    """
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    monkeypatch.setattr(agent_mod, "default_model",
                        lambda: RecordingModel(["请把你的姓名或工号告诉我，我帮你查假期余额。", []]))
    state = create_initial_state(user_query="我的年假还剩几天", user_id="u1", session_id="s")

    out = nodes.tool_node(state)

    assert out["intent_type"] == "direct"
    assert "姓名" in out["answer"]
    assert not out.get("need_human"), "缺参数是正常交互，不是「无法自动处理」"
    assert out["retrieve_docs"] == []


def test_ask_back_survives_a_rejected_call(monkeypatch, business_db):
    """**提出过调用但被护栏拒绝**，模型随后改口追问 —— 这句话不能被丢掉。

    判据是「本轮一次都没**成功**取到证据」（``used_tools`` 为空），
    而不是「一次工具都没提过」（``attempted``）。用后者的话，下面这条链路
    会退化成"零证据 → L4 拒答"，而它恰恰是规格第 3 条要求的行为。
    """
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    monkeypatch.setattr(agent_mod, "default_model", lambda: RecordingModel([
        {"name": "query_leave_balance", "args": {}},          # 缺 employee_id → 被护栏拒绝
        "请把你的工号告诉我，我帮你查假期余额。",              # ← 改口追问
        [],
    ]))
    state = create_initial_state(user_query="我的年假还剩几天", user_id="u1", session_id="s")

    out = nodes.tool_node(state)

    assert out["intent_type"] == "direct", "追问被丢掉的话会走 L4，用户拿到的是拒答"
    assert "工号" in out["answer"]
    assert out["soft_warnings"], "护栏拒绝过就要留痕（模型侧的格式稳定性信号）"


def test_closing_remark_is_discarded_once_evidence_exists(monkeypatch, business_db):
    """对照：**取到证据之后**，模型自己的收口话必须被丢掉，交 L4 重新成文。

    引用编号、置信度、拒答三项只在 L4 一处产生；让模型在工具循环里自由成文，
    等于把这四项能力拆成两处，两处必然漂移，且漂移是静默的（答案都很流利）。
    """
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    monkeypatch.setattr(agent_mod, "default_model", lambda: RecordingModel([
        {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
        "你的年假还有 5 天。",                                # ← 这句必须被丢掉
        [],
    ]))
    state = create_initial_state(user_query="我的年假还剩几天", user_id="u1", session_id="s")

    out = nodes.tool_node(state)

    assert out["intent_type"] == "tool"
    assert out["answer"] is None, "有证据时必须由 L4 成文"
    assert "annual_leave" in out["tool_result"]


def test_tool_generation_without_function_calling_is_a_soft_landing(monkeypatch):
    """模型不支持 function calling → 节点只如实报告 ``tool_degraded``，**不自己选兜底路径**。"""
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    monkeypatch.setattr(agent_mod, "default_model",
                        lambda: RecordingModel([[]], bind_error=RuntimeError("端点不支持 tools")))
    out = nodes.tool_node(create_initial_state(user_query="年假多少天", user_id="u1", session_id="s"))

    assert out["tool_degraded"] is True
    assert out["answer"] is None, "节点不得替下游预判结果（那会让拓扑消失）"
    assert out["intent_type"] is None
    assert out["soft_warnings"], "降级必须留痕"


def test_degraded_rerouting_is_visible_in_the_topology():
    """改道决策必须能**从拓扑上读出来**——落在边上而不是藏进节点里。"""
    assert tool_route_edge({"tool_degraded": True}) == SCENE_SIMPLE_RAG
    assert SCENE_SIMPLE_RAG in _route_edges("tool"), "tool 节点必须有一条通向 simple_rag 的边"
    # 前置图（流式）同样保留这条边：降级不因链路不同而改变行为
    assert SCENE_SIMPLE_RAG in _route_edges("tool", pre_generation_workflow)


def test_tool_route_edge_priority_order():
    """四分支的优先级：人工兜底 > 直答 > 改道 > **交给证据校验复核**。"""
    assert tool_route_edge({"need_human": True, "answer": "x", "tool_degraded": True}) == "human_fallback"
    assert tool_route_edge({"answer": "追问", "tool_degraded": True}) == "end"
    assert tool_route_edge({"tool_degraded": True}) == SCENE_SIMPLE_RAG
    assert tool_route_edge({}) == "verifier"
    # 空字符串 ≠ 没有答案：模型说了一句话但内容是空的，那是一次需要被下游
    # 按无依据处理的异常，不是一条答案（用真值判断会把两者混成一种）。
    assert tool_route_edge({"answer": ""}) == "end"


def test_business_tool_failure_still_reports_human(monkeypatch, business_db):
    """业务工具**执行失败**转人工——判据是「能不能从其他来源得到答案」。"""
    from app.core import tool_agent as agent_mod
    from app.graph import nodes

    class _BoomTool:
        name = "query_leave_balance"

        @staticmethod
        def invoke(_args):
            raise RuntimeError("业务库 502")

    monkeypatch.setitem(agent_mod._TOOLS_BY_NAME, "query_leave_balance", _BoomTool())
    monkeypatch.setattr(agent_mod, "default_model", lambda: RecordingModel([
        {"name": "query_leave_balance", "args": {"employee_id": "E1001"}},
        [],
    ]))
    state = create_initial_state(user_query="我的年假还剩几天", user_id="u1", session_id="s")

    out = nodes.tool_node(state)

    assert out["need_human"] is True
    assert tool_route_edge(out) == "human_fallback"


# ===========================================================================
# 七、拓扑不变量：场景名即节点名、两个编译产物同源
# ===========================================================================
def test_every_scene_is_a_registered_node():
    """``scene`` 的取值必须与节点名一一对应。

    这条约束让"五路分发"与"场景闭集"不可能漂移：``scene_route_edge`` 返回的
    值直接就是节点名，新增/改名场景时漏改映射表会被这里拦住。
    """
    assert set(SCENES) <= set(NODE_NAMES)


def test_router_branch_map_covers_the_whole_scene_set():
    """router 的出边必须**恰好**是那五个场景，不多不少。"""
    assert _route_edges("router") == set(SCENES)


def test_registered_nodes_match_the_declared_node_list():
    """``NODE_NAMES`` 是前端面板与拓扑校验的唯一来源，必须与图实际一致。"""
    actual = {n for n in enterprise_workflow.get_graph().nodes if not n.startswith("__")}
    assert actual == set(NODE_NAMES)


def test_two_direct_exits_end_the_graph_without_generation():
    """两个直答出口（闲聊 / 越界）必须直接结束，不经过 L4。"""
    assert _route_edges("smalltalk") == {"__end__"}
    assert _route_edges("out_of_scope") == {"__end__"}


def test_full_and_pre_generation_graphs_share_every_node():
    """流式链路的前置图必须与完整图**同源**（只少一个生成节点）。

    历史缺陷：旧实现在流式端点里重抄了整条前置链路，结果工具抛异常时图会把
    答案换成「已转接人工」，流式链路却照样去调模型生成——同一次提问，
    两条链路给出不一致的回答。
    """
    full = {n for n in enterprise_workflow.get_graph().nodes if not n.startswith("__")}
    pre = {n for n in pre_generation_workflow.get_graph().nodes if not n.startswith("__")}

    assert pre <= full
    assert full - pre == {"generate_answer"}


def test_pre_generation_graph_reroutes_the_evidence_exit_to_end():
    """    两条链路唯一允许的差别：**通过校验之后**通向 ``END`` 而不是
    ``generate_answer``。``generation_target`` 因此只有**一个**消费方 ——
    ``verifier_route_edge``。工具链路不直接消费它（它先被复核），这一点在
    2026-09-24 被改错过一次，见 `test_the_tool_path_must_pass_the_verifier`。

    注意三个证据出口（``simple_rag`` / ``complex_rag`` / ``tool``）两条链路
    **都**通向 ``verifier``：算出来的证据在流式链路上同样可能答非所问，
    校验不该只在非流式链路上存在。差别被推迟到了校验的出口。
    """
    assert _route_edges("simple_rag") == {"verifier"}
    assert _route_edges("complex_rag") == {"verifier"}
    assert _route_edges("simple_rag", pre_generation_workflow) == {"verifier"}
    assert _route_edges("complex_rag", pre_generation_workflow) == {"verifier"}
    assert _route_edges("verifier") == {"generate_answer", "router"}
    assert _route_edges("verifier", pre_generation_workflow) == {"__end__", "router"}
    # 工具链路：两个图都先经过 verifier（差别同样落在校验的出口）。
    assert _route_edges("tool") == {
        "__end__", "human_fallback", "simple_rag", "verifier"
    }
    assert _route_edges("tool", pre_generation_workflow) == {
        "__end__", "human_fallback", "simple_rag", "verifier"
    }


def test_verifier_back_edge_is_bounded_by_the_retry_budget():
    """``verifier → router`` 是图里**唯一**的回边，且必须有上界。

    没有上界就是死循环：LangGraph 撞上递归上限会以异常收场，那不是"降级"，
    是整轮失败。所以这里同时钉住三件事——回边存在（否则路由误判无人纠正）、
    不判不符时**不**回退（否则每一轮都白绕一圈）、以及超预算后必须收敛到生成。
    """
    from app.graph.edges import verifier_route_edge

    # 1. 回边存在
    assert "router" in _route_edges("verifier")
    # 2. 没判"不符"就不改道：None（没校验）与 True（校验通过）都直接生成
    assert verifier_route_edge({}) == "generate_answer"
    assert verifier_route_edge({"evidence_aligned": True}) == "generate_answer"
    # 3. 判"不符"且还有预算 → 退回；预算用尽 → 收敛到生成（正常路径，非异常）
    budget = config.ROUTE_RETRY_BUDGET
    assert verifier_route_edge({"evidence_aligned": False, "reroute_count": 1}) == "router"
    assert (
        verifier_route_edge({"evidence_aligned": False, "reroute_count": budget + 1})
        == "generate_answer"
    )


def test_reroute_budget_zero_means_verify_only(monkeypatch):
    """``ROUTE_RETRY_BUDGET=0`` 时只校验、不改道——校验结论仅进观测。

    这条守的是"关闭反馈通道"这个配置语义：关掉它不该顺带把校验也关掉
    （那是两件事：``VERIFIER_MODE`` 管校验做不做，预算管发现不符后改不改道）。
    """
    from app.graph.edges import verifier_route_edge

    assert verifier_route_edge({"evidence_aligned": False, "reroute_count": 1}) == "router"

    monkeypatch.setattr(config, "ROUTE_RETRY_BUDGET", 0)
    assert (
        verifier_route_edge({"evidence_aligned": False, "reroute_count": 1})
        == "generate_answer"
    )


# ===========================================================================
# 证据校验的触发判据 —— 它是**修复手段**，不是每轮都走的常规工序
#
# 这一组守的主张是"没理由怀疑时不许花钱"。它的失效是**不会报错**的那一类：
# 判据写漏了，校验照常返回"对齐"，功能测试全绿，只是每一轮都白付 1~3 秒。
# 所以下面的断言全部看 ``source``（有没有真的调用模型），而不是只看结论 ——
# 只看结论的话，"误调了模型、模型恰好答对齐"与"正确地跳过"完全同形。
# ===========================================================================
def _clean_docs():
    """一次正常检索的产物：没有任何"这批证据是凑的"的标记。"""
    return [{"content": "年休假为 5 天", "source": "员工手册.md", "fallback": False}]


def test_verifier_auto_skips_a_healthy_turn_without_calling_the_model(monkeypatch):
    """默认（auto）下，规则路由 + 正常检索 → **一次模型调用都不发**。

    这就是本机制存在的理由：线上 20 次采样里校验 0 次判出不符，却稳定占掉
    首 token 路径上 1.2~3.0 秒。

    ⚠️ 反向验证：把 ``should_verify`` 的最后一条判据改回
    "``route_gray_reason`` 非空即触发"，本条必须变红。
    """
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "auto")
    result = verifier.verify_evidence(
        "年假有多少天",
        docs=_clean_docs(),
        route_gray_reason="",
        model=_NamedModel("ALIGNED\n"),
    )

    assert result.source == verifier.SOURCE_SKIPPED, "健康的一轮不该调用校验模型"
    assert result.trigger == verifier.TRIGGER_NOT_NEEDED
    assert result.aligned is None


def test_verifier_only_treats_a_wobbling_route_as_suspicion(monkeypatch):
    """漏斗"没判"不等于"判不准"：只有 ``tight_margin`` 触发复核。

    这是本机制省不省钱的分水岭。漏斗的灰区原因分四类，只有 ``tight_margin``
    （顶两名咬得很紧）说明**这次判定本身不稳**；``low_floor`` / ``no_candidate``
    只说明这句话与例句不像——模型路由读能力描述本来就擅长这类，不该为它多花
    一次 1.2~3.0 秒的校验。

    实测（27 条正常提问）：漏斗只有 4 条被本地采信，其余 23 条里 ``low_floor``
    22 条、``no_candidate`` 1 条、``tight_margin`` **0 条**。若按"只要不是本地
    快通道拍的板就复核"，会有 85% 的正常轮次被拉回来，等于没做按需。

    ⚠️ 反向验证：把 ``should_verify`` 的最后一条判据改成
    ``if route_gray_reason:``，本条必须变红（前两个参数化用例会去调模型）。
    """
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "auto")
    model = _NamedModel("ALIGNED\n")

    for gray in ("low_floor", "no_candidate", "budget_exceeded"):
        skipped = verifier.verify_evidence(
            "年假有多少天", docs=_clean_docs(), route_gray_reason=gray, model=model
        )
        assert skipped.source == verifier.SOURCE_SKIPPED, f"{gray} 不该触发复核"
        assert skipped.aligned is None

    fired = verifier.verify_evidence(
        "年假有多少天",
        docs=_clean_docs(),
        route_gray_reason="tight_margin",
        model=model,
    )
    assert fired.trigger == TRIGGER_TIGHT_ROUTE
    assert fired.source == verifier.SOURCE_MODEL, "摇摆的一轮没有真的查"


@pytest.mark.parametrize(
    ("overrides", "trigger"),
    [
        ({"soft_warnings": ["检索失败已降级：TimeoutError"]}, TRIGGER_DEGRADED),
        ({"docs": [{"content": "x", "source": "a.md", "fallback": True}]}, TRIGGER_WEAK_EVIDENCE),
        ({"route_gray_reason": "tight_margin"}, TRIGGER_TIGHT_ROUTE),
    ],
)
def test_verifier_auto_fires_on_every_kind_of_suspicion(monkeypatch, overrides, trigger):
    """三条"值得查"的迹象各自都能**单独**触发，且都会真的调用模型。

    每一条都对应一种"这一轮确实可能有问题"的实证，而不是猜测：本轮发生了可降级
    故障（链路自己已经报了"质量已下降"——含工具不可用**而改道简单 RAG** 那一轮，
    那一次的证据是检索来的、带着退而求其次的背景）；证据含软回退片段（检索自己
    说了这批是硬凑的）；路由在通道间咬得很紧（判定本身就不稳）。

    少任何一条，对应的那类问题就会静默漏过去 —— 而漏过去的表现是"回答照常
    产出、只是错的"，不会有任何报错。

    注意这里**没有**"工具链路"这一条：它不是一条"迹象"，而是**代理信号**
    （判据和风险不是同一件事），已删除。从今往后工具轮走的是上面这几条同一套判据
    —— 干净的轮次零代价跳过，出过状况的轮次被复核。见
    `test_the_tool_path_must_pass_the_verifier`。
    """
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "auto")
    payload = {"docs": _clean_docs(), "route_gray_reason": "", **overrides}
    result = verifier.verify_evidence(
        "年假有多少天", model=_NamedModel("ALIGNED\n"), **payload
    )

    assert result.trigger == trigger
    assert result.source == verifier.SOURCE_MODEL, "该查的一轮没有真的查"
    assert result.aligned is True


def test_the_tool_path_must_pass_the_verifier():
    """**工具链路必须经过 verifier**，且 ``tool_result`` 必须传得进去。

    这条边在 2026-09-24 走过三段，整套钉在这里 —— 它是个**方向性错误**的标本：

    ① ``if tool_result:``（"走了工具链路就复核"）是**代理信号**：工具结果是自己
       库里的一行结构化记录，不存在让检索偏掉的机制（片段被切碎 / Top1 分不达标 /
       多路融合选错）。代价是每轮工具请求白付一次复核 —— ``verifier_node`` 中位
       **1551ms**、占整轮 17%~25%，而 54 次判定 MISMATCH = 0。
    ② 把判据删掉**顺手把这条边也摘了**——过度治疗。删判据解决的是"每轮都付"，
       边解决的是"**出问题时有没有人复核**"，两件事被一起丢了：工具侧真正可疑的
       信号（调用被护栏拒绝 / 执行抛异常 / 正文回捞）照旧进 ``soft_warnings``
       + trace + 前端面板，却**再也没有复核环节**。*留痕不等于兜底*。
    ③ 现在：边接回来，判据保持按需（那 1551ms 不会重现，干净轮次零代价跳过）。

    所以断言分两层，**缺一条这条边就是白接的**：

    - **拓扑层**：``tool`` 的路由目标里有 ``verifier``（两个图都要有 —— 流式链路
      取回的证据同样会被推给用户）；
    - **接口层**：``verify_evidence`` / ``should_verify`` **必须收** ``tool_result``。
      工具链路的 ``docs`` 恒为空，证据**全在** ``tool_result`` 里；不传，判据会在
      "没有证据可校验"那一条上短路，复核永远不会发生 —— 而症状与"干净轮次跳过"
      **完全同形**，不会有任何报错。

    ⚠️ 反向验证：把 ``tool_route_edge`` 的返回值改回 ``"generate_answer"``，
    第一条断言必须变红；把 ``verify_evidence`` 的 ``tool_result`` 入参删掉，
    第三条断言必须变红。
    """
    from app.core import verifier

    assert _route_edges("tool") == {
        "__end__", "human_fallback", "simple_rag", "verifier"
    }
    assert "verifier" in _route_edges("tool"), "工具链路脱离了证据校验（兜底没了）"
    assert "verifier" in _route_edges("tool", pre_generation_workflow)

    assert "tool_result" in inspect.signature(verifier.verify_evidence).parameters
    assert "tool_result" in inspect.signature(verifier.should_verify).parameters


def test_a_tool_turn_with_only_tool_result_is_actually_verifiable(monkeypatch):
    """``tool_result`` 单独一人也要能被判为"有证据"。

    这是上一条的**行为侧**护栏：拓扑接回了边、入参也在了，但如果判空只认
    ``retrieve_docs``，工具轮仍会静默跳过复核 —— ``needed=False``、
    ``aligned=None``、``source=verifier:skipped``，全都"看起来正常"。
    所以这里不查形状查行为：只有 ``tool_result``、没有任何 ``docs``，
    必须真的调用模型并给出判定。
    """
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "auto")

    # 干净的证据（无软回退）也没有任何告警 —— 不传 tool_result 时下面会跳过。
    skipped = verifier.should_verify(docs=[], tool_result=None)
    assert skipped.needed is False
    assert skipped.trigger == verifier.TRIGGER_NOT_NEEDED
    assert skipped.detail == "没有证据可校验"

    # 同样的状态，只多了 tool_result（且 soft_warnings 非空 = 本轮出过状况）：
    # 证据存在 + 有可疑迹象 → 必须动手。
    fired = verifier.verify_evidence(
        "李四的部门是什么",
        docs=[],
        tool_result='{"ok": true, "annual_leave_days": 3}',
        soft_warnings=["工具 annual_leave 调用被来源白名单拒绝"],
        model=_NamedModel("MISMATCH\n参数对不上，查的是年假不是部门"),
    )
    assert fired.source == verifier.SOURCE_MODEL, "工具轮没有真的查（边白接了）"
    assert fired.trigger == verifier.TRIGGER_DEGRADED
    assert fired.aligned is False


def test_verifier_off_is_distinguishable_from_auto_skipping(monkeypatch):
    """``off`` 是"这个机制不存在"，与 auto 的"查了、没有可疑迹象"必须能分开。

    合成一个的话，"关掉校验"会伪装成"证据都没问题"——排障时看不出区别，
    也就没人会发现它其实一直是关的。
    """
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "off")
    result = verifier.verify_evidence(
        "年假有多少天",
        docs=_clean_docs(),
        route_gray_reason="tight_margin",
        model=_NamedModel("ALIGNED\n"),
    )

    assert result.trigger == verifier.TRIGGER_DISABLED
    assert result.source == verifier.SOURCE_SKIPPED
    assert result.aligned is None


def test_verifier_always_keeps_the_every_turn_behaviour(monkeypatch):
    """``always`` 是回退路径：``auto`` 万一漏判，改一个值就能切回"每轮都查"。"""
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "always")
    result = verifier.verify_evidence(
        "年假有多少天",
        docs=_clean_docs(),
        model=_NamedModel("ALIGNED\n"),
    )

    assert result.trigger == verifier.TRIGGER_ALWAYS
    assert result.source == verifier.SOURCE_MODEL


def test_verifier_still_skips_when_there_is_nothing_to_check(monkeypatch):
    """没有证据时"是否对齐"这个问题不成立——开成 ``always`` 也变不出证据来。"""
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "always")
    result = verifier.verify_evidence(
        "年假有多少天", docs=[], route_gray_reason="tight_margin",
        model=_NamedModel("MISMATCH\n"),
    )

    assert result.source == verifier.SOURCE_SKIPPED
    assert result.aligned is None


def test_verifier_unknown_mode_falls_back_instead_of_crashing(monkeypatch):
    """``VERIFIER_MODE`` 写错时降级到默认值，而不是把整轮回答拖垮。"""
    from app.core import verifier

    monkeypatch.setattr(config, "VERIFIER_MODE", "banana")
    result = verifier.verify_evidence(
        "年假有多少天",
        docs=_clean_docs(),
        model=_NamedModel("ALIGNED\n"),
    )

    # 与默认值 auto 一致：健康的一轮照常跳过
    assert result.source == verifier.SOURCE_SKIPPED


def test_verifier_node_records_a_skip_as_not_checked(monkeypatch):
    """跳过必须如实写 ``None`` 与「未做校验」，且**不进 soft_warnings**。

    第一件事：写 ``True`` 会让"我们没查"与"查过、没问题"在观测上完全同形 ——
    前端面板把一个大面积跳过的链路显示成"全部对齐"，而实际上这件事本轮没人管。
    节点里那句 ``'对齐' if aligned else '不符'`` 有同一个毛病（``None`` 是假的，
    会被说成"证据不符"，比不写更糟）。

    第二件事：按需跳过是本轮的**正常结论**，不是"该做没做成"的降级。混进
    ``soft_warnings`` 之后，"今天有多少轮没人校验"就再也算不清了。

    ⚠️ 反向验证：把 ``VerifyResult.aligned`` 的默认值改回 ``True``，本条必须变红。
    """
    from app.graph.nodes import verifier_node

    monkeypatch.setattr(config, "VERIFIER_MODE", "auto")
    out = verifier_node(
        {
            "user_query": "年假有多少天",
            "retrieve_docs": _clean_docs(),
            "tool_result": None,
            "scene_source": "router:local",
        }
    )

    assert out["evidence_aligned"] is None
    assert out["verify_source"] == "verifier:skipped"
    assert out["route_decision"]["verify_trigger"] == "not_needed"
    assert "未做校验" in out["trace"][-1]["detail"]
    assert not out.get("soft_warnings")


def test_scene_route_edge_falls_back_for_an_unknown_scene():
    """状态里的场景名不认识时兜底，而不是把 ``KeyError`` 抛给 LangGraph。

    正常链路上 ``route_query`` 已经校验过；但状态也可能由脚本 / 测试 /
    未来的新入口直接构造，那时的异常会被误读成"图配置坏了"。
    """
    assert scene_route_edge({"scene": "weather"}) == SCENE_SIMPLE_RAG
    assert scene_route_edge({}) == SCENE_SIMPLE_RAG
    assert scene_route_edge({"scene": ""}) == SCENE_SIMPLE_RAG


@pytest.mark.parametrize("scene", SCENES)
def test_scene_route_edge_passes_known_scenes_through(scene):
    """已知场景**原样透传**——这里不重新翻译一次路由的判定。"""
    assert scene_route_edge({"scene": scene}) == scene


# ===========================================================================
# 八、API 契约：会话标识（user_id）与场景字段
# ===========================================================================
def test_workflow_execute_passes_user_id_into_the_graph(monkeypatch):
    """``/workflow/execute`` 必须把 ``user_id`` 送进图。

    它现在**只有一个用途**：隔离长期记忆（不同 ``user_id`` 各自一份记忆上下文）。
    服务端没有登录态，``user_id`` 不参与任何鉴权，也没有任何工具会去读它。
    漏传的表现很安静：接口照样 200、答案照样出得来，只是所有人的长期记忆
    会互相串台——所以必须在入口钉住。
    """
    from fastapi.testclient import TestClient

    import app.api.workflow as wf
    from app.main import app

    seen = {}

    class _StubGraph:
        def invoke(self, state):
            seen["user_id"] = state["user_id"]
            return {**state, "trace": []}

    monkeypatch.setattr(wf, "enterprise_workflow", _StubGraph())
    client = TestClient(app)  # 不用 with：避免 lifespan 真实调 LLM
    resp = client.post(
        "/workflow/execute",
        json={"query": "张三在哪个部门", "user_id": "u1"},
    )

    assert resp.status_code == 200
    assert seen["user_id"] == "u1", "user_id 没被送进图，长期记忆会串台"
    # 反向守卫：身份链已整条移除，响应体里不得再出现任何身份字段
    assert "current_employee_id" not in resp.json()


def test_workflow_execute_rejects_illegal_user_id():
    """``user_id`` 的字符约束与 ``ChatRequest`` 一致。

    它要参与长期记忆的文件名 / 键拼接，混进空白或引号会让记忆落到预期之外的位置
    ——症状是"记忆时有时无"而不是报错，所以必须在入口挡掉。
    """
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/workflow/execute",
        json={"query": "张三在哪个部门", "user_id": "u1 张三"},
    )
    assert resp.status_code == 422


def test_both_chat_chains_share_one_meta_builder_without_identity():
    """``/chat/ask`` 与 ``/chat/ask/stream`` 共用 ``_meta_common``，字段口径必须一致。

    这两条链路曾各自写一份字典，漏字段与语义漂移都真实发生过。
    身份链移除后这里同时钉住"不该再有身份字段"——避免有人顺手把
    ``current_employee_id`` 加回来，那会让前端以为服务端有登录态。
    """
    from app.api.chat import _meta_common

    hit = _meta_common({"answer": "答案"}, "答案")
    miss = _meta_common({}, "答案")

    assert hit["answer"] == "答案" and miss["answer"] == "答案"
    assert "current_employee_id" not in hit
    assert "current_employee_id" not in miss

