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

    monkeypatch.setattr(router_agent, "_default_model", _must_not_be_called)

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
    monkeypatch.setattr(router_agent, "_default_model", lambda: stub)

    decision = route_query("请帮我分析一下当前国际形势对我们部门明年预算的影响")

    assert decision.scene == SCENE_COMPLEX_RAG
    assert decision.source == "router"
    assert stub.calls, "漏斗判不了时应当落到模型上"


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
    monkeypatch.setattr(agent_mod, "_default_model",
                        lambda: RecordingModel(["请问你叫什么名字？我帮你查。", []]))
    direct = nodes.tool_node(state)
    assert direct["intent_type"] == "direct"

    # ② 取到证据：交 L4 → 证据出口
    monkeypatch.setattr(agent_mod, "_default_model",
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
    monkeypatch.setattr(agent_mod, "_default_model", boom)

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
    monkeypatch.setattr(agent_mod, "_default_model", boom)
    monkeypatch.setattr("app.providers.llm.get_chat_model", boom)

    out = nodes.smalltalk_node(create_initial_state(user_query="你好", session_id="s"))

    assert out["answer"], "闲聊必须给出回答"
    assert out["intent_type"] == "direct"
    assert out["intent_source"] == SCENE_SMALLTALK
    assert out["soft_warnings"] == [], "闲聊没有可降级的东西"
    assert out["retrieve_docs"] == []


def test_no_smalltalk_prompt_is_registered():
    """闲聊**没有提示词**——这不是遗漏，而是"不调模型"的结构性证据。

    一旦有人往注册表里加了 ``smalltalk``，说明闲聊又走回模型生成，
    那也意味着它重新获得了"编造事实"的能力。
    """
    assert SCENE_SMALLTALK not in PROMPTS


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

    monkeypatch.setattr(sub_agents, "_default_model",
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

    monkeypatch.setattr(sub_agents, "_default_model",
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

    monkeypatch.setattr(sub_agents, "_default_model", lambda: _TextModel('["A", "B"]'))
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

    monkeypatch.setattr(sub_agents, "_default_model", lambda: _TextModel('["A", "B"]'))

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

    monkeypatch.setattr(sub_agents, "_default_model", lambda: _TextModel('["A", "B"]'))

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

    monkeypatch.setattr(agent_mod, "_default_model",
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

    monkeypatch.setattr(agent_mod, "_default_model", lambda: RecordingModel([
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

    monkeypatch.setattr(agent_mod, "_default_model", lambda: RecordingModel([
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

    monkeypatch.setattr(agent_mod, "_default_model",
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
    """四分支的优先级：人工兜底 > 直答 > 改道 > 受控生成。"""
    assert tool_route_edge({"need_human": True, "answer": "x", "tool_degraded": True}) == "human_fallback"
    assert tool_route_edge({"answer": "追问", "tool_degraded": True}) == "end"
    assert tool_route_edge({"tool_degraded": True}) == SCENE_SIMPLE_RAG
    assert tool_route_edge({}) == "generate_answer"
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
    monkeypatch.setattr(agent_mod, "_default_model", lambda: RecordingModel([
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
    """两条链路唯一允许的差别：证据出口通向 ``END`` 而不是 ``generate_answer``。"""
    assert _route_edges("simple_rag") == {"generate_answer"}
    assert _route_edges("complex_rag") == {"generate_answer"}
    assert _route_edges("simple_rag", pre_generation_workflow) == {"__end__"}
    assert _route_edges("complex_rag", pre_generation_workflow) == {"__end__"}
    assert "generate_answer" not in _route_edges("tool", pre_generation_workflow)


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

