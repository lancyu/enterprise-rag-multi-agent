"""业务正确性修复的回归测试。

对应一次真实故障的若干症状：

    症状一 · 正常问答被拒绝回答
        根因 A：软回退片段写死 ``fused=0`` → 置信度恒为 0 → 必然拒答
        根因 B：意图规则里「邮箱」裸词同时命中 tool 与 knowledge，
                tool 组在前 → 知识问句被吞，工具侧又抽不出参数，
                只回一句「未识别到有效业务参数」（用户看到的等同拒答）
    症状二 · 工具调用一直加载
        根因 C：ChatOpenAI 未禁用 SDK 内置重试（默认 2 次），
                ``LLM_TIMEOUT=30`` 被放大到 100s 以上
        根因 D：单事实直问（如「查一下 E1001 的部门」）被硬升 Pro 档
        根因 E：流式链路没有复刻图上的 ``human_fallback`` 分支

架构改造后的对应关系（**读这四条才不会以为覆盖度下降了**）
----------------------------------------------------------
以 function calling 为核心后，有三条根因的**载体变了，症状没变**，
因此断言形态也跟着换，而不是删掉：

| 原根因 | 原载体 | 现载体 | 本文件中的对应 |
|---|---|---|---|
| B | 意图规则（谁先命中谁赢） | 路由提示词 + 工具描述 | ``test_*_forbid_guessing`` / ``test_router_prompt_*`` |
| D | 档位选型 | **概念消失**（只有一个模型） | 无——删掉才是正确的 |
| E | 流式端点复刻节点链 | 两条链路共用同一张**前置子图** | ``test_stream_*_human_fallback`` |

根因 B 值得展开：**"知识问句被工具吞掉 → 抽不出参数 → 等同拒答"这个失败模式
在五 Agent 架构下依然存在**（模型可能对「怎么申请邮箱扩容」误调
``find_employee_by_name``，护栏一拒就没有证据了）。变的只是阻挡它的东西——
从"规则优先级"变成了三样：① 工具描述里那句「用户没有指明是谁就不要调用本工具」；
② 路由提示词把制度类问题判给 RAG Agent；③ 工具清单里**不再有**检索工具，
制度问题根本不归工具 Agent 管。
既然契约搬到了 schema 与提示词里，测试就必须钉在那里；否则有人顺手改一句
描述，这个已经出过事故的失败模式就会悄无声息地回来。

全部为纯单元 / 进程内测试，不依赖真实大模型、向量库与网络。
"""
from __future__ import annotations

import pytest

from app import config
from app.core.prompts import get as get_prompt
from app.core.router_agent import SCENE_COMPLEX_RAG, SCENE_SIMPLE_RAG, SCENE_SMALLTALK, SCENE_TOOL
from app.core.tool_agent import AGENT_TOOLS
from app.rag import generator, retriever

# ---------------------------------------------------------------------------
# 根因 B（新载体之一）：禁止猜测对象的契约写在工具的 JSON Schema 里
# ---------------------------------------------------------------------------
# 原实现靠"规则优先级"保证知识问句不被员工查询吞掉。改成 function calling 后，
# 判断主体是模型，能约束它的只有**工具描述**——所以那几句话就是契约本身，
# 必须被测试钉住。改动它们等于改动业务行为。
def _schema(tool_name: str) -> dict:
    tool_obj = next(t for t in AGENT_TOOLS if t.name == tool_name)
    return tool_obj.args_schema.model_json_schema()


def _description_text(tool_name: str) -> str:
    """把工具的全部对模型可见文本（工具描述 + 参数描述）拼在一起。"""
    schema = _schema(tool_name)
    parts = [str(schema.get("description") or "")]
    for prop in (schema.get("properties") or {}).values():
        parts.append(str(prop.get("description") or ""))
    tool_obj = next(t for t in AGENT_TOOLS if t.name == tool_name)
    parts.append(str(getattr(tool_obj, "description", "") or ""))
    return "\n".join(parts)


@pytest.mark.parametrize(
    "tool_name",
    ["find_employee_by_name", "query_leave_balance"],
)
def test_object_lookup_tools_forbid_guessing(tool_name):
    """「没有指明对象就不要调用 / 不要编造」必须写在描述里——这是根因 B 的新防线。

    少了这句，模型会对「怎么申请邮箱扩容」调 ``find_employee_by_name``，
    然后护栏拒绝、没有证据、L4 拒答——用户看到的正是一次"正常问答被拒绝回答"。
    """
    text = _description_text(tool_name)

    assert "不要" in text or "不得" in text, f"{tool_name} 缺少禁止猜测的约束：{text!r}"
    assert any(
        word in text
        for word in ("没有指明", "没有给出", "不要猜测", "不要编造", "不要自行编造")
    ), f"{tool_name} 的描述没有覆盖「用户未指明对象」这一情形：{text!r}"


def test_tools_do_not_claim_policy_questions():
    """工具清单里**不能**出现"能查公司制度"这类主张。

    制度问题归简单/复杂 RAG Agent。工具描述里一旦出现"也可以查制度"，
    模型就会对制度问句调一个查不到制度的工具——正是根因 B 的形态。
    """
    for tool_name in ("find_employee_by_name", "query_employee_info",
                      "query_leave_balance"):
        text = _description_text(tool_name)
        assert "制度" not in text and "流程" not in text, (
            f"{tool_name} 的描述把制度类问题揽了过来（应由 RAG Agent 负责）：{text!r}"
        )


def test_router_prompt_owns_policy_questions():
    """路由提示词必须把「制度规定」明确判给 RAG 场景 —— 这是根因 B 的另一半。

    旧实现靠词表命中 ``knowledge``；现在靠路由 Agent 读提示词判断。提示词里
    少了这两类场景，制度问句就可能被判给工具 Agent，而工具查不到制度。
    """
    prompt = get_prompt("router")

    assert SCENE_SIMPLE_RAG in prompt and SCENE_COMPLEX_RAG in prompt
    assert "制度" in prompt
    # 个人数据与制度规定必须被明确分开，否则「年假有多少天」会被送进工具 Agent
    assert SCENE_TOOL in prompt
    # 越界拦截也在这份提示词里（边界管控集中在入口层）
    assert "out_of_scope" in prompt
    assert SCENE_SMALLTALK in prompt


def test_tool_prompt_forbids_answering_policy_questions():
    """工具 Agent 的系统提示必须明确「不负责制度类问题」。

    新架构里这不再是"能力问题"而是"**职责边界**问题"：工具 Agent 手里根本没有
    检索工具，让它在缺少依据的情况下作答，结果是编一份年假天数出来——
    那比拒答更糟（用户无从辨别）。
    """
    prompt = get_prompt("agent")
    # 去掉 markdown 强调标记再断言：提示词里用 `**不**` 强调，直接匹配子串会断在中间
    plain = prompt.replace("*", "")

    assert "制度" in plain
    assert "不负责" in plain or "不回答" in plain
    # 「不许编造参数」这条护栏的提示词版本
    assert "不要猜测" in plain or "不要编造" in plain

# 注：`extract_identifier`（离线人名抽取）与「判定 vs 抽取同源」两条用例已随
# 离线抽取层一起删除。抽取现在由模型完成，其回归用例见 tests/test_tool_agent.py。
# `test_looks_like_employee_query` 保留——判定仍在，且仍是路由的准入闸门。


# ---------------------------------------------------------------------------
# 根因 B 的兜底：模型不调工具 → 走直答出口，**绝不**回一句等同拒答的话术
# ---------------------------------------------------------------------------
# 旧实现的兜底是「工具抽不到参数 → 回退知识检索」。新架构下这个分支消失了：
# 模型不调工具时走**直答出口**（它自己的话就是答案），典型场景是"必填参数缺失 →
# 主动向用户追问"。但故障的原始症状——「用户看到的等同拒答」——必须继续被挡住，
# 所以这里守住直答出口的形状：有内容、不转人工、不写 error_msg。
def test_direct_exit_never_looks_like_a_refusal(monkeypatch):
    """模型判定无需工具 → 直答出口给出可用回答，而不是一句"未识别到参数"。"""
    from app.core import tool_agent as agent_mod
    from app.graph import nodes
    from app.graph.state import create_initial_state
    from tests.fakes import RecordingModel

    monkeypatch.setattr(
        agent_mod, "default_model",
        lambda: RecordingModel(["申请邮箱扩容请到 IT 服务台办理，或联系 IT 支持组。"]),
    )
    out = nodes.tool_node(
        create_initial_state(user_query="怎么申请邮箱扩容", user_id="u1", session_id="s1")
    )

    assert out["intent_type"] == "direct"
    assert out["answer"], "直答出口必须给出内容"
    assert "未识别到" not in out["answer"]
    assert not out.get("need_human"), "可降级情形不得转人工"
    assert not out.get("error_msg"), "不得写 error_msg（会被 error_route_edge 误用）"
    # 直答没有检索依据，但也不该被 L4 当成"零证据"拿去拒答——所以它不进 L4
    assert out["retrieve_docs"] == []


def test_retrieval_failure_is_degradable_not_fatal(monkeypatch):
    """检索抖动 → 记 soft_warnings、不阻断、不转人工。

    写入点已从"工具的 NO_HIT 约定"迁到 ``sub_agents.run_simple_rag_agent``：
    新架构下检索是**独立 Agent**（直接调 retriever）而不是工具，所以软降级的
    发生地也跟着搬了。**但留痕这件事不能少**——否则它与"知识库确实没这条"
    在调用方看来完全同形，检索质量一路下滑到彻底失效才会被发现。
    """
    from app.core import sub_agents
    from app.graph import nodes
    from app.graph.state import create_initial_state

    def boom(query, top_k=None, allowed_sources=None):
        raise RuntimeError("向量库连接超时")

    monkeypatch.setattr(sub_agents, "retrieve_knowledge_docs", boom)
    out = nodes.simple_rag_node(
        create_initial_state(user_query="怎么申请邮箱扩容", user_id="u1", session_id="s1")
    )

    assert out["retrieve_docs"] == []
    assert out.get("soft_warnings"), "降级信息应记入 soft_warnings"
    assert not out.get("need_human")
    assert not out.get("error_msg")
    assert out["intent_type"] == "knowledge", "降级后仍走证据出口，由 L4 诚实作答"


def test_business_tool_exception_reports_human(monkeypatch):
    """**业务工具**异常仍要转人工——fail-open 不能过度。

    与上一条形成对照：判据是「能不能从其他来源得到答案」。数据库挂了时，
    用户问的是系统里的确定事实，L4 无从回答；而检索失败时 L4 的「没找到」
    与真实语义一致。见 ``app/graph/nodes.py::_failed_business_tools``。
    """
    from app.core import tool_agent as agent_mod
    from app.graph import nodes
    from app.graph.state import create_initial_state
    from tests.fakes import RecordingModel

    class _BoomTool:
        name = "query_employee_info"

        @staticmethod
        def invoke(_args):
            raise RuntimeError("业务库 502")

    # StructuredTool 是 pydantic 模型，不能只替换其 invoke 方法，故整体换掉映射里的工具对象
    monkeypatch.setitem(agent_mod._TOOLS_BY_NAME, "query_employee_info", _BoomTool())
    monkeypatch.setattr(
        agent_mod, "default_model",
        lambda: RecordingModel([
            {"name": "query_employee_info", "args": {"employee_id": "E1001"}},
            [],
        ]),
    )
    state = create_initial_state(user_query="E1001 是哪个部门的", user_id="u1", session_id="s1")
    out = nodes.tool_node(state)

    assert out.get("need_human") is True
    assert out.get("error_msg")


# ---------------------------------------------------------------------------
# 根因 A：软回退不得把 fused 写成 0（否则置信度恒为 0 → 必拒答）
# ---------------------------------------------------------------------------
class _FakeHit:
    def __init__(self, content: str, source: str, score: float, chunk_index: int):
        self.content = content
        self.score = score
        self.metadata = {"source": source, "chunk_index": chunk_index}


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def count(self):
        return len(self._hits)

    def search(self, vector, k):
        return self._hits[:k]


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.1] * 8


class _FakeLexicalIndex:
    def search(self, query, top_k):
        return []

    def doc_meta(self, key):
        return {}

    def doc_text(self, key):
        return ""


@pytest.fixture
def _soft_fallback_env(monkeypatch):
    """构造「阈值过滤后无结果 → 触发软回退」的最小环境。"""
    hits = [
        # 内容与 query 无字面重合，保证纯走向量路，数学关系清晰
        _FakeHit("完全无关的段落内容甲乙丙", "doc/a.txt", 0.30, 0),
        _FakeHit("另一段无关内容丁戊己", "doc/b.txt", 0.20, 1),
        _FakeHit("第三段无关内容庚辛壬", "doc/c.txt", 0.10, 2),
    ]
    monkeypatch.setattr(retriever, "get_vector_store", lambda: _FakeStore(hits))
    monkeypatch.setattr(retriever, "get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr(retriever, "get_lexical_index", lambda: _FakeLexicalIndex())
    # 阈值设得远高于任何 RRF 分 → 全部被过滤 → 必然进软回退分支
    monkeypatch.setattr(config, "effective_score_threshold", lambda: 0.5)
    monkeypatch.setattr(config, "effective_fallback_min", lambda: 0.001)
    return hits


def test_soft_fallback_keeps_real_fused_score(_soft_fallback_env):
    """软回退片段必须带真实 RRF 分——写 0 会让置信度归一化结果恒为 0。"""
    docs = retriever.retrieve("年假有多少天")

    assert len(docs) == 3
    assert all(d["fallback"] is True for d in docs)
    assert docs[0]["fused"] > 0, "软回退不得把 fused 写成 0"

    # RRF 分上限 = (0.7 + 0.3) / (60 + 1)；rank1 向量贡献 = 0.7 / 61
    # （retriever 落库时对 fused 做了 6 位取整，期望值同步 round）
    fused_max = (retriever.DENSE_WEIGHT + retriever.LEXICAL_WEIGHT) / (retriever.RRF_K + 1)
    expected = round(retriever.DENSE_WEIGHT / (retriever.RRF_K + 1), 6)
    assert docs[0]["fused"] == expected
    assert fused_max > docs[0]["fused"]


def test_soft_fallback_docs_do_not_trigger_refusal(_soft_fallback_env):
    """命中 3 条却因 fused=0 而拒答，正是本次故障的核心表现。"""
    docs = retriever.retrieve("年假有多少天")
    confidence = generator.estimate_confidence(docs)

    # 降权仍然生效（软回退证据弱）：0.7 × 0.6 = 0.42，但必须高于拒答阈值
    assert confidence == pytest.approx(0.42, abs=0.01)
    assert confidence >= generator.REFUSE_THRESHOLD

    prepared = generator.prepare_generation("年假有多少天", docs)
    assert prepared["refused"] is False, "有检索结果就不该拒答"


def test_fused_zero_still_refuses_for_contrast():
    """对照：保留旧行为（fused=0）时置信度必为 0 —— 固化故障机理，防止回退。"""
    old_style = [{"fused": 0.0, "fallback": True, "lexical": 0.0}]
    assert generator.estimate_confidence(old_style) == 0.0
    prepared = generator.prepare_generation("年假有多少天", old_style)
    assert prepared["refused"] is True


# ---------------------------------------------------------------------------
# 根因 C：超时不得被放大（SDK 重试 + 流式回退两层叠加）
#
# 注：这一组原有三个用例，随自造的限流包装层一起删除：
#   · test_is_timeout_detection                    → _is_timeout 已删除
#   · test_stream_does_not_fall_back_on_timeout    → 流式回退逻辑已删除
#   · test_stream_still_falls_back_on_other_errors → 同上
# 现在模型直连 ChatOpenAI，"两层重试互相放大"这个结构本身不存在了。
# 唯一还需要守住的是下面这条：max_retries 必须显式为 0。
# ---------------------------------------------------------------------------
def test_raw_model_disables_sdk_retries(monkeypatch):
    """max_retries 必须显式为 0：SDK 默认重试 2 次会把 LLM_TIMEOUT=30 放大到 101s。

    放大在**只有 SDK 一层**时就已成立（30 × 3 + 退避），与是否存在自造包装无关——
    那条 trace 实证「用户看到一直加载」的根因还在，所以这个断言还得守着。
    """
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://127.0.0.1:9/v1")

    from app.providers.llm import _build_raw_model

    assert _build_raw_model().max_retries == 0


# ---------------------------------------------------------------------------
# 根因 D：单事实直问被硬升 Pro 档
# ---------------------------------------------------------------------------
# **本段已删除**，不是遗漏。档位机制（Flash / Pro）随「改用不限流模型 +
# function calling」整体移除：只有一个模型时，"降档/升档"没有落点，
# 阈值标定与会话 pin 也随之失去意义。
#
# 原用例守的是「单事实直问不该走最贵的档位」——这个**成本诉求**在新架构下
# 由另一件事满足：模型对单事实直问（如「E1001 的部门」）只发一次工具调用、
# 再过一次 L4 生成，路径固定且可预测，不再有"被规则误升到 Pro"的可能。
# 被删模块与原因见 _archive/removed-selfbuilt-routing-20260915-1314/README.md。


# ---------------------------------------------------------------------------
# 根因 E：流式链路必须复刻图上的「人工兜底」分支
# ---------------------------------------------------------------------------
def _force_human_fallback(monkeypatch):
    """把本轮强制引导到「工具 Agent 决策失败 → 转人工」。

    做法是替换两个**节点函数读的模块级引用**（``nodes.route_query`` 与
    ``nodes.run_tool_agent``），而不是替换节点本身——已编译的图里存的是节点
    函数的引用，换节点函数没用；而节点函数内部是按名查模块全局的，换全局有效。
    这也是"接线要能被测试"的一个前提：节点内部不得把依赖 hard-code 成闭包。
    """
    from app.core.router_agent import RouteDecision
    from app.core.tool_agent import ToolDecision
    from app.graph import nodes

    monkeypatch.setattr(
        nodes, "route_query",
        lambda **kwargs: RouteDecision(scene=SCENE_TOOL, reason="测试强制走工具路", confidence=1.0),
    )
    monkeypatch.setattr(
        nodes, "run_tool_agent",
        lambda **kwargs: ToolDecision(error="业务库 502"),
    )


def test_stream_emits_human_fallback_instead_of_generating(monkeypatch):
    """已判定转人工时，流式链路给用户的必须与非流式一致，且不再调模型生成。

    旧实现只复刻了意图分支，漏了 ``human_fallback``：节点已判定 need_human，
    流式链路却照样去调模型生成，meta 里还报 need_human=true —— 两条链路对
    同一状态给出不一致的回答。

    架构改造后的防线更强了：流式端点不再"重抄一遍节点链"，而是 invoke 一张
    **与完整图同源的前置子图**（``pre_generation_workflow``，由同一个装配函数
    编译，差别只有"证据出口通向哪"）。本用例因而同时钉住行为与接线。
    """
    import json
    from fastapi.testclient import TestClient

    import app.api.chat as chat_mod
    from app.main import app

    generated = {"called": False}

    async def fake_get_history(session_id):
        return []

    async def fake_save_message(*args, **kwargs):
        return None

    def fake_maybe_consolidate(*args, **kwargs):
        return None

    def fake_stream_answer_tokens(*args, **kwargs):
        generated["called"] = True
        yield "不应出现的内容"

    _force_human_fallback(monkeypatch)
    monkeypatch.setattr(chat_mod, "get_history", fake_get_history)
    monkeypatch.setattr(chat_mod, "save_message", fake_save_message)
    monkeypatch.setattr(chat_mod, "maybe_consolidate", fake_maybe_consolidate)
    monkeypatch.setattr(chat_mod, "stream_answer_tokens", fake_stream_answer_tokens)

    # 不用 with：避免触发 lifespan 的启动自检（会真实调 LLM）
    client = TestClient(app)
    resp = client.post(
        "/chat/ask/stream",
        json={"query": "E1001 是哪个部门的", "session_id": "s-human", "user_id": "u1"},
    )
    assert resp.status_code == 200

    meta = None
    for block in resp.text.split("\n\n"):
        if block.startswith("event: meta"):
            data_line = [ln for ln in block.splitlines() if ln.startswith("data: ")][0]
            meta = json.loads(data_line[len("data: "):])

    assert meta is not None, f"未收到 meta 事件：{resp.text[:400]}"
    assert meta["need_human"] is True
    assert "转接人工" in meta["answer"], meta["answer"]
    assert not generated["called"], "已判定转人工就不该再调模型生成"


def test_non_stream_graph_also_routes_to_human_fallback(monkeypatch):
    """非流式（完整图）路径同样转人工——两条链路口径一致（防止只修一边）。"""
    from app.graph.state import create_initial_state
    from app.graph.workflow_graph import enterprise_workflow

    _force_human_fallback(monkeypatch)
    out = enterprise_workflow.invoke(
        create_initial_state(user_query="E1001 是哪个部门的", user_id="u1", session_id="s")
    )

    assert out["need_human"] is True
    assert "转接人工" in out["answer"]
    assert out["intent_source"] == "human_fallback"


def test_stream_and_graph_share_the_same_pre_generation_nodes():
    """两条链路的前置节点必须**是同一批函数**，不是"长得一样的两份"。

    这是根因 E 的结构性防线：只要节点集合出现差异，说明有人又开始在流式端
    另写一套前置链路——那正是历史缺陷的成因。
    """
    from app.graph.workflow_graph import enterprise_workflow, pre_generation_workflow

    full = {n for n in enterprise_workflow.get_graph().nodes if not n.startswith("__")}
    pre = {n for n in pre_generation_workflow.get_graph().nodes if not n.startswith("__")}

    assert pre <= full, f"前置子图出现了完整图里没有的节点：{pre - full}"
    assert full - pre == {"generate_answer"}, (
        "两个图的差别只应有 generate_answer 一个节点（流式由端点自行生成）"
    )


# ---------------------------------------------------------------------------
# 症状三：回答不完整 + 很慢（推理模型的思考阶段）
# ---------------------------------------------------------------------------
# 故障现象：回答能出来，但「写到一半就断」且「首字要等十几秒」。
# 根因：默认模型 kimi-k2.6 是**推理模型**，正文之前会先产出一段隐藏的
#      reasoning_content，带来两个后果：
#        1) 首个可见字要等 11~24s（用户感知为「卡住/一直加载」）；
#        2) 思考 token 与正文**共享 max_tokens 预算**，思考吃掉一半后正文
#           更早撞顶 → finish_reason=length → 答案被拦腰截断；
#        3) 且截断此前**完全静默**：日志只有「L4 生成完成：N 字」，看不出异常。
#
# 实测对照（2026-09-11，同一问句，时长单位 ms）：
#     思考开启 temp=1.0   首字 11123   推理 535 字   正文 1115 字
#     思考关闭 temp=0.6   首字  1137   推理   0 字   正文  875 字
#     思考开启 + 1024     首字 15790   ——            正文  911 字   finish=length
#
# 下面用「固化故障机理」的方式把这三条钉死。
# ---------------------------------------------------------------------------
def test_truncation_is_visible_in_stats():
    """finish_reason=length 必须能被识别为「截断」——这是它不再静默的前提。"""
    stats = generator.StreamStats(finish_reason="length", chars=900)
    assert stats.truncated is True


def test_normal_stop_is_not_flagged_as_truncated():
    """正常结束不能被误报为截断，否则告警会失去意义（狼来了）。"""
    assert generator.StreamStats(finish_reason="stop").truncated is False
    # 未收到终止帧（供应商异常断流）同样不算「截断」——它是另一种故障，另有分支处理
    assert generator.StreamStats(finish_reason=None).truncated is False


def test_truncation_regression_keeps_old_silent_behaviour_visible():
    """对照：若仍沿用旧的「只有字数」统计，截断与正常结束无法区分。

    旧实现只记 ttft/chunks/chars 三项，长答案被截断后
    「L4 生成完成：900 字」与正常写完的 900 字在日志里长得一模一样，
    用户报「回答不完整」时运维无从定位。此用例固化这一差异。
    """
    truncated_run = generator.StreamStats(chunks=269, chars=900, finish_reason="length")
    complete_run = generator.StreamStats(chunks=269, chars=900, finish_reason="stop")
    assert truncated_run.chars == complete_run.chars          # 旧口径：完全一样
    assert truncated_run.truncated != complete_run.truncated  # 新口径：能区分


def _install_fake_model(monkeypatch, chunks, generate_text="回退生成的整段答案"):
    """把 ``get_chat_model`` 换成可控假模型，返回记录 ``_generate`` 调用次数的容器。

    ``chunks`` 形如 [("正文片段", "stop"), ("", "length")]：第二项是结束原因，
    用来模拟「思考吃满预算、正文为空」这类真实供应商行为。
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    calls = {"generate": 0}

    class FakeModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-reasoning-model"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            calls["generate"] += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=generate_text))])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            for content, reason in chunks:
                meta = {"finish_reason": reason} if reason else {}
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=content, response_metadata=meta)
                )

    monkeypatch.setattr("app.core.llm_factory.get_chat_model", lambda tier=None: FakeModel())
    return calls


_INPUTS = {"context": "参考内容", "user_query": "问题", "chat_history": "", "memory_context": ""}


def test_stream_captures_finish_reason(monkeypatch):
    """流式必须把供应商的结束原因记下来，否则截断无法被观测。"""
    _install_fake_model(monkeypatch, chunks=[("前半段", None), ("后半段", "length")])
    stats = generator.StreamStats()

    out = list(generator.stream_answer_tokens(dict(_INPUTS), stats=stats))

    assert "".join(out) == "前半段后半段"
    assert stats.finish_reason == "length"
    assert stats.truncated is True


def test_budget_exhausted_empty_body_does_not_retry(monkeypatch):
    """收到终止帧却无正文 = 预算耗尽，**不能**回退重试——那会把等待翻倍。

    故障现场：trace cae0cddcee3b4546 流式 28s 零产出 → 回退整段生成 →
    再等 28s，总耗时 57.7s。回退在这里毫无收益：重试只会再付一整个思考周期。
    """
    calls = _install_fake_model(monkeypatch, chunks=[("", "length")])
    stats = generator.StreamStats()

    out = list(generator.stream_answer_tokens(dict(_INPUTS), stats=stats))

    assert out == []                       # 不回退，正文确实为空
    assert stats.finish_reason == "length"
    assert calls["generate"] == 0          # 关键：没有触发 _generate 回退


def test_true_empty_stream_still_retries(monkeypatch):
    """对照：有 chunk 但内容全空、且**没有终止帧**时，仍要回退整段生成。

    这是原有容错，不能被上面的修复误伤——否则供应商偶发空流会直接变成空回答。
    必须区分两种「空」：
        - 空内容 chunk + 无终止帧 → 真·异常断流 → 回退（本用例）；
        - 空内容 chunk + 有终止帧 → 预算耗尽，回退只会重付一个思考周期（上一个用例）。
    （供应商返回**零个** chunk 时 langchain 会直接抛
      ``ValueError: No generation chunks were returned``，走不到这个分支。）
    """
    calls = _install_fake_model(monkeypatch, chunks=[("", None)])
    stats = generator.StreamStats()

    out = list(generator.stream_answer_tokens(dict(_INPUTS), stats=stats))

    assert out == ["回退生成的整段答案"]
    assert calls["generate"] == 1
    assert stats.finish_reason is None


def test_thinking_switch_is_forwarded_to_provider(monkeypatch):
    """关闭思考的开关必须真正落进请求体，否则配置只是摆设。"""
    from app.providers.llm import _build_raw_model

    monkeypatch.setattr(config, "LLM_DISABLE_THINKING", True)
    assert _build_raw_model().extra_body == {"thinking": {"type": "disabled"}}

    # 关闭时必须完全不传该字段——它是供应商特定参数，默认不传才最兼容
    monkeypatch.setattr(config, "LLM_DISABLE_THINKING", False)
    assert _build_raw_model().extra_body is None


# ---------------------------------------------------------------------------
# 检索侧：文档向量缓存的 O(n²) 写放大
# ---------------------------------------------------------------------------
# 故障现象：全量重建索引异常慢（与「回答慢」同源，都在检索链路上）。
# 根因：``put_doc_vec`` 每插入一条就把**整个缓存**序列化落盘，建索引逐条调用
#      即形成 O(n²) 写放大。实测缓存 7MB / 325 条时单次写入 128ms，
#      176 篇全量重建光序列化就白烧十几秒，且随语料增长更快。
# ---------------------------------------------------------------------------
def test_batch_doc_vec_write_saves_once(tmp_path):
    """批量写入只落盘一次——这是消除 O(n²) 写放大的关键。"""
    from app.utils.cache import EmbeddingCache

    cache = EmbeddingCache(tmp_path / "ec.json")
    saves: list = []
    cache._save_doc_cache = lambda: saves.append(1)

    texts = [f"chunk-{i}" for i in range(20)]
    cache.put_doc_vecs("m", texts, [[0.0] * 4] * 20)

    assert len(saves) == 1                                   # 20 条只写 1 次
    assert all(cache.get_doc_vec("m", t) is not None for t in texts)


def test_per_item_invocation_still_amplifies_for_contrast(tmp_path):
    """对照：**只要调用方逐条调用，写放大就会回来**。

    批量接口本身不保证性能——把 ``put_doc_vecs`` 当成单条接口循环调用，
    依旧是「一条一落盘」。这条用例固化机理，提醒后来者：
    真正的约束是「一次调用写一批」，不是「有个叫 vecs 的方法」。
    """
    from app.utils.cache import EmbeddingCache

    cache = EmbeddingCache(tmp_path / "ec.json")
    saves: list = []
    cache._save_doc_cache = lambda: saves.append(1)

    for i in range(20):
        cache.put_doc_vecs("m", [f"chunk-{i}"], [[0.0] * 4])

    assert len(saves) == 20


def test_embed_documents_uses_batch_cache_write(tmp_path, monkeypatch):
    """守住真正的调用点：embed_documents 必须批量回写缓存，而非逐条。"""
    from app.providers.embeddings import CachedAPIEmbeddings
    from app.utils.cache import EmbeddingCache

    class FakeInner:
        model = "fake-model"
        mode = "api"
        dim = 4

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    cache = EmbeddingCache(tmp_path / "ec.json")
    saves: list = []
    cache._save_doc_cache = lambda: saves.append(1)
    # _cache() 内部是「调用时导入」，故打桩 app.utils.cache 命名空间即可
    monkeypatch.setattr("app.utils.cache.get_embedding_cache", lambda: cache)

    vecs = CachedAPIEmbeddings(FakeInner()).embed_documents([f"doc-{i}" for i in range(12)])

    assert len(vecs) == 12
    assert len(saves) == 1                                   # 不是 12 次


def test_empty_batch_write_is_a_noop(tmp_path):
    """空批次不该产生任何落盘（避免无意义的整文件重写）。"""
    from app.utils.cache import EmbeddingCache

    cache = EmbeddingCache(tmp_path / "ec.json")
    saves: list = []
    cache._save_doc_cache = lambda: saves.append(1)

    cache.put_doc_vecs("m", [], [])

    assert saves == []


# ===========================================================================
# 截断必须「透到 state 与响应体」，不能只活在日志里
#
# 背景：「回答不完整」的元凶是 max_tokens 被思考 token 挤占而截断，但截断
# 原先完全静默——被截断的 900 字与正常写完的 900 字在日志里一模一样。
# 只在流式接口补 meta 是不够的：走 /chat/ask 的客户端同样要能判断。
# ===========================================================================
def _gen_state(**over):
    """按生产路径构造初始 state，再叠加本用例的输入。

    必须走 ``create_initial_state``：节点里是 ``dict(state)`` 拷进来的，
    生产环境的键来自初始 state；若测试手搓一个裸 dict，就测不出
    「LangGraph schema 有没有把新字段当非法通道丢掉」这类问题。

    检索片段的 ``fused`` 要够高才能过拒答阈值（0.25）：
    归一化上限 = (0.7+0.3)/(60+1) = 1/61，故 0.01 对应置信度约 0.61。
    """
    from app.graph.state import create_initial_state

    st = create_initial_state(user_query="报销流程怎么走", session_id="t-gen")
    st.update({
        "retrieve_docs": [{
            "content": "费用发生后，须在行程或事项结束之日起 7 个工作日内提交报销申请。",
            "source": "data/财务报销管理制度.txt",
            "fused": 0.01,
            "lexical": 0.6,
        }],
        "chat_history": [],
    })
    st.update(over)
    return st


def test_truncation_reaches_graph_state(monkeypatch):
    """节点必须把 stats 里的截断信息写进 state，供 API 层读取。"""
    from app.graph.nodes import generate_answer_node

    _install_fake_model(
        monkeypatch,
        chunks=[("这是被截断的正文内容，写到这里就撞上了 max_tokens 上限。", "length")],
    )
    state = generate_answer_node(_gen_state())

    assert state["finish_reason"] == "length"
    assert state["truncated"] is True


def test_normal_stop_state_is_not_truncated(monkeypatch):
    """对照组：正常收尾不得被误标截断（避免新指标变成常态噪声）。"""
    from app.graph.nodes import generate_answer_node

    _install_fake_model(
        monkeypatch,
        chunks=[("这里是正常写完的完整答案。", "stop")],
    )
    state = generate_answer_node(_gen_state())

    assert state["finish_reason"] == "stop"
    assert state["truncated"] is False


def test_initial_state_defaults_to_not_truncated():
    """初始 state 必须显式给出截断字段的默认值。

    否则 LangGraph 的 TypedDict schema 会把这些键当成非法通道直接丢弃，
    API 层读到的永远是 None —— 静默丢字段比报错更难查。
    """
    from app.graph.state import create_initial_state

    st = create_initial_state(user_query="你好", session_id="s1")

    assert st["truncated"] is False
    assert st["finish_reason"] is None
