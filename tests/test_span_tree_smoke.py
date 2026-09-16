#!/usr/bin/env python3
"""span 树耗时口径冒烟测试 —— 验证「根 span 覆盖整条消息处理链路」。

背景（本轮修复）：此前 span 树只记录前置节点（memory_load / intent_recognize /
knowledge_retrieve / model_route），漏掉了占 97%+ 耗时的 LLM 生成阶段，且没有单一
根 span，导致 trace.jsonl 的 elapsed_ms（≈几百 ms）与实际回消息时间（10s+）对不上。

本测试不启动服务、不调模型、不检索，用 mock 隔离所有外部依赖，验证两类关键行为：
    1. 非流式 /chat/ask：根 span `chat_request` 覆盖「历史加载→工作流→持久化→记忆
       整理」全链路；线程内节点 span 正确挂到根 span 之下（contextvars 跨线程透传）。
    2. 流式 /chat/ask/stream：`generate_answer` span 作为 `chat_request` 的子节点，
       覆盖逐 token 生成阶段（此前完全缺失）。

用法：
    cd langgraph-enterprise-bot && python -m tests.test_span_tree_smoke
"""
import asyncio
import json
import os
import sys
import time
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_results = []


def check(name, fn):
    try:
        detail = fn() or "OK"
        print(f"  {GREEN}PASS{RESET} {name} · {detail}")
        _results.append(True)
    except AssertionError as e:
        print(f"  {RED}FAIL{RESET} {name} · {e}")
        _results.append(False)


def find_child(tree, name):
    """在 span 树中按名字找节点（深度优先）。"""
    for node in tree:
        if node["name"] == name:
            return node
        found = find_child(node.get("children", []), name)
        if found:
            return found
    return None


def _empty_short_term(session_id, history, **_kw):
    """把短期窗口替换成空，让 span 测试不受真实裁剪逻辑影响。"""
    from app.memory.short_term import ShortTermMemory

    return ShortTermMemory(session_id, [])


# ---------------------------------------------------------------------------
print("\n[1] 非流式 /chat/ask：根 span 覆盖全链路 + 线程 span 正确挂载")
# ---------------------------------------------------------------------------
def test_chat_ask_root_span():
    from app.api.chat import chat_ask
    from app.utils.validator import ChatRequest

    # 模拟工作流：在 worker 线程内打开嵌套 span，验证跨线程挂载到 chat_request 下
    def fake_invoke(state):
        from app.core.tracing import span

        with span("workflow_run"):
            with span("generate_answer"):
                time.sleep(0.01)
        return {
            "answer": "年休假为五天。",
            "intent_type": "knowledge",
            "intent_source": "rule",
            "confidence": 0.9,
            "model_tier": "flash",
            "route_decision": {},
            "retrieve_docs": [],
            "citations": [],
            "refused": False,
            "tool_result": None,
            "need_human": False,
            "trace": [],
        }

    async def fake_get_history(sid):
        return []

    async def fake_save_message(sid, role, content):
        return None

    def fake_maybe_consolidate(user_id, session_id, history):
        return False

    class FakeWorkflow:
        def invoke(self, state):
            return fake_invoke(state)

    with mock.patch("app.api.chat.get_history", fake_get_history), \
         mock.patch("app.api.chat.save_message", fake_save_message), \
         mock.patch("app.api.chat.maybe_consolidate", fake_maybe_consolidate), \
         mock.patch("app.api.chat.enterprise_workflow", FakeWorkflow()):
        result = asyncio.run(chat_ask(ChatRequest(query="年假有多少天")))

    tree = result["span_tree"]
    assert len(tree) == 1, f"应只有一个根 span，实际 {[n['name'] for n in tree]}"
    root = tree[0]
    assert root["name"] == "chat_request", f"根 span 应为 chat_request，实际 {root['name']}"
    assert root["start_ms"] == 0.0, "根 span start_ms 应为 0（全链路基准）"

    # 线程内节点 span 挂到根 span 之下
    gen = find_child(tree, "generate_answer")
    assert gen is not None, "generate_answer 应挂到根 span 下（此前完全缺失）"
    assert find_child(tree, "workflow_run") is not None, "workflow_run 应作为 chat_request 子节点"
    assert gen["duration_ms"] >= 10, f"generate_answer 耗时应 ≥10ms（sleep 0.01s），实际 {gen['duration_ms']}"

    # 根 span 耗时与接口 elapsed_ms 同口径（误差 < 100ms）
    diff = abs(result["elapsed_ms"] - root["duration_ms"])
    assert diff < 100, f"根 span 耗时 {root['duration_ms']}ms 与 elapsed_ms {result['elapsed_ms']}ms 偏差过大"

    return f"根=chat_request · 含 workflow_run/generate_answer · 根耗时≈elapsed_ms（差 {diff}ms）"


check("非流式根 span", test_chat_ask_root_span)


# ---------------------------------------------------------------------------
print("\n[2] 流式 /chat/ask/stream：generate_answer 挂到 chat_request 下")
# ---------------------------------------------------------------------------
def test_chat_ask_stream_generate_span():
    from app.utils.validator import ChatRequest

    def passthrough(state):
        return state

    def fake_intent(state):
        state["intent_type"] = "knowledge"
        state["intent_source"] = "rule"
        return state

    def fake_route(state):
        state["model_tier"] = "flash"
        return state

    def fake_prepare(user_query, docs, chat_history, tool_result, memory_context):
        return {
            "citations": [],
            "confidence": 0.9,
            "refused": False,
            "refusal": None,
            "inputs": {"context": "年休假为五天。", "user_query": user_query},
        }

    def fake_stream_tokens(inputs, tier=None, stats=None):
        yield "年休假"
        yield "为五天。"

    async def fake_get_history(sid):
        return []

    async def fake_save_message(sid, role, content):
        return None

    def fake_maybe_consolidate(user_id, session_id, history):
        return False

    async def collect():
        from app.api.chat import chat_ask_stream

        with mock.patch("app.api.chat.get_history", fake_get_history), \
             mock.patch("app.api.chat.save_message", fake_save_message), \
             mock.patch("app.api.chat.maybe_consolidate", fake_maybe_consolidate), \
             mock.patch("app.api.chat.memory_load_node", passthrough), \
             mock.patch("app.api.chat.intent_recognize_node", fake_intent), \
             mock.patch("app.api.chat.knowledge_retrieve_node", passthrough), \
             mock.patch("app.api.chat.model_route_node", fake_route), \
             mock.patch("app.api.chat.get_short_term_memory", _empty_short_term), \
             mock.patch("app.api.chat.prepare_generation", fake_prepare), \
             mock.patch("app.api.chat.stream_answer_tokens", fake_stream_tokens):
            response = await chat_ask_stream(ChatRequest(query="年假有多少天"))
            frames = []
            async for chunk in response.body_iterator:
                frames.append(chunk)
        return frames

    frames = asyncio.run(collect())

    # 解析 meta 帧拿到 span_tree
    meta = None
    for frame in frames:
        if frame.startswith("event: meta"):
            data_line = [ln for ln in frame.splitlines() if ln.startswith("data: ")][0]
            meta = json.loads(data_line[len("data: "):])
    assert meta is not None, "应产出 meta 帧"

    tree = meta["span_tree"]
    root = find_child(tree, "chat_request")
    assert root is not None, f"应存在 chat_request 根 span，实际顶层 {[n['name'] for n in tree]}"

    gen = find_child(tree, "generate_answer")
    assert gen is not None, "generate_answer 应挂到 chat_request 下（此前流式路径完全缺失）"
    # generate_answer 是 chat_request 的直接子节点
    assert any(c["name"] == "generate_answer" for c in root["children"]), \
        "generate_answer 应是 chat_request 的直接子节点"

    # 生成阶段确实覆盖了 token 输出：answer 完整
    assert meta["answer"] == "年休假为五天。", f"流式拼接结果错误：{meta['answer']!r}"

    return f"根=chat_request · 子节点含 generate_answer（{gen['duration_ms']}ms）· answer 完整"


check("流式 generate_answer span", test_chat_ask_stream_generate_span)


# ---------------------------------------------------------------------------
print("\n[3] 生成统计 StreamStats：TTFT / chunk / 字符数（模型无关）")
# ---------------------------------------------------------------------------
def test_stream_stats():
    import time
    from types import SimpleNamespace

    from app.rag.generator import StreamStats, stream_answer_tokens

    class FakeChain:
        def __init__(self, chunks):
            self._chunks = chunks

        def stream(self, inputs):
            for c in self._chunks:
                time.sleep(0.01)
                yield c

        def invoke(self, inputs):
            return SimpleNamespace(content="".join(c.content for c in self._chunks))

    class FakePrompt:
        def __or__(self, other):
            return FakeChain([
                SimpleNamespace(content="年休假"),
                SimpleNamespace(content="为五天。"),
                SimpleNamespace(content=""),  # 空 chunk 应被过滤，不计数
            ])

    with mock.patch("app.rag.generator.ANSWER_PROMPT", FakePrompt()), \
         mock.patch("app.core.llm_factory.get_chat_model", return_value=None):
        stats = StreamStats()
        out = "".join(stream_answer_tokens({"context": "x"}, tier="flash", stats=stats))

    assert out == "年休假为五天。", f"拼接错误: {out!r}"
    assert stats.ttft_ms >= 10, f"ttft 应≥10ms（首个 chunk 前 sleep 0.01s），实际 {stats.ttft_ms}"
    assert stats.chunks == 2, f"应 2 个有效 chunk（空 chunk 过滤），实际 {stats.chunks}"
    assert stats.chars == len("年休假为五天。"), f"字符数应为 7，实际 {stats.chars}"
    return f"ttft={stats.ttft_ms}ms chunks={stats.chunks} chars={stats.chars} · 空 chunk 被过滤"


check("StreamStats 统计", test_stream_stats)


# ---------------------------------------------------------------------------
print()
passed, total = sum(_results), len(_results)
print(f"span 树耗时口径冒烟测试：{passed}/{total} 通过")
sys.exit(0 if passed == total else 1)
