"""短期记忆对称化回归测试（2026-09-11）。

背景：参考 HKUDS/nanobot 的 Session 写了 ``ShortTermMemory``，但它长期零调用——
根因是持久化层 ``app/core/chat_memory.py`` 跑出了 ``app/memory/`` 包，导致
``short_term.py`` 只剩算法、领域对象无处落地。本轮把它归位为
``app/memory/chat_history.py``，并恢复领域对象 + 对称门面。

本文件守三件事：
1. **对称性**：长短期各层一一对应，谁也不缺；
2. **对象有真实消费点**：恢复的 ``ShortTermMemory`` 必须被生产代码使用，
   否则又会长成新的死代码（这正是它上一版的下场）；
3. **不重复 I/O**：``history`` 由构造注入，对象自己不读 Redis。
4. **裁剪点与格式化各只有一份**：``build_window`` 负责裁、``format_history`` 负责写，
   生成层与路由层都只是消费者（第 4 节）。
"""

from __future__ import annotations

import inspect
from typing import Dict, List

from app import memory as memory_pkg
from app.memory import short_term as short_term_mod


def _history(n: int) -> List[Dict[str, str]]:
    """造 n 轮问答（一问一答 = 2 条）。"""
    out: List[Dict[str, str]] = []
    for i in range(n):
        out.append({"role": "user", "content": f"第 {i} 个问题"})
        out.append({"role": "assistant", "content": f"第 {i} 个回答"})
    return out


# ---------------------------------------------------------------------------
# 1. 对称性：长短期各层一一对应
# ---------------------------------------------------------------------------
def test_long_and_short_term_have_symmetric_facades() -> None:
    """长期有 get_long_term_memory，短期必须有 get_short_term_memory。"""
    assert callable(memory_pkg.get_long_term_memory), "长期记忆门面缺失"
    assert callable(memory_pkg.get_short_term_memory), "短期记忆门面缺失"


def test_long_and_short_term_have_symmetric_domain_objects() -> None:
    """长期有 LongTermMemory，短期必须有 ShortTermMemory，且都在 memory 包内。"""
    long_obj = memory_pkg.get_long_term_memory.__annotations__.get("return")
    short_obj = memory_pkg.get_short_term_memory.__annotations__.get("return")
    assert long_obj is not None and short_obj is not None
    assert long_obj.__module__.startswith("app.memory")
    assert short_obj.__module__.startswith("app.memory"), "领域对象必须在 app.memory 包内"


def test_persistence_layers_are_both_inside_memory_package() -> None:
    """两层持久化都必须在 app/memory/ 下——不对称会让领域对象无处落地。"""
    import app.memory.chat_history as chat_history_mod
    import app.memory.store as store_mod

    assert chat_history_mod.__name__ == "app.memory.chat_history"
    assert store_mod.__name__ == "app.memory.store"


def test_both_are_exported() -> None:
    """两个领域对象都要在 __all__ 里，否则对外契约不完整。"""
    assert "LongTermMemory" in memory_pkg.__all__
    assert "ShortTermMemory" in memory_pkg.__all__


# ---------------------------------------------------------------------------
# 2. 对象有真实消费点（防止又长成死代码）
# ---------------------------------------------------------------------------
def test_short_term_memory_is_used_by_generation_prestage() -> None:
    """生成前置必须走门面，而不是绕过对象直接调纯函数。

    这是本轮的核心：上一版 ShortTermMemory 就是因为没有消费点才被删掉的。

    架构改造（function calling 为核心）后，消费点从 ``generate_answer_node``
    上移到了 ``build_generation_inputs`` —— 这是**刻意的**：非流式节点与流式
    端点共用这一处前置，窗口裁剪与拒答阈值才有唯一产生点。断言因此跟着移动到
    新的协作者上，而不是放宽标准。
    """
    from app.graph import nodes as nodes_mod

    src = inspect.getsource(nodes_mod.build_generation_inputs)
    assert "get_short_term_memory" in src, "生成前置未使用短期记忆门面"
    assert "build_short_term_window" not in src, "应走门面，不应再直接调纯函数"

    # 消费点必须真的在链路上：非流式节点要调它
    node_src = inspect.getsource(nodes_mod.generate_answer_node)
    assert "build_generation_inputs" in node_src, "非流式节点绕过了共用的生成前置"


def test_stream_path_uses_the_same_prestage() -> None:
    """流式链路与非流式链路必须走**同一个**生成前置，避免两条口径。

    这条断言比"两处都 import 了某个门面"更强：过去两条链路的失效方式恰恰是
    「各自 import、各自重抄一遍前置逻辑」，import 相同并不能阻止逻辑漂移。
    真正要钉的是**同一个函数**。
    """
    from app.api import chat as chat_mod
    from app.graph import nodes as nodes_mod

    src = inspect.getsource(chat_mod)
    assert "build_generation_inputs" in src, "流式链路未复用共用的生成前置"
    # 流式端点不得自己再裁一次窗口（裁剪点必须唯一）
    assert "build_short_term_window" not in src, "流式端点自行裁剪了短期记忆窗口"
    assert "get_short_term_memory" not in src, "流式端点自行组装了短期记忆"
    # 复用必须是"引用同一个函数"，不是"另写一个同名实现"
    assert chat_mod.build_generation_inputs is nodes_mod.build_generation_inputs


# ---------------------------------------------------------------------------
# 3. 不重复 I/O：history 由构造注入
# ---------------------------------------------------------------------------
def test_short_term_memory_does_not_read_redis() -> None:
    """构造过程不得触发任何 Redis 读取——历史由调用方注入。"""
    import app.memory.chat_history as chat_history_mod

    calls: List[str] = []
    original = chat_history_mod.get_history

    async def spy(session_id: str):
        calls.append(session_id)
        return []

    chat_history_mod.get_history = spy  # type: ignore[assignment]
    try:
        stm = memory_pkg.get_short_term_memory("s1", _history(3))
        _ = stm.window
        _ = stm.stats()
    finally:
        chat_history_mod.get_history = original  # type: ignore[assignment]

    assert calls == [], f"对象自己读了 Redis，会造成每轮多一次往返：{calls}"


def test_window_matches_pure_function() -> None:
    """对象走门面与直接调纯函数，结果必须一致。"""
    history = _history(10)
    stm = memory_pkg.get_short_term_memory("s1", history)
    assert stm.window == memory_pkg.build_short_term_window(history)


def test_stats_are_consistent_with_window() -> None:
    """stats 与 window 必须基于同一次裁剪，不能各算各的。"""
    history = _history(10)
    stm = memory_pkg.get_short_term_memory("s1", history)
    stats = stm.stats()
    assert stats["window_messages"] == len(stm.window)
    assert stats["total_messages"] == len(history)
    assert stats["dropped"] == len(history) - len(stm.window)
    assert stats["window_chars"] == sum(len(m["content"]) for m in stm.window)


def test_empty_history_is_safe() -> None:
    stm = memory_pkg.get_short_term_memory("s1", [])
    assert stm.window == []
    assert stm.stats()["window_messages"] == 0


def test_signature_takes_history_as_construction_arg() -> None:
    """history 必须是构造参数而非方法参数——方法参数形态正是当初没人用的样子。"""
    params = inspect.signature(short_term_mod.ShortTermMemory.__init__).parameters
    assert "history" in params, "history 必须是构造参数"
    for name in ("window", "stats"):
        method = getattr(short_term_mod.ShortTermMemory, name)
        target = method.fget if isinstance(method, property) else method
        assert "history" not in inspect.signature(target).parameters, f"{name} 不该再要 history"


def test_no_format_method_by_design() -> None:
    """刻意不提供 format()：它会二次裁剪，让 SHORT_TERM_WINDOW 静默失效。

    裁剪点必须唯一。生产链路是「先 window 裁剪 → 再 generator 格式化」两步。
    """
    assert not hasattr(short_term_mod.ShortTermMemory, "format")


def test_window_is_property_not_method() -> None:
    """window 做成属性：读起来是「这个视图的窗口」，不是一次动作。"""
    assert isinstance(
        inspect.getattr_static(short_term_mod.ShortTermMemory, "window"), property
    )


# ---------------------------------------------------------------------------
# 4. 格式化实现只有一份（曾经三份并存：memory / generator / router_agent）
# ---------------------------------------------------------------------------
def test_format_history_does_not_crop() -> None:
    """format_history 只格式化，绝不再裁一刀。

    裁剪点必须唯一：本函数若二次裁剪，上游 SHORT_TERM_WINDOW 会静默失效。
    """
    history = _history(20)
    text = memory_pkg.format_history(history)
    assert text.count("个问题") == 20, "format_history 不该再裁剪"
    assert text.count("个回答") == 20


def test_generator_delegates_to_memory_formatter(monkeypatch) -> None:
    """generator 必须委托 memory 的格式化实现，而不是自己留一份。"""
    from app.rag import generator

    calls: List[int] = []
    original = short_term_mod.format_history

    def spy(history):
        calls.append(len(history))
        return original(history)

    monkeypatch.setattr(short_term_mod, "format_history", spy)
    text = generator._format_history(_history(4))

    assert calls == [8], f"generator 未走 memory 的格式化实现：{calls}"
    assert "第 0 个问题" in text


def test_two_formatter_outputs_are_identical() -> None:
    """两条路径的输出必须逐字一致——若不一致说明有人又写了一份。"""
    from app.rag import generator

    history = _history(5)
    assert generator._format_history(history) == memory_pkg.format_history(history)
    assert memory_pkg.format_history([]) == "（无历史对话）"


# ---------------------------------------------------------------------------
# 4b. 路由层也是同一份（曾经是**第三份**独立实现，见下）
# ---------------------------------------------------------------------------
def test_router_delegates_to_memory_formatter(monkeypatch) -> None:
    """路由看历史用的也是同一份格式化实现，只是**取得更少轮数**。

    路由只为一件事看历史：消解代词（「他的邮箱是多少」里的"他"）。
    所以它该做的是"少取几轮"，而不是"另写一套拼行逻辑"。
    """
    from app.core import router_agent

    calls: List[int] = []
    original = short_term_mod.format_history

    def spy(history):
        calls.append(len(history))
        return original(history)

    monkeypatch.setattr(short_term_mod, "format_history", spy)
    text = router_agent._format_history(_history(10))

    assert calls == [8], f"路由未走 memory 的格式化实现：{calls}"
    assert "第 9 个问题" in text, "路由应当能看到最近一轮"
    assert "第 0 个问题" not in text, "路由只该取最近 4 轮（_ROUTER_HISTORY_TURNS）"


def test_router_and_generator_see_the_same_text() -> None:
    """同一条历史，路由与生成拿到的**文本逐字相同**。

    这条是"格式化第二处实现"的探测器：谁在路由里另写一份拼行逻辑，
    换行方式、角色前缀、空历史哨兵里的任何一个就会对不上。

    ⚠️ 样本必须**含长消息与内嵌换行**，否则本条测不出东西：
    两份实现的分歧点正是"每行截断 200 字"与"把换行压成空格"。
    拿几条短句去比，两份实现的输出会**恰好相同**，护栏恒真——
    这不是假设，是本用例第一版真实发生过的（把修复退回时它没变红）。
    """
    from app.core import router_agent
    from app.rag import generator

    history = [
        {"role": "user", "content": "长" * 300},
        {"role": "assistant", "content": "第一行\n第二行"},
        {"role": "user", "content": "短句"},
        {"role": "assistant", "content": "答"},
    ]
    assert router_agent._format_history(history) == generator._format_history(history)


def test_empty_history_sentinel_has_one_source() -> None:
    """空历史的哨兵文案也只有一个来源。

    曾经是「（无）」（路由那份）与「（无历史对话）」（memory 那份）两种写法——
    两份实现的最早裂缝就是从这种"无关紧要的小字"开始的。
    """
    from app.core import router_agent

    assert router_agent._format_history([]) == memory_pkg.format_history([])
    assert router_agent._format_history(None) == memory_pkg.format_history([])
