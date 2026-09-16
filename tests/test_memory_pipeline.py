"""记忆链路回归测试 —— 钉住三处曾经静默失效的行为。

这三个缺陷的共同特征是「不报错、只是悄悄做错事」，因此必须有测试兜住，
否则任何一次重构都可能让它们复发：

1. **裁剪点必须唯一**——上游 `build_window` 裁完后，生成层不得再砍一刀，
   否则 `SHORT_TERM_WINDOW` 名存实亡（改大改小都只有 3 轮生效）。
2. **归档必须按「新增量」判定**——历史列表被 ltrim 封顶，用总长度做阈值
   会让长会话退化成「每轮都触发一次模型调用」。
3. **长期记忆的写入链路必须自动闭合**——归档攒够一批就该自动蒸馏，
   不能依赖人工调接口，否则 MEMORY.md / USER.md 永远是空的。
"""
import asyncio
from typing import Any, Dict, List

from app import config
from app.memory import short_term
from app.memory.consolidator import Consolidator
from app.memory.dream import Dream
from app.memory.long_term import LongTermMemory
from app.memory.store import MemoryStore


def _dialog(turns: int) -> List[Dict[str, Any]]:
    """造 N 轮对话，seq 从 1 起连续编号（与真实 save_message 一致）。"""
    msgs: List[Dict[str, Any]] = []
    for i in range(turns):
        msgs.append({"role": "user", "content": f"问题{i + 1}", "seq": len(msgs) + 1})
        msgs.append({"role": "assistant", "content": f"回答{i + 1}", "seq": len(msgs) + 1})
    return msgs


# ---------------------------------------------------------------------------
# 1. 上下文裁剪点唯一
# ---------------------------------------------------------------------------
def test_format_history_does_not_re_truncate() -> None:
    """生成层只做格式化，不再二次裁剪。

    回归：`_format_history` 曾硬编码 `history[-6:]`，把上游按
    `SHORT_TERM_WINDOW=6` 轮裁好的 12 条又砍成 6 条（3 轮）。
    """
    from app.rag import generator

    history = _dialog(6)  # 12 条
    text = generator._format_history(history)
    assert text.count("问题") == 6
    assert text.count("回答") == 6


def test_short_term_window_is_respected(monkeypatch) -> None:
    """`SHORT_TERM_WINDOW` 是唯一生效的轮数开关。"""
    monkeypatch.setattr(config, "SHORT_TERM_WINDOW", 3)
    monkeypatch.setattr(config, "SHORT_TERM_MAX_CHARS", 100000)
    window = short_term.build_window(_dialog(10))
    assert len(window) == 6, "3 轮应保留 6 条消息"


def test_char_budget_is_a_second_line_of_defense(monkeypatch) -> None:
    """轮数相同但每条很长时，字符预算必须兜住。"""
    monkeypatch.setattr(config, "SHORT_TERM_WINDOW", 6)
    monkeypatch.setattr(config, "SHORT_TERM_MAX_CHARS", 100)
    history = [{"role": "user", "content": "长" * 60, "seq": 1},
               {"role": "assistant", "content": "长" * 60, "seq": 2}]
    window = short_term.build_window(history)
    assert sum(len(m["content"]) for m in window) <= 100


# ---------------------------------------------------------------------------
# 2. 归档按新增量判定
# ---------------------------------------------------------------------------
def test_select_unarchived_uses_seq_not_length() -> None:
    """只有 seq 超过水位的消息才算待归档。"""
    pending = short_term.select_unarchived(_dialog(5), last_archived_seq=6)
    assert [m["seq"] for m in pending] == [7, 8, 9, 10]


def test_messages_without_seq_are_never_archived() -> None:
    """客户端直传的历史没有 seq，不应被服务端归档。"""
    history = [{"role": "user", "content": "外来的"}, {"role": "assistant", "content": "外来的"}]
    assert short_term.select_unarchived(history, last_archived_seq=0) == []


def test_archive_watermark_is_persisted_per_session(tmp_path) -> None:
    """归档水位按会话区分，避免同一用户的多会话互相污染。"""
    store = MemoryStore("u-test", base_dir=tmp_path)
    store.append_history("批次A", session_key="s1", last_seq=8)
    store.append_history("批次B", session_key="s2", last_seq=3)
    assert store.get_last_archived_seq("s1") == 8
    assert store.get_last_archived_seq("s2") == 3
    assert store.get_last_archived_seq("s3") == 0


def test_consolidation_is_incremental(tmp_path, monkeypatch) -> None:
    """同一批内容只能被归档一次：水位推进后不再重复归档。"""
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    monkeypatch.setattr(config, "CONSOLIDATE_ENABLED", True)
    monkeypatch.setattr(config, "CONSOLIDATE_THRESHOLD", 4)
    monkeypatch.setattr(config, "CONSOLIDATE_KEEP_TAIL", 2)
    # 不打真实模型：摘要内容不是本用例的观测对象
    monkeypatch.setattr(Consolidator, "_summarize_with_llm", lambda self, msgs: "摘要占位")

    cons = Consolidator(LongTermMemory("u-test", store=MemoryStore("u-test", base_dir=tmp_path)))

    first = cons.maybe_consolidate(_dialog(3), session_key="s1")  # 6 条待归档 >= 4
    assert first is not None
    assert first["last_seq"] == 4, "6 条里尾部 2 条应留下，归档到 seq=4"

    # 同样的历史再喂一次：seq>4 的只剩 2 条，不足阈值 → 不再触发
    assert cons.maybe_consolidate(_dialog(3), session_key="s1") is None


def test_saturated_session_does_not_consolidate_every_turn(tmp_path, monkeypatch) -> None:
    """长会话的核心回归：窗口饱和后不得每轮都触发归档。

    回归：旧实现 `CONSOLIDATE_THRESHOLD`(=20) 与 `MAX_CHAT_HISTORY*2`(=20)
    数值相等，而历史被 ltrim 封顶后长度恒为 20，`len(history) > 20` 永远成立，
    于是每一轮都调一次模型做摘要，并反复写高度重叠的归档。
    """
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    monkeypatch.setattr(config, "CONSOLIDATE_ENABLED", True)
    monkeypatch.setattr(Consolidator, "_summarize_with_llm", lambda self, msgs: "摘要占位")

    cons = Consolidator(LongTermMemory("u-test", store=MemoryStore("u-test", base_dir=tmp_path)))
    capacity = config.MAX_CHAT_HISTORY * 2  # 模拟 ltrim 封顶后的窗口长度

    triggered = sum(
        1
        for turn in range(1, 41)
        if cons.maybe_consolidate(_dialog(turn)[-capacity:], session_key="s1")
    )

    # 每攒够 CONSOLIDATE_THRESHOLD 条新增才归档一次；旧行为是 ~39 次
    assert triggered <= 10, f"40 轮里归档触发了 {triggered} 次，疑似退化为每轮触发"


def test_no_consolidation_when_history_has_no_new_content(tmp_path, monkeypatch) -> None:
    """没有新增内容时零开销返回 None（绝大多数轮次走这条路）。"""
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    monkeypatch.setattr(config, "CONSOLIDATE_ENABLED", True)
    cons = Consolidator(LongTermMemory("u-test", store=MemoryStore("u-test", base_dir=tmp_path)))
    assert cons.maybe_consolidate(_dialog(1), session_key="s1") is None


# ---------------------------------------------------------------------------
# 3. 长期记忆写入链路自动闭合
# ---------------------------------------------------------------------------
def test_save_message_assigns_monotonic_seq(monkeypatch) -> None:
    """每条消息拿到严格递增的 seq —— 归档水位依赖它保持单调。"""
    from app.memory import chat_history
    from app.db import redis_db

    monkeypatch.setattr(redis_db, "_redis_client", redis_db.MemoryRedis())

    async def _run() -> List[Dict[str, Any]]:
        await chat_history.save_message("s1", "user", "你好")
        await chat_history.save_message("s1", "assistant", "您好")
        return await chat_history.get_history("s1")

    history = asyncio.run(_run())
    assert [m["seq"] for m in history] == [1, 2]


def test_auto_dream_fires_when_batch_is_ready(tmp_path, monkeypatch) -> None:
    """归档攒够一批后自动蒸馏，长期记忆不再依赖人工触发接口。"""
    from app import memory as memory_pkg

    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    monkeypatch.setattr(config, "DREAM_ENABLED", True)
    monkeypatch.setattr(config, "DREAM_BATCH_SIZE", 2)

    ltm = LongTermMemory("u-test", store=MemoryStore("u-test", base_dir=tmp_path))
    ltm.archive("第一批归档", session_key="s1", last_seq=2)
    ltm.archive("第二批归档", session_key="s1", last_seq=4)

    monkeypatch.setattr(memory_pkg, "get_long_term_memory", lambda user_id="default": ltm)
    # 走规则降级路径，避免真实模型调用
    monkeypatch.setattr(Dream, "_extract_with_llm", lambda self, entries: None)

    result = memory_pkg._maybe_auto_dream("u-test")
    assert result is not None
    assert result["consumed"] == 2
    assert result["degraded"] is True


def test_auto_dream_waits_until_batch_is_full(tmp_path, monkeypatch) -> None:
    """没攒够一整批就不蒸馏，避免把模型调用摊到每一轮。"""
    from app import memory as memory_pkg

    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    monkeypatch.setattr(config, "DREAM_ENABLED", True)
    monkeypatch.setattr(config, "DREAM_BATCH_SIZE", 5)

    ltm = LongTermMemory("u-test", store=MemoryStore("u-test", base_dir=tmp_path))
    ltm.archive("只有一条", session_key="s1", last_seq=2)
    monkeypatch.setattr(memory_pkg, "get_long_term_memory", lambda user_id="default": ltm)

    assert memory_pkg._maybe_auto_dream("u-test") is None
