"""记忆系统 —— 短期记忆 + 长期记忆。

设计参考 HKUDS/nanobot（47k stars），核心理念是「记忆分层」：

    短期记忆（short_term.py）
        当前会话的活跃对话窗口，滑动窗口 + 字符预算 + 问答成对裁剪。
        特点：快、会过期、只服务于当下这一轮对话。

    长期记忆（long_term.py + store.py）
        MEMORY.md（业务事实） / USER.md（用户画像） / SOUL.md（人格）。
        特点：慢、持久、跨会话累积。

    两阶段流转
        Consolidator（consolidator.py）：对话变长时，把旧消息压缩成归档
            → history.jsonl（追加写 + 游标，机器可增量消费）
        Dream（dream.py）：把归档提炼成长期知识
            → 增量写入 MEMORY.md / USER.md

为什么不每轮对话都读写长期记忆：
    短期记忆在内存/Redis 里，零成本；长期记忆涉及文件 I/O 与（可选的）
    大模型调用。把两者分开，才能让"记住东西"这件事既及时又便宜——
    绝大多数轮次只走短期记忆，长期记忆只在归档与蒸馏时才被触碰。

数据流：
    用户提问 → 短期窗口（裁剪后历史）
             → 长期记忆上下文（MEMORY/USER/SOUL）
             → 组装进 prompt → 生成回答
             → 未归档的新增消息攒够阈值？→ Consolidator 压缩归档
             → 归档攒够一批？→ 自动 Dream 蒸馏成长期知识

    两条闸门都以「新增量」而非「总量」判定：历史窗口被 ltrim 封顶，
    总量在会话饱和后不再变化，用总量做判据会退化成「每轮都触发」。
"""
from typing import Any, Dict, List, Optional

from app import config
from app.memory.consolidator import Consolidator
from app.memory.dream import Dream
from app.memory.long_term import LongTermMemory
from app.memory.short_term import (
    ShortTermMemory,
    build_window,
    format_history,
    select_unarchived,
    should_consolidate,
    split_for_consolidation,
)
from app.memory.store import MemoryStore
from app.utils.logger import logger

# user_id → LongTermMemory 实例缓存，避免每轮请求重复构造与磁盘探测
_LTM_CACHE: Dict[str, LongTermMemory] = {}


def get_long_term_memory(user_id: str = "default") -> LongTermMemory:
    """获取（并缓存）用户的长期记忆实例。"""
    key = user_id or "default"
    if key not in _LTM_CACHE:
        _LTM_CACHE[key] = LongTermMemory(key)
    return _LTM_CACHE[key]


def get_short_term_memory(
    session_id: str,
    history: List[Dict[str, str]],
    max_turns: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> ShortTermMemory:
    """获取会话的短期记忆视图 —— 与 :func:`get_long_term_memory` 对称。

    与长期记忆的关键差异：**不做实例缓存**。长期记忆按 user_id 缓存是因为
    构造要探测磁盘；短期视图只是一次窗口计算的句柄，缓存反而会让 history
    停在旧数据上。history 由调用方注入（GraphState 已在传），不重复读 Redis。
    """
    return ShortTermMemory(session_id, history, max_turns=max_turns, max_chars=max_chars)


# ---------------------------------------------------------------------------
# 供工作流节点调用的高层 API
# ---------------------------------------------------------------------------
def get_memory_context(user_id: str = "default", max_chars: Optional[int] = None) -> str:
    """取长期记忆上下文，用于注入 prompt。"""
    if not config.MEMORY_ENABLED:
        return ""
    try:
        return get_long_term_memory(user_id).get_context(max_chars=max_chars)
    except Exception:  # noqa: BLE001
        # 记忆不可用绝不能阻断主对话链路
        logger.exception("长期记忆读取失败，已跳过记忆注入")
        return ""


def build_short_term_window(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """裁剪出注入 prompt 的短期对话窗口。

    保留为兼容入口：等价于 ``get_short_term_memory(sid, history).window``，
    供不需要会话标识的调用方（如 RAG 引擎的对外函数）直接使用。
    """
    return build_window(history)


def maybe_consolidate(
    user_id: str,
    session_id: str,
    history: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """按需把过期对话压缩归档，并在攒够一批后自动蒸馏进长期记忆。

    绝大多数轮次什么都不做（未达阈值时直接返回 None，零模型调用）。
    """
    if not config.MEMORY_ENABLED or not config.CONSOLIDATE_ENABLED:
        return None
    try:
        result = Consolidator(get_long_term_memory(user_id)).maybe_consolidate(
            history, session_key=session_id
        )
    except Exception:  # noqa: BLE001
        logger.exception("记忆整理失败，已跳过")
        return None

    if result:
        # 归档刚产生新内容，才有必要看一眼是否该蒸馏
        result["dreamed"] = bool(_maybe_auto_dream(user_id))
    return result


def _maybe_auto_dream(user_id: str) -> Optional[Dict[str, Any]]:
    """归档攒够一整批时自动蒸馏，把长期记忆的写入链路闭合成自动的。

    为什么不每次归档都蒸馏：蒸馏要额外一次模型调用。用 DREAM_BATCH_SIZE
    作闸门（积够一批才触发），使一次蒸馏的成本摊薄到很多轮对话上；
    配合 CONSOLIDATE_THRESHOLD，长会话的整体模型开销仍然可控。
    """
    if not config.DREAM_ENABLED:
        return None
    try:
        ltm = get_long_term_memory(user_id)
        if len(ltm.pending_entries()) < config.DREAM_BATCH_SIZE:
            return None
        return Dream(ltm).run()
    except Exception:  # noqa: BLE001
        logger.exception("自动蒸馏失败，已跳过")
        return None


def run_dream(user_id: str = "default", batch_size: Optional[int] = None) -> Dict[str, Any]:
    """手动执行一次记忆蒸馏（忽略批量闸门，供运维/调试随时触发）。"""
    try:
        return Dream(get_long_term_memory(user_id)).run(batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001
        logger.exception("记忆蒸馏失败")
        return {"consumed": 0, "error": str(exc)}


def get_memory_stats(user_id: str = "default") -> Dict[str, Any]:
    """记忆系统状态（供 /stats 展示）。"""
    try:
        stats = get_long_term_memory(user_id).stats()
    except Exception:  # noqa: BLE001
        stats = {"user_id": user_id, "error": "unavailable"}
    stats["enabled"] = config.MEMORY_ENABLED
    stats["consolidate_enabled"] = config.CONSOLIDATE_ENABLED
    stats["dream_enabled"] = config.DREAM_ENABLED
    return stats


__all__ = [
    "MemoryStore",
    "LongTermMemory",
    "ShortTermMemory",
    "Consolidator",
    "Dream",
    "get_long_term_memory",
    "get_short_term_memory",
    "get_memory_context",
    "build_short_term_window",
    "maybe_consolidate",
    "run_dream",
    "get_memory_stats",
    "build_window",
    "format_history",
    "select_unarchived",
    "should_consolidate",
    "split_for_consolidation",
]
