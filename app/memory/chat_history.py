"""短期记忆 · 持久化层 —— 多轮对话历史存取（Redis 优先，自动降级内存）。

位置（与长期记忆对称）：

    长期记忆：store.py（文件 I/O）→ long_term.py（LongTermMemory）→ 门面
    短期记忆：**本模块**（Redis I/O）→ short_term.py（ShortTermMemory）→ 门面

    本模块原先在 ``app/core/chat_memory.py``。持久化层跑出 ``app/memory/`` 包，
    会让 ``short_term.py`` 只剩裁剪算法、领域对象无处落地（``ShortTermMemory``
    因此长期无人调用）。2026-09-11 归位，让长短期四层完全对齐。

能力：
1. 持久化存储：按 session_id 维度保存会话消息
2. 自动裁剪：仅保留最近 N 轮（一问一答 = 2 条）
3. 过期清理：会话 24 小时自动失效
4. 单调序号：每条消息带会话内唯一的 seq，供记忆整理做**增量**归档

关于 seq（为什么必须有）：
    历史列表会被 ltrim 封顶在固定长度，因此「长度」无法表达进度——
    一旦会话饱和，每一轮读到的长度都相同，归档逻辑无从判断哪些内容
    已经归档过，只能整段重来。seq 是那个能跨越窗口滑动的稳定锚点。
"""
import json
from typing import Dict, List

from app import config
from app.db.redis_db import get_redis
from app.utils.logger import logger

KEY_PREFIX = "chat:history:"
SEQ_PREFIX = "chat:seq:"


async def save_message(session_id: str, role: str, content: str) -> None:
    """保存单条消息到会话历史，并分配会话内单调递增的 seq。"""
    from app.utils.validator import sanitize_text

    redis = await get_redis()
    key = f"{KEY_PREFIX}{session_id}"
    seq_key = f"{SEQ_PREFIX}{session_id}"
    # INCR 是原子的：并发写同一会话不会拿到重复 seq
    seq = await redis.incr(seq_key)
    msg = json.dumps(
        {"role": role, "content": sanitize_text(content), "seq": seq}, ensure_ascii=False
    )
    await redis.rpush(key, msg)
    await redis.ltrim(key, -config.MAX_CHAT_HISTORY * 2, -1)  # 保留最近 N 轮
    await redis.expire(key, config.SESSION_TTL)
    await redis.expire(seq_key, config.SESSION_TTL)


async def get_history(session_id: str) -> List[Dict[str, str]]:
    """获取会话完整对话历史（含 seq 字段）。"""
    redis = await get_redis()
    raw = await redis.lrange(f"{KEY_PREFIX}{session_id}", 0, -1)
    history = []
    for item in raw:
        try:
            history.append(json.loads(item))
        except json.JSONDecodeError:
            logger.warning("会话历史反序列化失败，已跳过：%s", item[:50])
    return history


async def clear_history(session_id: str) -> None:
    """清除指定会话历史（含序号计数器，避免重建会话时 seq 从中间续上）。"""
    redis = await get_redis()
    await redis.delete(f"{KEY_PREFIX}{session_id}")
    await redis.delete(f"{SEQ_PREFIX}{session_id}")
    logger.info("已清除会话历史：%s", session_id)
