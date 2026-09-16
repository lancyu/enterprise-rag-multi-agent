"""Redis 连接池 —— 不可用时自动降级为进程内内存缓存。

对上层暴露与 redis.asyncio 兼容的最小方法集：
    rpush / lrange / ltrim / delete / expire / ttl / incr / ping / keys / dbsize

增删命令时必须同步更新上面这份清单与 `MemoryRedis` 的实现，
否则真实 Redis 环境正常、降级环境才炸 AttributeError（最难排查的一类问题）。
"""
import asyncio
import time
from typing import Dict, List, Optional

from app import config
from app.utils.logger import logger


class MemoryRedis:
    """进程内内存缓存（带 TTL），接口对齐 redis.asyncio，用于无 Redis 环境降级。

    内部用两个字典分别承载 Redis 的两种值类型——列表与字符串。
    真实 Redis 共用同一 keyspace，故 `keys`/`dbsize`/`delete` 必须同时看两边。
    """

    def __init__(self):
        self._data: Dict[str, List[str]] = {}
        self._kv: Dict[str, str] = {}
        self._expire_at: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def _drop_if_expired(self, key: str) -> None:
        expire = self._expire_at.get(key)
        if expire and time.time() > expire:
            self._data.pop(key, None)
            self._kv.pop(key, None)
            self._expire_at.pop(key, None)

    async def rpush(self, key: str, value: str) -> int:
        async with self._lock:
            await self._drop_if_expired(key)
            self._data.setdefault(key, []).append(value)
            return len(self._data[key])

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        async with self._lock:
            await self._drop_if_expired(key)
            items = self._data.get(key, [])
            if end == -1:
                return list(items[start:])
            return list(items[start: end + 1])

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        async with self._lock:
            await self._drop_if_expired(key)
            items = self._data.get(key, [])
            self._data[key] = items[start:] if end == -1 else items[start: end + 1]
            return True

    async def incr(self, key: str, amount: int = 1) -> int:
        """原子自增（`redis.asyncio` 同名方法）。

        用于给会话消息分配单调递增序号。**必须是原子的**：并发写入同一会话时，
        非原子的「读尾部 + 加一」会产生重复序号，进而让按序号归档的记忆整理
        重复消费或漏消费。
        """
        async with self._lock:
            await self._drop_if_expired(key)
            try:
                value = int(self._kv.get(key, "0")) + amount
            except (TypeError, ValueError):
                value = amount
            self._kv[key] = str(value)
            return value

    async def delete(self, key: str) -> int:
        async with self._lock:
            existed = key in self._data or key in self._kv
            self._data.pop(key, None)
            self._kv.pop(key, None)
            self._expire_at.pop(key, None)
            return int(existed)

    async def expire(self, key: str, seconds: int) -> bool:
        async with self._lock:
            self._expire_at[key] = time.time() + seconds
            return True

    async def ttl(self, key: str) -> int:
        """剩余存活秒数（-1 表示无过期时间）。

        注意：本方法与 `dbsize` 当前**没有调用方**，但刻意保留——`MemoryRedis` 是
        `redis.asyncio` 的降级替身，价值在于「可替换性」：若删掉它们，日后有人写
        `await redis.ttl(k)` 时，真实 Redis 环境正常、无 Redis 的降级环境才炸
        AttributeError，属于最难排查的环境相关 bug。故保持模块 docstring 中
        声明的「最小方法集」完整。已在 tests/deadcode_allowlist.py 登记豁免。
        """
        async with self._lock:
            expire = self._expire_at.get(key)
            if not expire:
                return -1
            return max(-1, int(expire - time.time()))

    async def keys(self, pattern: str = "*") -> List[str]:
        async with self._lock:
            prefix = pattern.rstrip("*")
            return [k for k in {**self._data, **self._kv} if k.startswith(prefix)]

    async def dbsize(self) -> int:
        """键总数（`redis.asyncio` 同名方法）；保留理由见上方 `ttl`。"""
        async with self._lock:
            return len({**self._data, **self._kv})

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


_redis_client = None
_redis_mode: Optional[str] = None


async def get_redis():
    """获取全局 Redis 客户端（真实 Redis 优先，失败降级为内存缓存）。"""
    global _redis_client, _redis_mode
    if _redis_client is not None:
        return _redis_client

    try:
        import redis.asyncio as aioredis

        client = aioredis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD or None,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await client.ping()
        _redis_client = client
        _redis_mode = "redis"
        logger.info("Redis 连接成功：%s:%s/%s", config.REDIS_HOST, config.REDIS_PORT, config.REDIS_DB)
        return _redis_client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 不可用（%s），自动降级为进程内内存缓存", type(exc).__name__)

    _redis_client = MemoryRedis()
    _redis_mode = "memory"
    return _redis_client


def get_redis_mode() -> str:
    return _redis_mode or "unknown"


async def close_redis() -> None:
    global _redis_client, _redis_mode
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:  # noqa: BLE001
            pass
    _redis_client = None
    _redis_mode = None
