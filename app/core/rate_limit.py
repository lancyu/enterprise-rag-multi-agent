"""入站限流 —— 内存滑动窗口，按 IP 维度。

为什么用内存滑动窗口而非 Redis 令牌桶：
    demo 是单机部署，内存滑动窗口零依赖、零运维，Redis 不可用时还要处理
    降级分支，属过度设计。若未来多机部署再换 Redis 令牌桶（接口不变）。

为什么默认 20 次/分钟：
    账号上游限流是 RPM=3（免费档），单用户高频调试时很容易把额度打满，
    入站限流要能拦住「同 IP 无脑刷」，但阈值不能太低以免误伤正常提问。
    20 次/分钟是按 IP 的粗粒度保护，真正精细的配额仍由上游 RPM 兜底。
"""
import threading
import time
from collections import defaultdict, deque
from typing import DefaultDict, Deque

from app import config
from app.core.errors import RateLimitExceeded

RATE_LIMIT_PER_MINUTE: int = int(getattr(config, "RATE_LIMIT_PER_MINUTE", 20))
_WINDOW_SECONDS: float = 60.0


class RateLimiter:
    """按 key（IP）维度的滑动窗口限流器。"""

    def __init__(self, per_minute: int = RATE_LIMIT_PER_MINUTE):
        self.per_minute = per_minute
        self._hits: DefaultDict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """判断该 key 当前是否允许通过；允许则记录一次命中。"""
        now = time.time()
        with self._lock:
            window = self._hits[key]
            # 滑出窗口外的旧命中丢弃
            while window and now - window[0] > _WINDOW_SECONDS:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True


_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def check_rate_limit(key: str) -> None:
    """检查限流，超限抛 RateLimitExceeded。"""
    if not get_rate_limiter().allow(key):
        raise RateLimitExceeded()
