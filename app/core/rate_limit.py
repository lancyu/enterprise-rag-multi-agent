"""入站限流 —— 内存滑动窗口，按 IP 维度。

为什么用内存滑动窗口而非 Redis 令牌桶：
    demo 是单机部署，内存滑动窗口零依赖、零运维，Redis 不可用时还要处理
    降级分支，属过度设计。若未来多机部署再换 Redis 令牌桶（接口不变）。

为什么默认 20 次/分钟：
    入站限流要拦住「同 IP 无脑刷」，但阈值不能太低以免误伤正常提问。

    ⚠️ **这个数字已经没有标定依据了**：它当初是按「上游账号 RPM 极低（3 次/分钟）」
    倒推出来的，而那个前提早已不成立（账号已更换，配置里也没有任何 RPM 项）。
    它现在保护的是**服务端自身**——一次对话要跑检索 + 多次模型调用，成本与并发
    都不低，所以限流仍然必要；但阈值应当按服务端实际能承受的并发**重新定**，
    而不是继续沿用这个失去来源的旧值。
    另注意它限制的是**用户请求数**，不是模型调用数（一次对话内部要发多次）。
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
