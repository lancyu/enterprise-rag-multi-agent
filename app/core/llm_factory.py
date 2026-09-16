"""向后兼容封装 —— 大模型提供者已迁移到 app/providers/llm.py。

本文件保留原有 import 路径（``from app.core.llm_factory import get_chat_model`` 等），
内部直接 re-export 新实现，新旧路径完全等价。新代码请直接使用 app.providers。

注意：``_is_rate_limit`` 为内部符号，但被自检模块跨模块引用，故一并 re-export，
保持向后兼容。（``_RateLimitRetryModel`` 已随自造限流包装层一起移除。）
"""
from app.providers.llm import (
    MockChatModel,
    _is_rate_limit,
    get_chat_model,
    get_llm_mode,
    reset_chat_model,
)

__all__ = [
    "MockChatModel",
    "_is_rate_limit",
    "get_chat_model",
    "get_llm_mode",
    "reset_chat_model",
]
