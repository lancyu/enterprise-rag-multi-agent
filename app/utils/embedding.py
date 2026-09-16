"""向后兼容封装 —— Embedding 提供者已迁移到 app/providers/embeddings.py。

本文件保留原有 import 路径（``from app.utils.embedding import get_embeddings`` 等），
内部直接 re-export 新实现，新旧路径完全等价。新代码请直接使用 app.providers。
"""
from app.providers.embeddings import (
    APIEmbeddings,
    CachedAPIEmbeddings,
    LocalHashEmbeddings,
    get_embeddings,
    get_embedding_mode,
    reset_embeddings,
)

__all__ = [
    "APIEmbeddings",
    "CachedAPIEmbeddings",
    "LocalHashEmbeddings",
    "get_embeddings",
    "get_embedding_mode",
    "reset_embeddings",
]
