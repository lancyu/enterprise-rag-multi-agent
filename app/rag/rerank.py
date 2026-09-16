"""向后兼容封装 —— Rerank 精排提供者已迁移到 app/providers/rerank.py。

本文件保留原有 import 路径（``from app.rag.rerank import rerank_docs`` 等），
内部直接 re-export 新实现，新旧路径完全等价。新代码请直接使用 app.providers。
"""
from app.providers.rerank import rerank_docs, rerank_info

__all__ = ["rerank_docs", "rerank_info"]
