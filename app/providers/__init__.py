"""统一模型抽象层 —— 项目里所有「模型相关」能力的唯一入口。

三类模型 provider 均为「配置驱动」：切换模型只需改 .env（或环境变量），无需改动任何
业务代码。各 provider 的实现见同目录：

    - embeddings.py  向量化（Embedder：api / local-hash 双模式降级）
    - llm.py         大模型生成/分类（真实 / Mock 双模式降级）
    - rerank.py      精排（可选，sentence-transformers cross-encoder）

业务代码（RAG 五层、图节点、API）建议优先从这里 import，例如：

    from app.providers import get_embeddings, get_chat_model, rerank_docs

历史原因，`app/core/llm_factory.py`、`app/rag/rerank.py` 仍保留为向后兼容的薄封装
（re-export 本包同名符号）。`app/utils/embedding.py` **已删除**（P1-6 解环时一并处理：
它的唯一存在理由是让 `app/utils` 这一最底层去 import `app/providers`，而配置层为了读
「当前 embedding 模式」又得经过它，形成环）。
"""
from app.providers.base import Embedder, Reranker
from app.providers.embeddings import (
    APIEmbeddings,
    CachedAPIEmbeddings,
    LocalHashEmbeddings,
    get_embeddings,
    get_embedding_mode,
    reset_embeddings,
)
from app.providers.llm import (
    MockChatModel,
    get_chat_model,
    get_llm_mode,
    reset_chat_model,
)
from app.providers.rerank import rerank_docs, rerank_info

__all__ = [
    # 抽象接口
    "Embedder",
    "Reranker",
    # Embedding
    "APIEmbeddings",
    "CachedAPIEmbeddings",
    "LocalHashEmbeddings",
    "get_embeddings",
    "get_embedding_mode",
    "reset_embeddings",
    # LLM
    "MockChatModel",
    "get_chat_model",
    "get_llm_mode",
    "reset_chat_model",
    # Rerank
    "rerank_docs",
    "rerank_info",
]
