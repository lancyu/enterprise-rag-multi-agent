"""RAG 引擎统一入口 —— 五层架构的兼容门面。

历史版本把「加载、切片、向量化、检索、兜底」全部塞在一个文件里，
导致任何一次调优都要在单文件里做全局改动，且无法归因问题属于哪个环节。

现版本按「数据准备 → 索引构建 → 检索优化 → 生成控制 → 评估迭代」五层重构
（见 app/rag/ 包）。本模块保留原有对外函数签名，内部委托给各层实现，
使 API 层、工作流节点、自检模块无需改动即可完成升级。
"""
from typing import Any, Dict, List, Optional

from app import config
from app.db.vector_db import get_vector_store
from app.rag import evaluator, generator, indexer, prepare, retriever

# ============================================================
# L1 + L2：索引侧
# ============================================================
def build_index() -> Dict[str, Any]:
    """全量重建向量索引（L1 数据准备 + L2 索引构建）。"""
    return indexer.build_index()


def add_document(file_name: str, content: str) -> int:
    """增量入库单篇文档（幂等覆盖）。"""
    return indexer.add_document(file_name, content)


def delete_document(file_name: str) -> int:
    """删除指定文档的全部片段。"""
    return indexer.delete_document(file_name)


# ============================================================
# L3：检索
# ============================================================
def retrieve_knowledge_docs(
    query: str,
    top_k: Optional[int] = None,
    allowed_sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """混合检索（L3）：多路召回 → RRF 融合 → 去重 → 软回退。

    Args:
        allowed_sources: 来源白名单，非空时只返回该集合内的文档来源（按部门/来源做知识隔离）。

    Returns:
        [{content, source, score, lexical, fused, fallback, chunk_index}, ...]
    """
    return retriever.retrieve(query, top_k=top_k, allowed_sources=allowed_sources)


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """纯语义检索（供知识库管理面板调试使用）。"""
    return retriever.retrieve(query, top_k=top_k)


# ============================================================
# L4：生成
# ============================================================
def generate_answer(
    user_query: str,
    docs: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    tool_result: Optional[str] = None,
):
    """受控生成（L4）：引用溯源 + 置信度拒答。

    ``chat_history`` 接受**未裁剪**的完整历史——本函数是 RAG 引擎的对外
    统一入口，调用方不应被迫了解窗口参数，故在此统一裁剪一次。
    这是「谁负责裁剪」的唯一定论：门面层裁剪，生成层只做格式化。
    """
    from app.memory import build_short_term_window

    return generator.generate_answer(
        user_query, docs, build_short_term_window(chat_history or []), tool_result
    )


# ============================================================
# 统计与评估
# ============================================================
def get_stats() -> Dict[str, Any]:
    """知识库统计信息（含五层状态快照）。"""
    store = get_vector_store()
    return {
        "total_chunks": store.count(),
        "sources": store.list_sources(),
        "config": {
            "top_k": config.SIMILARITY_TOP_K,
            "score_threshold": config.effective_score_threshold(),
            "fallback_min_score": config.effective_fallback_min(),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        "layers": {
            "L1_prepare": prepare.get_last_stats(),
            "L2_index": indexer.get_index_stats(),
            "L3_retrieval": retriever.retrieval_stats(),
            "L4_generation": generator.generator_stats(),
            "L5_feedback": evaluator.feedback_stats(),
        },
        "architecture": ["数据准备", "索引构建", "检索优化", "生成控制", "评估迭代"],
    }


def record_feedback(session_id: str, question: str, answer: str, rating: str, comment: str = "") -> bool:
    """记录用户反馈（L5）。"""
    return evaluator.record_feedback(session_id, question, answer, rating, comment)
