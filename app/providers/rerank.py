"""Rerank 精排提供者（可选增强）—— cross-encoder 对召回候选做二次评分重排。

与 reorder 正交，二者解决不同问题：
    - reorder（app/rag/reorder.py）：零模型调用，只把「1,3,5,…,6,4,2」交错重排，
      缓解长上下文的 lost-in-the-middle，但不改变片段的相关性排序；
    - rerank（本模块）：cross-encoder 对 (query, doc) 逐对打分，真正提升
      「该进 Top-K 却排在后面」的片段，代价是多一次模型前向（CPU 上约 80~120ms）。

参考 dify 的 RerankRunnerFactory（模型精排 / 加权重排双模式）与 LlamaIndex 的
SentenceTransformerRerank：候选粗召回后取 top-N 交给 cross-encoder 精排，再截 top-k。

依赖 sentence-transformers（可选）：未安装或加载失败时自动降级为「不做精排」，
绝不拖垮检索主链路。默认关闭（config.RERANK_ENABLED=False）。
"""
import threading
from typing import Any, Dict, List, Optional

from app import config
from app.providers.base import Reranker
from app.utils.logger import logger


class SentenceTransformerReranker(Reranker):
    """基于 sentence-transformers cross-encoder 的精排实现。

    模型名由 config.RERANK_MODEL 指定（任意 sentence-transformers 兼容 cross-encoder），
    代码不识别具体模型名——换重排模型改配置即可。
    """

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """对前 top_n 条候选精排，重排后按 rerank_score 降序返回（其余原样拼回）。"""
        candidates, rest = docs[:top_n], docs[top_n:]
        pairs = [(query, d.get("content", "")) for d in candidates]
        scores = self._model.predict(pairs)
        for d, s in zip(candidates, scores):
            d["rerank_score"] = round(float(s), 4)
        candidates.sort(key=lambda d: d["rerank_score"], reverse=True)
        return candidates + rest


_reranker: Optional[SentenceTransformerReranker] = None
_reranker_tried = False
_reranker_lock = threading.Lock()


def _load_reranker() -> Optional[SentenceTransformerReranker]:
    """懒加载 cross-encoder 单例；失败返回 None 并记住，避免每次请求重复尝试。"""
    global _reranker, _reranker_tried
    if not config.RERANK_ENABLED:
        # 未启用时不加载模型（避免 rerank_info 查询时也白加载一次重排模型）
        return None
    if _reranker is not None or _reranker_tried:
        return _reranker
    with _reranker_lock:
        if _reranker is not None or _reranker_tried:
            return _reranker
        _reranker_tried = True
        try:
            _reranker = SentenceTransformerReranker(config.RERANK_MODEL)
            logger.info("Rerank 精排已启用：model=%s", config.RERANK_MODEL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rerank 精排不可用（%s），已降级为不精排。可 pip install sentence-transformers 启用", exc)
            _reranker = None
    return _reranker


def rerank_docs(query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """对候选 docs 做 cross-encoder 精排。

    只对前 top_n 条候选精排（控制计算量），重排后按 rerank_score 降序返回。
    精排不可用或失败时原样返回 docs（保住原 RRF 顺序），绝不因此丢片段。

    Args:
        query: 用户提问
        docs: 已按 fused 降序的候选列表（dict 含 content 等字段）
        top_n: 参与精排的候选条数
    """
    if not config.RERANK_ENABLED or not docs:
        return docs
    model = _load_reranker()
    if model is None:
        return docs
    try:
        return model.rerank(query, docs, top_n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rerank 精排失败，回退原顺序：%s", exc)
        return docs


def rerank_info() -> Dict[str, Any]:
    """rerank 配置快照（供观测接口）。"""
    return {
        "enabled": bool(config.RERANK_ENABLED),
        "model": config.RERANK_MODEL,
        "top_n": config.RERANK_TOP_N,
        "available": _load_reranker() is not None,
    }
