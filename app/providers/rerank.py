"""Rerank 精排提供者（可选增强）—— cross-encoder 对召回候选做二次评分重排。

与 reorder 正交，二者解决不同问题：
    - reorder（app/rag/reorder.py）：零模型调用，只把「1,3,5,…,6,4,2」交错重排，
      缓解长上下文的 lost-in-the-middle，但不改变片段的相关性排序；
    - rerank（本模块）：cross-encoder 对 (query, doc) 逐对打分，真正提升
      「该进 Top-K 却排在后面」的片段，代价是多一次模型前向（CPU 上约 80~120ms）。

参考 dify 的 RerankRunnerFactory（模型精排 / 加权重排双模式）与 LlamaIndex 的
SentenceTransformerRerank：候选粗召回后取 top-N 交给 cross-encoder 精排，再截 top-k。

两种后端，按配置自动选（与 embedding「有 Key 用 API、没 Key 用本地」同一套路）：

    - ``APIReranker``   调 OpenAI 兼容的 ``/rerank`` 接口（SiliconFlow / Cohere / vLLM
                        等均提供）。**这是本项目的实际路径**：本机没装
                        sentence-transformers，只靠本地后端的话「支持 rerank」这句话
                        在本机永远是假的（P1-3 要解决的正是这个"空壳"）。
    - ``SentenceTransformerReranker``  本地 cross-encoder，仅在 ``RERANK_API_KEY`` /
                        ``RERANK_BASE_URL`` 都没配时才会走到。

两者都不识别具体模型名：换重排模型改 ``RERANK_MODEL`` 即可，代码不做任何
``"bge" in model`` 之类的判断（那是把特定模型知识写死在代码里）。
"""
import threading
from typing import Any, Dict, List, Optional

from app import config
from app.providers.base import Reranker
from app.utils.logger import logger


class APIReranker(Reranker):
    """OpenAI 兼容 ``/rerank`` 接口的精排实现（SiliconFlow / Cohere / vLLM 等）。

    请求体 ``{"model", "query", "documents", "top_n"}``；
    响应体 ``{"results": [{"index": i, "relevance_score": s}]}``。

    ⚠️ **只按分数重排，不拿返回值替换内容**。部分服务会在 ``results[].document`` 里
    回显（可能是截断的）文本；若拿它覆盖原片段，检索结果的正文本身就取决于服务端
    实现 —— 而下游（生成、引用页码）全都依赖原始 chunk 文本。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int):
        import httpx

        self._url = base_url.rstrip("/") + "/rerank"
        self._model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # 复用连接池，别每次请求新建一条连接（这条路径在检索主链路上）
        self._client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        candidates, rest = docs[:top_n], docs[top_n:]
        payload = {
            "model": self._model,
            "query": query,
            # ① **原始 chunk 文本**，不是分词结果：cross-encoder 吃的是自然语言，
            # 喂分词碎片会让打分失去意义（P1-3 的三条约束之一，有用例钉住）。
            "documents": [d.get("content", "") for d in candidates],
            "top_n": len(candidates),
        }
        resp = self._client.post(self._url, json=payload, headers=self._headers)
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []

        scored: Dict[int, float] = {}
        for item in results:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                scored[idx] = float(item.get("relevance_score", 0.0))

        # 服务端没返回分的候选保持原顺序排在后面 —— 绝不因为"分没回来"就丢片段。
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        ordered = [candidates[i] for i, _ in ranked]
        got = {i for i, _ in ranked}
        ordered += [d for i, d in enumerate(candidates) if i not in got]

        for d, (_i, s) in zip(ordered, ranked):
            d["rerank_score"] = round(s, 4)
        return ordered + rest


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


_reranker: Optional[Reranker] = None
_reranker_tried = False
_reranker_lock = threading.Lock()


def _build_reranker() -> Optional[Reranker]:
    """按配置建一个精排实例。

    选后端的判据是**有没有配服务端**，不是「本地依赖装没装」——
    后者只有在两者都没配时才作为兜底。反过来写的话，本机没装 sentence-transformers
    就会一路降级到"不精排"，而配置里明明配好了可用的精排服务。
    """
    if config.RERANK_BASE_URL and config.RERANK_API_KEY:
        return APIReranker(
            config.RERANK_BASE_URL, config.RERANK_API_KEY,
            config.RERANK_MODEL, config.RERANK_TIMEOUT,
        )
    return SentenceTransformerReranker(config.RERANK_MODEL)


def _load_reranker() -> Optional[Reranker]:
    """懒加载精排单例；失败返回 None 并记住，避免每次请求重复尝试。"""
    global _reranker, _reranker_tried
    if not config.RERANK_ENABLED:
        # 未启用时不建实例（避免 rerank_info 查询时也白连一次精排服务）
        return None
    if _reranker is not None or _reranker_tried:
        return _reranker
    with _reranker_lock:
        if _reranker is not None or _reranker_tried:
            return _reranker
        _reranker_tried = True
        try:
            _reranker = _build_reranker()
            logger.info("Rerank 精排已启用：model=%s backend=%s",
                        config.RERANK_MODEL, type(_reranker).__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Rerank 精排不可用（%s），已降级为不精排。"
                "API 路径请检查 RERANK_BASE_URL / RERANK_API_KEY；"
                "本地路径需 pip install sentence-transformers", exc,
            )
            _reranker = None
    return _reranker


def reset_reranker() -> None:
    """清掉单例（测试与配置热切换用）。"""
    global _reranker, _reranker_tried
    with _reranker_lock:
        _reranker = None
        _reranker_tried = False


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
    model = _load_reranker()
    return {
        "enabled": bool(config.RERANK_ENABLED),
        "model": config.RERANK_MODEL,
        "candidates": config.RERANK_CANDIDATES,
        "available": model is not None,
        # 报**实际**后端（api / local），而不是"配了什么" ——
        # 「配了 API 却静默走成本地兜底」是要在观测里一眼看出来的。
        "backend": type(model).__name__ if model is not None else None,
    }
