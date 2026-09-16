"""Embedding 缓存 —— 两层结构：查询向量内存 LRU+TTL，文档向量内容哈希持久化。

为什么要缓存：
    每次问答都重新 embed query，每次建索引都重新 embed 全量文档。
    文档内容不变时向量永远不变，重复付费纯粹是浪费配额——尤其对限频账号
    （RPM=3）来说，缓存命中的那一次调用可能就是用户能否正常提问的关键。

cache key 含模型名（借鉴 dify cached_embedding）：
    换 embedding 模型后旧向量在新模型下无效，若 key 只按文本哈希，会拿到
    跨模型的脏向量。key = f"{model}:{sha1(text)}" 让换模型自动失效。

并发安全：
    词面路独立后 embedding 会在线程池里并发调用，用 threading.Lock 保护。
"""
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app import config
from app.utils.logger import logger

_QUERY_TTL: float = float(getattr(config, "EMBEDDING_CACHE_QUERY_TTL", 600.0))
_QUERY_MAX_SIZE: int = int(getattr(config, "EMBEDDING_CACHE_QUERY_MAX_SIZE", 1000))


def _cache_key(model: str, text: str) -> str:
    return f"{model}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"


class EmbeddingCache:
    """查询向量内存缓存（LRU + TTL），文档向量落盘持久化（无 TTL）。"""

    def __init__(self, persist_path: Optional[Path] = None):
        self._lock = threading.Lock()
        # key -> (vec, expire_ts)，插入顺序即 LRU 顺序
        self._query_cache: Dict[str, Tuple[List[float], float]] = {}
        self._doc_cache: Dict[str, List[float]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load_doc_cache()

    # ---------------- 文档向量持久化 ----------------
    def _load_doc_cache(self) -> None:
        try:
            self._doc_cache = json.loads(self._persist_path.read_text(encoding="utf-8"))
            logger.info("文档向量缓存已加载：%d 条", len(self._doc_cache))
        except Exception:  # noqa: BLE001
            logger.exception("文档向量缓存加载失败，忽略")
            self._doc_cache = {}

    def _save_doc_cache(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(self._doc_cache), encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.exception("文档向量缓存落盘失败")

    # ---------------- 查询向量 ----------------
    def get_query_vec(self, model: str, text: str) -> Optional[List[float]]:
        key = _cache_key(model, text)
        with self._lock:
            item = self._query_cache.get(key)
            if item is None:
                return None
            vec, expire = item
            if time.time() > expire:
                self._query_cache.pop(key, None)
                return None
            # 命中续期：把 key 移到末尾（LRU 最近使用）
            self._query_cache.pop(key, None)
            self._query_cache[key] = item
            return list(vec)

    def put_query_vec(self, model: str, text: str, vec: List[float]) -> None:
        key = _cache_key(model, text)
        with self._lock:
            self._query_cache[key] = (list(vec), time.time() + _QUERY_TTL)
            while len(self._query_cache) > _QUERY_MAX_SIZE:
                # LRU 淘汰最久未用（dict 保序，第一个即最旧）
                self._query_cache.pop(next(iter(self._query_cache)), None)

    # ---------------- 文档向量 ----------------
    def get_doc_vec(self, model: str, text: str) -> Optional[List[float]]:
        key = _cache_key(model, text)
        with self._lock:
            vec = self._doc_cache.get(key)
            return list(vec) if vec is not None else None

    def put_doc_vecs(self, model: str, texts: List[str], vecs: List[List[float]]) -> None:
        """批量写入文档向量，**只在最后落盘一次**。

        只提供批量接口、不提供单条接口，是刻意的：``_save_doc_cache`` 会把**整个
        缓存**序列化后覆盖写盘，单条接口一旦被逐条调用就是 O(n²) 写放大——
        实测缓存 7MB / 325 条时单次写入 135ms，全量重建 176 篇光序列化就要 23.8 秒，
        改批量后降到 0.14 秒（168x）。**没有单条接口，就没有被逐条调用的机会。**
        """
        if not texts:
            return
        with self._lock:
            for text, vec in zip(texts, vecs):
                self._doc_cache[_cache_key(model, text)] = list(vec)
        self._save_doc_cache()


_cache_instance: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance
    persist_path = Path(config.CHROMA_PERSIST_DIR) / "memory" / "embedding_cache.json"
    _cache_instance = EmbeddingCache(persist_path)
    return _cache_instance
