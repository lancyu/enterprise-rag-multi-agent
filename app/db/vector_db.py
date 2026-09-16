"""向量库统一封装 —— 支持 Milvus（分布式）/ Chroma（单机）/ 内存库（零依赖）三后端。

对外暴露一致接口，上层 RAG 引擎无需感知底层差异：
    add / search / delete_by_source / count / clear / list_sources / get_texts

选型按规模走，靠 ``VECTOR_DB_TYPE`` 一个开关切换，不改调用方代码：
    memory  万级以内，零依赖，开发调试
    chroma  十万级，单机 HNSW
    milvus  百万级以上，分布式、标量过滤、多租户

三者的接口是**鸭子类型**（无 ABC/Protocol）。新增后端必须实现全部 7 个方法，
漏一个只会在运行到那条路径时才炸——所以配了 ``tests/test_vector_store_backends.py``
做接口一致性钉死（见 docs/history/redundancy-review.md 对「新增后端要抄一份」的记录）。
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app import config
from app.utils.logger import logger

# 内存库持久化文件
_MEMORY_INDEX_FILE = "memory_index.npz"
_MEMORY_META_FILE = "memory_meta.json"


def _fsync_file(path: Path) -> None:
    """把单个文件刷到磁盘（os.replace 前必须先 fsync，否则替换后仍可能丢内容）。"""
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    """刷目录项，确保 rename/replace 本身持久化（POSIX 要求）。"""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # 某些平台（如 Windows）不支持目录 fsync，忽略即可
        pass


def _atomic_write(target: Path, write_fn) -> None:
    """写临时文件 → fsync → 原子替换，杜绝「写一半」的中间态。

    `write_fn(tmp_path)` 负责真正落盘。os.replace 在同一文件系统内是原子的，
    因此任何时刻外部看到的 target 要么是旧完整版，要么是新完整版。
    """
    tmp = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        write_fn(tmp)
        _fsync_file(tmp)
        os.replace(tmp, target)
    except Exception:
        # 清理残留临时文件，避免下次加载时困惑
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


class SearchResult:
    """检索结果（文档内容 + 元数据 + 相似度）。"""

    __slots__ = ("content", "metadata", "score")

    def __init__(self, content: str, metadata: Dict[str, Any], score: float):
        self.content = content
        self.metadata = metadata
        self.score = score


class MemoryVectorStore:
    """零依赖内存向量库：numpy 暴力余弦检索 + 本地持久化。

    适用于开发调试与中小规模知识库（万级片段内性能充足），无需任何外部服务。
    """

    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._vectors: Optional[np.ndarray] = None
        self._metas: List[Dict[str, Any]] = []
        self._texts: List[str] = []
        self._ids: List[str] = []
        self._load()

    # ---------------- 持久化 ----------------
    def _load(self) -> None:
        vec_file = self.persist_dir / _MEMORY_INDEX_FILE
        meta_file = self.persist_dir / _MEMORY_META_FILE
        if vec_file.exists() and meta_file.exists():
            try:
                data = np.load(vec_file, allow_pickle=False)
                vectors = data["vectors"]
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                texts, metas, ids = meta["texts"], meta["metas"], meta["ids"]

                # 一致性校验：四个序列必须等长，否则说明上次写入被中断。
                # 旧实现不校验，错位会被当成正常数据加载，直到检索时才崩或返回错内容。
                n = len(ids)
                if not (len(texts) == len(metas) == n and vectors.shape[0] == n):
                    raise ValueError(
                        f"索引文件不一致：vectors={vectors.shape[0]} "
                        f"ids={n} texts={len(texts)} metas={len(metas)}"
                    )

                self._vectors, self._texts, self._metas, self._ids = vectors, texts, metas, ids
                logger.info("内存向量库已加载：%d 条片段", len(self._ids))
            except Exception:  # noqa: BLE001
                logger.exception("内存向量库加载失败，将重建索引")
                self._quarantine(vec_file, meta_file)
                self._reset_state()

    def _quarantine(self, *files: Path) -> None:
        """把损坏的索引文件改名保留，而不是直接删除。

        旧行为是加载失败后直接重置为空库 —— 用户只会看到「知识库空了」，
        既没有告警，也失去了人工抢救的机会。保留现场后：
        1. 运维能从日志与文件时间戳定位到「何时损坏」；
        2. 数据恢复人员还有原始数据可用（重建需要先 `POST /knowledge/rebuild`）。
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        saved = []
        for f in files:
            if not f.exists():
                continue
            try:
                target = f.with_name(f"{f.name}.corrupt-{stamp}")
                os.replace(f, target)
                saved.append(str(target))
            except OSError as exc:
                logger.warning("无法隔离损坏文件 %s：%s", f, exc)
        if saved:
            logger.error(
                "索引文件已损坏并被隔离（未删除）：%s。"
                "请检查磁盘/上次写入是否被中断，并通过 /knowledge/rebuild 重建索引。",
                "、".join(saved),
            )

    def _persist(self) -> None:
        """原子持久化：临时文件 → fsync → os.replace。

        旧实现把 npz 与 meta 分两次独立写入、无 fsync、无原子替换：
        写到一半进程崩溃就会留下「两个文件不一致」的残局，
        下次加载必然抛异常 → 被当成「索引不存在」而静默清空。
        """
        meta_file = self.persist_dir / _MEMORY_META_FILE
        vec_file = self.persist_dir / _MEMORY_INDEX_FILE
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        if self._vectors is not None and len(self._ids) > 0:
            vectors = self._vectors

            def _write_vec(tmp: Path) -> None:
                # 必须传文件对象：np.savez_compressed 对不以 .npz 结尾的**路径**
                # 会自动追加 .npz 后缀，导致实际文件名与 tmp 不符、os.replace 找不到文件。
                with open(tmp, "wb") as f:
                    np.savez_compressed(f, vectors=vectors)

            _atomic_write(vec_file, _write_vec)
        elif vec_file.exists():
            vec_file.unlink()

        payload = {"texts": self._texts, "metas": self._metas, "ids": self._ids}
        _atomic_write(
            meta_file,
            lambda tmp: tmp.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            ),
        )
        # 刷目录项，保证两次 replace 本身都已落盘
        _fsync_dir(self.persist_dir)

    def _reset_state(self) -> None:
        self._vectors = None
        self._texts, self._metas, self._ids = [], [], []

    # ---------------- 接口 ----------------
    def add(self, ids: List[str], texts: List[str], vectors: List[List[float]], metas: List[dict]) -> int:
        with self._lock:
            # 幂等：同 id 先删后增
            self._delete_ids(set(ids))
            new_mat = np.asarray(vectors, dtype=np.float32)
            self._vectors = new_mat if self._vectors is None or self._vectors.size == 0 else np.vstack([self._vectors, new_mat])
            self._ids.extend(ids)
            self._texts.extend(texts)
            self._metas.extend(metas)
            self._persist()
            return len(ids)

    def _delete_ids(self, id_set: set) -> int:
        if not id_set:
            return 0
        keep = [i for i, _id in enumerate(self._ids) if _id not in id_set]
        removed = len(self._ids) - len(keep)
        if removed:
            if self._vectors is not None and self._vectors.size:
                self._vectors = self._vectors[keep]
            self._ids = [self._ids[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._metas = [self._metas[i] for i in keep]
        return removed

    def delete_by_source(self, source: str) -> int:
        with self._lock:
            id_set = {_id for _id, meta in zip(self._ids, self._metas) if meta.get("source") == source}
            removed = self._delete_ids(id_set)
            if removed:
                self._persist()
            return removed

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        with self._lock:
            if not self._ids or self._vectors is None:
                return []
            q = np.asarray(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q) or 1.0
            scores = (self._vectors @ q) / q_norm  # 向量已归一化，点积即余弦
            # 只取 Top-K 时用 argpartition（O(N)）替代全排序 argsort（O(N log N)）。
            # 量级：万级片段差异不大，十万级以上能省掉一次全表排序。
            # 注意必须再做一次 **候选内部** 排序 —— RRF 融合依赖 rank 顺序，
            # 直接把 argpartition 的无序结果交出去会让排名失真。
            k_eff = min(int(k), len(scores))
            if k_eff <= 0:
                return []
            if k_eff >= len(scores):
                top_idx = np.argsort(-scores)[:k_eff]
            else:
                part = np.argpartition(-scores, k_eff - 1)[:k_eff]
                top_idx = part[np.argsort(-scores[part])]
            return [
                SearchResult(content=self._texts[i], metadata=self._metas[i], score=float(scores[i]))
                for i in top_idx
            ]

    def count(self) -> int:
        return len(self._ids)

    def get_texts(self) -> List[str]:
        """返回已入库文本（供 IDF 增量统计使用）。"""
        with self._lock:
            return list(self._texts)

    def clear(self) -> None:
        with self._lock:
            self._reset_state()
            self._persist()

    def list_sources(self) -> List[dict]:
        agg: Dict[str, int] = {}
        for meta in self._metas:
            src = meta.get("source", "unknown")
            agg[src] = agg.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(agg.items(), key=lambda x: -x[1])]


class ChromaVectorStore:
    """Chroma 向量库：基于 HNSW 的近似检索，适配生产环境海量文档。"""

    def __init__(self, persist_dir: Path, collection_name: str):
        import chromadb

        from chromadb.config import Settings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, ids: List[str], texts: List[str], vectors: List[List[float]], metas: List[dict]) -> int:
        existing = {i for i in ids if i in set(self._collection.get(ids=ids).get("ids", []))}
        if existing:
            self._collection.delete(ids=list(existing))
        self._collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
        return len(ids)

    def delete_by_source(self, source: str) -> int:
        got = self._collection.get(where={"source": source})
        ids = got.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        res = self._collection.query(
            query_embeddings=[query_vector], n_results=min(k, max(1, self.count())), include=["documents", "metadatas", "distances"]
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma 默认返回 L2 距离，归一化向量下转换为余弦相似度
            score = 1.0 - (float(dist) ** 2) / 2.0
            out.append(SearchResult(content=doc, metadata=meta or {}, score=max(-1.0, min(1.0, score))))
        return out

    def count(self) -> int:
        return self._collection.count()

    def get_texts(self) -> List[str]:
        got = self._collection.get(include=["documents"])
        return list(got.get("documents") or [])

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(name=self._collection.name)

    def list_sources(self) -> List[dict]:
        got = self._collection.get(include=["metadatas"])
        agg: Dict[str, int] = {}
        for meta in got.get("metadatas") or []:
            src = (meta or {}).get("source", "unknown")
            agg[src] = agg.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(agg.items(), key=lambda x: -x[1])]


# Milvus VARCHAR 字段上限（Milvus 规定 max_length ≤ 65535）。
# 切片尺寸默认 300 字符，这里取上限是为了「切片调大后不至于静默截断」。
_MILVUS_MAX_TEXT_LEN = 65535
# 全量扫描（get_texts 供 IDF 统计用）的行数上限，防止误连超大集合时把内存打爆。
_MILVUS_SCAN_CAP = 200000


def _loopback_proxy_hint(uri: str) -> str:
    """诊断「本机地址 + 设了 HTTP 代理」这个高频连不上原因，返回提示语（无问题则空串）。

    gRPC 会读取 ``http_proxy`` / ``https_proxy`` 环境变量。公司网络里普遍配了全局代理，
    此时连 localhost 的 Milvus 会被代理转发，代理返回 502，而 pymilvus 只报
    「illegal connection params or server unavailable」——真正的原因（代理握手失败）
    只出现在一行 stderr 里，极易被误判成「Milvus 没启动」。
    """
    from urllib.parse import urlparse

    host = urlparse(uri).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1"):
        return ""
    proxy = (
        os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        or os.getenv("http_proxy") or os.getenv("HTTP_PROXY") or ""
    )
    if not proxy:
        return ""
    no_proxy = os.getenv("no_proxy") or os.getenv("NO_PROXY") or ""
    if any(h in no_proxy for h in ("127.0.0.1", "localhost", host)):
        return ""
    return (
        f"检测到 Milvus 地址是本机（{host}）却设置了 HTTP 代理（{proxy}）："
        f"gRPC 会走代理连接从而失败。请把本机地址加入 NO_PROXY 后重试，"
        f"例如 export NO_PROXY=localhost,127.0.0.1"
    )


class MilvusVectorStore:
    """Milvus 向量库：分布式、支持标量过滤与百万级以上规模。

    与内存 / Chroma 后端暴露完全相同的 7 个方法。两个关键设计：

    1. **集合惰性创建**：维度在首次写入时从向量长度推断（或由 ``MILVUS_DIM``
       显式指定）。这样「配置里的维度」与「实际 embedding 模型」不一致时，
       错误会出现在第一次写入而非某次检索，且报错能直接指向维度。
    2. **构造期主动探测**：``MilvusClient`` 的构造是惰性的，不探测的话
       「Milvus 没起」要等到第一次检索才暴露，上层工厂也就无法降级。
       因此 ``__init__`` 里主动发一次 RPC，失败即抛异常交给工厂。

    检索距离语义：``COSINE`` 下 Milvus 返回的 ``distance`` 就是余弦相似度
    （越大越相似），与内存库 / Chroma 归一化后的分值同量纲，无需再换算。

    依赖可选包 ``pymilvus``；未安装时 ``import`` 抛 ImportError，由工厂降级。
    """

    def __init__(
        self,
        uri: str,
        collection_name: str,
        token: str = "",
        dim: int = 0,
        index_type: str = "HNSW",
        metric_type: str = "COSINE",
    ):
        from pymilvus import MilvusClient  # 延迟导入：未安装时由工厂降级

        self.collection_name = collection_name
        self.index_type = index_type or "HNSW"
        self.metric_type = metric_type or "COSINE"
        self.dim = int(dim or 0)
        try:
            self._client = MilvusClient(uri=uri, token=token or None)

            # 主动探测连接（见类文档第 2 点）。集合已存在时顺带确保加载。
            if self._client.has_collection(self.collection_name):
                self._ensure_loaded()
        except Exception:
            # 失败时补一条针对「本机地址 + 全局代理」的诊断，否则真正的原因会被
            # pymilvus 笼统的「illegal connection params or server unavailable」盖住。
            hint = _loopback_proxy_hint(uri)
            if hint:
                logger.warning("%s", hint)
            raise

    # ---------------- 内部辅助 ----------------
    def _ensure_loaded(self) -> None:
        """确保集合处于 loaded 状态（检索/查询的前置条件）。

        已加载时 Milvus 会报「already loaded」，属正常情况，忽略即可。
        """
        try:
            self._client.load_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass

    def _ensure_collection(self, dim: int) -> None:
        """集合不存在则按 ``dim`` 建表建索引并加载；存在则只做维度一致性校验。"""
        if self._client.has_collection(self.collection_name):
            if self.dim and dim and self.dim != dim:
                raise ValueError(
                    f"向量维度与已有集合不一致：集合 {self.collection_name} 为 "
                    f"{self.dim} 维，本次写入 {dim} 维。请对齐 embedding 模型"
                    f"（或设置 MILVUS_DIM / 删除并重建集合）。"
                )
            return

        from pymilvus import DataType

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=int(dim))
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=_MILVUS_MAX_TEXT_LEN)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="meta", datatype=DataType.JSON)

        index_params = self._client.prepare_index_params()
        # 只有 HNSW 认这组参数；换 IVF/FLAT 时留空，避免非法参数导致建索引失败。
        params = {"M": 16, "efConstruction": 200} if self.index_type == "HNSW" else {}
        index_params.add_index(
            field_name="vector",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=params,
        )
        self._client.create_collection(
            collection_name=self.collection_name, schema=schema, index_params=index_params
        )
        self.dim = int(dim)
        logger.info(
            "Milvus 集合已创建：%s dim=%d metric=%s index=%s",
            self.collection_name, dim, self.metric_type, self.index_type,
        )

    @staticmethod
    def _row_meta(meta: Optional[dict]) -> dict:
        """元数据必须是可 JSON 序列化的 dict（Milvus JSON 字段要求）。"""
        if not isinstance(meta, dict):
            return {}
        return meta

    # ---------------- 接口 ----------------
    def add(self, ids: List[str], texts: List[str], vectors: List[List[float]], metas: List[dict]) -> int:
        if not ids or not vectors:
            return 0
        dim = len(vectors[0])
        if dim <= 0:
            return 0
        # 无条件调用：它同时负责「不存在则创建」与「已存在则校验维度」。
        self._ensure_collection(dim)

        rows = [
            {
                "id": str(_id),
                "vector": [float(x) for x in vec],
                "text": text or "",
                "source": str(self._row_meta(meta).get("source", "unknown")),
                "meta": self._row_meta(meta),
            }
            for _id, text, vec, meta in zip(ids, texts, vectors, metas)
        ]
        # 用 upsert 而不是 insert：同 id 覆盖，与内存库「先删后增」的幂等语义一致。
        self._client.upsert(collection_name=self.collection_name, data=rows)
        return len(rows)

    def delete_by_source(self, source: str) -> int:
        if not self._client.has_collection(self.collection_name):
            return 0
        expr = f"source == {json.dumps(source, ensure_ascii=False)}"
        res = self._client.delete(collection_name=self.collection_name, filter=expr)
        # pymilvus 的返回值形态随版本 / 后端而异，三种都要兜住：
        #   MilvusClient（2.4~2.6 实测）→ list[str]，被删除的**主键列表**
        #   部分版本 → dict，如 {"delete_count": N}
        #   ORM / 老版本 → MutationResult，带 .delete_count 属性
        # 只认其中一种会让「删除确实生效、返回值却是 0」——本函数在真实
        # Milvus 引擎上实测踩到过（Milvus Lite 返回 list，被当成 0）。
        if isinstance(res, dict):
            return int(res.get("delete_count", res.get("delete_cnt", 0)) or 0)
        if isinstance(res, (list, tuple, set)):
            return len(res)
        return int(getattr(res, "delete_count", 0) or 0)

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        if k <= 0 or not self._client.has_collection(self.collection_name):
            return []
        res = self._client.search(
            collection_name=self.collection_name,
            data=[[float(x) for x in query_vector]],
            limit=int(k),
            output_fields=["text", "source", "meta"],
            search_params={"metric_type": self.metric_type},
        )
        hits = res[0] if res else []
        out = []
        for hit in hits:
            entity = hit.get("entity") or {}
            meta = entity.get("meta")
            if not isinstance(meta, dict):
                meta = {"source": entity.get("source", "unknown")}
            out.append(
                SearchResult(
                    content=entity.get("text", ""),
                    metadata=meta,
                    score=float(hit.get("distance", 0.0)),
                )
            )
        return out

    def count(self) -> int:
        if not self._client.has_collection(self.collection_name):
            return 0
        stats = self._client.get_collection_stats(self.collection_name)
        return int((stats or {}).get("row_count", 0))

    def get_texts(self) -> List[str]:
        """返回已入库文本（供 IDF 增量统计使用）。

        注意：Milvus 没有「只取一列全部行」的廉价接口，这里是**全量分页扫描**。
        十万级可以接受，百万级会明显变慢——真到那个规模应改为把 IDF 词频
        单独落一份，而不是每次回捞全文。
        """
        return [row.get("text", "") for row in self._iter_all(["text"])]

    def clear(self) -> None:
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)

    def list_sources(self) -> List[dict]:
        agg: Dict[str, int] = {}
        for row in self._iter_all(["source"]):
            src = row.get("source", "unknown")
            agg[src] = agg.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(agg.items(), key=lambda x: -x[1])]

    # ---------------- 全量扫描 ----------------
    def _iter_all(self, output_fields: List[str]):
        if not self._client.has_collection(self.collection_name):
            return
        page = 1000
        offset = 0
        while offset < _MILVUS_SCAN_CAP:
            rows = self._client.query(
                collection_name=self.collection_name,
                # Milvus 不接受空表达式；主键非空，这个条件恒真。
                filter="id != ''",
                output_fields=output_fields,
                limit=page,
                offset=offset,
            )
            if not rows:
                return
            yield from rows
            if len(rows) < page:
                return
            offset += len(rows)
        logger.warning(
            "Milvus 全量扫描达到上限 %d 行，结果可能不完整：%s",
            _MILVUS_SCAN_CAP, self.collection_name,
        )


_store_instance = None
_store_type: Optional[str] = None


def _handle_backend_failure(backend: str, exc: Exception) -> None:
    """后端不可用时的统一出口：默认降级，严格模式下快速失败。

    「配了生产后端但服务没起」是部署期最常见的故障。默认处理是**降级而不是
    启动失败**：服务照常起来，日志留明确告警，检索退化为内存库（可能是空的），
    这样前台可用性问题与后端配置问题不会互相掩盖。

    但默认行为有个代价：数据会被悄悄写进本地内存库，而运维以为在用 Milvus。
    因此提供 ``VECTOR_DB_STRICT``——显式选了生产后端时，让它直接启动失败，
    把问题挡在写入之前，而不是等发现「数据怎么没了」。
    """
    hint = (
        f"向量库 {backend} 不可用：{exc}。"
        f"请确认服务已启动、客户端依赖已安装（pip install -r requirements-vector.txt），"
        f"或显式改用 VECTOR_DB_TYPE=memory。"
    )
    if config.VECTOR_DB_STRICT:
        raise RuntimeError(f"VECTOR_DB_STRICT=true，拒绝降级。{hint}") from exc
    logger.warning("向量库 %s 不可用，自动降级为内存向量库：%s", backend, exc)


def get_vector_store():
    """获取全局向量库实例（按 ``VECTOR_DB_TYPE`` 选择，默认降级为内存库）。

    ``VECTOR_DB_TYPE`` 的合法值见 ``config.VECTOR_DB_CHOICES``；非法值会直接
    抛 ``ValueError``（配置写错必须立刻暴露，不能混进降级流程）。

    后端构造失败时的行为由 ``VECTOR_DB_STRICT`` 决定，见 ``_handle_backend_failure``。
    """
    global _store_instance, _store_type
    if _store_instance is not None:
        return _store_instance

    db_type = config.validate_vector_db_type()

    if db_type == "milvus":
        try:
            _store_instance = MilvusVectorStore(
                uri=config.MILVUS_URI,
                collection_name=config.MILVUS_COLLECTION,
                token=config.MILVUS_TOKEN,
                dim=config.MILVUS_DIM,
                index_type=config.MILVUS_INDEX_TYPE,
                metric_type=config.MILVUS_METRIC_TYPE,
            )
            _store_type = "milvus"
            logger.info(
                "向量库初始化成功：type=milvus collection=%s uri=%s",
                config.MILVUS_COLLECTION, config.MILVUS_URI,
            )
            return _store_instance
        except Exception as exc:  # noqa: BLE001
            _handle_backend_failure("milvus", exc)

    if db_type == "chroma":
        try:
            _store_instance = ChromaVectorStore(config.CHROMA_PERSIST_DIR, config.COLLECTION_NAME)
            _store_type = "chroma"
            logger.info("向量库初始化成功：type=chroma collection=%s", config.COLLECTION_NAME)
            return _store_instance
        except Exception as exc:  # noqa: BLE001
            _handle_backend_failure("chroma", exc)

    persist_dir = Path(config.CHROMA_PERSIST_DIR) / "memory"
    _store_instance = MemoryVectorStore(persist_dir)
    _store_type = "memory"
    logger.info("向量库初始化完成：type=memory dir=%s", persist_dir)
    return _store_instance


def get_store_type() -> str:
    if _store_instance is None:
        get_vector_store()
    return _store_type or "unknown"
