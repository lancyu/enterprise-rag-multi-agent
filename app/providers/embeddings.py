"""Embedding 提供者 —— 双模式自动降级，配置驱动、与具体模型解耦。

模式一（默认）：调用 OpenAI 兼容的 /embeddings 接口，语义效果最佳，需配置 API Key。
模式二（降级）：本地零依赖哈希向量，无需联网与模型下载，保证服务在无 Key 环境下仍可完整跑通。

两种模式输出均做 L2 归一化，检索统一使用余弦相似度。

与旧实现的区别（本次重构）：
    1. 移除「bge 模型需拼接检索指令前缀」的硬编码——原实现用 ``"bge" in model.lower()``
       判断是否为 BGE 模型，这是把特定模型知识写死在代码里。现改为配置项
       ``EMBEDDING_QUERY_PREFIX``：需要前缀的模型（如 bge-m3）在 .env 里显式配置，
       换非 bge 模型时清空该配置即可，代码无需任何改动。
    2. 每个实现都带 ``mode`` 属性（见 base.Embedder），检索阈值等自适应逻辑通过
       ``mode`` 判断而非 ``isinstance`` 具体类名——新增 provider 无需改动判断处。
"""
import hashlib
import json
import math
import re
from collections import Counter
from typing import List

from app import config
from app import runtime_flags
from app.providers.base import Embedder
from app.utils.logger import logger
from app.utils.text import CJK_CHAR, CJK_RANGE

# 汉字区间从 app.utils.text 取，不在这里重写一遍——同一区间用 ``一-鿿`` 与
# ``\u4e00-\u9fff`` 两种写法表达，读代码的人会以为它们不是一回事。
_TOKEN_PATTERN = re.compile(f"[{CJK_RANGE}]|[A-Za-z]+|\\d+")


def _tokenize(text: str) -> List[str]:
    """中英文混合分词：中文按字 + 二元组，英文数字按词。

    ⚠️ **刻意不剔虚词**，与 ``app/rag/lexical.py::_tokenize`` 不同——这不是遗漏：
    - 倒排索引剔虚词，是因为「的/了/是」会让所有片段的词面分一起冲到 1.0，
      词面路彻底失去排序能力；
    - 这里的输出要喂给**本地哈希向量**：token 集合直接决定向量，剔掉虚词等于
      改变全部历史向量的取值（缓存里每一个向量都要重算），收益却为零——
      向量路本来就不靠字面区分度排序。

    两者的差别是**消费方决定的**，故各自保留；共用的只有「哪些字算中文」。
    """
    text = text.lower()
    units = _TOKEN_PATTERN.findall(text)
    tokens: List[str] = list(units)
    # 中文二元组，提升短语匹配能力
    for i in range(len(units) - 1):
        if _TOKEN_PATTERN.fullmatch(units[i]) and CJK_CHAR.match(units[i]) \
                and CJK_CHAR.match(units[i + 1]):
            tokens.append(units[i] + units[i + 1])
    return tokens


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class LocalHashEmbeddings(Embedder):
    """零依赖本地哈希向量：hashing trick + 亚线性 TF + IDF 加权 + L2 归一化。

    说明：这是无 API Key 时的兜底方案，语义泛化能力弱于神经向量模型，
    但足以支撑「字面/近义短语重叠」类企业制度检索，保证链路可运行。

    IDF 加权是关键：不做加权时，"的 / 员工 / 系统" 这类高频词会淹没
    "年假 / VPN / 报销" 等真正有区分度的关键词，导致段落级检索失准。
    IDF 表在建索引时基于全语料统计并持久化，查询时复用同一套权重。
    """

    mode: str = "local-hash"

    def __init__(self, dim: int = 1024, idf_path=None):
        from pathlib import Path

        self.dim = dim
        self.idf_path = Path(idf_path) if idf_path else None
        self.idf: dict = {}
        self._corpus_size = 0
        self._default_idf = 1.0
        self._load_idf()

    # ---------------- IDF ----------------
    def _load_idf(self) -> None:
        if not self.idf_path or not self.idf_path.exists():
            return
        try:
            data = json.loads(self.idf_path.read_text(encoding="utf-8"))
            self.idf = data.get("idf", {})
            self._corpus_size = data.get("corpus_size", 0)
            self._default_idf = math.log(self._corpus_size + 1) + 1.0 if self._corpus_size else 1.0
        except Exception:  # noqa: BLE001
            self.idf, self._corpus_size, self._default_idf = {}, 0, 1.0

    def _save_idf(self) -> None:
        if not self.idf_path:
            return
        try:
            self.idf_path.parent.mkdir(parents=True, exist_ok=True)
            self.idf_path.write_text(
                json.dumps({"idf": self.idf, "corpus_size": self._corpus_size}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    def has_idf(self) -> bool:
        return bool(self.idf)

    def fit(self, corpus: List[str]) -> None:
        """基于语料统计 IDF（应在全量建索引时调用）。"""
        df: Counter = Counter()
        for text in corpus:
            df.update(set(_tokenize(text)))
        n = len(corpus) or 1
        self._corpus_size = n
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._default_idf = math.log(n + 1) + 1.0  # 未见词按最高区分度处理
        self._save_idf()

    def _weight(self, token: str) -> float:
        return self.idf.get(token, self._default_idf)

    # ---------------- 向量化 ----------------
    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return vec
        tf: Counter = Counter(tokens)
        for token, count in tf.items():
            weight = (1.0 + math.log(count)) * self._weight(token)  # 亚线性 TF × IDF
            # 双哈希降低碰撞影响
            for salt in ("", "#"):
                digest = hashlib.md5((token + salt).encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign * weight
        return _l2_normalize(vec)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class APIEmbeddings(Embedder):
    """OpenAI 兼容 /embeddings 接口封装（httpx 直连，避免额外依赖）。"""

    mode: str = "api"

    def __init__(self, api_key: str, base_url: str, model: str, dim: int = 384):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        # 查询侧前缀：由配置驱动。部分模型（如 bge 系列）要求查询拼接检索指令前缀，
        # 否则余弦相似度偏低。是否拼接由 EMBEDDING_QUERY_PREFIX 决定，代码不识别任何
        # 具体模型名——换模型时改配置即可。
        self.query_prefix = getattr(config, "EMBEDDING_QUERY_PREFIX", "") or ""

    def _request(self, texts: List[str]) -> List[List[float]]:
        import httpx

        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=config.LLM_TIMEOUT) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        vectors = [item["embedding"] for item in items]
        return [_l2_normalize(v) for v in vectors]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 分批请求，避免单次 payload 过大
        batch, results = 32, []
        for i in range(0, len(texts), batch):
            results.extend(self._request(texts[i: i + batch]))
        return results

    def embed_query(self, text: str) -> List[float]:
        # 只对 query 拼接前缀，文档侧不加（标准 bge 用法；前缀由配置决定，模型无关）
        query = f"{self.query_prefix}{text}" if self.query_prefix else text
        return self._request([query])[0]


class CachedAPIEmbeddings(Embedder):
    """带缓存的 API Embedding 代理：查询向量内存 LRU+TTL，文档向量内容哈希持久化。

    只对付费的 APIEmbeddings 包缓存，LocalHashEmbeddings 本地零成本无需缓存。
    缓存 key 含模型名，换模型自动失效，避免跨模型的脏向量。
    """

    def __init__(self, inner: "APIEmbeddings"):
        self._inner = inner
        self._model = inner.model
        self.mode = inner.mode
        self.dim = inner.dim

    def _cache(self):
        from app.utils.cache import get_embedding_cache

        return get_embedding_cache()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cache = self._cache()
        vectors: List[List[float]] = []
        missing_idx: List[int] = []
        for i, text in enumerate(texts):
            hit = cache.get_doc_vec(self._model, text)
            if hit is not None:
                vectors.append(hit)
            else:
                vectors.append([])
                missing_idx.append(i)
        if missing_idx:
            miss_texts = [texts[i] for i in missing_idx]
            miss_vecs = self._inner.embed_documents(miss_texts)
            for i, vec in zip(missing_idx, miss_vecs):
                vectors[i] = vec
            # 批量回写：缓存每次写入都会全量落盘，逐条写会形成 O(n²) 写放大
            # （实测 176 条从 23.8s 降到 0.14s，见 EmbeddingCache.put_doc_vecs）。
            cache.put_doc_vecs(self._model, miss_texts, miss_vecs)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        cache = self._cache()
        hit = cache.get_query_vec(self._model, text)
        if hit is not None:
            return hit
        vec = self._inner.embed_query(text)
        cache.put_query_vec(self._model, text, vec)
        return vec


_embeddings_instance: Embedder | None = None


def get_embeddings() -> Embedder:
    """获取全局单例 Embedding 客户端（真实接口优先，失败自动降级）。"""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    if config.USE_REAL_EMBEDDING:
        try:
            client = APIEmbeddings(
                api_key=config.EMBEDDING_API_KEY,
                base_url=config.EMBEDDING_BASE_URL,
                model=config.EMBEDDING_MODEL_NAME,
                dim=config.LOCAL_EMBEDDING_DIM,
            )
            _ = client.embed_query("健康检查")
            _embeddings_instance = CachedAPIEmbeddings(client)
            logger.info("Embedding 初始化成功：mode=api model=%s（已启用缓存）", config.EMBEDDING_MODEL_NAME)
            return _publish(_embeddings_instance)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding 接口不可用，自动降级为本地哈希向量：%s", exc)

    from pathlib import Path

    idf_path = Path(config.CHROMA_PERSIST_DIR) / "memory" / "idf.json"
    _embeddings_instance = LocalHashEmbeddings(dim=config.LOCAL_EMBEDDING_DIM, idf_path=idf_path)
    logger.info(
        "Embedding 初始化完成：mode=local-hash dim=%d idf=%s",
        config.LOCAL_EMBEDDING_DIM, "已加载" if _embeddings_instance.has_idf() else "未训练",
    )
    return _publish(_embeddings_instance)


def _publish(instance: Embedder) -> Embedder:
    """把「实际生效的模式」发布给配置层，然后原样返回实例。

    这是 `app/runtime_flags` 的**唯一写入点** —— 只有这里知道 embedder
    到底是 api 还是 local-hash（降级是初始化失败后才发生的，看配置看不出来）。
    config 的自适应阈值只读那个值，不再反过来 import 本模块（原先形成环）。
    """
    runtime_flags.set_embedding_mode(instance.mode)
    return instance


def reset_embeddings() -> None:
    """重置单例（用于配置变更后的热切换）。"""
    global _embeddings_instance
    _embeddings_instance = None
    # 同时清掉已发布的模式：重置之后「实际模式」确实又变成未知了。
    runtime_flags.clear_embedding_mode()


def get_embedding_mode() -> str:
    """返回实际生效的 embedding 模式：api / local-hash。

    注意：这是「实际初始化成功」的模式，而非配置里写着的模式。
    配置了 Key 但接口不可用时，会回退到 local-hash，本函数如实返回，
    供检索阈值等依赖 embedding 分数分布的参数做自适应。

    实现上直接读实例的 ``mode`` 属性，而非 isinstance 具体类名——新增 provider
    时无需改动本函数。
    """
    return get_embeddings().mode
