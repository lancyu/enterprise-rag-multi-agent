"""BM25 词面倒排索引 —— 与向量库并行的独立召回路。

为什么需要它：
    现有的词面打分（retriever.lexical_score）只对「向量候选集」重打分，
    向量池外的纯词面命中根本进不来。当查询是精确的专有名词 / 制度条款
    （"年假""VPN"）而向量路把无关段落排到前面时，这条独立路能把它们救回来。

打分用 BM25（k1=1.5, b=0.75）而非 TF-IDF：
    BM25 多了文档长度归一化（参数 b），避免长片段仅因字数多就占便宜。
    标准库即可实现，不引入 jieba / rank_bm25 依赖。

分词复用 ``app.utils.text.cjk_runs`` 的「连续中文串 + bigram」思路：
    中文无空格，直接按字符集切词会退化成字符倒排，故拆成连续串后再叠 bigram；
    英文/数字按 ``\\w+`` 小写化，与 retriever._latin_lexical_score 保持一致。

    虚词表（``CJK_STOP``）与汉字区间也**只在 ``app.utils.text`` 定义一次**——
    此前本模块与 retriever 各抄一份，注释里还写着「与 retriever._CJK_STOP 一致」，
    等于用注释记录了重复。两处各改一遍的最终结局必然是分叉。

索引更新采用「增量」而非全量重建：
    add / remove 只增删受影响文档的倒排项，删除文档时无需重建整个索引。
    代价是 doc_len / avg_len 需在增删时同步维护，代码多几行，但避免文档
    频繁增删时每次全量重建的 O(N) 成本。
"""
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app import config
from app.utils.logger import logger
from app.utils.text import CJK_STOP, cjk_runs

# BM25 参数：k1 控制词频饱和，b 控制文档长度归一化强度（0 不归一化，1 全归一化）。
# 从 config 读取（config.LEXICAL_BM25_K1 / LEXICAL_BM25_B），使 BM25 调参可通过 .env 生效，
# 消除「定义了、写了文档、但没人读」的死配置（评估 P1-2）。
_K1: float = getattr(config, "LEXICAL_BM25_K1", 1.5)
_B: float = getattr(config, "LEXICAL_BM25_B", 0.75)

_LATIN = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> List[str]:
    """中英混合分词：中文「单字 + bigram」叠英文/数字词。

    中文同时产出单字与 bigram，是为了兼顾两种命中：
    - 单字：让「年假」能命中「年休假」（跨字组合，bigram 覆盖不到）；
    - bigram：提升短语精确度，让「年假」优先命中含「年假」连写的片段。
    虚词（的/了/是）已剔除，剩余单字都有区分度，不会退化成字符倒排。
    """
    tokens: List[str] = []
    lowered = text.lower()
    for run in cjk_runs(lowered):
        meaningful = [ch for ch in run if ch not in CJK_STOP]
        if not meaningful:
            continue
        tokens.extend(meaningful)  # 单字
        if len(meaningful) >= 2:
            tokens.extend("".join(meaningful[i: i + 2]) for i in range(len(meaningful) - 1))
    tokens.extend(_LATIN.findall(lowered))
    return tokens


def chunk_key(meta: Dict[str, Any]) -> str:
    """片段的检索键：``source::chunk_index``。

    **dense 路（``retriever._hit_key``）与 lexical 路（``indexer.index_chunks``
    写入）必须共用本函数。** 两条路各自实现时，只要有一处细节不同（例如缺失
    ``chunk_index`` 时的兜底取值），同一片段就会被认成两个不同片段，
    RRF 融合随之退化成「两份互不相干的结果」。

    该键成立的前提是 ``chunk_index`` 在**同一 source 内唯一**——由
    ``indexer.chunk_documents`` 按 source 全局递增编号保证。注意 PDF 加载器
    每页产出一条 ``Document``（source 相同），若各页各自从 0 编号，
    键就会撞（实测 176 条切片只得到 174 个键，丢 2 条）。
    """
    return f"{meta.get('source', 'unknown')}::{meta.get('chunk_index', -1)}"


class LexicalIndex:
    """BM25 倒排索引。与向量库同生命周期，json 落盘便于调试与增量恢复。"""

    def __init__(self, persist_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._postings: Dict[str, Dict[str, int]] = {}   # term -> {doc_key: 词频}
        self._doc_len: Dict[str, int] = {}               # doc_key -> 词数
        self._doc_text: Dict[str, str] = {}              # doc_key -> 原文（组装结果用）
        self._doc_meta: Dict[str, Dict[str, Any]] = {}   # doc_key -> 元数据（source/chunk_index）
        self._avg_len_cache: Optional[float] = None      # 平均文档长度缓存（避免每文档重算，见 _avg_len）
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    # ---------------- 持久化 ----------------
    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._postings = data.get("postings", {})
            self._doc_len = data.get("doc_len", {})
            self._doc_text = data.get("doc_text", {})
            self._doc_meta = data.get("doc_meta", {})
            self._recompute_avg_len()
            logger.info("词面索引已加载：%d 条文档", len(self._doc_len))
        except Exception:  # noqa: BLE001
            logger.exception("词面索引加载失败，将重建")
            self._postings, self._doc_len, self._doc_text, self._doc_meta = {}, {}, {}, {}

    def save(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps(
                    {
                        "postings": self._postings,
                        "doc_len": self._doc_len,
                        "doc_text": self._doc_text,
                        "doc_meta": self._doc_meta,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.exception("词面索引落盘失败")

    # ---------------- 增量更新 ----------------
    def add(self, doc_key: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """新增/覆盖一篇文档（幂等：同 key 先移除旧倒排项）。"""
        with self._lock:
            self.remove(doc_key)
            tokens = _tokenize(content)
            self._doc_len[doc_key] = len(tokens)
            self._doc_text[doc_key] = content
            self._doc_meta[doc_key] = meta or {}
            for term in tokens:
                bucket = self._postings.setdefault(term, {})
                bucket[doc_key] = bucket.get(doc_key, 0) + 1
            self._recompute_avg_len()

    def remove(self, doc_key: str) -> None:
        with self._lock:
            if doc_key not in self._doc_len:
                return
            for term in _tokenize(self._doc_text[doc_key]):
                bucket = self._postings.get(term)
                if bucket and doc_key in bucket:
                    del bucket[doc_key]
                    if not bucket:
                        self._postings.pop(term, None)
            self._doc_len.pop(doc_key, None)
            self._doc_text.pop(doc_key, None)
            self._doc_meta.pop(doc_key, None)
            self._recompute_avg_len()

    def remove_by_source(self, source: str) -> int:
        """删除某来源的全部文档（增量删除，返回删除条数）。"""
        keys = [k for k, m in self._doc_meta.items() if m.get("source") == source]
        for k in keys:
            self.remove(k)
        return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._postings, self._doc_len, self._doc_text, self._doc_meta = {}, {}, {}, {}
            self._avg_len_cache = 1.0

    # ---------------- 检索 ----------------
    def _recompute_avg_len(self) -> None:
        """重算平均文档长度并缓存；add/remove/clear/_load 之后调用。

        旧实现用 ``@property`` 每次调用 _bm25 都重算一次全量均值，
        在 N 篇文档上各算一遍 → 整体 O(N²)。缓存后 _bm25 内读取 O(1)，降为 O(N)。
        """
        if not self._doc_len:
            self._avg_len_cache = 1.0
            return
        self._avg_len_cache = sum(self._doc_len.values()) / len(self._doc_len)

    @property
    def _avg_len(self) -> float:
        if self._avg_len_cache is None:
            self._recompute_avg_len()
        return self._avg_len_cache

    def _bm25(self, query_terms: List[str], doc_key: str) -> float:
        dl = self._doc_len.get(doc_key, 0)
        if dl == 0:
            return 0.0
        total_docs = len(self._doc_len)
        score = 0.0
        for term in query_terms:
            bucket = self._postings.get(term)
            if not bucket or doc_key not in bucket:
                continue
            tf = bucket[doc_key]
            df = len(bucket)
            # 逆文档频率：出现于越多文档，区分度越低
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf + _K1 * (1 - _B + _B * dl / self._avg_len)
            score += idf * (tf * (_K1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """返回 [(doc_key, bm25分)]，按分数降序。

        bm25 分数可能为 0（查询词未命中任何文档），此时返回空。

        性能：用倒排表 ``_postings`` 先把候选集剪枝到「至少命中一个查询词」的文档，
        复杂度从全表扫描 O(N) 降到 O(命中数)。文档规模大（万级以上）时差异显著。
        """
        with self._lock:
            if not self._postings:
                return []
            terms = _tokenize(query)
            if not terms:
                return []
            # 倒排剪枝：候选 = 各查询词 postings 的并集
            candidates: set = set()
            for term in terms:
                bucket = self._postings.get(term)
                if bucket:
                    candidates.update(bucket.keys())
            if not candidates:
                return []
            scored = [(k, self._bm25(terms, k)) for k in candidates]
            scored = [(k, s) for k, s in scored if s > 0.0]
            scored.sort(key=lambda item: -item[1])
            return scored[:top_k]

    def doc_text(self, doc_key: str) -> str:
        return self._doc_text.get(doc_key, "")

    def doc_meta(self, doc_key: str) -> Dict[str, Any]:
        return dict(self._doc_meta.get(doc_key, {}))

    def __len__(self) -> int:
        return len(self._doc_len)

    def list_sources(self) -> List[Dict[str, Any]]:
        """按来源聚合片段数 —— 与向量库 ``list_sources()`` **同一返回形态**。

        对账（``app/rag/indexer.reconcile``）要拿两处存储的来源集合做差集；
        两处各自定义一种返回结构，比较的那一行就得写两份适配代码，
        而这类"适配器"正是最容易漏掉一边的东西。
        """
        agg: Dict[str, int] = {}
        for meta in self._doc_meta.values():
            src = str(meta.get("source", "unknown"))
            agg[src] = agg.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(agg.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
_lexical_index: Optional[LexicalIndex] = None


def get_lexical_index() -> LexicalIndex:
    """获取全局词面索引单例（持久化到与向量库同目录）。"""
    global _lexical_index
    if _lexical_index is not None:
        return _lexical_index
    persist_dir = Path(config.CHROMA_PERSIST_DIR) / "memory"
    persist_dir.mkdir(parents=True, exist_ok=True)
    _lexical_index = LexicalIndex(persist_dir / "lexical_index.json")
    return _lexical_index
