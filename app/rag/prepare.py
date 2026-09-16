"""RAG 五层架构 · L1 数据准备层。

职责边界：把「原始文件」变成「可索引的高质量语料」。不涉及切片与向量化。

处理流水线：
    加载 → 归一化 → 清洗 → 质量校验 → 近重复去重 → 元数据注入

为什么单独成层：
    检索质量的上限由语料质量决定。噪声（页眉页脚、空白行、乱码、重复片段）
    会被下一层的切片放大成大量低信息量向量，既浪费索引空间又污染召回排序。
    把清洗前移到数据入口，是唯一能在源头治理的位置。

设计取舍：
    近重复检测采用「字符级 Jaccard 相似度」而非向量相似度——
    数据准备阶段还没有 embedding，且 Jaccard 对排版差异（空格/换行）不敏感，
    能稳定识别「同一段话的两种排版」这类真实重复。
"""
import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Set

from langchain_core.documents import Document

from app import config
from app.utils.logger import logger

# ---------------------------------------------------------------------------
# 可调参数（可通过 config 覆盖，缺省值面向企业制度类文档调优）
# ---------------------------------------------------------------------------
MIN_DOC_CHARS: int = getattr(config, "PREPARE_MIN_DOC_CHARS", 10)          # 短于此长度的文档视为噪声
NEAR_DUP_THRESHOLD: float = getattr(config, "PREPARE_NEAR_DUP_THRESHOLD", 0.95)  # Jaccard 近重复阈值

# 清洗用正则
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # 保留 \t \n \r
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
# 页码行 / 纯分隔线噪声
_PAGE_NOISE = re.compile(r"^\s*(第?\s*\d+\s*页?(\s*/\s*共\s*\d+\s*页?)?|[-=_—]{3,})\s*$", re.MULTILINE)

# 最近一次流水线统计（供 L5 评估与 /stats 消费）
_LAST_STATS: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 文本归一化
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """统一文本表示：全角兼容 → 去控制字符 → 压缩空白 → 去页码噪声。

    顺序很重要：必须先做 NFKC 全角转半角，否则全角空格（U+3000）会逃过
    后续的空白压缩规则，导致指纹把「同样内容、不同输入法」判为两篇文档。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _PAGE_NOISE.sub("", text)
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_SPACES.sub(" ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def content_fingerprint(text: str) -> str:
    """归一化文本的稳定指纹（sha1），用于精确去重与增量入库判定。"""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def _char_set(text: str) -> Set[str]:
    """字符集（去空白），用于 Jaccard 近重复检测。"""
    return {ch for ch in text if not ch.isspace()}


def jaccard_similarity(a: str, b: str) -> float:
    """字符级 Jaccard 相似度（0~1）。"""
    set_a, set_b = _char_set(a), _char_set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_near_duplicate(a: str, b: str, threshold: float = NEAR_DUP_THRESHOLD) -> bool:
    """判定两段文本是否为近重复。"""
    return jaccard_similarity(a, b) >= threshold


# ---------------------------------------------------------------------------
# 元数据注入
# ---------------------------------------------------------------------------
def _infer_file_type(source: str) -> str:
    lowered = (source or "").lower()
    for ext in (".pdf", ".md", ".markdown", ".txt"):
        if lowered.endswith(ext):
            return ext.lstrip(".")
    return "unknown"


def enrich_metadata(doc: Document) -> Document:
    """为文档注入检索期可用的元数据。

    注入的字段会被 L2 切片继承，最终出现在检索结果里，
    是 L4 引用溯源与 L5 评估归因的数据基础。
    """
    content = doc.page_content
    meta: Dict[str, Any] = dict(doc.metadata or {})
    source = str(meta.get("source", "unknown"))
    meta.setdefault("source", source)
    meta["file_type"] = _infer_file_type(source)
    meta["file_name"] = source.split("/")[-1].split("\\")[-1]
    meta["char_count"] = len(content)
    meta["fingerprint"] = content_fingerprint(content)
    return Document(page_content=content, metadata=meta)


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------
def prepare_documents(documents: Iterable[Document]) -> List[Document]:
    """执行完整的数据准备流水线，返回可直接进入 L2 的语料。

    返回文档保证：已归一化、已去噪、已去重、元数据完整。

    去重在本批次内完成（局部状态，无跨调用副作用）。若未来语料膨胀到
    万级文档，O(n²) 的近重复比对应替换为 MinHash / SimHash 分桶。
    """
    global _LAST_STATS

    prepared: List[Document] = []
    kept_texts: List[str] = []          # 本批次已保留文本，用于近重复比对
    seen_fingerprints: Set[str] = set()  # 精确去重
    stats = {
        "input": 0,
        "empty_dropped": 0,
        "short_dropped": 0,
        "exact_dup_dropped": 0,
        "near_dup_dropped": 0,
        "output": 0,
    }

    for doc in documents:
        stats["input"] += 1
        cleaned_text = normalize_text(doc.page_content or "")
        if not cleaned_text:
            stats["empty_dropped"] += 1
            continue
        if len(cleaned_text) < MIN_DOC_CHARS:
            stats["short_dropped"] += 1
            continue

        fingerprint = content_fingerprint(cleaned_text)
        if fingerprint in seen_fingerprints:
            stats["exact_dup_dropped"] += 1
            continue
        seen_fingerprints.add(fingerprint)

        if any(is_near_duplicate(cleaned_text, kept) for kept in kept_texts):
            stats["near_dup_dropped"] += 1
            continue

        prepared.append(enrich_metadata(Document(page_content=cleaned_text, metadata=doc.metadata or {})))
        kept_texts.append(cleaned_text)

    stats["output"] = len(prepared)
    _LAST_STATS = stats
    logger.info(
        "L1 数据准备完成：输入 %d → 输出 %d（空 %d / 过短 %d / 精确重复 %d / 近重复 %d）",
        stats["input"], stats["output"], stats["empty_dropped"], stats["short_dropped"],
        stats["exact_dup_dropped"], stats["near_dup_dropped"],
    )
    return prepared


def get_last_stats() -> Dict[str, Any]:
    """返回最近一次数据准备的统计信息。"""
    return dict(_LAST_STATS)


def prepare_single(file_name: str, content: str) -> List[Document]:
    """准备单篇文档（增量入库场景）。"""
    from app.utils.doc_loader import load_text_content

    return prepare_documents(load_text_content(file_name, content))


def quality_report(documents: List[Document]) -> Dict[str, Any]:
    """语料质量报告：长度分布与来源分布，用于 L5 评估归因。"""
    if not documents:
        return {"count": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0, "sources": 0}
    lengths = [len(d.page_content) for d in documents]
    sources = {str(d.metadata.get("source", "unknown")) for d in documents}
    return {
        "count": len(documents),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "sources": len(sources),
    }
