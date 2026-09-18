"""RAG 五层架构 · L2 索引构建层。

职责边界：把 L1 产出的语料变成「可高效检索的向量索引」。
    自适应切片 → 短片段治理 → 向量化 → 幂等写入 → IDF 训练

关键设计：
1. 短片段合并——机械切片会产生「3. 考勤」这类只有标题没有正文的碎片，
   它们向量化后是纯噪声（既答不了问题，又会挤占 Top-K 名额）。
   本层把过短片段向前合并，合并后仍超限的才丢弃。
2. 幂等写入——chunk_id 由「来源 + 内容哈希」确定性生成，向量库 add 本身幂等；
   增量入库前再按 source 清理，确保同一文档重复上传是覆盖而非累积。
3. IDF 训练时机——本地哈希向量的 IDF 必须基于全量语料重算，
   但增量入库时不能重算（会破坏历史向量的可比性），故用 refit_idf 显式区分。
"""
import hashlib
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.db.vector_db import get_vector_store
from app.providers.embeddings import get_embeddings
from app.rag import parent_store
from app.rag.lexical import chunk_key, get_lexical_index
from app.rag.prepare import prepare_documents, prepare_single, quality_report
from app.rag.structure import (
    FLAT,
    parse_document,
    split_by_section,
)
from app.utils.doc_loader import load_all_documents
from app.utils.logger import logger

# 短于该长度的片段视为碎片，尝试向上一片段合并
MIN_CHUNK_CHARS: int = getattr(config, "INDEX_MIN_CHUNK_CHARS", 40)
# 合并后的软上限（允许略微超出 chunk_size，避免二次切碎）
MERGE_SOFT_LIMIT_RATIO: float = 1.3

_LAST_INDEX_STATS: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 切片
# ---------------------------------------------------------------------------
def _splitter(overlap: Optional[int] = None) -> RecursiveCharacterTextSplitter:
    """递归切分器。

    Args:
        overlap: 覆盖 default 的块间重叠字符数。为 None 时用 ``CHUNK_OVERLAP``。
            供「降级路径」传入按比例换算出的更小重叠（见 ``_fallback_overlap``）。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP if overlap is None else overlap,
        separators=config.SEPARATORS,
        length_function=len,
    )


def _fallback_overlap() -> int:
    """降级递归路径（structure 遇到无结构文档）使用的块间重叠。

    为什么降级路径要用比常规更小的重叠（``CHUNK_FALLBACK_OVERLAP_RATIO``，
    默认 0.12 × 300 = 36 字，而常规是 ``CHUNK_OVERLAP`` = 60 字）：

        重叠的收益是「保住被切断的语义」，代价是「同一段内容被多个块重复持有，
        既占向量库空间又会挤占 Top-K 名额」。结构感知路径靠章节边界天然隔离，
        重叠本就设为 0；而走到递归降级路径的文档（无标题锚点的 PDF 抽取文本等）
        段落边界弱、内容密度低，用大重叠换来的语义连续性很有限，
        却会明显放大重复——故单独给一个更保守的比例。

    本函数是该项目「回退 = 改配置，不改代码」原则的落点：把比例调回
    ``CHUNK_OVERLAP / CHUNK_SIZE``（默认配置下即 0.2）即可精确复现接线前行为。
    """
    ratio = max(0.0, min(1.0, float(config.CHUNK_FALLBACK_OVERLAP_RATIO)))
    return int(round(config.CHUNK_SIZE * ratio))


def _merge_short_chunks(pieces: List[str], min_chars: int = MIN_CHUNK_CHARS) -> List[str]:
    """把过短碎片向前合并，返回治理后的片段列表。

    合并规则：
    - 片段长度 >= min_chars：直接保留；
    - 片段过短：尝试拼到上一个片段末尾（用换行连接），只要不超出软上限；
    - 首片段就过短：暂存，等下一个片段来「收养」；
    - 全程只有一个过短片段：兜底保留，避免整篇文档被清空。
    """
    if not pieces:
        return []

    soft_limit = int(config.CHUNK_SIZE * MERGE_SOFT_LIMIT_RATIO)
    merged: List[str] = []
    pending = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if pending:
            piece = f"{pending}\n{piece}"
            pending = ""

        if len(piece) >= min_chars:
            merged.append(piece)
            continue
        # 过短：能并进上一个就并，否则暂存等下一个
        if merged and len(merged[-1]) + len(piece) + 1 <= soft_limit:
            merged[-1] = f"{merged[-1]}\n{piece}"
        else:
            pending = piece

    if pending:
        # 收尾：能并就并，否则兜底保留（宁可留一个短片段，也不要丢内容）
        if merged and len(merged[-1]) + len(pending) + 1 <= soft_limit:
            merged[-1] = f"{merged[-1]}\n{pending}"
        else:
            merged.append(pending)
    return merged


def _effective_strategy(doc: Document) -> str:
    """按 CHUNK_STRATEGY_SCOPE 决定单篇文档走哪种切分策略。

    scope 为空 → 全部文档用 CHUNK_STRATEGY；
    scope 非空 → 仅列出的文件名走 CHUNK_STRATEGY，其余走 recursive（按文档灰度）。

    注意：灰度档位取自 `config.CHUNK_STRATEGY` 而非硬编码 "structure"。
    此前硬编码会导致「CHUNK_STRATEGY=recursive + 设了 scope」时自相矛盾 ——
    scope 内的文档被切成 structure，与配置表达的意图相反。
    """
    scope = [s.strip() for s in (config.CHUNK_STRATEGY_SCOPE or "").split(",") if s.strip()]
    if not scope:
        return config.CHUNK_STRATEGY
    name = doc.metadata.get("file_name", "")
    return config.CHUNK_STRATEGY if name in scope else "recursive"


def _chunk_recursive(
    doc: Document, *, degraded: bool = False, start_index: int = 0
) -> List[Document]:
    """递归字符切分 + 短片段治理。

    Args:
        degraded: 是否为「结构感知路径遇到无结构文档」的降级调用。
            True 时块间重叠取 ``CHUNK_FALLBACK_OVERLAP_RATIO`` 换算值
            （见 ``_fallback_overlap``），False 时取常规 ``CHUNK_OVERLAP``。
            两者默认不等价，理由见 ``_fallback_overlap`` 的说明。
        start_index: 块序号的起点。同一 source 有多条 Document 时（PDF 每页一条），
            由 ``chunk_documents`` 传入累加游标，避免各页都从 0 开始而撞键。
            理由见 ``chunk_documents``。
    """
    splitter = _splitter(_fallback_overlap() if degraded else None)
    raw_pieces = splitter.split_text(doc.page_content)
    pieces = _merge_short_chunks(raw_pieces)
    chunks: List[Document] = []
    for idx, piece in enumerate(pieces):
        meta = dict(doc.metadata or {})
        meta["chunk_index"] = start_index + idx
        meta["chunk_chars"] = len(piece)
        chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


def _structure_merge_blocks(blocks: List[dict]) -> List[dict]:
    """结构感知合并：相邻小节拼到接近目标块长；单节超限则递归拆。

    核心算法（T2-2）：
    - 仅合并「同章（L1）内」的相邻小节，跨章边界一律切分（防章级串味）；
    - 缓冲不超过 TARGET_CHARS，超限即 flush 后另起新块；
    - 缓冲超过 HARD_MAX_CHARS 时用递归切分器拆成 <= HARD_MAX 的片；
    - 单块本身就超 HARD_MAX 且缓冲为空 → 直接拆，不入缓冲；
    - overlap 固定为 0（结构边界天然隔离，无需重叠）。
    """
    target = config.CHUNK_TARGET_CHARS
    hard_max = config.CHUNK_HARD_MAX_CHARS

    pieces: List[dict] = []
    buf = ""
    buf_head = ""
    buf_chapter = ""

    def _chapter_of(head: str) -> str:
        return head.split(" > ")[0].strip() if head else ""

    def flush():
        nonlocal buf, buf_head
        if not buf:
            return
        if len(buf) > hard_max:
            for sub in _splitter().split_text(buf):
                sub = sub.strip()
                if sub:
                    pieces.append({"content": sub, "heading_path": buf_head})
        else:
            pieces.append({"content": buf, "heading_path": buf_head})
        buf = ""
        buf_head = ""

    for blk in blocks:
        content = blk["content"].strip()
        if not content:
            continue
        head = blk.get("heading_path") or ""
        cur_chapter = _chapter_of(head)

        if buf:
            # 跨章保护：章节变化先 flush，绝不把两章拼进一块
            if cur_chapter and buf_chapter and cur_chapter != buf_chapter:
                flush()
            if buf:   # flush 后可能已空
                candidate = f"{buf}\n{content}"
                if len(candidate) > target:
                    flush()
                    buf = content
                    buf_head = head
                    buf_chapter = cur_chapter
                    continue
                buf = candidate
                buf_head = head
                buf_chapter = cur_chapter
        else:
            if len(content) > hard_max:
                for sub in _splitter().split_text(content):
                    sub = sub.strip()
                    if sub:
                        pieces.append({"content": sub, "heading_path": head})
                continue
            buf = content
            buf_head = head
            buf_chapter = cur_chapter

    flush()
    return pieces


def _chunk_structure(doc: Document, *, start_index: int = 0) -> List[Document]:
    """结构感知切分：section 聚合为正文块 → 按目标块长合并/裁剪。

    - 无结构文档（flat）回退递归（与 baseline 一致，保证兼容）；
    - 上下文头（CHUNK_CONTEXT_HEADER）与元数据增强（CHUNK_ENRICH_METADATA）
      各自独立开关，默认关闭，开启即生效、关闭即回退。

    Args:
        start_index: 块序号起点，语义见 ``_chunk_recursive`` / ``chunk_documents``。
    """
    text = doc.page_content
    tree = parse_document(text)
    if tree.structure == FLAT:
        # 降级路径：无标题锚点的文档交给递归切分，但用更保守的重叠比例
        # （CHUNK_FALLBACK_OVERLAP_RATIO），理由见 _fallback_overlap。
        return _chunk_recursive(doc, degraded=True, start_index=start_index)

    blocks = split_by_section(text)
    merged = _structure_merge_blocks(blocks)

    add_header = config.CHUNK_CONTEXT_HEADER
    enrich = config.CHUNK_ENRICH_METADATA
    base_meta = dict(doc.metadata or {})
    base_meta["structure"] = "hierarchical"
    if enrich and tree.doc_title:
        base_meta["doc_title"] = tree.doc_title
        if tree.doc_version:
            base_meta["doc_version"] = tree.doc_version
        if tree.doc_effective_date:
            base_meta["doc_effective_date"] = tree.doc_effective_date
        if tree.doc_updated_date:
            base_meta["doc_updated_date"] = tree.doc_updated_date

    chunks: List[Document] = []
    for idx, piece in enumerate(merged):
        content = piece["content"]
        heading_path = piece.get("heading_path") or ""
        meta = dict(base_meta)
        meta["chunk_index"] = start_index + idx
        meta["chunk_chars"] = len(content)

        if enrich and heading_path:
            meta["heading_path"] = heading_path
            segs = [s for s in heading_path.split(" > ") if s]
            if segs:
                meta["chapter"] = segs[0]
                meta["section"] = segs[-1]

        out_content = f"【章节】{heading_path}\n\n{content}" if (add_header and heading_path) else content
        chunks.append(Document(page_content=out_content, metadata=meta))
    return chunks


def chunk_documents(documents: List[Document]) -> List[Document]:
    """切片 + 短片段治理 + 元数据继承。

    策略分发：
    - CHUNK_STRATEGY=recursive（默认）→ 改造前行为，零变化；
    - CHUNK_STRATEGY=structure → 结构感知切分（无结构文档自动回退递归）。

    **块序号按 source 全局递增**，而不是「每条 Document 各自从 0 开始」。
    为什么必须这样：同一个 source 可以对应**多条 Document** —— PDF 加载器每页
    产出一条（``doc_loader._load_pdf_pdfplumber`` / ``_load_pdf_pypdf``），
    它们的 ``source`` 完全相同。若各页都从 0 编号，两页的块就共用
    ``source::0``、``source::1`` 这些检索键，后果有两层：

    1. 词面索引 ``add()`` 对同 key 幂等覆盖 → 先写入的块被后写入的顶掉，
       该块**从词面召回里彻底消失**；
    2. 向量库因为用内容哈希作 id 不会丢，但检索期 ``_hit_key`` 撞键会把两条
       不同片段当成同一条，在 RRF 融合时被去重挤掉。

    实测：12 篇语料 176 条切片只得到 174 个键（那份 2 页 PDF 恰好丢 2 条）。
    """
    chunks: List[Document] = []
    next_index: Dict[str, int] = {}      # source -> 下一个可用块序号
    for doc in documents:
        source = str((doc.metadata or {}).get("source", "unknown"))
        start = next_index.get(source, 0)
        if _effective_strategy(doc) == "structure":
            part = _chunk_structure(doc, start_index=start)
        else:
            part = _chunk_recursive(doc, start_index=start)
        next_index[source] = start + len(part)
        chunks.extend(part)
    return chunks


# ---------------------------------------------------------------------------
# 父子双层索引（T6-1，默认关闭）
# ---------------------------------------------------------------------------
def _ctx_header_of(text: str) -> str:
    """取块首的【章节】上下文头（没有则返回空串）。"""
    if text.startswith("【章节】"):
        return text.split("\n\n", 1)[0]
    return ""


def _strip_ctx_header(text: str) -> str:
    """去掉块首的【章节】上下文头。"""
    if text.startswith("【章节】"):
        return text.split("\n\n", 1)[1] if "\n\n" in text else text
    return text


def attach_parents(chunks: List[Document]) -> int:
    """把同章连续子块合并成父块，写入旁路存储，并给子块打 `parent_id`。

    解决的问题：子块粒度细、检索准，但单块信息量常常不足以支撑完整回答
    （「命中正确但答案不完整」）。父块把同章相邻子块拼成完整上下文，
    检索仍用子块、生成用父块 —— 即 small-to-big。

    设计约束：
    - **仅在 PARENT_CHUNK_ENABLED 时调用**；未启用时切分产物零变化；
    - 父块**不进向量库**（理由见 parent_store 模块 docstring），向量数量零增长；
    - 跨章 / 跨源必切，保证父块语义内聚；
    - 合并时去掉重复的章头（同章相邻子块头相同），避免浪费 prompt token；
    - 任何异常只告警不抛错，退回「无父块」状态，绝不打断主链路。
    """
    if not chunks or not config.PARENT_CHUNK_ENABLED:
        return 0

    from app.rag.parent_store import add_many, save

    max_chars = config.PARENT_MAX_CHARS
    parents: List[dict] = []
    buf: List[Document] = []
    buf_chars = 0
    buf_chapter: str | None = None
    buf_source: str | None = None
    seq = 0

    def flush() -> None:
        nonlocal buf, buf_chars, buf_chapter, buf_source, seq
        if not buf:
            return
        # 组装父块：相邻块章头相同则只保留一次
        parts: List[str] = []
        prev_head = ""
        for d in buf:
            text = d.page_content
            head = _ctx_header_of(text)
            if head:
                if head == prev_head:
                    text = _strip_ctx_header(text)
                else:
                    prev_head = head
            parts.append(text)
        content = "\n\n".join(parts).strip()

        source = buf_source or "unknown"
        pid = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{source}::parent::{buf_chapter or 'flat'}::{seq}",
        ))
        parents.append({
            "parent_id": pid,
            "content": content,
            "source": source,
            "chapter": buf_chapter or "",
            "child_count": len(buf),
            "chars": len(content),
        })
        for d in buf:
            d.metadata["parent_id"] = pid
        seq += 1
        buf, buf_chars, buf_chapter, buf_source = [], 0, None, None

    try:
        for d in chunks:
            chapter = d.metadata.get("chapter", "") or ""
            source = d.metadata.get("source", "unknown")
            if buf and (source != buf_source or chapter != buf_chapter):
                flush()
            if buf and buf_chars + len(d.page_content) > max_chars:
                flush()
            buf.append(d)
            buf_chars += len(d.page_content)
            buf_chapter, buf_source = chapter, source
        flush()
        add_many(parents)
        save()
        logger.info("父子索引：%d 个子块 → %d 个父块（上限 %d 字/父块）",
                    len(chunks), len(parents), max_chars)
    except Exception as exc:  # noqa: BLE001
        logger.warning("父块构建失败，退回无父块模式（不影响检索）：%s", exc)
        return 0
    return len(parents)


def _chunk_id(source: str, content: str) -> str:
    """确定性 chunk id：同来源同内容 → 同 id。

    用内容哈希而非位置序号（旧实现 uuid5(source::chunk_index)）：
    位置型 ID 是「伪增量」——同一文档前插一段后，后续所有 chunk 的序号偏移，
    ID 全变，导致重复入库时本应不变的片段被误删重写，也破坏向量库的增量语义。
    内容哈希让「内容未变的 chunk」ID 稳定，只有真正变化的片段才会更新。
    """
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{digest}"))


# ---------------------------------------------------------------------------
# 向量化与写入
# ---------------------------------------------------------------------------
def _existing_texts(store) -> List[str]:
    getter = getattr(store, "get_texts", None)
    return list(getter()) if callable(getter) else []


def _warn_on_key_collision(chunks: List[Document]) -> int:
    """检测检索键冲突并告警，返回冲突键个数（0 表示健康）。

    键冲突 = 「同一个 ``source::chunk_index`` 被多条片段共用」。这是**静默丢数据**
    的典型形态：词面索引 ``add()`` 幂等覆盖，后写的把先写的顶掉，该片段从此
    召不回来；向量路虽不丢，却会在 RRF 融合时被当成同一条去重。

    这里是**观测点而非拦截点** —— 索引链路不因冲突中断（与「可观测性永不
    raise」的约定一致），但必须留下可排查的痕迹，否则这类问题只会表现为
    「某个问法就是问不出答案」。真正的修复在 ``chunk_documents``（按 source
    全局编号），本函数只负责让回归立刻可见。
    """
    counts = Counter(chunk_key(c.metadata or {}) for c in chunks)
    dup = {k: n for k, n in counts.items() if n > 1}
    if dup:
        sample = ", ".join(f"{k}×{n}" for k, n in list(dup.items())[:3])
        logger.warning(
            "检索键冲突：%d 个 key 被多条片段共用，这些片段会被静默覆盖"
            "（样例：%s）。检查 chunk_documents 是否按 source 全局编号。",
            len(dup), sample,
        )
    return len(dup)


def index_chunks(chunks: List[Document], refit_idf: bool = False) -> int:
    """对切片做向量化并写入向量库。

    Args:
        refit_idf: 是否基于本次语料重算 IDF。全量建索引时置 True；
                   增量入库时沿用已有 IDF，避免破坏历史向量的一致性。
    """
    if not chunks:
        return 0
    store = get_vector_store()
    embeddings = get_embeddings()

    # 父子双层（T6-1）：必须在向量化**之前**调用 —— parent_id 要随 metadata
    # 一起落进向量库/词面索引，检索时才有得回捞。未启用时是空操作。
    parent_count = attach_parents(chunks)

    texts = [c.page_content for c in chunks]

    # 只有 local-hash 模式需要 IDF 训练（神经向量模型无需、也提供不了 has_idf）。
    # 用 mode 而非 isinstance 判断，与具体实现类解耦（见 app/providers/base.py）。
    if getattr(embeddings, "mode", "") == "local-hash" and (refit_idf or not embeddings.has_idf()):
        corpus = texts if refit_idf else list(texts) + _existing_texts(store)
        embeddings.fit(corpus)
        logger.info("IDF 训练完成：语料 %d 条，词表 %d 个", len(corpus), len(embeddings.idf))

    vectors = embeddings.embed_documents(texts)
    ids = [_chunk_id(c.metadata.get("source", "unknown"), c.page_content)
           for c in chunks]
    store.add(ids=ids, texts=texts, vectors=vectors, metas=[c.metadata for c in chunks])

    # 同步维护词面倒排索引（增量：逐条 add，幂等覆盖）
    _warn_on_key_collision(chunks)
    lexical = get_lexical_index()
    for c in chunks:
        # **全量继承** chunk 元数据，而不是另立一份白名单。
        # 向量库落库的就是 c.metadata，词面索引若只同步其中几个字段，两条召回路
        # 的 metadata 口径就会分叉——曾只同步 6 个字段，导致词面路命中的片段
        # 缺 page / doc_version / doc_title / file_type 等对下游（Dify 的
        # metadata_condition 过滤、时效判断、引用展示）有意义的字段，且每次
        # 新增字段都必然漏改。与 chunk_key 同理：同一语义只定义一次。
        # 父子块回捞依赖的 parent_id 也随 attach_parents 一并带过来
        # （未启用该特性时本就不写，消费方用 .get(...) or "" 兜底）。
        lexical_meta = dict(c.metadata or {})
        lexical_meta.setdefault("source", "unknown")
        lexical_meta.setdefault("chunk_index", -1)
        # key 必须由 chunk_key 统一生成：dense 路（retriever._hit_key）用的是
        # 同一个函数，两处算法分叉会让同一片段在融合时被当成两条。
        lexical.add(chunk_key(lexical_meta), c.page_content, lexical_meta)
    lexical.save()
    if parent_count:
        logger.info("L2 索引写入完成：%d 子块 / %d 父块", len(chunks), parent_count)
    return len(chunks)


# ---------------------------------------------------------------------------
# 对外：全量 / 增量 / 删除
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 增量索引的对账（P1-5）
# ---------------------------------------------------------------------------
def resolve_docstore_strategy() -> str:
    """解析实际生效的对账策略；配置写了未知值时**降级并告警**。

    为什么不能静默：策略名拼错（`upsert` 少个 s）时，"静默用默认值"与"配置生效了"
    在结果上完全一样，排查时没有任何线索 —— 写错必须留痕。
    """
    raw = config.DOCSTORE_STRATEGY
    if raw in config.DOCSTORE_STRATEGY_CHOICES:
        return raw
    logger.warning(
        "DOCSTORE_STRATEGY=%r 不是合法取值（只认 %s），已降级为 %r。请检查 .env",
        raw, " / ".join(config.DOCSTORE_STRATEGY_CHOICES), config.DOCSTORE_STRATEGY_DEFAULT,
    )
    return config.DOCSTORE_STRATEGY_DEFAULT


def _source_set(entries: List[Dict[str, Any]]) -> set:
    return {str(e.get("source", "unknown")) for e in entries}


def reconcile() -> Dict[str, Any]:
    """只读对账：向量库 / 词面索引 / 父块三处的**条数与来源集合**是否一致。

    为什么要单独有一个对账函数：三处存储是分别维护的（向量库、词面倒排、父块旁路），
    任何一处漏删都不会报错，只会表现为「某条知识怎么问都检索不到」或
    「删掉的文档还在召回里」。这类故障的共同点是**没有任何一方报错** ——
    只有把三处摆在一起比一次才看得见。

    返回 ``consistent=False`` 时调用方应当告警，而不是继续当没事。
    """
    store = get_vector_store()
    lexical = get_lexical_index()

    vec_sources = store.list_sources()
    lex_sources = lexical.list_sources()
    parent_count = parent_store.count() if config.PARENT_CHUNK_ENABLED else None

    vec_set, lex_set = _source_set(vec_sources), _source_set(lex_sources)
    return {
        "vector_chunks": store.count(),
        "lexical_chunks": len(lexical),
        "parent_sources": parent_count,
        # 条数不等**不一定**是故障（父块不进词面索引），但来源集合不等一定是。
        "only_in_vector": sorted(vec_set - lex_set),
        "only_in_lexical": sorted(lex_set - vec_set),
        "vector_sources": len(vec_set),
        "lexical_sources": len(lex_set),
        "consistent": vec_set == lex_set,
    }


def build_index() -> Dict[str, Any]:
    """重建索引：L1 数据准备 → L2 切片写入 → 对账。

    对账策略由 ``config.DOCSTORE_STRATEGY`` 决定（三态，见
    ``config.DOCSTORE_STRATEGY_CHOICES``）；默认 ``upserts_and_delete``。

    ⚠️ 三态里只有 ``upserts_and_delete`` 能保证「源删了索引里也跟着删」。
    另两种是**有意保留**的弱语义（语料只会增加、或不做删除的场景），
    不是缺陷 —— 但选了它们就必须接受「消失的来源会留在索引里」。
    """
    global _LAST_INDEX_STATS

    strategy = resolve_docstore_strategy()
    raw_documents = load_all_documents()
    prepared = prepare_documents(raw_documents)   # L1
    store = get_vector_store()

    if strategy == "duplicates_only":
        # 只补新片段。已存在的片段靠 **chunk_id（= source + 内容哈希）** 覆盖：
        # id 本身就是内容指纹，不需要另存一份 sha1 —— 同一语义只定义一次。
        pass
    elif strategy == "upserts":
        # 覆盖同名来源，但**不**删除语料里已消失的来源
        for source in _source_set(store.list_sources()):
            store.delete_by_source(source)
            get_lexical_index().remove_by_source(source)
    else:  # upserts_and_delete：清空三处再重建，是唯一能保证不留孤儿的实现
        store.clear()
        get_lexical_index().clear()               # 词面索引与向量库同步全量重建
        if config.PARENT_CHUNK_ENABLED:
            parent_store.clear()                  # 父块旁路存储同步重建

    chunks = chunk_documents(prepared)
    count = index_chunks(chunks, refit_idf=True)

    stats = {
        "documents": len(raw_documents),
        "prepared": len(prepared),
        "chunks": count,
        "total_chunks": store.count(),
        "quality": quality_report(prepared),
        "prepare": None,
    }
    try:
        from app.rag.prepare import get_last_stats
        stats["prepare"] = get_last_stats()
    except Exception:  # noqa: BLE001
        pass

    # 对账：建完就比一次。不一致时**告警**而不是静默 ——
    # 「重跑一次会不会多一份、源删了还在不在」此前没有任何机制会回答这两个问题。
    stats["strategy"] = strategy
    stats["reconcile"] = reconcile()
    if not stats["reconcile"]["consistent"]:
        logger.warning(
            "索引对账不一致：仅向量库有 %s；仅词面索引有 %s。"
            "同一片段应当在两处同时存在，否则会出现「某条知识怎么问都检索不到」",
            stats["reconcile"]["only_in_vector"] or "无",
            stats["reconcile"]["only_in_lexical"] or "无",
        )

    _LAST_INDEX_STATS = stats
    logger.info(
        "L2 索引构建完成：原始 %d 篇 → 清洗后 %d 篇 → 片段 %d 条（策略=%s）",
        len(raw_documents), len(prepared), count, strategy,
    )
    return stats


def add_document(file_name: str, content: str) -> int:
    """增量入库单篇文档（幂等：先清理同名来源，再写入）。"""
    store = get_vector_store()
    # 幂等覆盖：先删除该 source 的旧片段，避免重复上传造成片段累积膨胀
    removed = store.delete_by_source(file_name)
    get_lexical_index().remove_by_source(file_name)   # 词面索引同步增量删除
    if config.PARENT_CHUNK_ENABLED:
        parent_store.remove_by_source(file_name)      # 父块同步删除（否则残留孤儿父块）

    prepared = prepare_single(file_name, content)   # L1
    chunks = chunk_documents(prepared)
    count = index_chunks(chunks, refit_idf=False)
    logger.info("L2 文档入库完成：%s → %d 条片段（替换旧片段 %d 条）", file_name, count, removed)
    return count


def delete_document(file_name: str) -> int:
    """删除指定文档的全部片段。"""
    removed = get_vector_store().delete_by_source(file_name)
    get_lexical_index().remove_by_source(file_name)   # 词面索引同步增量删除
    get_lexical_index().save()
    if config.PARENT_CHUNK_ENABLED:
        parent_store.remove_by_source(file_name)
    logger.info("L2 文档删除完成：%s → 移除 %d 条片段", file_name, removed)
    return removed


def get_index_stats() -> Dict[str, Any]:
    """索引层统计（供 /stats 与 L5 评估消费）。"""
    return dict(_LAST_INDEX_STATS)
