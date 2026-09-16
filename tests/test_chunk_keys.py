"""检索键（``source::chunk_index``）唯一性回归护栏（P1-1）。

背景：为什么单独立一个测试文件
------------------------------
同一个 ``source`` 可以对应**多条 Document** —— PDF 加载器每页产出一条
（``doc_loader._load_pdf_pdfplumber`` / ``_load_pdf_pypdf``），它们的 ``source``
完全相同。这层「一对多」关系此前没有被意识到，``chunk_documents`` 对每条
Document 独立从 0 编号，于是两页的块共用 ``source::0`` / ``source::1``。
后果是**静默丢数据**，而非报错：

- 词面索引 ``add()`` 对同 key 幂等覆盖 → 先写入的块被顶掉，从词面召回里消失；
- 向量库因用内容哈希作 id 不丢，但检索期 ``_hit_key`` 撞键会把两条不同片段
  当成同一条，在 RRF 融合时被去重挤掉。

实测：12 篇语料 176 条切片只得到 174 个键——正好是那份 2 页 PDF 丢的 2 条。
**这类缺陷不会让任何测试变红，只会让「某个问法就是问不出答案」。**
所以这里钉三层：编号语义、真实语料不变量、以及「再撞键时能被发现」。
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from app.rag import indexer
from app.rag.lexical import LexicalIndex, chunk_key


def _long_text(tag: str, paragraphs: int = 12) -> str:
    """构造足够长的中文正文，保证切出多于 1 块（避免被短片段合并吸收）。"""
    unit = (
        f"【{tag}】本节说明相关制度与流程：员工应按要求在系统内提交申请，"
        "经直属主管审批后生效；涉及跨部门的，需补充会签意见后方可流转。"
    )
    return "\n\n".join(f"{unit}（第 {i + 1} 段）" for i in range(paragraphs))


# ---------------------------------------------------------------------------
# 1. 编号语义：同一 source 的多条 Document 必须共用一条递增游标
# ---------------------------------------------------------------------------
def test_chunk_index_continues_across_same_source_documents():
    """模拟 PDF 两页：两条 Document 同 source，块序号必须接续而非各自归零。"""
    docs = [
        Document(page_content=_long_text("第一页"), metadata={"source": "a.pdf", "page": 1}),
        Document(page_content=_long_text("第二页"), metadata={"source": "a.pdf", "page": 2}),
    ]
    chunks = indexer.chunk_documents(docs)

    first_page = [c for c in chunks if c.metadata.get("page") == 1]
    second_page = [c for c in chunks if c.metadata.get("page") == 2]
    assert first_page and second_page, "两页都应切出块（否则用例前提不成立）"

    first_idx = sorted(c.metadata["chunk_index"] for c in first_page)
    second_idx = sorted(c.metadata["chunk_index"] for c in second_page)

    assert first_idx == list(range(len(first_page))), f"首页应从 0 连续编号：{first_idx}"
    assert second_idx[0] == len(first_page), (
        f"第二页应从 {len(first_page)} 接续，实际从 {second_idx[0]} 开始——"
        "这正是「各页各自从 0 编号」的撞键成因"
    )
    assert not set(first_idx) & set(second_idx), "两页的块序号不得重叠"

    keys = [chunk_key(c.metadata) for c in chunks]
    assert len(set(keys)) == len(chunks), f"键冲突：{len(chunks)} 条切片只有 {len(set(keys))} 个键"


def test_single_document_source_still_starts_at_zero():
    """单 Document 来源（.md/.txt）编号行为不变——修复不得误伤多数文档。"""
    chunks = indexer.chunk_documents(
        [Document(page_content=_long_text("单篇"), metadata={"source": "b.txt"})]
    )
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# 2. 真实语料不变量：切片数必须等于唯一键数
# ---------------------------------------------------------------------------
def test_real_corpus_has_no_duplicate_keys():
    """真实语料上「切片数 == 唯一键数」—— P1-1 的验收线。

    这条用例在修复前会失败（176 条 / 174 键）。语料里那份 2 页 PDF
    （星河智能云平台产品白皮书）就是触发条件，换语料后若不变量被破坏
    同样会红，不需要专门依赖那个文件名。
    """
    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    chunks = indexer.chunk_documents(prepare_documents(load_all_documents()))
    keys = [chunk_key(c.metadata) for c in chunks]

    assert len(set(keys)) == len(chunks), (
        f"{len(chunks)} 条切片只得到 {len(set(keys))} 个唯一键，"
        f"将有 {len(chunks) - len(set(keys))} 条片段在词面索引里被静默覆盖"
    )


def test_multi_page_pdf_chunks_have_distinct_keys():
    """把触发条件单独钉住：同 source 的多页 PDF，其块键必须互不相同。"""
    from collections import Counter

    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    raw = load_all_documents()
    pages_per_source = Counter(str(d.metadata.get("source")) for d in raw)
    multi = [s for s, n in pages_per_source.items() if n > 1]
    assert multi, "测试语料应含「一个 source 多条 Document」的文件（多页 PDF）"

    chunks = indexer.chunk_documents(prepare_documents(raw))
    for src in multi:
        idx = [c.metadata["chunk_index"] for c in chunks
               if str(c.metadata.get("source")) == src]
        assert len(set(idx)) == len(idx), f"{src} 的块序号仍有重复：{sorted(idx)}"


# ---------------------------------------------------------------------------
# 3. 两条召回路的键必须同源
# ---------------------------------------------------------------------------
def test_dense_and_lexical_paths_share_the_same_key():
    """dense 路（retriever._hit_key）与 lexical 路必须算出同一个键。

    两处算法一旦分叉（哪怕只是缺失 chunk_index 时的兜底取值不同），
    同一片段就会被认成两条，RRF 融合退化成两份互不相干的结果。
    """
    from app.rag.retriever import _hit_key

    class _Hit:
        def __init__(self, meta):
            self.metadata = meta

    meta = {"source": "c.txt", "chunk_index": 7}
    assert _hit_key(_Hit(meta)) == chunk_key(meta), "两条召回路的检索键算法已分叉"

    # 缺 chunk_index 时也必须一致（历史实现此处一处是 -1、一处是 enumerate 下标）
    bare = {"source": "c.txt"}
    assert _hit_key(_Hit(bare)) == chunk_key(bare)


# ---------------------------------------------------------------------------
# 4. 观测网：再撞键时必须留下痕迹，且不打断索引链路
# ---------------------------------------------------------------------------
def test_collision_detector_reports_duplicates_without_raising(caplog):
    """冲突检测是「观测点」不是「拦截点」：报数 + 告警，绝不抛错。"""
    dup = [
        Document(page_content="甲", metadata={"source": "x.pdf", "chunk_index": 0}),
        Document(page_content="乙", metadata={"source": "x.pdf", "chunk_index": 0}),
        Document(page_content="丙", metadata={"source": "x.pdf", "chunk_index": 1}),
    ]
    with caplog.at_level(logging.WARNING):
        n = indexer._warn_on_key_collision(dup)
    assert n == 1, f"应报出 1 个冲突键，实际 {n}"
    assert "检索键冲突" in caplog.text, "冲突必须留下 WARNING 痕迹"

    ok = [
        Document(page_content="甲", metadata={"source": "y.pdf", "chunk_index": 0}),
        Document(page_content="乙", metadata={"source": "y.pdf", "chunk_index": 1}),
    ]
    assert indexer._warn_on_key_collision(ok) == 0


def test_index_chunks_keeps_every_chunk(tmp_path, monkeypatch, caplog):
    """端到端：N 条切片写入后，词面索引必须恰好有 N 条文档。

    向量库与 embedding 用替身隔离（本用例只验证词面侧不丢片段，
    真实向量化不是它要覆盖的东西）。
    """
    idx = LexicalIndex(persist_path=tmp_path / "lex.json")

    class _Store:
        def add(self, ids, texts, vectors, metas):
            self.count = len(ids)

        def get_texts(self):
            return []

    class _Emb:
        mode = "neural"          # 非 local-hash → 跳过 IDF 训练分支

        def embed_documents(self, texts):
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(indexer, "get_lexical_index", lambda: idx)
    monkeypatch.setattr(indexer, "get_vector_store", lambda: _Store())
    monkeypatch.setattr(indexer, "get_embeddings", lambda: _Emb())
    monkeypatch.setattr(indexer.config, "PARENT_CHUNK_ENABLED", False)

    chunks = [
        Document(page_content="年假天数", metadata={"source": "p.pdf", "chunk_index": 0}),
        Document(page_content="报销流程", metadata={"source": "p.pdf", "chunk_index": 1}),
        Document(page_content="VPN 申请", metadata={"source": "p.pdf", "chunk_index": 2}),
    ]
    with caplog.at_level(logging.WARNING):
        n = indexer.index_chunks(chunks)

    assert n == 3
    assert len(idx) == 3, f"词面索引只有 {len(idx)} 条，应有 3 条（片段被覆盖丢失）"
    assert "检索键冲突" not in caplog.text, "健康数据不该报冲突"

    # 反向验证：故意制造冲突时，必须能被检测到（否则这条护栏是空的）
    bad = [
        Document(page_content="甲", metadata={"source": "q.pdf", "chunk_index": 0}),
        Document(page_content="乙", metadata={"source": "q.pdf", "chunk_index": 0}),
    ]
    with caplog.at_level(logging.WARNING):
        indexer.index_chunks(bad)
    assert len(idx) == 4, "冲突时后写覆盖先写 → 只应留下 1 条新文档"
    assert "检索键冲突" in caplog.text, "冲突未被检测到"
