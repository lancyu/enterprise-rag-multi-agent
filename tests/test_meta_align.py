"""两条召回路的片段元数据必须同源。

背景：向量库落库的是**完整** `chunk.metadata`，而词面索引曾另立一份 6 字段
白名单，使同一条片段在两条路上的 metadata 不一致——词面路命中的片段缺
`page` / `doc_version` / `doc_title` / `file_type` 等字段，且每次新增字段
都必然漏改。Dify 的 `metadata_condition` 过滤据此会「条件恒假、静默返回空」。

与检索键冲突（`test_chunk_keys.py`）同源：都是**同一语义在两处各定义一遍**。
一处是键，一处是字段。

本组用例钉三件事：
1. 词面路 metadata 键集必须覆盖 chunk.metadata（全量继承）；
2. 关键业务字段确实能到达词面路（不是"继承了空值"）；
3. 真实语料上同样成立——避免只在构造数据上成立。
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.rag import indexer
from app.rag.lexical import LexicalIndex, chunk_key


def _install_fakes(monkeypatch, tmp_path):
    """隔离向量库与 embedding：本组用例只验证词面侧口径，不消耗 API 配额。"""
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
    return idx


def test_lexical_meta_covers_chunk_meta(tmp_path, monkeypatch):
    """词面路 metadata 的键集必须覆盖 chunk.metadata——全量继承，不是白名单。"""
    idx = _install_fakes(monkeypatch, tmp_path)
    chunk = Document(
        page_content="年假天数与折算规则",
        metadata={
            "source": "data/员工手册.txt", "chunk_index": 0,
            "page": 3, "doc_title": "员工手册", "doc_version": "V3.2",
            "doc_effective_date": "2026-01-01", "file_type": "txt",
            "fingerprint": "abc123", "chunk_chars": 12,
        },
    )

    indexer.index_chunks([chunk])

    meta = idx.doc_meta(chunk_key(chunk.metadata))
    missing = set(chunk.metadata) - set(meta)
    assert not missing, f"词面路 metadata 缺字段 {sorted(missing)}——两路口径又分叉了"


def test_business_fields_reach_lexical_path(tmp_path, monkeypatch):
    """关键业务字段要真的带上值，而不是"继承了同名的空字段"。"""
    idx = _install_fakes(monkeypatch, tmp_path)
    chunk = Document(
        page_content="报销标准与审批权限",
        metadata={"source": "data/财务报销管理制度.pdf", "chunk_index": 4, "page": 2,
                  "doc_version": "V2.0", "doc_effective_date": "2025-09-01",
                  "doc_title": "财务报销管理制度", "file_type": "pdf"},
    )

    indexer.index_chunks([chunk])

    meta = idx.doc_meta(chunk_key(chunk.metadata))
    for field in ("page", "doc_version", "doc_effective_date", "doc_title", "file_type"):
        assert meta.get(field) not in (None, ""), f"词面路丢了 {field}"


def test_parent_id_is_inherited_not_injected(tmp_path, monkeypatch):
    """`parent_id` 由上游继承而来，而不是白名单里硬塞的空串。

    未启用父子块时 `attach_parents` 直接返回、上游本就不写该字段，
    消费方（`retriever._attach_parent_content`）用 `.get(...) or ""` 兜底。
    硬塞空串会掩盖"该字段其实不存在"这一事实。
    """
    idx = _install_fakes(monkeypatch, tmp_path)
    chunks = [
        Document(page_content="甲", metadata={"source": "s.txt", "chunk_index": 0,
                                              "parent_id": "pid-1"}),
        Document(page_content="乙", metadata={"source": "s.txt", "chunk_index": 1}),
    ]

    indexer.index_chunks(chunks)

    assert idx.doc_meta(chunk_key(chunks[0].metadata))["parent_id"] == "pid-1"
    assert "parent_id" not in idx.doc_meta(chunk_key(chunks[1].metadata)), (
        "上游没有 parent_id 时不该凭空造一个空串键"
    )


def test_real_corpus_lexical_meta_matches_chunk_meta(tmp_path, monkeypatch):
    """真实语料复验：每条切片的词面 metadata 都要覆盖 chunk.metadata。

    只用构造数据验证有盲区——真实文档（PDF 多页、结构化/扁平混排、
    带/不带文档级元数据）才会暴露字段缺失。
    """
    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    idx = _install_fakes(monkeypatch, tmp_path)
    chunks = indexer.chunk_documents(prepare_documents(load_all_documents()))
    if not chunks:
        pytest.skip("语料目录为空，无法复验")

    indexer.index_chunks(chunks)

    assert len(idx) == len(chunks), (
        f"词面索引 {len(idx)} 条 != 切片 {len(chunks)} 条（有片段被覆盖丢失）"
    )
    for c in chunks:
        key = chunk_key(c.metadata)
        meta = idx.doc_meta(key)
        assert meta, f"词面索引缺 {key}"
        missing = set(c.metadata) - set(meta)
        assert not missing, f"{key} 缺字段 {sorted(missing)}"
