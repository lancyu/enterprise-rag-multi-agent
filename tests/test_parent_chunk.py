"""父子双层索引测试（T6-1）。

覆盖两件事：
1. **默认关闭时必须是彻底的空操作**（零变化），这是接线的硬约束；
2. 启用时父块按章内聚、跨章/跨源必切，且生成端优先消费父块。
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app import config
from app.rag import indexer, parent_store
from app.rag.generator import build_context


@pytest.fixture
def docs() -> list:
    """4 个子块：a.txt 两章（第一章 2 块）、b.txt 一章。"""
    return [
        Document(page_content="【章节】第一章 总则\n\n第一条 适用范围",
                 metadata={"source": "a.txt", "chapter": "第一章 总则", "chunk_index": 0}),
        Document(page_content="【章节】第一章 总则\n\n第二条 基本原则",
                 metadata={"source": "a.txt", "chapter": "第一章 总则", "chunk_index": 1}),
        Document(page_content="【章节】第二章 细则\n\n第三条 具体要求",
                 metadata={"source": "a.txt", "chapter": "第二章 细则", "chunk_index": 2}),
        Document(page_content="【章节】第三章 附则\n\n第四条 生效",
                 metadata={"source": "b.txt", "chapter": "第三章 附则", "chunk_index": 0}),
    ]


@pytest.fixture(autouse=True)
def _isolated_parent_store(tmp_path, monkeypatch):
    """把父块存储隔离到临时目录，避免测试污染真实 vector_store/。"""
    monkeypatch.setattr(config, "PARENT_STORE_PATH", str(tmp_path / "parents.json"))
    parent_store._loaded = False
    parent_store._store.clear()
    yield
    parent_store._loaded = False
    parent_store._store.clear()


def test_disabled_is_complete_noop(docs, monkeypatch):
    """关闭时：返回 0、不写 parent_id、不落盘。"""
    monkeypatch.setattr(config, "PARENT_CHUNK_ENABLED", False)
    before = [dict(d.metadata) for d in docs]

    assert indexer.attach_parents(docs) == 0
    assert [d.metadata for d in docs] == before, "关闭时不得修改任何 metadata"
    assert parent_store.count() == 0


def test_enabled_splits_by_chapter_and_source(docs, monkeypatch):
    """启用时：同章合并，跨章与跨源必切 → 3 个父块。"""
    monkeypatch.setattr(config, "PARENT_CHUNK_ENABLED", True)
    monkeypatch.setattr(config, "PARENT_MAX_CHARS", 1000)

    n = indexer.attach_parents(docs)

    assert n == 3, "a:第一章(2 子块) / a:第二章 / b:第三章"
    # 前两个子块共享同一父块
    assert docs[0].metadata["parent_id"] == docs[1].metadata["parent_id"]
    # 第三章（跨章）与 b.txt（跨源）各自独立
    assert docs[2].metadata["parent_id"] != docs[0].metadata["parent_id"]
    assert docs[3].metadata["parent_id"] != docs[2].metadata["parent_id"]


def test_parent_content_merges_children_and_dedupes_header(docs, monkeypatch):
    """父块正文应包含全部子块内容，且重复的章头只保留一次。"""
    monkeypatch.setattr(config, "PARENT_CHUNK_ENABLED", True)
    monkeypatch.setattr(config, "PARENT_MAX_CHARS", 1000)

    indexer.attach_parents(docs)
    content = parent_store.get_content(docs[0].metadata["parent_id"])

    assert "第一条 适用范围" in content
    assert "第二条 基本原则" in content
    # 同章相邻子块章头相同 → 只出现一次（省 prompt token）
    assert content.count("【章节】第一章 总则") == 1


def test_parent_respects_max_chars(monkeypatch):
    """超过 PARENT_MAX_CHARS 时切分，保证父块不会无限膨胀。"""
    monkeypatch.setattr(config, "PARENT_CHUNK_ENABLED", True)
    monkeypatch.setattr(config, "PARENT_MAX_CHARS", 120)

    docs = [
        Document(page_content=f"内容{'一二三四五六七八九十' * 4}段落{i}",
                 metadata={"source": "x.txt", "chapter": "第一章", "chunk_index": i})
        for i in range(6)
    ]
    n = indexer.attach_parents(docs)
    assert n >= 2, "每块约 44 字，上限 120 字 → 必然切成多个父块"
    assert parent_store.stats()["mean_chars"] <= 200


def test_build_context_prefers_parent_content(docs, monkeypatch):
    """生成端优先用父块正文；没有 parent_content 时回退子块。"""
    monkeypatch.setattr(config, "PARENT_CHUNK_ENABLED", True)
    monkeypatch.setattr(config, "PARENT_MAX_CHARS", 1000)
    indexer.attach_parents(docs)
    parent_text = parent_store.get_content(docs[0].metadata["parent_id"])

    doc = {"source": "a.txt", "content": "子块内容", "parent_content": parent_text}
    context, _citations = build_context([doc])
    assert parent_text in context
    assert "子块内容" not in context

    # 未启用（无 parent_content 字段）→ 行为与改造前一致
    plain = {"source": "a.txt", "content": "子块内容"}
    context2, _ = build_context([plain])
    assert "子块内容" in context2
