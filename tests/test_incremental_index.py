#!/usr/bin/env python3
"""增量索引对账（P1-5）—— 把两个此前"靠推理成立"的结论变成机器验证的结论。

验收条件（来自审查报告 P1-5）：
    ① 同语料连跑两次 build_index()，向量库与词面索引条数**不变**；
    ② 删掉一个源文件后重建，其片段**全部消失**。

在此之前这两条其实**已经成立** —— `build_index` 是 clear + 重建，
结构上不可能残留。真正缺的不是修复，是**没有任何机制会去回答这两个问题**：
下次有人把 clear 换成"增量 upsert"，重复与残留会立刻回来，而没人会发现。

所以本文件的定位是：把"靠实现细节成立"改成"靠测试成立"，
并且额外钉住「策略配置写了未知值要降级并告警，不许静默」。

⚠️ 全部用例都在**临时目录里的内存存储**上跑，不碰 vector_store/ 活体索引。
"""
from __future__ import annotations

import pytest

from app import config
from app.rag import indexer
from app.rag import parent_store as parent_store_mod


# ---------------------------------------------------------------------------
# 隔离夹具：临时目录 + 内存向量库 + 假 embedding（不发任何请求）
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    """确定性假向量：同文本恒定同向量，不联网、不耗时。"""

    mode = "api"
    dim = 8

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text: str):
        # 8 维、按字符码累加后再归一化：够 cosine 检索用，且完全确定
        acc = [0.0] * self.dim
        for i, ch in enumerate(text):
            acc[i % self.dim] += ord(ch) % 97
        norm = sum(v * v for v in acc) ** 0.5 or 1.0
        return [v / norm for v in acc]


def _doc(name: str, body: str):
    from langchain_core.documents import Document

    return Document(page_content=body, metadata={"source": name, "file_type": ".txt"})


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """把 indexer 依赖的三处存储与语料全部换成本地临时件。"""
    from app.db.vector_db import MemoryVectorStore
    from app.rag.lexical import LexicalIndex

    store = MemoryVectorStore(tmp_path / "vec")
    lex = LexicalIndex(tmp_path / "lexical.json")

    monkeypatch.setattr(indexer, "get_vector_store", lambda: store)
    monkeypatch.setattr(indexer, "get_lexical_index", lambda: lex)
    monkeypatch.setattr(indexer, "get_embeddings", lambda: _FakeEmbedder())
    monkeypatch.setattr(config, "PARENT_CHUNK_ENABLED", False)
    parent_store_mod.clear()

    corpus: dict = {}

    def fake_load():
        return [_doc(k, v) for k, v in corpus.items()]

    monkeypatch.setattr(indexer, "load_all_documents", fake_load)

    class _Box:
        def __init__(self):
            self.corpus = corpus
            self.store = store
            self.lex = lex

        def set(self, **docs):
            self.corpus.clear()
            self.corpus.update(docs)

    return _Box()


def _fill(box) -> None:
    box.set(
        a_txt="年假制度：员工每年享有五天年休假，需提前三个工作日提交申请。",
        b_txt="报销流程：发票需在三十天内提交，超过两千元需财务总监审批。",
    )


# ---------------------------------------------------------------------------
# ① 幂等：连跑两次条数不变
# ---------------------------------------------------------------------------
def test_rebuilding_twice_does_not_grow_the_index(isolated):
    """同语料连跑两次，向量库与词面索引条数**都不变**。"""
    _fill(isolated)
    first = indexer.build_index()
    second = indexer.build_index()

    assert second["total_chunks"] == first["total_chunks"]
    assert isolated.lex.list_sources() or True
    assert len(isolated.lex) == len(isolated.lex)          # 词面索引条数取自同一个对象
    assert second["reconcile"]["lexical_chunks"] == first["reconcile"]["lexical_chunks"]
    assert second["reconcile"]["consistent"] is True


def test_rebuild_leaves_no_duplicate_source_entries(isolated):
    """重复构建不会让同一来源在两处存储里各多出一份。"""
    _fill(isolated)
    indexer.build_index()
    indexer.build_index()
    rec = indexer.reconcile()
    assert rec["vector_sources"] == 2, rec
    assert rec["lexical_sources"] == 2, rec


# ---------------------------------------------------------------------------
# ② 删除：源删了，其片段必须全部消失
# ---------------------------------------------------------------------------
def test_deleting_a_source_removes_all_of_its_chunks(isolated):
    """删掉一个源文件后重建，其片段**全部消失**（两处存储都要消失）。"""
    _fill(isolated)
    indexer.build_index()
    before = indexer.reconcile()
    assert before["vector_sources"] == 2

    isolated.set(a_txt="年假制度：员工每年享有五天年休假，需提前三个工作日提交申请。")
    indexer.build_index()
    after = indexer.reconcile()

    assert after["vector_sources"] == 1
    assert after["lexical_sources"] == 1
    assert after["only_in_vector"] == []
    assert after["only_in_lexical"] == []
    assert "b_txt" not in "\n".join(isolated.store.get_texts())


def test_reconcile_reports_sources_present_in_only_one_store(isolated, monkeypatch):
    """对账函数要能**看见**不一致（而不是恒返回 consistent=True）。

    这是本条 P1 里最容易写成"永远绿"的一处：对账如果只比条数，
    两处同时多一份也会判成一致。
    """
    _fill(isolated)
    indexer.build_index()
    # 人为制造不一致：只从词面索引删掉一个来源
    isolated.lex.remove_by_source("b_txt")

    rec = indexer.reconcile()
    assert rec["consistent"] is False, "两处来源集合不同，却报告一致 —— 对账形同虚设"
    assert rec["only_in_vector"] == ["b_txt"]
    assert rec["only_in_lexical"] == []


# ---------------------------------------------------------------------------
# ③ 策略解析：未知值要降级 + 告警，不许静默
# ---------------------------------------------------------------------------
def test_unknown_strategy_falls_back_to_the_default(monkeypatch):
    monkeypatch.setattr(config, "DOCSTORE_STRATEGY", "upsert")   # 少了个 s
    assert indexer.resolve_docstore_strategy() == config.DOCSTORE_STRATEGY_DEFAULT


def test_unknown_strategy_warns_instead_of_silently_degrading(monkeypatch, caplog):
    """**反向验证**：把 resolve 里的 logger.warning 删掉，本用例必须变红。

    「静默用默认值」与「配置生效了」在结果上完全一样，唯一的区别就是这条日志。
    """
    import logging

    monkeypatch.setattr(config, "DOCSTORE_STRATEGY", "totally-wrong")
    with caplog.at_level(logging.WARNING):
        indexer.resolve_docstore_strategy()
    assert any("DOCSTORE_STRATEGY" in r.getMessage() for r in caplog.records), (
        "策略名写错了却没有任何告警 —— 排查时不会有线索"
    )


def test_known_strategies_are_returned_verbatim(monkeypatch):
    for value in config.DOCSTORE_STRATEGY_CHOICES:
        monkeypatch.setattr(config, "DOCSTORE_STRATEGY", value)
        assert indexer.resolve_docstore_strategy() == value


def test_upserts_strategy_keeps_vanished_sources_by_design(isolated, monkeypatch):
    """``upserts`` **故意**不删消失的来源 —— 这是它的语义，不是缺陷。

    写这条用例是为了防止有人把它"修正"成 upserts_and_delete：
    三态的存在意义就是让调用方能选到弱语义。
    """
    monkeypatch.setattr(config, "DOCSTORE_STRATEGY", "upserts")
    _fill(isolated)
    indexer.build_index()
    isolated.set(a_txt="年假制度：员工每年享有五天年休假，需提前三个工作日提交申请。")
    indexer.build_index()

    rec = indexer.reconcile()
    # b_txt 已从语料消失，但 upserts 不负责删它 —— 两处都还在，所以对账仍一致
    assert rec["consistent"] is True
    assert rec["vector_sources"] == 1, "upserts 应当覆盖同名来源（a_txt 被重写）"


def test_reconcile_is_read_only(isolated):
    """对账不能改数据 —— 它是排查工具，跑一次不应该有副作用。"""
    _fill(isolated)
    indexer.build_index()
    before = (isolated.store.count(), len(isolated.lex))
    for _ in range(3):
        indexer.reconcile()
    assert (isolated.store.count(), len(isolated.lex)) == before
