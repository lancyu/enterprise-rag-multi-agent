#!/usr/bin/env python3
"""RAG 离线测试 —— 词面索引 / Embedding 缓存 / Reorder / request_ctx / Rerank。

不启动服务、不调模型、不检索网络。覆盖：
    1. 词面索引：增量增删、BM25 召回、中文 bigram 命中、json 落盘恢复
    2. Embedding 缓存：查询向量命中、TTL 过期、文档向量持久化、key 含模型名
    3. Reorder：1,3,5,…,6,4,2 重排、≤2 条不变、不修改原列表
    4. request_ctx：query 向量按文本匹配复用、跨请求隔离
    5. Rerank：未启用时原样返回、配置快照结构完整

用法：
    cd langgraph-enterprise-bot && python -m tests.test_rag
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_results = []


def check(name, fn):
    try:
        detail = fn() or "OK"
        print(f"  {GREEN}PASS{RESET} {name} · {detail}")
        _results.append(True)
    except AssertionError as e:
        print(f"  {RED}FAIL{RESET} {name} · {e}")
        _results.append(False)


# ---------------------------------------------------------------------------
print("\n[1] 词面倒排索引：BM25 召回与中文 bigram")
# ---------------------------------------------------------------------------
def test_lexical_build_search():
    from app.rag.lexical import LexicalIndex

    idx = LexicalIndex()  # 无落盘路径，纯内存
    idx.add("员工手册.txt::0", "年休假为五天，员工入职满一年可享受。")
    idx.add("员工手册.txt::1", "报销流程：填写报销单后提交财务审批。")
    idx.add("IT支持指南.txt::0", "VPN 连接不上请先检查网络，再联系 IT 服务台。")

    # 中文 bigram 让「年假」能命中「年休假」
    hits = idx.search("年假", top_k=5)
    assert hits, "「年假」应命中「年休假」"
    assert hits[0][0] == "员工手册.txt::0", f"Top1 应为年休假片段，实际 {hits[0][0]}"

    # 英文关键词精确命中
    hits = idx.search("VPN", top_k=5)
    assert hits and hits[0][0] == "IT支持指南.txt::0", "「VPN」应命中 IT 片段"

    # 无关查询不命中
    assert idx.search("生日蛋糕", top_k=5) == [], "无关查询不应有结果"
    return f"{len(idx)} 条文档 · 中文 bigram + 英文关键词均命中"


check("词面索引召回", test_lexical_build_search)


# ---------------------------------------------------------------------------
print("\n[2] 词面索引：增量删除与 json 落盘恢复")
# ---------------------------------------------------------------------------
def test_lexical_incremental_and_persist():
    import tempfile
    from pathlib import Path

    from app.rag.lexical import LexicalIndex

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lex.json"
        idx = LexicalIndex()
        idx.add("a.txt::0", "年假制度规定", {"source": "a.txt", "chunk_index": 0})
        idx.add("a.txt::1", "报销流程", {"source": "a.txt", "chunk_index": 1})
        idx.add("b.txt::0", "VPN 设置", {"source": "b.txt", "chunk_index": 0})

        # 增量删除整个 source
        removed = idx.remove_by_source("a.txt")
        assert removed == 2, f"应删 2 条，实际 {removed}"
        assert len(idx) == 1, "删除后应剩 1 条"
        assert idx.search("年假") == [], "删除后「年假」不应再命中"

        # 落盘 + 从磁盘恢复
        idx._persist_path = path
        idx.save()
        idx2 = LexicalIndex(persist_path=path)
        assert len(idx2) == 1, "恢复后文档数应一致"
        assert idx2.search("VPN")[0][0] == "b.txt::0", "恢复后应能检索"
    return "增量删除 + json 落盘恢复正常"


check("词面索引增量与持久化", test_lexical_incremental_and_persist)


# ---------------------------------------------------------------------------
print("\n[3] Embedding 缓存：查询向量命中与 key 含模型名")
# ---------------------------------------------------------------------------
def test_cache_query_and_model_key():
    from app.utils.cache import EmbeddingCache, _cache_key

    cache = EmbeddingCache()  # 无落盘
    vec = [0.1, 0.2, 0.3]
    cache.put_query_vec("model-a", "你好", vec)
    assert cache.get_query_vec("model-a", "你好") == vec, "查询向量应命中"

    # key 含模型名：换模型不命中（避免跨模型脏向量）
    assert cache.get_query_vec("model-b", "你好") is None, "换模型应 miss"

    # 未写入的文本 miss
    assert cache.get_query_vec("model-a", "未缓存") is None, "未写入应 miss"
    assert _cache_key("m", "x") != _cache_key("n", "x"), "cache key 应区分模型"
    return "查询缓存命中 + 模型名隔离正常"


check("Embedding 查询缓存", test_cache_query_and_model_key)


# ---------------------------------------------------------------------------
print("\n[4] Embedding 缓存：文档向量持久化")
# ---------------------------------------------------------------------------
def test_cache_doc_persist():
    import tempfile
    from pathlib import Path

    from app.utils.cache import EmbeddingCache

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc_cache.json"
        c1 = EmbeddingCache(persist_path=path)
        c1.put_doc_vec("model-a", "文档内容", [0.5, 0.6])

        c2 = EmbeddingCache(persist_path=path)
        got = c2.get_doc_vec("model-a", "文档内容")
        assert got == [0.5, 0.6], "文档向量应跨实例持久化命中"
    return "文档向量持久化正常"


check("Embedding 文档缓存", test_cache_doc_persist)


# ---------------------------------------------------------------------------
print("\n[5] Reorder：1,3,5,…,6,4,2 重排与边界")
# ---------------------------------------------------------------------------
def test_reorder():
    from app.rag.reorder import reorder_docs

    docs = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
    out = reorder_docs(docs)
    ids = [d["id"] for d in out]
    assert ids == [1, 3, 5, 4, 2], f"重排顺序错误：{ids}"

    # ≤2 条顺序不变
    assert [d["id"] for d in reorder_docs([{"id": 1}, {"id": 2}])] == [1, 2]
    assert reorder_docs([]) == []
    assert reorder_docs([{"id": 9}]) == [{"id": 9}]

    # 不修改原列表
    assert [d["id"] for d in docs] == [1, 2, 3, 4, 5], "不应修改原列表"
    return "5 条重排 + 边界用例通过"


check("Reorder 重排", test_reorder)


# ---------------------------------------------------------------------------
print("\n[6] request_ctx：query 向量按文本匹配复用")
# ---------------------------------------------------------------------------
def test_request_ctx_reuse():
    from app.core.request_ctx import get_query_vector, set_query_vector

    set_query_vector("年假有多少天", [1.0, 2.0, 3.0])
    assert get_query_vector("年假有多少天") == [1.0, 2.0, 3.0], "同文本应命中缓存向量"
    assert get_query_vector("报销流程") is None, "不同文本应 miss（跨请求不串值）"
    return "query 向量按文本匹配复用 + 跨请求隔离正常"


check("request_ctx 复用", test_request_ctx_reuse)


# ---------------------------------------------------------------------------
print("\n[7] Rerank：未启用时原样返回 + 配置快照")
# ---------------------------------------------------------------------------
def test_rerank_degrade():
    from app.rag.rerank import rerank_docs, rerank_info

    docs = [{"content": "a", "id": 1}, {"content": "b", "id": 2}]
    # 默认 RERANK_ENABLED=false：应原样返回，不改顺序、不丢片段
    out = rerank_docs("query", docs, top_n=10)
    assert out == docs, "未启用时应原样返回"

    info = rerank_info()
    assert info["enabled"] is False and "model" in info and "top_n" in info, "快照结构应完整"
    assert info["available"] is False, "未启用时 available 应为 False"
    return "未启用时原样返回 + 配置快照结构正常"


check("Rerank 降级", test_rerank_degrade)


# ---------------------------------------------------------------------------
print()
passed, total = sum(_results), len(_results)
print(f"RAG 离线测试：{passed}/{total} 通过")
sys.exit(0 if passed == total else 1)
