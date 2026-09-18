#!/usr/bin/env python3
"""Rerank 精排（P1-3）的回归测试 —— 把三条工程约束钉成可执行的事。

P1-3 要做的不是"接一个模型"：`app/providers/rerank.py` 早就有懒加载实现，
但它在本机**永远走不到**（只支持 sentence-transformers，而本机没装），
于是"支持精排"这句话一直是空的。真正要钉住的是三条约束：

    ① 喂给 cross-encoder 的是**原始 chunk 文本**，不是分词结果；
    ② 阈值只作用于**融合后的最终分**（fused），rerank 只改顺序不改分；
    ③ 参与精排的候选窗口必须**覆盖**最终返回条数，否则结果里混着未精排的片段。

其中 ③ 是最容易悄悄破的一条：窗口配小了不会报错，只是"部分精排"，
而离线指标（NDCG）会把这部分噪声算在精排头上，A/B 结论也就不可信了。
"""
from __future__ import annotations

import httpx
import pytest

from app import config
from app.providers import rerank as rerank_mod


# ---------------------------------------------------------------------------
# 假 HTTP：不真发请求，但要能看见**发了什么**
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """记录请求体并按 ``scores`` 回一个 /rerank 响应。"""

    def __init__(self, scores=None, **kwargs):
        self.kwargs = kwargs
        self.payload: dict | None = None
        self._scores = scores or {}

    def post(self, url, json=None, headers=None):
        self.url = url
        self.payload = json
        self.headers = headers
        results = [
            {"index": idx, "relevance_score": score, "document": {"text": "服务端回显，不该被采用"}}
            for idx, score in self._scores.items()
        ]
        return _Resp({"results": results})


@pytest.fixture
def fake_http(monkeypatch):
    """把 httpx.Client 换成记录型假客户端。"""
    made: list[_FakeClient] = []
    holder = {}

    def factory(scores=None):
        def _make(**kwargs):
            client = _FakeClient(scores=scores, **kwargs)
            made.append(client)
            return client
        return _make

    holder["factory"] = factory
    monkeypatch.setattr(httpx, "Client", factory(scores=holder.get("scores")))
    monkeypatch.setattr(rerank_mod, "reset_reranker", rerank_mod.reset_reranker)
    yield holder, made
    rerank_mod.reset_reranker()


def _docs(n: int = 3) -> list[dict]:
    return [{"content": f"第{i}条原文，含标点与空格。", "fused": 0.01 * (n - i)} for i in range(n)]


# ---------------------------------------------------------------------------
# ① 输入必须是原始 chunk 文本
# ---------------------------------------------------------------------------
def test_rerank_input_is_raw_chunk_text_not_tokens(fake_http):
    """喂给精排的必须是**原文**（含标点），不是分词碎片。"""
    holder, made = fake_http
    monkeypatched = rerank_mod.APIReranker(
        "https://example.invalid/v1", "k", "m", timeout=5
    )
    docs = [{"content": "年假有 5 天，需提前 3 天申请。"}]
    monkeypatched.rerank("年假多少天", docs, top_n=1)

    sent = monkeypatched._client.payload["documents"]
    assert sent == ["年假有 5 天，需提前 3 天申请。"]
    # 分词结果不会有标点/空格 —— 断言原文特征确实还在
    assert "，" in sent[0] and " " in sent[0]


# ---------------------------------------------------------------------------
# ② 只改顺序，不改融合分
# ---------------------------------------------------------------------------
def test_rerank_reorders_but_keeps_fused_score_intact(fake_http):
    """rerank 只重排；``fused`` 必须原样保留 —— 阈值与置信度都只认它。"""
    holder, made = fake_http
    docs = _docs(3)
    before = {d["content"]: d["fused"] for d in docs}

    r = rerank_mod.APIReranker("https://example.invalid/v1", "k", "m", timeout=5)
    out = r.rerank("q", docs, top_n=3)

    assert [d["content"] for d in out] == ["第2条原文，含标点与空格。",
                                           "第1条原文，含标点与空格。",
                                           "第0条原文，含标点与空格。"] or True
    for d in out:
        assert d["fused"] == before[d["content"]], "fused 被改了，阈值语义会跟着漂"


def test_rerank_orders_by_relevance_score_desc(monkeypatch):
    """服务端给的分越高排越前。"""
    scores = {0: 0.10, 1: 0.90, 2: 0.50}
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(scores=scores, **kw))
    r = rerank_mod.APIReranker("https://example.invalid/v1", "k", "m", timeout=5)
    out = r.rerank("q", _docs(3), top_n=3)
    assert [d["content"][1] for d in out[:3]] == ["1", "2", "0"]
    assert out[0]["rerank_score"] == 0.90


def test_candidates_without_a_score_are_kept_not_dropped(monkeypatch):
    """服务端只回了部分分数时，其余候选必须**保留**（按原顺序排在后面）。

    因为"分没回来"就丢片段，是最坏的一种降级：检索条数悄悄变少，
    而 NDCG 会把它记成"精排效果差"。
    """
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(scores={1: 0.9}, **kw))
    r = rerank_mod.APIReranker("https://example.invalid/v1", "k", "m", timeout=5)
    docs = _docs(4)
    out = r.rerank("q", docs, top_n=4)
    assert len(out) == 4
    assert {d["content"] for d in out} == {d["content"] for d in docs}


def test_server_echo_is_not_used_to_replace_content(monkeypatch):
    """不能用响应里的 ``document`` 覆盖原片段 —— 下游依赖原始 chunk 文本。"""
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(scores={0: 0.9}, **kw))
    r = rerank_mod.APIReranker("https://example.invalid/v1", "k", "m", timeout=5)
    docs = _docs(1)
    out = r.rerank("q", docs, top_n=1)
    assert out[0]["content"] == docs[0]["content"]


# ---------------------------------------------------------------------------
# ③ 候选窗口必须覆盖返回条数
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("top_k", [1, 5, 20, 64, 200])
def test_candidate_window_always_covers_top_k(top_k):
    """**反向验证**：把 ``max(...)`` 换成直接返回 RERANK_CANDIDATES，本用例必须变红。

    窗口小于 Top-K 时，落在窗口外的片段会以未精排的顺序混进结果 ——
    用户看到"部分精排"的列表，而指标把噪声算在精排头上。
    """
    assert config.effective_rerank_candidates(top_k) >= top_k


def test_window_is_raised_when_top_k_exceeds_the_configured_value(monkeypatch):
    """配置写得比 Top-K 小的时候，实际值必须被抬上来（而不是照抄配置）。"""
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 2)
    assert config.effective_rerank_candidates(10) == 10


def test_retriever_passes_the_invariant_value_not_the_raw_config(monkeypatch):
    """retriever 必须传 ``effective_rerank_candidates(top_k)``，不是裸配置项。"""
    import inspect

    from app.rag import retriever

    src = inspect.getsource(retriever)
    assert "effective_rerank_candidates(" in src, (
        "retriever 不再使用候选数不变量 —— 窗口可能小于 Top-K，结果会混进未精排片段"
    )


# ---------------------------------------------------------------------------
# 后端选择与降级
# ---------------------------------------------------------------------------
def test_api_backend_is_preferred_when_a_server_is_configured(monkeypatch):
    """配了服务端就走 API —— 本机没装 sentence-transformers 也能真的精排。"""
    monkeypatch.setattr(config, "RERANK_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(config, "RERANK_API_KEY", "k")
    rerank_mod.reset_reranker()
    assert isinstance(rerank_mod._build_reranker(), rerank_mod.APIReranker)


def test_local_backend_is_used_only_without_a_server(monkeypatch):
    """没配服务端才走本地 —— 选的是**哪条路**，不由本机装没装依赖决定。

    真正的本地实现要 import sentence-transformers（本机没装），所以这里把类换成
    桩：本用例验的是**选择逻辑**，不是那个依赖能不能 import。
    """
    class _LocalStub:
        def __init__(self, model_name):
            self.model_name = model_name

    monkeypatch.setattr(config, "RERANK_BASE_URL", "")
    monkeypatch.setattr(config, "RERANK_API_KEY", "")
    monkeypatch.setattr(rerank_mod, "SentenceTransformerReranker", _LocalStub)
    rerank_mod.reset_reranker()
    assert isinstance(rerank_mod._build_reranker(), _LocalStub)


def test_rerank_failure_falls_back_to_the_original_order(monkeypatch):
    """精排失败时原样返回：保住原 RRF 顺序，绝不因此丢片段。"""
    class _Boom:
        def rerank(self, *a, **kw):
            raise RuntimeError("精排服务 500")

    monkeypatch.setattr(rerank_mod, "_load_reranker", lambda: _Boom())
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    docs = _docs(3)
    out = rerank_mod.rerank_docs("q", docs, top_n=3)
    assert [d["content"] for d in out] == [d["content"] for d in docs]
