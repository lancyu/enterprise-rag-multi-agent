"""Dify 外部知识库接口测试（T-DIFY）。

只测**适配层**：鉴权、分数归一化、响应字段格式、过滤与限流。
检索本身由其他用例覆盖，这里把 `retriever.retrieve` 打桩 —— 测试跑得快，
也不消耗 embedding 配额。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.rag import retriever

client = TestClient(app)   # 不用 with，避免触发 lifespan 的启动自检（会真实调 LLM）

# fused 取 RRF 理论上限与其一半，用于验证归一化结果约等于 1.0 / 0.5
FAKE_HITS = [
    {
        "content": "年假天数正文",
        "source": "data/员工手册.txt",
        "score": 0.6, "lexical": 0.4, "fused": 0.0164,
        "metadata": {"source": "data/员工手册.txt", "chapter": "第三章 假期管理",
                     "section": "3.2 年休假", "heading_path": "第三章 假期管理 > 3.2 年休假",
                     "chunk_index": 3,
                     "doc_title": "星河科技有限公司 · 员工手册", "doc_version": "V3.2",
                     "doc_effective_date": "2026-01-01", "file_type": "txt"},
    },
    {
        "content": "报销正文",
        "source": "data/财务报销管理制度.txt",
        "score": 0.5, "lexical": 0.3, "fused": 0.0082,
        "metadata": {"source": "data/财务报销管理制度.txt", "chapter": "第三章 报销流程",
                     "section": "3.1 提交申请", "heading_path": "第三章 报销流程 > 3.1 提交申请",
                     "chunk_index": 5,
                     "doc_title": "星河科技有限公司 · 财务报销管理制度", "doc_version": "V2.0",
                     "doc_effective_date": "2025-09-01", "file_type": "txt"},
    },
]


@pytest.fixture(autouse=True)
def _stub_retrieve(monkeypatch):
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=None: FAKE_HITS)


@pytest.fixture
def _auth(monkeypatch):
    monkeypatch.setattr(config, "DIFY_API_KEY", "test-key-123")
    monkeypatch.setattr(config, "DIFY_KNOWLEDGE_ID", "")
    return "test-key-123"


def _post(payload: dict, headers: dict | None = None, url: str = "/retrieval"):
    return client.post(url, json=payload, headers=headers or {})


# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------
def test_server_key_missing(monkeypatch):
    """服务端未配 DIFY_API_KEY → 拒绝服务（安全默认），且带明确提示。"""
    monkeypatch.setattr(config, "DIFY_API_KEY", "")
    r = _post({"knowledge_id": "kb", "query": "年假"}, {"Authorization": "Bearer anything"})
    assert r.status_code == 200
    assert r.json()["error_code"] == 1002
    assert "DIFY_API_KEY" in r.json()["error_msg"]


def test_missing_auth_header(_auth):
    r = _post({"knowledge_id": "kb", "query": "年假"})
    assert r.json()["error_code"] == 1001


def test_bad_auth_format(_auth):
    r = _post({"knowledge_id": "kb", "query": "年假"}, {"Authorization": "Basic abc"})
    assert r.json()["error_code"] == 1001


def test_wrong_key(_auth):
    r = _post({"knowledge_id": "kb", "query": "年假"}, {"Authorization": "Bearer wrong"})
    assert r.json()["error_code"] == 1002


def test_correct_key_passes(_auth):
    r = _post({"knowledge_id": "kb", "query": "年假"}, {"Authorization": f"Bearer {_auth}"})
    assert "records" in r.json()


# ---------------------------------------------------------------------------
# 响应契约
# ---------------------------------------------------------------------------
def test_response_shape(_auth):
    r = _post({"knowledge_id": "kb", "query": "年假",
               "retrieval_setting": {"top_k": 5, "score_threshold": 0.0}},
              {"Authorization": f"Bearer {_auth}"})
    records = r.json()["records"]
    assert len(records) == 2
    for rec in records:
        assert set(rec) == {"content", "score", "title", "metadata"}
        assert isinstance(rec["metadata"], dict), "Dify 要求 metadata 必须是对象，不能为 null"
        assert rec["content"]
        assert rec["title"]


def test_score_normalized_to_0_1(_auth):
    """RRF 分必须归一化到 0~1，否则会被 Dify 的 score_threshold 全过滤掉。"""
    r = _post({"knowledge_id": "kb", "query": "年假"},
              {"Authorization": f"Bearer {_auth}"})
    scores = [rec["score"] for rec in r.json()["records"]]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > 0.9, "fused 取 RRF 理论上限，归一化后应接近 1.0"
    assert 0.45 <= scores[1] <= 0.55, "fused 取上限一半，归一化后应接近 0.5"


def test_score_threshold_filters(_auth):
    body = {"knowledge_id": "kb", "query": "年假",
            "retrieval_setting": {"top_k": 5, "score_threshold": 0.7}}
    r = _post(body, {"Authorization": f"Bearer {_auth}"})
    records = r.json()["records"]
    assert len(records) == 1, "阈值 0.7 应只留下归一化分约 1.0 的那条"


def test_top_k_limits(_auth):
    body = {"knowledge_id": "kb", "query": "年假",
            "retrieval_setting": {"top_k": 1, "score_threshold": 0.0}}
    assert len(_post(body, {"Authorization": f"Bearer {_auth}"}).json()["records"]) == 1


def test_empty_query_returns_empty_records(_auth):
    r = _post({"knowledge_id": "kb", "query": ""}, {"Authorization": f"Bearer {_auth}"})
    assert r.json() == {"records": []}


def test_knowledge_id_mismatch(_auth, monkeypatch):
    monkeypatch.setattr(config, "DIFY_KNOWLEDGE_ID", "expected-kb")
    r = _post({"knowledge_id": "other-kb", "query": "年假"}, {"Authorization": f"Bearer {_auth}"})
    assert r.json()["error_code"] == 2001


def test_knowledge_id_required(_auth):
    r = _post({"knowledge_id": "", "query": "年假"}, {"Authorization": f"Bearer {_auth}"})
    assert r.json()["error_code"] == 2001


# ---------------------------------------------------------------------------
# metadata_condition 过滤
# ---------------------------------------------------------------------------
def test_metadata_condition_is(_auth):
    body = {"knowledge_id": "kb", "query": "年假",
            "metadata_condition": {"logical_operator": "and",
                                   "conditions": [{"name": "chapter",
                                                   "comparison_operator": "is",
                                                   "value": "第三章 假期管理"}]}}
    records = _post(body, {"Authorization": f"Bearer {_auth}"}).json()["records"]
    assert len(records) == 1 and "员工手册" in records[0]["title"]


def test_metadata_condition_contains(_auth):
    body = {"knowledge_id": "kb", "query": "年假",
            "metadata_condition": {"conditions": [{"name": "heading_path",
                                                   "comparison_operator": "contains",
                                                   "value": "报销"}]}}
    records = _post(body, {"Authorization": f"Bearer {_auth}"}).json()["records"]
    assert len(records) == 1 and "财务" in records[0]["title"]


def test_unsupported_operator_passes_through(_auth):
    """不支持的运算符放行而不是让整次检索失败。"""
    body = {"knowledge_id": "kb", "query": "年假",
            "metadata_condition": {"conditions": [{"name": "chapter",
                                                   "comparison_operator": "≥",
                                                   "value": "3"}]}}
    assert len(_post(body, {"Authorization": f"Bearer {_auth}"}).json()["records"]) == 2


def test_metadata_condition_on_document_level_field(_auth):
    """按 `doc_version` 这类文档级字段过滤必须生效。

    回归：对外白名单曾只含 6 个字段，`_match_conditions` 取不到 `doc_version`
    → `is` 比较恒假 → 记录被静默丢弃、返回空且不报错。Dify 侧只会表现为
    「知识库明明有这份文档，却检索不到」，排查成本极高。
    """
    body = {"knowledge_id": "kb", "query": "年假",
            "metadata_condition": {"logical_operator": "and",
                                   "conditions": [{"name": "doc_version",
                                                   "comparison_operator": "is",
                                                   "value": "V3.2"}]}}
    records = _post(body, {"Authorization": f"Bearer {_auth}"}).json()["records"]
    assert len(records) == 1 and "员工手册" in records[0]["title"]


def test_metadata_condition_absent_field_yields_empty(_auth):
    """反向验证：落在对外白名单之外的字段，过滤必然落空。

    这条同时坐实了「字段缺失 → 静默返回空」这一失效模式真实存在，
    上面那条用例才不是空转。若哪天白名单全量放开了，本用例会红，
    应当连同用例一起重新设计而不是删掉。
    """
    body = {"knowledge_id": "kb", "query": "年假",
            "metadata_condition": {"conditions": [{"name": "fingerprint",
                                                   "comparison_operator": "is",
                                                   "value": "abc"}]}}
    assert _post(body, {"Authorization": f"Bearer {_auth}"}).json()["records"] == []


def test_payload_exposes_document_level_fields(_auth):
    """对外契约：文档级业务字段必须出现在 `records[].metadata` 里。

    这些字段是 Dify 侧过滤与引用展示的依据，漏掉即等于该能力不可用。
    """
    records = _post({"knowledge_id": "kb", "query": "年假"},
                    {"Authorization": f"Bearer {_auth}"}).json()["records"]
    meta = records[0]["metadata"]
    for field in ("doc_title", "doc_version", "doc_effective_date", "file_type"):
        assert field in meta, f"对外白名单漏了 {field}，Dify 无法按它过滤"


def test_payload_meta_omits_empty_but_keeps_zero():
    """空值剔除：空串会让 Dify 的过滤界面出现无意义选项。

    但 `0` 是合法下标，不能被当成空值一起剔掉——哨兵值是 -1，不是 0。
    """
    from app.api.dify import _build_payload_meta

    payload = _build_payload_meta({"source": "a/b.txt", "chapter": "", "chunk_index": 0})
    assert payload["source"] == "a/b.txt"
    assert payload["file_name"] == "b.txt"
    assert "chapter" not in payload, "空串应被剔除"
    assert payload["chunk_index"] == 0, "0 是合法下标，不能被剔除"
    assert "page" not in payload, "缺席的 page 应被剔除"


# ---------------------------------------------------------------------------
# 路由与自检
# ---------------------------------------------------------------------------
def test_prefixed_route_works(_auth):
    r = _post({"knowledge_id": "kb", "query": "年假"},
              {"Authorization": f"Bearer {_auth}"}, url="/dify/retrieval")
    assert "records" in r.json()


def test_info_endpoint_exposes_status_without_secret(monkeypatch):
    monkeypatch.setattr(config, "DIFY_API_KEY", "super-secret")
    data = client.get("/dify/info").json()
    assert data["enabled"] is True
    assert "super-secret" not in str(data), "自检接口不得泄露密钥"


def test_info_endpoint_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "DIFY_API_KEY", "")
    assert client.get("/dify/info").json()["enabled"] is False
