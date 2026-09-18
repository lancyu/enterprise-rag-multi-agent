"""Dify 外部知识库接口测试（T-DIFY）。

只测**适配层**：鉴权、分数归一化、响应字段格式、过滤与限流。
检索本身由其他用例覆盖，这里把 `retriever.retrieve` 打桩 —— 测试跑得快，
也不消耗 embedding 配额。
"""
from __future__ import annotations

from typing import Dict

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
# 鉴权解析：**一条规则，两个入口**（曾经两套，且对制表符判定相反）
# ---------------------------------------------------------------------------
#: 一批 Authorization 头写法。**制表符那一条是关键**：入站中间件过去用
#: ``startswith("bearer ")``（拒），Dify 端点用 ``split(None, 1)``（收），
#: 同一个 token 走两个入口得到相反结论。
_AUTH_HEADER_CASES = [
    "Bearer test-key-123",
    "bearer test-key-123",
    "BEARER test-key-123",
    "Bearer   test-key-123",      # 多个空格
    "  Bearer test-key-123  ",    # 头两端有空白
    "Bearer\ttest-key-123",       # ← 制表符
    "Bearer",                      # 只有 scheme，没令牌
    "Basic test-key-123",          # 别的 scheme
    "",                            # 空头
]


def _middleware_request(headers: dict):
    """造一个只带请求头的最小 Starlette 请求，用来直接驱动入站鉴权。

    走 ``_check_auth`` 而不是 TestClient：后者要挑一个"必然存在、又不豁免鉴权、
    还不会真跑业务逻辑"的路由，条件比被测代码还难维护。
    """
    from starlette.requests import Request

    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "POST", "path": "/chat/ask", "headers": raw})


@pytest.mark.parametrize("header", _AUTH_HEADER_CASES)
def test_middleware_and_dify_reach_the_same_verdict(_auth, header, monkeypatch):
    """两个入口对同一个 Authorization 头的结论必须一致（这里用**正确的令牌**）。

    这是"解析又被抄了一份"的探测器：只要有人把 ``split(None, 1)`` 或
    ``startswith("bearer ")`` 写回任一处，制表符那一格就会立刻分叉。
    """
    from app.api import dify
    from app.main import _check_auth as middleware_check

    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    monkeypatch.setattr(config, "AUTH_API_KEY", _auth)

    middleware_ok = middleware_check(_middleware_request({"Authorization": header})) is None
    dify_ok = dify._check_auth(header) is None

    assert middleware_ok == dify_ok, (
        f"Authorization={header!r} 时两个入口结论相反："
        f"中间件={'放行' if middleware_ok else '拒绝'}，"
        f"Dify={'放行' if dify_ok else '拒绝'} —— 解析规则又变成两套了"
    )


def test_tab_is_not_a_valid_separator(_auth, monkeypatch):
    """制表符不算合法分隔符，两个入口都要拒——取 RFC 6750 的严格一侧。

    ``str.split()`` 按**任意空白**切，会把 ``"Bearer\\txxx"`` 也认成合法。
    这条钉住"更严格"这个方向：哪天有人为了"兼容"把两边一起放宽，
    本用例会红，逼他先想清楚放宽的代价。
    """
    from app.api import dify
    from app.main import _check_auth as middleware_check

    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    monkeypatch.setattr(config, "AUTH_API_KEY", _auth)

    assert dify._check_auth(f"Bearer\t{_auth}") is not None, "Dify 端点不应用制表符分隔"
    assert middleware_check(_middleware_request({"Authorization": f"Bearer\t{_auth}"})) is not None


def test_middleware_prefers_x_api_key(_auth, monkeypatch):
    """``X-API-Key`` 优先——这是**入站入口的策略**，不是解析规则，故留在 main.py。

    两个入口共用的是"怎么解 Bearer"，不是"认哪些头"：Dify 端点的对外契约里
    只有 ``Authorization``，多认一个头等于悄悄扩大它的攻击面。
    """
    from app.main import _extract_api_key

    assert _extract_api_key(_middleware_request({"X-API-Key": "from-header"})) == "from-header"
    assert _extract_api_key(_middleware_request({
        "X-API-Key": "from-header", "Authorization": f"Bearer {_auth}",
    })) == "from-header"


def _looks_like_bearer_parsing(node) -> str:
    """判断一个 AST 节点是不是"在解析 Authorization 头"。返回说明，不是则空串。

    ⚠️ 判据必须**只认真正会执行的解析动作**，不能是"字符串里出现 bearer"：
    本仓库里 ``'Bearer '`` 这种字面量还出现在别处且完全正当——
    ``app/providers/embeddings.py`` 用它**拼出站请求头**（方向相反的一件事），
    ``dify.py`` 用它拼错误文案，模块 docstring 里也写着这个词。
    第一版判据就是"字符串包含 bearer"，一上来报了 7 处误报。
    **护栏误报的下场是被关掉**，所以这里收紧到三种语法特征：

    ① ``x.startswith("bearer…")`` —— 入站中间件原来的写法；
    ② ``x.split(None, 1)`` —— Dify 端点原来的写法（按任意空白切）；
    ③ ``re.match/compile/...("…bearer…")`` —— 改成正则也算重写了一份。
    """
    import ast

    # ① / ②：方法调用
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr == "startswith" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if "bearer" in arg.value.lower():
                    return f'startswith({arg.value!r})'
        if attr == "split" and len(node.args) == 2:
            first, second = node.args
            if (
                isinstance(first, ast.Constant) and first.value is None
                and isinstance(second, ast.Constant) and second.value == 1
            ):
                return "split(None, 1)"
    # ③：正则里写 bearer
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"compile", "match", "search", "fullmatch"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and "bearer" in node.args[0].value.lower()
    ):
        return f"re.{node.func.attr}({node.args[0].value!r})"
    return ""


#: 「非解析用途」的豁免 —— 逐条写明理由，且必须**真的被报出**才算数
#: （`test_bearer_parsing_lives_in_exactly_one_module` 会反向校验僵尸豁免）。
#: 为什么需要它：判据认的是**语法形态**（正则里含 bearer），而形态本身分不出
#: "把凭证抽出来给鉴权用"与"把凭证替换成占位符"。前者是重复实现，后者不是。
_BEARER_OTHER_PURPOSE: Dict[str, str] = {
    "core/trace_mask.py": (
        "脱敏用的替换规则，不是解析：它从不把 token 交给任何调用方，"
        "只把 ``Bearer <token>`` 里的 token 换成占位符。"
        "与 auth_header.py 的语义方向相反（一个是取出，一个是抹掉），"
        "不构成同一语义的重复实现。"
    ),
}


def test_bearer_parsing_lives_in_exactly_one_module():
    """Bearer 的**解析**只许出现在 ``app/utils/auth_header.py``。

    判据走 AST 而不是正则扫文本：本仓库里 main.py / dify.py 的注释**特意**写着
    "这里曾经用 startswith('bearer ')/split(None, 1)"，正则会把说明文字当成违规。
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    owner = "utils/auth_header.py"
    offenders = []
    seen_exempt = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == owner:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            why = _looks_like_bearer_parsing(node)
            if why:
                if rel in _BEARER_OTHER_PURPOSE:
                    seen_exempt.add(rel)
                    continue
                offenders.append(f"{rel}:{node.lineno} {why}")

    # 兜底：本判据若哪天认不出东西，本用例会毫无意义地变绿 —— 那是"静默失效"，
    # 正是这份文件存在的理由。所以先确认它确实能认出**已知的那两种写法**。
    import ast as _ast

    for probe, expect in [
        ('x.lower().startswith("bearer ")', "startswith"),
        ('h.split(None, 1)', "split"),
    ]:
        found = _looks_like_bearer_parsing(_ast.parse(probe).body[0].value)
        assert expect in found, f"判据认不出已知写法 {probe!r}（得到 {found!r}）—— 本用例已失效"

    # 僵尸豁免反向校验：清单里的条目若不再被报出，说明它的理由已经过期
    # （文件删了、规则改了），必须同步删掉 —— 与 tests/deadcode_allowlist.py 同一套语义。
    assert seen_exempt == set(_BEARER_OTHER_PURPOSE), (
        f"豁免清单与实际不符：未触发的 {sorted(set(_BEARER_OTHER_PURPOSE) - seen_exempt)}，"
        f"未登记的 {sorted(seen_exempt - set(_BEARER_OTHER_PURPOSE))}"
    )

    assert offenders == [], (
        "Bearer 解析在下面这些地方又出现了 —— 它只该有 app/utils/auth_header.py 一处：\n  "
        + "\n  ".join(offenders)
    )


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
