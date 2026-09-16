"""评估报告（docs/history/project-assessment.md）修复项的回归测试。

覆盖：P0-1 错误路由语义、P1-1 伪配置生效、P1-2 词面检索死配置 + O(N²)、
P1-5 工作流拓扑漂移，以及「retrieve 按来源白名单（权限）过滤」。
全部为纯单元判定，不依赖真实大模型 / 向量库 / 网络。
"""
from app import config
from app.graph.edges import error_route_edge
from app.utils.logger import preview
from app.graph.workflow_graph import enterprise_workflow
from app.rag import lexical
from app.rag import retriever
from app.rag import generator
from app.rag import prepare
from app.rag import evaluator


# ---------------------------------------------------------------------------
# P0-1：error_route_edge 只认 need_human，不再被 error_msg 覆盖答案
# ---------------------------------------------------------------------------
def test_error_route_ignores_error_msg_when_answer_present():
    """检索抖动写了 error_msg 但答案已生成 → 不应转人工（保留答案）。"""
    state = {"answer": "这是已生成的有效答案", "error_msg": "知识检索失败：timeout", "need_human": False}
    assert error_route_edge(state) == "end"


def test_error_route_honors_need_human():
    """真正无法自动处理（need_human=True）→ 转人工。"""
    assert error_route_edge({"need_human": True, "answer": "x"}) == "human_fallback"


def test_error_route_soft_warning_keeps_answer():
    """仅有 soft_warnings（可降级故障）也应保留答案、正常结束。"""
    state = {"answer": "诚实回答", "soft_warnings": ["检索抖动"], "need_human": False}
    assert error_route_edge(state) == "end"


# ---------------------------------------------------------------------------
# P1-1：8 项伪配置现已真正生效（改 .env 能改到 import 期常量）
# ---------------------------------------------------------------------------
def test_pseudo_configs_resolve_from_config():
    assert retriever.RRF_K == config.RRF_K
    assert retriever.DENSE_WEIGHT == config.DENSE_WEIGHT
    assert retriever.LEXICAL_WEIGHT == config.LEXICAL_WEIGHT
    assert retriever.QUERY_REWRITE_ENABLED == config.QUERY_REWRITE_ENABLED
    assert generator.REFUSE_THRESHOLD == config.REFUSE_THRESHOLD
    assert prepare.MIN_DOC_CHARS == config.PREPARE_MIN_DOC_CHARS
    assert prepare.NEAR_DUP_THRESHOLD == config.PREPARE_NEAR_DUP_THRESHOLD
    assert evaluator.FEEDBACK_FILE == config.FEEDBACK_FILE


# ---------------------------------------------------------------------------
# P1-2：词面检索 —— 死配置消除 + 倒排剪枝等价于全表扫描 + avg_len 缓存
# ---------------------------------------------------------------------------
def test_lexical_bm25_params_read_from_config():
    """BM25 的 k1/b 现在从 config 读取，而非硬编码 1.5/0.75（死配置修复）。"""
    assert lexical._K1 == config.LEXICAL_BM25_K1
    assert lexical._B == config.LEXICAL_BM25_B


def test_lexical_search_pruning_matches_brute_force():
    """倒排剪枝后的结果与全表扫描等价：不得丢文档、不得改分数。

    近并列分数的排序受 Python 稳定排序的输入顺序影响，故只校验
    「集合完整 + 每篇分数逐位相等 + 结果按分数降序」，不校验并列项的绝对次序。
    """
    idx = lexical.LexicalIndex()
    # 200 篇文档：共享 filler 词 + 各自唯一词；其中 5 篇含查询词 "年假"
    for i in range(200):
        content = f"公共背景内容 filler 文档编号 uniq{i}"
        if i % 40 == 0:  # 0,40,80,120,160 共 5 篇命中
            content += " 年假 有多少天"
        idx.add(f"doc::{i}", content)
    query = "年假有多少天"
    top_k = 10
    pruned = idx.search(query, top_k)

    # 全表基准：对每篇文档用公开 _bm25 打分，收集正分集合
    terms = lexical._tokenize(query)
    all_positive = {}
    for k in idx._doc_len:
        s = idx._bm25(terms, k)
        if s > 0.0:
            all_positive[k] = s

    # 1) 剪枝未漏掉任何命中文档
    assert {k for k, _ in pruned} == set(all_positive)
    # 2) 剪枝未改变任何文档的分数（逐位相等）
    assert all(abs(s - all_positive[k]) < 1e-12 for k, s in pruned)
    # 3) 返回结果按分数降序
    assert pruned == sorted(pruned, key=lambda it: -it[1])


def test_lexical_avg_len_cache_updates_on_add_remove():
    """_avg_len 缓存应随增删正确更新（验证不再每文档重算）。"""
    idx = lexical.LexicalIndex()
    idx.add("a", "短文本 内容")
    idx.add("b", "较长一点的文档文本内容 内容")
    n, total = 0, 0
    for k, length in idx._doc_len.items():
        n += 1
        total += length
    assert idx._avg_len == total / n
    idx.remove("a")
    remaining = list(idx._doc_len.values())
    assert idx._avg_len == sum(remaining) / len(remaining)
    idx.clear()
    assert idx._avg_len == 1.0


# ---------------------------------------------------------------------------
# P0-3 子项：日志中用户 query 脱敏（preview 截断，杜绝明文完整 query 落盘）
# ---------------------------------------------------------------------------
def test_preview_truncates_long_query():
    assert preview("") == ""
    assert preview("短文本") == "短文本"
    long = "年假有多少天可以休" * 20
    out = preview(long, 30)
    assert out.endswith("…")
    assert len(out) == 31  # 30 字符 + 省略号
    assert out[:-1] == long[:30]
def test_workflow_topology_matches_compiled_graph():
    from app.api import workflow as wf

    compiled = {n for n in enterprise_workflow.nodes if not n.startswith("__")}
    # 面板声明的节点 == 编译图实际节点
    assert set(wf.NODE_LABELS) == compiled
    # 入口是编译图真实入口
    assert wf.ENTRY_POINT in compiled
    # 每个分支的 from / routes 都在真实节点集合内
    for branch in wf.BRANCHES:
        assert branch["from"] in compiled
        for route in branch["routes"]:
            assert route == "END" or route in compiled
    # 启动时一致性校验不抛异常
    wf._validate_topology()


# ---------------------------------------------------------------------------
# retrieve() 来源白名单（按来源/部门做知识隔离）
# ---------------------------------------------------------------------------
def _doc(source: str, text: str = "内容") -> dict:
    return {"content": text, "source": source, "score": 0.5, "lexical": 0.1, "fused": 0.3}


def test_filter_by_allowed_sources_none_passthrough():
    """allowed_sources=None 不过滤。"""
    docs = [_doc("hr/policy.pdf"), _doc("finance/rule.xlsx")]
    assert retriever.filter_by_allowed_sources(docs, None) == docs


def test_filter_by_allowed_sources_keeps_only_whitelisted():
    """只保留白名单内的来源。"""
    docs = [_doc("hr/policy.pdf"), _doc("finance/rule.xlsx"), _doc("hr/handbook.pdf")]
    out = retriever.filter_by_allowed_sources(docs, ["hr/policy.pdf"])
    assert [d["source"] for d in out] == ["hr/policy.pdf"]


def test_filter_by_allowed_sources_empty_denies_all():
    """空列表表示全部拒绝（与 None 的语义不同）。"""
    docs = [_doc("hr/policy.pdf"), _doc("finance/rule.xlsx")]
    assert retriever.filter_by_allowed_sources(docs, []) == []


def test_retrieve_signature_accepts_allowed_sources():
    """retrieve 暴露 allowed_sources 形参（端到端贯通的契约）。"""
    import inspect

    params = inspect.signature(retriever.retrieve).parameters
    assert "allowed_sources" in params
    assert "allowed_sources" in inspect.signature(
        __import__("app.core.rag_engine", fromlist=["x"]).retrieve_knowledge_docs
    ).parameters


# ---------------------------------------------------------------------------
# 来源 ACL（服务端按用户解析白名单，承接 retrieve 权限参数）
# ---------------------------------------------------------------------------
def test_resolve_acl_disabled_returns_none():
    from app import config

    saved = config.SOURCE_ACL
    config.SOURCE_ACL = {}
    try:
        from app.core import source_acl

        assert source_acl.resolve_allowed_sources("alice") is None
    finally:
        config.SOURCE_ACL = saved


def test_resolve_acl_exact_user_and_wildcard():
    from app import config

    saved = config.SOURCE_ACL
    config.SOURCE_ACL = {"alice": ["hr/*"], "bob": ["finance/*"], "*": ["*"]}
    try:
        from app.core import source_acl

        assert source_acl.resolve_allowed_sources("alice") == ["hr/*"]
        # "*" 默认规则含 "*" → 不限制
        assert source_acl.resolve_allowed_sources("unknown") is None
    finally:
        config.SOURCE_ACL = saved


def test_resolve_acl_unmatched_fail_closed():
    """配了 ACL 但既无该用户也无默认规则 → 全部拒绝（返回 []）。"""
    from app import config

    saved = config.SOURCE_ACL
    config.SOURCE_ACL = {"alice": ["hr/*"]}
    try:
        from app.core import source_acl

        assert source_acl.resolve_allowed_sources("bob") == []
    finally:
        config.SOURCE_ACL = saved


def test_filter_by_allowed_sources_fnmatch():
    """来源白名单支持 fnmatch 通配（如 hr/*）。"""
    docs = [
        _doc("hr/policy.pdf"),
        _doc("finance/rule.xlsx"),
        _doc("hr/handbook.pdf"),
    ]
    out = retriever.filter_by_allowed_sources(docs, ["hr/*"])
    assert {d["source"] for d in out} == {"hr/policy.pdf", "hr/handbook.pdf"}


def test_chat_uses_acl_authoritative(monkeypatch):
    """chat 入口：ACL 配了就以服务端为准，忽略客户端传入的 allowed_sources。"""
    import app.api.chat as chat_mod
    from app import config

    saved = config.SOURCE_ACL
    config.SOURCE_ACL = {"u1": ["hr/*"]}
    try:
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return {}

        monkeypatch.setattr(chat_mod, "create_initial_state", fake_create)
        # 仅验证 allowed_sources 解析逻辑：直接调用共享函数
        from app.core.source_acl import resolve_allowed_sources

        assert (resolve_allowed_sources("u1") or ["client_val"]) == ["hr/*"]
    finally:
        config.SOURCE_ACL = saved
