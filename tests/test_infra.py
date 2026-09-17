#!/usr/bin/env python3
"""横切离线测试 —— 异常基类 / Prompt 注册表 / trace / span 树 / 限流。

不启动服务、不调模型。覆盖：
    1. 异常基类：默认文案 + 自定义文案 + 子类可被 AppError 捕获
    2. Prompt 注册表：5 个 prompt 齐全、render 变量渲染、缺失变量报错
    3. trace：trace_id 生成/设置/获取、contextvars 隔离
    4. span 树：嵌套父子关系、序列化结构、trace.jsonl 持久化
    5. 限流：窗口内超限拦截、窗口滑出后放行
    6. LLM 限流（429）：_is_rate_limit 三层精确判定

用法：
    cd langgraph-enterprise-bot && python -m tests.test_infra
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
print("\n[1] 异常基类：默认文案与捕获")
# ---------------------------------------------------------------------------
def test_errors():
    from app.core.errors import AppError, KnowledgeBaseEmpty, RateLimitExceeded

    assert KnowledgeBaseEmpty().message == "知识库为空，请先上传文档"
    assert RateLimitExceeded("自定义").message == "自定义"
    # 子类可被基类捕获
    try:
        raise KnowledgeBaseEmpty()
    except AppError:
        caught = True
    assert caught, "子类应可被 AppError 捕获"
    return "默认/自定义文案 + 基类捕获正常"


check("异常基类", test_errors)


# ---------------------------------------------------------------------------
print("\n[2] Prompt 注册表：5 个 prompt 齐全")
# ---------------------------------------------------------------------------
def test_prompts_registry():
    from app.core.prompts import PROMPTS, get

    expected = {"answer", "dream", "summarize", "intent", "query_rewrite"}
    assert set(PROMPTS.keys()) == expected, f"注册表 key 应为 {expected}"
    # 关键变量占位符存在
    assert "{user_query}" in get("answer")
    assert "{context}" in get("answer")
    assert "{memory}" in get("dream")
    assert "{conversation}" in get("summarize")
    return "5 个 prompt 齐全 · 关键占位符存在"


check("Prompt 注册表", test_prompts_registry)


# ---------------------------------------------------------------------------
print("\n[3] Prompt 注册表：render 渲染与缺失变量报错")
# ---------------------------------------------------------------------------
def test_prompts_render():
    from app.core.errors import AppError
    from app.core.prompts import render

    out = render("summarize", conversation="你好")
    assert "你好" in out and "归档" in out, "渲染应包含变量值"

    try:
        render("answer")  # 缺全部变量
        raised = False
    except AppError:
        raised = True
    assert raised, "缺变量应抛 AppError"
    return "render 渲染 + 缺失变量报错正常"


check("Prompt 渲染", test_prompts_render)


# ---------------------------------------------------------------------------
print("\n[4] trace：trace_id 生成/设置/获取")
# ---------------------------------------------------------------------------
def test_tracing():
    from app.core.tracing import get_trace_id, new_trace_id, set_trace_id

    tid = new_trace_id()
    assert tid and len(tid) == 16, "trace_id 应为 16 位 hex"
    assert get_trace_id() == tid

    set_trace_id("abc123")
    assert get_trace_id() == "abc123"
    return "trace_id 生成与读写正常"


check("trace_id", test_tracing)


# ---------------------------------------------------------------------------
print("\n[5] span 树：嵌套父子 + 序列化 + trace.jsonl 持久化")
# ---------------------------------------------------------------------------
def test_span_tree():
    import json
    import tempfile
    import time
    from pathlib import Path

    from app import config
    from app.core.tracing import begin_trace, end_trace, span

    # 嵌套 span 应形成父子树，且顺序保持
    begin_trace("test-span-tree")
    with span("root"):
        with span("child_a"):
            time.sleep(0.005)
        with span("child_b"):
            time.sleep(0.01)
    tree = end_trace(persist=False)

    assert len(tree) == 1, "应只有一个根 span"
    root = tree[0]
    assert root["name"] == "root"
    assert [c["name"] for c in root["children"]] == ["child_a", "child_b"], "子 span 顺序应保持"
    assert root["duration_ms"] > 0, "根 span 耗时应大于 0"
    assert root["start_ms"] >= 0 and "children" in root, "序列化结构应含 start_ms/children"

    # 持久化到 trace.jsonl（写临时目录，不污染真实日志）
    old_log_dir = config.LOG_DIR
    config.LOG_DIR = Path(tempfile.mkdtemp())
    try:
        begin_trace("persist-check")
        with span("p"):
            pass
        end_trace(persist=True)
        lines = (config.LOG_DIR / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1, "应写入一行 trace 记录"
        rec = json.loads(lines[0])
        assert rec["trace_id"] == "persist-check"
        assert rec["spans"][0]["name"] == "p"
    finally:
        config.LOG_DIR = old_log_dir
    return "嵌套父子 + 序列化 + trace.jsonl 持久化正常"


check("span 树", test_span_tree)


# ---------------------------------------------------------------------------
print("\n[5] 限流：窗口内超限拦截")
# ---------------------------------------------------------------------------
def test_rate_limit():
    from app.core.errors import RateLimitExceeded
    from app.core.rate_limit import RateLimiter, check_rate_limit

    limiter = RateLimiter(per_minute=3)
    for _ in range(3):
        assert limiter.allow("ip-1") is True, "前 3 次应放行"
    assert limiter.allow("ip-1") is False, "第 4 次应被拦截"
    assert limiter.allow("ip-2") is True, "不同 key 不受影响"

    # check_rate_limit 抛异常
    from app.core import rate_limit as rl

    rl._limiter = RateLimiter(per_minute=1)
    check_rate_limit("x")
    try:
        check_rate_limit("x")
        raised = False
    except RateLimitExceeded:
        raised = True
    assert raised, "超限应抛 RateLimitExceeded"
    rl._limiter = None  # 清理，避免影响其他测试
    return "窗口内超限拦截 + 异常抛出正常"


check("入站限流", test_rate_limit)


# ---------------------------------------------------------------------------
print("\n[6] LLM 限流（429）判定")
# ---------------------------------------------------------------------------
def test_llm_rate_limit_detection():
    from app.core.llm_factory import _is_rate_limit

    class Status429(Exception):
        status_code = 429

    class RequestIdWith429(Exception):
        # 异常串里 request id 恰好含 "429"，但并非限流——旧版裸子串匹配会误判
        def __str__(self):
            return "request id org-293cc4be5ba6402799803b8035db740d failed"

    class ErrCode429(Exception):
        def __str__(self):
            return "Error code: 429 - rate limit exceeded"

    assert _is_rate_limit(Status429()) is True, "status_code=429 应判定为限流"
    assert _is_rate_limit(RequestIdWith429()) is False, "request id 含 429 不应误判为限流"
    assert _is_rate_limit(ErrCode429()) is True, "错误码语境 code: 429 应判定为限流"
    return "三层限流判定正确"


check("LLM 限流判定", test_llm_rate_limit_detection)


# ---------------------------------------------------------------------------
print()
passed, total = sum(_results), len(_results)
print(f"横切离线测试：{passed}/{total} 通过")
sys.exit(0 if passed == total else 1)
