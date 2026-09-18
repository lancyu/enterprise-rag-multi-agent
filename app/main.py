"""FastAPI 入口 —— 生命周期管理、路由注册、静态面板与启动自检。"""
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.api import chat, dify, evaluation, knowledge, memory, routing, test, workflow
from app.core.errors import RateLimitExceeded, trace_ref
from app.core.rate_limit import check_rate_limit
from app.core.self_check import get_health, run_self_check
from app.core.tracing import new_trace_id, set_trace_id
from app.db.redis_db import close_redis, get_redis
from app.utils.auth_header import api_key_matches, bearer_token
from app.utils.logger import logger

_startup_report: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时执行全量自检，关闭时释放资源。"""
    start = time.perf_counter()
    logger.info("=" * 68)
    logger.info("企业智能助手 · RAG 知识引擎 启动中 | 端口 %d", config.APP_PORT)
    logger.info("=" * 68)

    # 预热缓存连接，使自检报告能正确显示缓存模式（Redis / 内存降级）
    await get_redis()

    report = await _run_startup_check()
    _startup_report.update(report)

    logger.info("-" * 68)
    for item in report["items"]:
        flag = "PASS" if item["status"] == "pass" else "FAIL"
        logger.info("[%s] %-14s %s（%dms）", flag, item["label"], item["detail"], item["elapsed_ms"])
    logger.info("-" * 68)
    logger.info(
        "自检结果：%d/%d 通过 | 耗时 %dms | 环境 llm=%s vector=%s cache=%s",
        report["passed"], report["total"], int((time.perf_counter() - start) * 1000),
        report["env"]["llm_mode"], report["env"]["vector_db"], report["env"]["cache"],
    )
    # 降级体检：配置要求的后端与实际生效的后端不一致，说明后端没连上、已退回内存库。
    # 不显式告警的话，数据会被静静写进本地文件，而运维以为在用 Milvus。
    if report["env"]["vector_db"] != report["env"]["vector_db_requested"]:
        logger.error(
            "向量库降级：配置 VECTOR_DB_TYPE=%s，实际生效 %s —— "
            "写入会落到本地内存库而非 %s。请检查后端服务与客户端依赖；"
            "若要「连不上就拒绝启动」，请设置 VECTOR_DB_STRICT=true。",
            report["env"]["vector_db_requested"], report["env"]["vector_db"],
            report["env"]["vector_db_requested"],
        )
    if not report["success"]:
        logger.warning("存在未通过的自检项，相关功能可能不可用，请检查日志定位问题")

    # 安全配置体检：把「配错了但看不出来」的情况显式暴露在启动日志里
    if config.AUTH_ENABLED and not config.AUTH_API_KEY:
        logger.error(
            "鉴权已开启（AUTH_ENABLED=true）但未配置 AUTH_API_KEY —— "
            "当前所有请求都会被拒绝（fail-closed）。请配置 AUTH_API_KEY 或关闭 AUTH_ENABLED。"
        )
    elif not config.AUTH_ENABLED:
        logger.warning(
            "入站鉴权当前关闭，任何可访问本服务的人都能调用全部接口（含上传/重建索引）。"
            "部署到非本机环境前请设置 AUTH_ENABLED=true 与 AUTH_API_KEY。"
        )
    if "*" in config.CORS_ORIGINS:
        logger.warning("CORS 来源为通配（*），已自动关闭凭据模式；生产环境建议显式配置 CORS_ORIGINS。")

    logger.info("服务已就绪 → http://localhost:%d", config.APP_PORT)

    yield

    await close_redis()
    logger.info("服务已停止")


async def _run_startup_check() -> dict:
    """启动自检（异步线程中执行，避免阻塞事件循环）。"""
    import asyncio

    return await asyncio.to_thread(run_self_check)


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description="基于 LangGraph + FastAPI + 大模型的企业智能助手 RAG 知识引擎",
    lifespan=lifespan,
)

# CORS：来源可配置。**不允许** origins=* 与 credentials=True 同时生效 ——
# 浏览器会直接拒绝该组合（带凭据的请求不能通配来源），且它是典型的
# 「本地能跑、上线跨域全挂」的坑。来源为通配时自动关闭凭据。
_CORS_ORIGINS: list = config.CORS_ORIGINS
_CORS_ALLOW_CREDENTIALS: bool = "*" not in _CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_api_key(request: Request) -> str:
    """从 X-API-Key 或 Authorization: Bearer 中提取密钥。

    **解析与比较都委托 :mod:`app.utils.auth_header`**（全项目唯一实现处）。
    ``X-API-Key`` 优先是**本入口的策略**，不是解析规则，故留在本地：
    Dify 兼容端点只读 Authorization（那是它的对外契约），两者共用的是"怎么解"。

    这里曾经自己写 ``startswith("bearer ")``，而 Dify 端点写 ``split(None, 1)``——
    后者按任意空白切，于是 ``"Bearer\\txxx"`` 在两条入口上一个放行、一个 401。
    """
    provided = request.headers.get("X-API-Key", "").strip()
    if provided:
        return provided
    return bearer_token(request.headers.get("Authorization"))


def _check_auth(request: Request) -> Optional[JSONResponse]:
    """入站鉴权。返回 None 表示放行，否则返回 401 响应。

    默认关闭（AUTH_ENABLED=false），保持零行为变化。
    开启后若未配 AUTH_API_KEY，则**一律拒绝**（fail-closed）：
    「开着鉴权但谁都能过」比不开更危险，因为管理员会误以为已防护。
    """
    if not config.AUTH_ENABLED:
        return None
    path = request.url.path
    if path.startswith("/static") or path in config.AUTH_EXEMPT_PATHS:
        return None

    # fail-closed（"配了鉴权但没配密钥 → 一律拒绝"）写在 api_key_matches 里，
    # 不在这里再判一次：两处各判一遍时，漏掉任何一处都只会**静默放宽**。
    if not api_key_matches(_extract_api_key(request), config.AUTH_API_KEY):
        logger.warning(
            "鉴权失败：%s %s（来源 %s）",
            request.method, path,
            request.client.host if request.client else "-",
        )
        return JSONResponse(
            status_code=401,
            content={"code": 401, "message": "未授权：请提供有效的 API Key"},
        )
    return None


@app.middleware("http")
async def trace_and_rate_limit(request: Request, call_next):
    """为每个请求注入 trace_id，做入站鉴权与限流（按 IP）。

    trace_id 用 contextvars 贯穿到后续所有日志；限流超限返回 429。
    静态资源与健康检查不参与限流（它们是运维探针，不是业务流量）。
    """
    path = request.url.path
    set_trace_id(new_trace_id())

    # 鉴权先于业务逻辑：未授权请求不该消耗下游资源
    unauthorized = _check_auth(request)
    if unauthorized is not None:
        return unauthorized

    # /dify/info 是接入自检探针（只返回配置状态、不含密钥），与 /health 同级免限流
    if path.startswith("/static") or path in ("/health", "/", "/dify/info"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    try:
        check_rate_limit(client_ip)
    except RateLimitExceeded as exc:
        return JSONResponse(status_code=429, content={"code": 429, "message": exc.message})

    return await call_next(request)

# ---------------- 路由注册 ----------------
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(workflow.router)
app.include_router(memory.router)        # 记忆系统（长期/短期记忆管理）
app.include_router(evaluation.router)    # 评估迭代（RAG L5）
app.include_router(test.router)
app.include_router(dify.router)          # Dify 外部知识库兼容接口（/retrieval）
# /routing/*（意图预演 + 目录快照）：**预演专用**，不进入任何子 Agent、无副作用。
#
# 注：上一版的 /routing/*（档位统计、阈值标定、意图预演）随动态路由与自造意图路由
# 一并删除，因为它们服务的是一套已经删掉的机制。这里重新开一个不是"把旧的搬回来"：
# 它服务的是 docs/intent-routing-hybrid-design.md 的四层漏斗，
# 存在的唯一理由是回答"日志回答不了"的问题——**这次为什么这么判、阈值当时是多少**。
# 生产链路的观测证据仍然是 /chat/ask 响应里的 agent_steps + route_decision + span_tree。
app.include_router(routing.router)

# ---------------- 静态面板 ----------------
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """前端可视化面板入口。"""
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/health", tags=["对话服务"])
async def health():
    """服务健康状态检测。"""
    data = get_health()
    data["startup_check"] = {
        "passed": _startup_report.get("passed"),
        "total": _startup_report.get("total"),
        "success": _startup_report.get("success"),
    }
    return {"code": 0, **data}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常兜底。

    只回显通用文案 + trace_id，**不外泄异常原文**：异常字符串常含文件路径、
    内网地址、依赖版本等信息，直接返回等于把内部拓扑暴露给匿名调用方。
    完整堆栈已由 logger.exception 落盘，凭 trace_id 即可检索。
    """
    logger.exception("未捕获异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": "服务内部异常，请稍后重试",
            # 与 /chat/ask、/chat/ask/stream 共用同一个凭证取值处
            # （app/core/errors.py）。这里多了一层结构：message 与 trace_id
            # 分列两个字段，而 HTTPException 的 detail 是拼成一句话。
            "trace_id": trace_ref(),
        },
    )
