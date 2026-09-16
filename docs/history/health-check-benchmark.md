# 健康检查设计调研报告（对标业界开源项目）
- 日期：2026-09-04 ｜ 方式：`curl -sL raw.githubusercontent.com` 取源码后本地精读（未用 GitHub API）
- 对象：dify、ragflow、LiteLLM、Langflow + Kubernetes 官方探针规范
- 声明：代码片段均为 main 分支原文。未找到的已明确标注，不做推测性填充。

---

## 0. 一句话结论
> **没有任何一个成熟项目会在进程启动路径或 liveness 探针里真实调用付费 LLM API。**
> 涉及 LLM 连通性的检查，统一做法是：**挪到后台周期任务 / 独立可选端点，结果缓存，且可单独关闭**（LiteLLM 是唯一做了 LLM 健康检查节流全套手段的项目）。

你项目里 70 秒的启动自检 + `get_chat_model()` 里的 `model.invoke("ping")`，在四个对标项目里**找不到同类实现**。

---

## 1. Kubernetes 官方探针职责划分
来源：`https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/`

| 探针 | 官方定义（原文摘抄） | 失败后果 |
|---|---|---|
| startup | "verify whether the application within a container is started. **If a startup probe is configured, Kubernetes does not execute liveness or readiness probes until the startup probe succeeds**... only executed at startup." | 杀容器重启 |
| liveness | "determine when to **restart** a container... could catch a **deadlock**, where an application is running, but unable to make progress." | 杀容器重启 |
| readiness | "determine when a container is **ready to accept traffic**... run on the container during its **whole lifecycle**." | **不重启**，从 EndpointSlice 摘除、停止导流 |

**分层原则（最值得抄的两段原文）**

> "**When your app has a strict dependency on back-end services, you can implement both a liveness and a readiness probe. The liveness probe passes when the app itself is healthy, but the readiness probe additionally checks that each required back-end service is available.**"
>
> "**A common pattern for liveness probes is to use the same low-cost HTTP endpoint as for readiness probes, but with a higher `failureThreshold`.** This ensures that the pod is observed as not-ready for some period of time before it is hard killed."

**liveness 的官方警告（原文）**

> "Liveness probes... should be used with caution. **Liveness probes must be configured carefully to ensure that they truly indicate unrecoverable application failure**, for example a deadlock.
>
> **Incorrect implementation of liveness probes can lead to cascading failures.** This results in restarting of container under high load... and increased workload on remaining pods due to some failed pods."

**诚实说明**：社区流传的"liveness 绝对不能依赖外部服务"这句**在当前官方文档里没有原话**。真正的物理约束来自默认值：`timeoutSeconds=1`、`periodSeconds=10`、`failureThreshold=3`、`initialDelaySeconds=0`——把慢依赖放进 liveness 必然放大成级联重启。

---

## 2. LiteLLM（最值得抄：唯一系统化了 LLM 健康检查节流）
仓库 `BerriAI/litellm` @ main。它本身就在 LLM API 前面，与健康检查耦合度最高。

### 2.1 liveness：极简、零依赖
`litellm/proxy/health_endpoints/_health_endpoints.py:1813-1831`

```python
@router.get("/health/liveliness", tags=["health"])   # 历史名，保留兼容
@router.get("/health/liveness", tags=["health"])     # Kubernetes has "liveness" probes (https://kubernetes.io/docs/tasks/...)
async def health_liveliness(response: Response):
    if GracefulShutdownManager.is_shutting_down():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "shutting_down"}
    return "I'm alive!"                              # ← 不查 DB / Redis / 任何 LLM
```
注释显式引用 K8s 官方 URL；另有 `/health/drain`（`:1769`）配合优雅下线。

### 2.2 readiness：查 DB，但带 15 秒缓存 + 双层超时预算
`_health_endpoints.py:1373-1432`

```python
# Bounds each DB round-trip on the probe path so a hung connection during a
# failover cannot make the probe fail by timeout (k8s default timeoutSeconds: 5).
DB_READINESS_CHECK_TIMEOUT_SECONDS: Final = 2.0     # 单次 DB 往返硬超时
DB_READINESS_PROBE_DEADLINE_SECONDS: Final = 4.0    # 整条探针路径总预算
db_health_cache: DBHealthCache = {"status": "unknown", "last_updated": datetime.now()}

async def _db_health_readiness_check_unbounded() -> DBHealthCache:
    time_diff: Final = datetime.now() - db_health_cache["last_updated"]
    if db_health_cache["status"] == "connected" and time_diff < timedelta(seconds=15):
        return db_health_cache                      # ← 15 秒内复用上次结果
    await asyncio.wait_for(prisma_client.health_check(), timeout=DB_READINESS_CHECK_TIMEOUT_SECONDS)
    ...                                             # 失败时 attempt_db_reconnect 自愈后重试一次
```

这是**问题 5（结果缓存）最直接的答案**：TTL 15 秒；单次往返 2s 硬超时；整条路径 4s 总预算（注释明写要卡在 K8s 默认 5s 内）；失败还会尝试重连自愈。

### 2.3 `/health`（模型级）：会调用 LLM，但全套节流
`_health_endpoints.py:997` 的 docstring 第一句就是警告：

```python
async def health_endpoint(...):
    """
    🚨 USE `/health/liveliness` to health check the proxy 🚨
    To run health checks in the background, add this to config.yaml:
        general_settings:
            background_health_checks: True
    else, the health checks will be run on models when /health is called.
    """
```

**即：LiteLLM 承认"健康检查会打 LLM"，但把它做成后台任务，而不是请求路径上的阻塞操作。**

后台循环 `litellm/proxy/proxy_server.py:3751-3924`（节选）：

```python
async def _run_background_health_check():
    """Periodically run health checks in the background on the endpoints."""
    if health_check_interval is None or not isinstance(health_check_interval, int) or health_check_interval <= 0:
        return                                          # ① 间隔可配，<=0 直接关闭
    while True:
        # filter out models that have disabled background health checks  → ② 可按模型单独关闭
        _llm_model_list = [m for m in _llm_model_list
                           if not m.get("model_info", {}).get("disable_background_health_check", False)]
        ...                                             # ③ health_check_concurrency 限并发
        health_check_results["healthy_endpoints"] = healthy_endpoints
        health_check_results["unhealthy_endpoints"] = unhealthy_endpoints
        if cycle_duration_ms > (health_check_interval * 1000):
            verbose_proxy_logger.warning("background_health_check_cycle_duration_exceeded_interval ...")
        await asyncio.sleep(health_check_interval)
```

节流手段：① **间隔可配**（`general_settings.health_check_interval`，`proxy_server.py:5679`），`<=0` 即关闭；② **可按模型关闭**（`model_info.disable_background_health_check: true`）；③ **并发上限**（`health_check_concurrency`）；④ **结果缓存**——`GET /health` 在 `use_background_health_checks` 时直接读 `health_check_results`，不真跑（`_health_endpoints.py:1109-1148`）；⑤ **多副本去重**——`litellm/proxy/health_check_utils/shared_health_check_manager.py` 用 Redis 锁 + TTL 缓存（*"Caches health check results with configurable TTL / Uses Redis locks to ensure only one pod runs health checks at a time"*）；⑥ **限制单次成本**——`litellm/constants.py` 里 `DEFAULT_HEALTH_CHECK_PROMPT="test from litellm"`、`HEALTH_CHECK_TIMEOUT_SECONDS=60`、`BACKGROUND_HEALTH_CHECK_MAX_TOKENS`（后台检查单独限制 max_tokens）。

**未找到**：`DEFAULT_HEALTH_CHECK_INTERVAL`、`DEFAULT_SHARED_HEALTH_CHECK_TTL` 的具体数值。二者从 `litellm.constants` 导入，但 main 分支 `litellm/constants.py`（1986 行）中搜不到定义（仓库正在做 `model_runtime` → `graphon` 改名迁移）。故不引用具体默认值。

### 2.4 降级：per-deployment 冷却（熔断器）
`litellm/router_utils/cooldown_handlers.py`：每个 deployment 可配 `allowed_fails`、`allowed_fails_policy`（按异常类型分别配）、`cooldown_time`，从 `model_info` 或 `litellm_params` 读取，实现"失败 N 次后冷却 X 秒"的熔断语义。

---

## 3. ragflow（三层端点 + 明确"启动不阻塞外部网络"）
仓库 `infiniflow/ragflow` @ main。

### 3.1 三个端点，语义严格分层（`api/apps/restful_apis/system_api.py`）
`/system/ping`（39-41，`return "pong", 200`，无鉴权）→ `/system/healthz`（269-272，无鉴权，K8s 探针）→ `/system/status`（73-179，`@login_required`，含 task executor 心跳 + elapsed）。

```python
@manager.route("/system/healthz", methods=["GET"])
def healthz():
    result, all_ok = run_health_checks()
    return jsonify(result), (200 if all_ok else 500)
```

### 3.2 检查项：只查自己的基础设施，完全不碰 LLM
`api/utils/health_utils.py:44-79`——四个探针，每个都带耗时：

```python
def check_db() -> tuple[bool, dict]:
    st = timer()
    try:
        # lightweight probe; works for MySQL/Postgres
        DB.execute_sql("SELECT 1")
        return True, {"elapsed": f"{(timer() - st) * 1000.0:.1f}"}
    except Exception as e:
        return False, {"elapsed": ..., "error": str(e)}

def check_redis() -> tuple[bool, dict]: ...        # REDIS_CONN.health()
def check_doc_engine() -> tuple[bool, dict]: ...   # settings.docStoreConn.health()
def check_storage() -> tuple[bool, dict]: ...      # settings.STORAGE_IMPL.health()
```

汇总 `run_health_checks()`（`health_utils.py:454-488`）逐项 try，任一 nok 则整体 nok：
`all_ok = (db=="ok") and (redis=="ok") and (doc_engine=="ok") and (storage=="ok")` → `result["status"] = "ok" if all_ok else "nok"`。

**关键观察**：ragflow 是 RAG 系统，LLM 是核心依赖，但 healthz **一项 LLM 检查都没有**。

### 3.3 启动策略（直接回应你的问题 3）
`api/ragflow_server.py:25-27`

```python
# LiteLLM fetches a model cost map from GitHub during import unless this is set.
# The API server should not block startup on external network access.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
```

ragflow **主动关掉一个会在 import 期访问外部网络的动作**，理由就是"API server 不应该在启动时被外部网络阻塞"。`run_server()` 里只做本地/内网动作：`init_web_db()` → `init_web_data()` → `RuntimeConfig.init_env()` → `GlobalPluginManager.load_plugins()`，**没有任何外部 API 调用**。

---

## 4. Langflow（FastAPI 生态：端点分层 + 不泄露错误细节）
仓库 `langflow-ai/langflow` @ main，文件 `src/backend/base/langflow/api/health_check_router.py`（123 行，全文精读）。

```python
# /health is also supported by uvicorn
# it means uvicorn's /health serves first before the langflow instance is up
# therefore it's not a reliable health check for a langflow instance
# we keep this for backward compatibility
@health_check_router.get("/health")
async def health():
    return {"status": "ok"}
```

**这条注释对你很关键**：ASGI 层自带的 `/health` 会在应用初始化完成前就返回 200。你的 FastAPI 里若 `/health` 只 return 常量，它在 70 秒自检跑完之前就已经 200 了——**探针完全失效**。

`/health_check`（可靠版）的探针 `_probe_services`（L41-60）：

```python
    try:
        # Check database to query a bogus flow
        stmt = select(Flow).where(Flow.id == uuid.uuid4())   # 空查询、不写入、不锁表
        (await session.exec(stmt)).first()
        response.db = "ok"
    except Exception:
        await logger.aexception("Error checking database")
    try:
        chat = get_chat_service()
        await chat.set_cache("health_check", _HEALTH_CHECK_PROBE_KEY)   # 固定探针 key
        await chat.get_cache("health_check")
        response.chat = "ok"
    except Exception:
        await logger.aexception("Error checking chat service")
```
`/healthz`（K8s 风格 readiness，L90-123）：在之上遍历企业版插件注册表 `_enterprise_readiness_checks`，每项套 `await asyncio.wait_for(check(), timeout=check_timeout)`——**健康检查必须有超时，超时即失败**；任一返回 `error:` 则 `raise HTTPException(503)`（Unready 但不重启，与 K8s readiness 语义对齐）。

### 安全细节（值得抄）
```python
class HealthResponse(BaseModel):
    status: str = "nok"
    chat: str = "error check the server logs"
    db: str = "error check the server logs"
    """
    Do not send exceptions and detailed error messages to the client because it might contain
    credentials and other sensitive server information.
    """
```

健康检查端点通常**不鉴权**（要给 K8s / LB 用），绝不能把 exception 原文吐出去。

---

## 5. dify（降级策略最完整：provider 轮询 + 冷却）
仓库 `langgenius/dify` @ main。

### 5.1 容器探针：极简 + start_period 豁免窗口
`docker/docker-compose.yaml:264-269`——`test: ["CMD","curl","-f","http://localhost:5001/health"]` + `interval: 30s` / `timeout: 5s` / `retries: 3` / **`start_period: 30s`（compose 版 startupProbe：慢启动豁免窗口）**。
worker（`:340-346`）用 `celery -A celery_healthcheck.celery inspect ping`，且 `disable: ${COMPOSE_WORKER_HEALTHCHECK_DISABLED:-true}`（**默认关闭**）。

### 5.2 为健康检查专门做"轻量入口"（很有启发）
`api/celery_healthcheck.py`（全文 19 行）

```python
# This module provides a lightweight Celery instance for use in Docker health checks.
# Unlike celery_entrypoint.py, this does NOT import app.py and therefore avoids
# initializing all Flask extensions (DB, Redis, storage, blueprints, etc.).
# Using this module keeps the health check fast and low-cost.
celery = Celery(broker=dify_config.CELERY_BROKER_URL)
```

**健康检查不应该复用业务启动路径**——dify 宁可为 healthcheck 单独写一个最小实例。你的 `self_check.py` 恰恰是"复用完整业务栈跑端到端"，方向相反。

### 5.3 降级：轮询 + 冷却（问题 4 的最佳参考）
`api/core/model_manager.py:429-480`

```python
    def _round_robin_invoke(self, function, *args, **kwargs):
        while True:
            lb_config = self.load_balancing_manager.fetch_next()
            if not lb_config:
                raise last_exception if last_exception else ProviderTokenNotInitError(...)
            try:
                kwargs["credentials"] = lb_config.credentials
                return function(*args, **kwargs)
            except InvokeRateLimitError as e:
                self.load_balancing_manager.cooldown(lb_config, expire=60)     # 429 → 冷却 60s
                last_exception = e
                continue
            except (InvokeAuthorizationError, InvokeConnectionError) as e:
                self.load_balancing_manager.cooldown(lb_config, expire=10)     # 连接/鉴权错 → 10s
                last_exception = e
                continue
            except Exception as e:
                raise e                                                        # 其他异常不重试
```

`LBModelManager.fetch_next`（`:1036-1090`）用 Redis `incr` 轮询并跳过冷却中的配置；全部冷却则 `return None`（→ 抛错，而非给假响应）。

另：`api/core/helper/model_provider_cache.py` 把 provider 凭据缓存到 Redis，**TTL 86400 秒**。

### 5.4 未找到
**dify 的 `/health` 路由实现未定位到。** 已核对 main / 1.4.0 / 1.9.0 三个 ref 下的 `api/app.py`、`api/app_factory.py`、`api/dify_app.py`、`api/extensions/ext_blueprints.py`、`api/controllers/__init__.py`、`api/controllers/web/__init__.py`，以及 `api/controllers/{health,healthz,ping}.py`（均 404），均无该路由。判断注册在 fork/企业版分发层或已迁移。为避免编造，此处只引用可验证的 compose 配置与 celery_healthcheck 注释。

---

## 6. 横向对比表
| 维度 | K8s 官方 | LiteLLM | ragflow | Langflow | dify |
|---|---|---|---|---|---|
| 端点分层 | startup / liveness / readiness | `/health/liveness`、`/health/readiness`、`/health`（模型级）、`/health/drain` | `/system/ping`、`/system/healthz`、`/system/status` | `/health`（自认不可靠）、`/health_check`、`/healthz` | 单一 `/health`（实现未见）+ worker celery ping |
| liveness 查什么 | 进程死锁/不可恢复 | 只返回一个字符串 | `return "pong", 200` | `{"status":"ok"}` | HTTP 200 |
| readiness 查什么 | 后端依赖可用性 | DB（15s 缓存 + 4s 预算 + 重连自愈） | DB/Redis/文档引擎/对象存储 | DB 空查询 + 缓存读写 + 插件注册表 | — |
| **健康检查里调 LLM？** | — | **会，但只在后台周期任务，结果缓存，可按模型关闭，限 max_tokens** | **否** | **否** | 未找到（无法判断） |
| 结果缓存 | — | **是**：DB 15s TTL；模型结果后台缓存 + Redis 共享 TTL | 否（都很轻，每次真跑） | 否 | 凭据缓存 86400s |
| 健康检查超时 | `timeoutSeconds` 默认 1s | 2s 单次 / 4s 总预算（注释明写卡 5s 内） | 无显式超时 | `asyncio.wait_for(timeout=worker_timeout)` | compose `timeout: 5s` |
| 启动期策略 | startupProbe 给豁免窗口，期间 liveness/readiness 不执行 | 无启动自检 | **明确规避**："should not block startup on external network access" | 无启动自检 | `start_period: 30s` 豁免窗口 |
| 启动失败策略 | startup 失败 → 重启容器 | — | `depends_on: condition: service_healthy` 交给编排层 | — | 交编排层；`required: false` 让可选依赖不阻断 |
| 降级/熔断 | — | per-deployment `allowed_fails` + `cooldown_time` | 组件级 red/green 状态上报 | 失败即 500/503，不做降级 | **凭据轮询 + 429→冷却60s / 连接错→冷却10s** |
| 错误泄露防护 | — | readiness 默认低信息量，详情需鉴权 | `_meta` 仅失败时附带 | **明确不回传 exception 原文** | — |

---

## 7. 针对你项目的落地改进建议
### 7.1 三个硬伤
| 现状 | 对标结论 |
|---|---|
| 启动自检 70 秒且调用 LLM/embedding | 与 ragflow 的 "should not block startup on external network access" 直接冲突；dify 用 `start_period: 30s` 而非自己做重检查 |
| RPM=3 期间自检打满配额 | LiteLLM 的解法是后台周期任务 + 间隔可配 + `max_tokens` 限制 + 可按模型关闭。你的自检是"启动瞬间一次性打满"，无任何节流维度 |
| `get_chat_model()` 里 `model.invoke("ping")` | 四个项目无同类做法。把外部依赖放进关键路径；失败即降级 Mock 还会**掩盖真实故障** |

### 7.2 启动自检 7 项重新分配
| 原检查项 | 建议归属 | 理由 |
|---|---|---|
| 服务健康 | **保留在启动**（瞬时、纯内存） | 唯一零成本项 |
| 知识库资源（文件/索引存在性） | **保留在启动** | 本地磁盘 IO，毫秒级 |
| 向量库索引 | **保留在启动，但只做"能连上 + 集合存在"**，不做真实检索 | 对标 ragflow `check_doc_engine` 只调 `health()` |
| LangGraph 工作流（图可编译） | **保留在启动** | 纯本地、不调 LLM，是真正的"代码正确性"检查 |
| 大模型加载 | **降级为启动期只校验配置非空 + 凭据格式**，不调 API | 对标 dify `ProviderCredentialsCache` |
| **语义检索能力（真调 embedding）** | **移出启动** → 后台周期任务（建议 300s，结果缓存） | 对标 LiteLLM 后台健康检查 |
| **端到端对话链路（真调 LLM）** | **移出启动** → 独立 `POST /admin/self-check` 手动触发，或后台任务 | 对标 LiteLLM `/health` 与 `/health/liveness` 分离 |

**启动自检退出策略改分级**：
- 致命项（DB/向量库连不上、工作流编译失败）→ **阻断启动**（fail fast）
- 非致命项（LLM 不通、检索为空）→ **告警但放行**，服务带降级标记启动，由 readiness 端点反映

### 7.3 健康检查端点改造为三层
```
GET /health/live       → 200 {"status":"alive"}      进程存活；不查任何外部依赖
GET /health/ready      → DB + 向量库 + Redis，结果缓存 15s，单次 2s 超时，失败 503
GET /health/details    → 需鉴权；各依赖 elapsed/状态 + 后台 LLM 探针最近一次结果
GET /health/drain      → 优雅下线（可选）
```

实现要点（直接抄）：

1. **liveness 绝不查外部依赖**——K8s `timeoutSeconds` 默认 1s，查了就是级联重启
2. **readiness 每步都要超时**：参考 LiteLLM `DB_READINESS_CHECK_TIMEOUT_SECONDS = 2.0` / `DB_READINESS_PROBE_DEADLINE_SECONDS = 4.0`
3. **结果缓存 15 秒**：直接抄 `time_diff < timedelta(seconds=15)` 模式
4. **不回传 exception 原文**：抄 Langflow `HealthResponse` 注释
5. **`/health/live` 不能只 return 常量**：Langflow 的注释提醒 ASGI 可能在应用初始化前就响应。要让 liveness 依赖一个**由 lifespan 设置的就绪标志**

### 7.4 删掉 `get_chat_model()` 里的 ping，改运行时降级
现在的问题：ping 失败 → 静默降级 Mock → 用户拿到假答案且无感知。改法：

1. **构建模型时不 ping**，只做构造
2. **连通性判断放到调用侧**，用 dify 的冷却模式：

```python
# 结构对齐 dify api/core/model_manager.py::_round_robin_invoke
for cred in credentials_pool:
    if cooldown_registry.in_cooldown(cred):
        continue
    try:
        return model.invoke(...)
    except RateLimitError:                              # 429
        cooldown_registry.cooldown(cred, expire=60)     # 限流冷却 60s
        continue
    except (AuthenticationError, APIConnectionError):
        cooldown_registry.cooldown(cred, expire=10)     # 连接/鉴权错冷却 10s
        continue
    except Exception:
        raise                                           # 其他异常不重试
raise AllProvidersUnavailable(...)                      # 全部冷却 → 显式报错，不给假答案
```

3. **Mock 模型只在显式配置 `USE_MOCK_LLM=true` 时启用**，不要作为 ping 失败的自动兜底。自动降级掩盖故障比故障本身更糟。
4. `LLM_HEALTH_CHECK` 开关**改变语义**：从"构建时是否 ping"改为"是否启用后台 LLM 探测任务（含间隔）"，即 LiteLLM 的 `background_health_checks` + `health_check_interval`。

### 7.5 后台 LLM 探测器（针对 RPM=3）
```python
# app/core/health/llm_probe.py —— 结构对齐 litellm/proxy/proxy_server.py::_run_background_health_check
async def _run_background_llm_probe():
    if llm_probe_interval <= 0:
        return                                                                 # 可关闭
    while True:
        try:
            await asyncio.wait_for(
                chat_model.ainvoke("hi", max_tokens=1),                        # 限制 max_tokens
                timeout=LLM_PROBE_TIMEOUT_SECONDS)
            llm_probe_state.update(ok=True, at=now())
        except RateLimitError:
            llm_probe_state.update(ok=None, note="rate_limited")               # 429 不算故障
            await asyncio.sleep(llm_probe_interval * 2)                        # 退避
        except Exception as e:
            llm_probe_state.update(ok=False, error=type(e).__name__)
        await asyncio.sleep(llm_probe_interval)
```

三条硬约束（来自 LiteLLM）：**限制 `max_tokens`**；**间隔可配 + 可关闭**（RPM=3 时调到 60s 以上或直接关）；**429 不能判成"服务不健康"**（对齐 dify：429 触发冷却而非报错）。

### 7.6 优先级
| 优先级 | 动作 |
|---|---|
| **P0** | 把"端到端对话链路"和"语义检索能力"移出启动路径（消除 429 打满，启动 70s → 秒级） |
| **P0** | 删除 `get_chat_model()` 里的 `model.invoke("ping")` |
| **P1** | 拆出 `/health/live` + `/health/ready`，readiness 带 15s 缓存和 2s 超时 |
| **P1** | 启动自检改分级（致命阻断 / 非致命告警放行） |
| **P2** | 凭据冷却降级（429→60s，连接错→10s），Mock 改为显式开关 |
| **P2** | 后台 LLM 探测器 + `GET /health/details`（鉴权） |

---

## 8. "未找到"清单（诚实声明）
| 项目 | 未找到内容 |
|---|---|
| dify | `/health` 路由的具体实现。已核对 main/1.4.0/1.9.0 三个 ref 的 `api/app.py`、`api/app_factory.py`、`api/dify_app.py`、`api/extensions/ext_blueprints.py`、`api/controllers/**`，均未找到。只验证了 compose 中存在该探针调用。 |
| LiteLLM | `DEFAULT_HEALTH_CHECK_INTERVAL`、`DEFAULT_SHARED_HEALTH_CHECK_TTL` 的具体数值（main 分支 `litellm/constants.py` 无定义，仓库处于改名迁移中）。 |
| ragflow | main 分支 `docker/docker-compose.yml` 中 ragflow 自身容器的 healthcheck 段——只有 `depends_on: condition: service_healthy` 引用外部依赖状态，自身探针未找到。 |
| FastAPI 官方生态 | **未找到**：FastAPI 官方文档没有健康检查的规范性章节，tiangolo 模板项目也没做分层健康检查。本报告改用 Langflow（同为 FastAPI 技术栈的成熟项目）作为生态参考。 |
| 所有项目 | **未找到**任何一个项目在进程启动路径上同步调用真实 LLM API 做自检的实现。这一点是比较确定的"不存在"，而非"没搜到"。 |

---

## 附：本次拉取的源码（本地 `/tmp/hc/`）
`dify`：`api/app.py`、`api/app_factory.py`、`api/dify_app.py`、`api/celery_healthcheck.py`、`api/extensions/ext_blueprints.py`、`api/core/model_manager.py`(1159行)、`api/core/helper/model_provider_cache.py`、`docker/docker-compose.yaml`(1326行) ｜ `ragflow`：`api/ragflow_server.py`、`api/utils/health_utils.py`(488行)、`api/apps/restful_apis/system_api.py`(462行)、`docker/docker-compose.yml` ｜ `litellm`：`litellm/proxy/health_endpoints/_health_endpoints.py`(2085行)、`litellm/proxy/proxy_server.py`(15365行)、`litellm/proxy/health_check_utils/shared_health_check_manager.py`(371行)、`litellm/router_utils/cooldown_handlers.py`(625行)、`litellm/constants.py`(1986行) ｜ `langflow`：`src/backend/base/langflow/api/health_check_router.py`(123行)
