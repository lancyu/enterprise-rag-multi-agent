"""启动自检与服务健康检测 —— 全量检查，异常项精准定位。

覆盖：服务健康 / 知识库资源 / 大模型加载 / 向量库索引 / **Agent 工具与路由配置** /
**业务只读库与工具 Schema** / 检索能力 / 工作流 / 对话链路
"""
import time
from typing import Callable, List

from app import config
from app.core.llm_factory import _is_rate_limit, get_llm_mode
from app.core.rag_engine import build_index, get_stats, retrieve_knowledge_docs
from app.db.redis_db import get_redis_mode
from app.db.vector_db import get_store_type, get_vector_store
from app.graph.workflow_graph import enterprise_workflow
from app.utils.doc_loader import file_name, list_data_files
from app.utils.logger import logger

FAST_ITEMS = {"service", "vector_db", "agent_config", "sqlite_db", "workflow"}

# 深度检查项：真实调用 LLM / Embedding，启动时默认跳过（避免真实调用 + 避免卡启动）。
# 对齐 Dify / LangGraph 的健康检查实践——启动 readiness 只探基础设施，绝不真实调用模型。
# 仅在显式 deep=True（如 /test/all 手动全量自测）时执行。
DEEP_ITEMS = {"retrieval", "chat"}


def _item(name: str, label: str) -> Callable:
    """装饰器：把检查函数注册为带元信息的检查项。"""

    def decorator(func):
        func._check_name = name
        func._check_label = label
        return func

    return decorator


@_item("service", "服务健康状态")
def _check_service() -> dict:
    return {"detail": f"FastAPI 运行中 · 端口 {config.APP_PORT}", "extra": {"version": config.APP_VERSION}}


@_item("data_dir", "知识库资源")
def _check_data_dir() -> dict:
    files = list_data_files()
    if not files:
        raise RuntimeError(f"知识库目录为空：{config.DATA_DIR}")
    return {"detail": f"检出 {len(files)} 篇原始文档", "extra": {"files": [f["file_name"] for f in files]}}


@_item("llm", "大模型配置")
def _check_llm() -> dict:
    # 仅校验配置，不真实 ping / 构建模型。LLM_HEALTH_CHECK 默认关闭，不在启动时发真实请求。
    # get_llm_mode 首次会惰性构造模型对象，但 ChatOpenAI 构造不发网络请求，安全。
    mode = get_llm_mode()
    detail = f"mode={mode}"
    if mode == "real":
        detail += f" · model={config.LLM_MODEL_NAME}"
    return {
        "detail": detail,
        "extra": {"mode": mode, "model": config.LLM_MODEL_NAME if mode == "real" else "mock-rule-engine"},
    }


@_item("vector_db", "向量库与索引")
def _check_vector_db() -> dict:
    store = get_vector_store()
    count = store.count()
    if count == 0:
        logger.info("向量库为空，触发首次自动建索引")
        result = build_index()
        count = result["chunks"]
        if count == 0:
            raise RuntimeError("索引构建后仍为空，请检查 data/ 目录内容")
        return {"detail": f"自动建索引完成 · {count} 条片段", "extra": {"auto_built": True, "type": get_store_type()}}
    return {"detail": f"索引就绪 · {count} 条片段", "extra": {"type": get_store_type(), "chunks": count}}


@_item("retrieval", "语义检索能力")
def _check_retrieval() -> dict:
    hits = retrieve_knowledge_docs("年假有多少天", top_k=3)
    if not hits:
        raise RuntimeError("检索返回空结果，请检查向量索引")
    best = hits[0]
    return {
        "detail": f"Top1 相似度 {best['score']} · 来源 {file_name(best['source'])}",
        "extra": {"hits": len(hits), "top_score": best["score"]},
    }


@_item("workflow", "LangGraph 工作流")
def _check_workflow() -> dict:
    # 只校验图已编译 + 节点齐全，不真实 invoke——invoke 会真实触发 LLM 生成，
    # 无网络时会把启动拖住（旧版曾卡到 70s+，主因是当时那层重试与超时叠加；
    # 重试虽已移除，但单次 invoke 仍可能等满 LLM_TIMEOUT=30s）。
    # 对齐 LangGraph 的 /ok 无副作用探活实践。
    graph = enterprise_workflow.get_graph()
    nodes = [n for n in getattr(graph, "nodes", {}).keys() if not n.startswith("__")]
    if not nodes:
        raise RuntimeError("工作流图为空")
    return {
        "detail": f"编译就绪 · {len(nodes)} 个节点",
        "extra": {"nodes": nodes},
    }


@_item("chat", "端到端对话链路")
def _check_chat() -> dict:
    from app.graph.state import create_initial_state

    state = create_initial_state(user_query="年假有多少天")
    result = enterprise_workflow.invoke(state)
    answer = result.get("answer") or ""
    return {
        "detail": f"问答链路通过 · 回答 {len(answer)} 字",
        "extra": {
            "scene": result.get("scene"),
            "intent": result.get("intent_type"),
            "tools": result.get("intent_capability"),
            "need_human": result.get("need_human", False),
        },
    }


@_item("agent_config", "Agent 工具与路由配置")
def _check_agent_config() -> dict:
    """工具清单与参数护栏的合法性（纯本地、零依赖，可进 only_fast）。

    检查的是三类**确定性错误**，它们的共同点是不报错、只表现为"答得不对"：

    1. **工具名重复**：``AGENT_TOOLS`` 里两个工具同名时，``_TOOLS_BY_NAME``
       会静默丢掉一个，现象是「模型选了某个工具，执行时却说没有这个工具」。
    2. **护栏登记了不存在的工具**：``GROUNDED_ARGS`` 写错工具名时校验**永不触发**，
       而它守的正是「模型幻觉出另一个人名」这类静默错答——护栏空转最难发现。
    3. **路由兜底场景不合法**：``router_agent`` 的确定性兜底必须落在一个真实存在的
       场景上，否则模型不可用时整张图会因未知场景名崩掉。

    另注：Mock 模式（未配 Key）下 ``bind_tools`` 必然 NotImplementedError，
    这是刻意的（见 providers/llm.py），因此**不在本项里断言真实模型可绑定**——
    那会让离线环境启动即失败。
    """
    from app.core.router_agent import DEFAULT_SCENE, SCENES, route_query
    from app.core.tool_agent import AGENT_TOOLS, GROUNDED_ARGS

    names = [t.name for t in AGENT_TOOLS]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RuntimeError(f"工具名重复（会导致候选集静默丢工具）：{dupes}")
    unknown = sorted(set(GROUNDED_ARGS) - set(names))
    if unknown:
        raise RuntimeError(f"落地校验登记了不存在的工具（护栏将空转）：{unknown}")
    if DEFAULT_SCENE not in SCENES:
        raise RuntimeError(f"路由兜底场景 {DEFAULT_SCENE!r} 不在合法场景集合内：{SCENES}")

    # 兜底路由必须真的能产出合法场景（它只在离线/模型故障时生效，平时跑不到）。
    fallback_scene = route_query("你好").scene
    if fallback_scene not in SCENES:
        raise RuntimeError(f"路由兜底产出了非法场景：{fallback_scene!r}")

    return {
        "detail": (
            f"{len(names)} 个工具 · 工具决策上限 {config.TOOL_AGENT_MAX_STEPS} 轮 · "
            f"{len(SCENES)} 个场景"
        ),
        "extra": {
            "tools": names,
            "grounded_args": {k: list(v) for k, v in GROUNDED_ARGS.items()},
            "scenes": list(SCENES),
            "tool_max_steps": config.TOOL_AGENT_MAX_STEPS,
        },
    }


@_item("sqlite_db", "业务数据库与工具 Schema")
def _check_sqlite_db() -> dict:
    """只读 SQLite 库与 3 个工具的可用性（纯本地、零依赖，可进 only_fast）。

    检查三件事，都是"不检查就会在用户提问时才炸"的：

    1. **库文件存在且两张表齐全**：缺表时工具会抛 ``EnterpriseDBError``，
       而它对用户表现为"业务系统暂时查不了 → 转人工"，不会指出是哪张表缺了；
    2. **只读真的生效**：连接串带 ``mode=ro``，写操作必须被 SQLite 拒绝。
       这是本项目对"工具全部只读"的**硬保证**——靠代码自觉不写是约定，
       靠 URI 拒绝写才是约束（见 app/db/enterprise_db.py）；
    3. **3 个工具能构建参数 schema**：schema 构建失败会在 ``bind_tools`` 时才暴露，
       那时用户已经在等了。
    """
    from app.core.tool_agent import AGENT_TOOLS
    from app.db import enterprise_db as db

    path = db.db_path()
    tables = set(db.table_names())
    required = {"employee", "leave_balance"}
    missing = sorted(required - tables)
    if missing:
        raise RuntimeError(
            f"业务库缺少表 {missing}（路径 {path}）。"
            f"请执行：python scripts/seed_enterprise_db.py"
        )

    # 只读校验：不是"看代码里写没写 INSERT"，而是**真的写一次**看是否被拒绝
    # （见 app/db/enterprise_db.py::is_read_only）。
    read_only = db.is_read_only()
    if not read_only:
        raise RuntimeError(f"业务库可写！只读约束未生效（路径 {path}）")

    for tool in AGENT_TOOLS:
        # 注意 `StructuredTool.args` 返回的是**属性映射本身**（`{"name": {...}}`），
        # 不是含 `properties` 键的完整 JSON Schema。写成 `.get("properties")` 会恒为假，
        # 使本项在工具完全正常时也永远判失败——自检变成常态噪声，真故障反而被淹没。
        properties = getattr(tool, "args", None)
        if not isinstance(properties, dict) or not properties:
            raise RuntimeError(f"工具 {tool.name} 未能构建参数 schema")

    counts = db.table_counts(sorted(required))
    return {
        "detail": "只读库就绪 · " + " / ".join(f"{k} {v} 行" for k, v in counts.items()),
        "extra": {"path": path, "tables": sorted(tables), "rows": counts, "read_only": read_only},
    }


def _collect_checks() -> List[Callable]:
    return [
        _check_service,
        _check_data_dir,
        _check_llm,
        _check_vector_db,
        _check_agent_config,
        _check_sqlite_db,
        _check_retrieval,
        _check_workflow,
        _check_chat,
    ]


def run_self_check(only_fast: bool = False, deep: bool = False) -> dict:
    """执行自检，返回结构化报告。

    Args:
        only_fast: 只跑本地快速项（service / vector_db / workflow）。
        deep: 是否执行深度检查项（retrieval / chat，会真实调用 LLM / Embedding）。
            默认 False——启动自检只探本地基础设施，不发真实模型调用、不卡启动。

    限流（429）类异常会被标记为 skip 而非 fail：429 的语义是"暂时不可用"，
    不是"配置错误"，自检期间触发它不构成"服务不健康"的证据，故不阻断启动。
    这个判断与账号是否限流无关——任何供应商都可能返回 429。

    ⚠️ 不要指望"真实对话时有重试兜底"：**重试已随自造包装层一并移除**
    （见 app/providers/llm.py 的模块 docstring），真实对话遇到 429 会直接失败降级。
    skip 只表示"这次没测出来"，**不等于**这条链路在真实流量下一定可用。
    """
    start = time.perf_counter()
    items = []
    passed = 0
    skipped = 0

    for check in _collect_checks():
        name = getattr(check, "_check_name")
        if only_fast and name not in FAST_ITEMS:
            continue
        if name in DEEP_ITEMS and not deep:
            items.append(
                {
                    "name": name,
                    "label": getattr(check, "_check_label"),
                    "status": "skip",
                    "detail": "深度检查未启用（避免启动时发真实调用），需 deep 模式手动触发",
                    "extra": {},
                    "elapsed_ms": 0,
                }
            )
            skipped += 1
            continue
        item_start = time.perf_counter()
        try:
            info = check()
            items.append(
                {
                    "name": name,
                    "label": getattr(check, "_check_label"),
                    "status": "pass",
                    "detail": info.get("detail", "OK"),
                    "extra": info.get("extra", {}),
                    "elapsed_ms": int((time.perf_counter() - item_start) * 1000),
                }
            )
            passed += 1
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc):
                logger.warning("自检跳过 [%s]（上游返回 429，属暂时不可用，不影响服务启动）", getattr(check, "_check_label"))
                items.append(
                    {
                        "name": name,
                        "label": getattr(check, "_check_label"),
                        "status": "skip",
                        "detail": "限流跳过：上游暂时不可用（429），不影响服务启动",
                        "extra": {},
                        "elapsed_ms": int((time.perf_counter() - item_start) * 1000),
                    }
                )
                skipped += 1
                continue
            logger.warning("自检未通过 [%s]：%s", getattr(check, "_check_label"), exc)
            items.append(
                {
                    "name": name,
                    "label": getattr(check, "_check_label"),
                    "status": "fail",
                    "detail": str(exc),
                    "extra": {},
                    "elapsed_ms": int((time.perf_counter() - item_start) * 1000),
                }
            )

    total = len(items)
    failed = total - passed - skipped
    report = {
        "passed": passed,
        "skipped": skipped,
        "total": total,
        "success": failed == 0,
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
        "items": items,
        "env": {
            "llm_mode": get_llm_mode(),
            "vector_db": get_store_type(),
            "vector_db_requested": config.VECTOR_DB_TYPE,
            "cache": get_redis_mode(),
        },
    }
    return report


def get_health() -> dict:
    """轻量健康检查（不触发重计算）。"""
    return {
        "status": "ok",
        "service": config.APP_TITLE,
        "version": config.APP_VERSION,
        "llm_mode": get_llm_mode(),
        "vector_db": get_store_type(),
        # 与 vector_db 并列返回「配置里要的后端」：两者不一致即说明发生了降级。
        # 只报生效值的话，「配了 milvus 却在用内存库」从健康检查里看不出来。
        "vector_db_requested": config.VECTOR_DB_TYPE,
        "cache": get_redis_mode(),
        "knowledge": get_stats(),
        "config": config.dump_config(),
    }
