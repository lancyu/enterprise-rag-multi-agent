"""工作流管理接口 —— 状态查询与手动触发执行。"""
import asyncio
import time

from fastapi import APIRouter, HTTPException

from app.core.errors import public_detail
from app.core.tracing import begin_trace, end_trace, span
from app.graph.state import create_initial_state
from app.graph.workflow_graph import enterprise_workflow, get_mermaid
from app.utils.logger import logger
from app.utils.validator import WorkflowExecuteRequest

router = APIRouter(prefix="/workflow", tags=["工作流引擎"])

# ============================================================
# 单一事实来源：工作流拓扑声明。
# 必须与 app/graph/workflow_graph.build_workflow_graph 完全对应。
# 通过 _validate_topology() 在模块加载时与编译图节点的实际集合做断言，
# 一旦有人新增/改名节点却忘了同步这里，启动日志会立刻告警 —— 杜绝面板与
# 真实图漂移（评估 P1-5）。
# ============================================================
ENTRY_POINT = "memory_load"

NODE_LABELS = {
    "memory_load": "请求初始化·身份解析·记忆加载",
    "router": "路由 Agent（意图识别·边界管控）",
    "smalltalk": "闲聊 Agent（模板直答）",
    "out_of_scope": "越界拦截（常量话术）",
    "simple_rag": "简单 RAG Agent（单次检索）",
    "complex_rag": "复杂 RAG Agent（拆解·多次检索）",
    "tool": "工具 Agent（function calling）",
    "generate_answer": "受控生成",
    "human_fallback": "人工兜底",
}

# 条件分支：from 节点 → 路由可达的目标节点列表（END 表示图终点）。
BRANCHES = [
    {
        "from": "router",
        "type": "conditional",
        "routes": ["smalltalk", "out_of_scope", "simple_rag", "complex_rag", "tool"],
        "note": "五个场景各自独立；越界在入口拦下，不进任何子 Agent",
    },
    {
        "from": "tool",
        "type": "conditional",
        "routes": ["human_fallback", "END", "simple_rag", "generate_answer"],
        "note": "决策失败→人工兜底；反问用户→结束；不支持 function calling→改道简单 RAG；有证据→受控生成",
    },
    {
        "from": "generate_answer",
        "type": "conditional",
        "routes": ["human_fallback", "END"],
    },
]


def _validate_topology() -> None:
    """校验拓扑声明与编译图实际节点集合一致，捕获漂移。

    仅在编译图暴露 ``nodes`` 属性时生效；属性缺失则静默跳过（不阻断启动）。
    """
    # LangGraph 会在编译图中注入 __start__ / __end__ 等伪节点，过滤掉再比较
    compiled_nodes = {
        name
        for name in (getattr(enterprise_workflow, "nodes", {}) or {})
        if not name.startswith("__")
    }
    if not compiled_nodes:
        return
    declared = set(NODE_LABELS)
    if compiled_nodes != declared:
        missing = declared - compiled_nodes
        extra = compiled_nodes - declared
        logger.warning(
            "工作流拓扑声明与编译图不一致（面板可能展示过时拓扑）：缺少=%s 多余=%s",
            sorted(missing),
            sorted(extra),
        )


_validate_topology()


@router.get("/status")
async def workflow_status() -> dict:
    """查询工作流引擎状态与拓扑结构。"""
    return {
        "code": 0,
        "engine": "LangGraph",
        "entry_point": ENTRY_POINT,
        "nodes": [{"name": k, "label": v} for k, v in NODE_LABELS.items()],
        "branches": BRANCHES,
        "mermaid": get_mermaid(),
    }


@router.post("/execute")
async def workflow_execute(req: WorkflowExecuteRequest) -> dict:
    """手动触发一次工作流执行（不写入会话历史，用于调试与演示）。"""
    start = time.perf_counter()
    try:
        # user_id 只用于隔离长期记忆（不同会话不互相污染），与"谁在提问"无关。
        state = create_initial_state(user_query=req.query, user_id=req.user_id or "default")

        def _run():
            begin_trace()
            try:
                with span("workflow_request"):
                    res = enterprise_workflow.invoke(state)
            except Exception:
                end_trace()
                raise
            res["span_tree"] = end_trace()
            return res

        result = await asyncio.to_thread(_run)
        trace = [
            {**t, "label": NODE_LABELS.get(t.get("node"), t.get("node"))} for t in result.get("trace", [])
        ]
        return {
            "code": 0,
            "query": req.query,
            # 路由 Agent 的场景判定（本轮判给哪个 Agent）
            "scene": result.get("scene"),
            "scene_reason": result.get("scene_reason"),
            "scene_source": result.get("scene_source"),
            "sub_queries": result.get("sub_queries", []),
            # 答案实际走出的出口（direct / knowledge / tool）
            "intent": result.get("intent_type"),
            # 实际执行成功的工具名，与 /chat/ask 口径一致
            "intent_capability": result.get("intent_capability"),
            "intent_source": result.get("intent_source"),
            "agent_steps": result.get("agent_steps", []),
            "answer": result.get("answer"),
            "tool_result": result.get("tool_result"),
            "sources_count": len(result.get("retrieve_docs", [])),
            "need_human": result.get("need_human", False),
            # ``error_msg`` 是图内部的**诊断串**（可含异常原文，见 graph/nodes.py），
            # 不原样外发：它此前直接把 ``str(exc)`` 送到了客户端，而同一个故障在
            # /chat/ask 上只给 trace_id —— 两条链路两种口径（P0-2）。
            # 这里保留字段名与"有无错误"的语义，只把原文换成通用文案。
            "has_error": bool(result.get("error_msg")),
            "error_msg": public_detail("本轮遇到致命错误，请查看服务日志") if result.get("error_msg") else None,
            "trace": trace,
            "span_tree": result.get("span_tree", []),
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
        }
    except Exception:  # noqa: BLE001
        logger.exception("工作流执行异常")
        raise HTTPException(status_code=500, detail=public_detail("工作流执行失败"))
