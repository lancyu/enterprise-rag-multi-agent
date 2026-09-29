"""工作流管理接口 —— 状态查询与手动触发执行。"""
import asyncio
import time

from fastapi import APIRouter, HTTPException

from app.core.errors import public_detail
from app.core.tracing import begin_trace, end_trace, span
from app.graph.state import create_initial_state
from app.graph.topology import (
    branch_declarations,
    compiled_edges,
    compiled_nodes,
    conditional_sources,
    node_labels,
    ungrounded,
)
from app.graph.workflow_graph import enterprise_workflow, get_mermaid
from app.utils.logger import logger
from app.utils.validator import WorkflowExecuteRequest

router = APIRouter(prefix="/workflow", tags=["工作流引擎"])

# ============================================================
# 拓扑呈现：节点中文名与条件分支**都从编译图派生**。
#
# 这里曾经是**手写声明**（注释还自称「单一事实来源」）—— 它其实是第三份手写副本：
# `workflow_graph.get_mermaid()` 一份、本文件一份、`static/index.html` 的手画 SVG 一份。
# 三份守同一件事，于是 2026-09-24 那条 `tool → verifier` 边摘掉又接回时真的分叉过。
#
# 现在：事实 = 编译图（唯一）；文案 = `app/graph/topology.py` 的呈现表。
# `_validate_topology()` 在模块加载时做**双向**接地检查 —— 呈现表漏一个节点/一条边
# （图上会缺东西）、或多一个（图上会出现不存在的流转），启动日志立刻告警。
# ============================================================
ENTRY_POINT = "memory_load"

NODE_LABELS = node_labels()
BRANCHES = branch_declarations(
    compiled_edges(enterprise_workflow), conditional_sources(enterprise_workflow)
)


def _validate_topology() -> None:
    """校验呈现表与编译图**双向**一致，捕获漂移。

    仅在编译图暴露 ``nodes`` 属性时生效；属性缺失则静默跳过（不阻断启动）。
    """
    nodes = compiled_nodes(enterprise_workflow)
    if not nodes:
        return
    problems = ungrounded(nodes, compiled_edges(enterprise_workflow))
    if problems:
        logger.warning(
            "工作流拓扑呈现表与编译图不一致（面板/接口会展示过时拓扑）：%s",
            "；".join(problems),
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
