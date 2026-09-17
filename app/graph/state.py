"""工作流全局状态定义 —— 与 LangGraph StateGraph 原生兼容。

围绕「一次提问的完整生命周期」组织字段：
    输入（用户/问题/历史） → 记忆（长期上下文） → **路由**（场景判定 + 边界管控）
    → 子 Agent（取证据 / 直答） → 产出（答案/引用/置信度） → 观测（轨迹/异常）

场景（``scene``）与出口（``intent_type``）是两个问题
---------------------------------------------------
五 Agent 架构下**必须**把这两件事分开，合并会得到"路由要预判执行结果"的悖论：

==================  ==============================================
字段                 回答的问题
==================  ==============================================
``scene``           **这件事该由谁干** —— 由路由 Agent 在入口判定
``intent_type``     **答案是怎么来的** —— 由子 Agent 实际做了什么决定
==================  ==============================================

例：工具 Agent 拿到证据交 L4 组织成话 → ``scene=tool``、``intent_type=tool``；
同一个 Agent 因参数不全而反问用户 → ``scene`` 仍是 ``tool``，``intent_type``
却是 ``direct``。若只用 ``scene`` 驱动下游分支，"反问用户"会被当成"有证据"
送进 L4，把一句正常的追问改写成「知识库中没有找到相关信息」。

关于 ``intent_*`` 三个字段名
---------------------------
字段名保留（API 响应与前端徽章依赖它们），但**语义随架构一起变了**：

============  ==========================  ====================================
字段           旧含义                       新含义
============  ==========================  ====================================
intent_type    规则/模型判出的路由通道       本轮**实际走出的出口**：
                                           ``direct``（答案已就绪，不进 L4）/
                                           ``knowledge``（取了检索片段，
                                           答案必须有引用）/
                                           ``tool``（取了业务工具数据）
intent_capability 目录里的能力名（开集）     **实际执行成功的工具名**，如
                                           ``query_leave_balance``
intent_source   rule/lexical/fused/llm      **产出答案的那个 Agent**：
                                           ``smalltalk`` / ``simple_rag`` /
                                           ``complex_rag`` / ``tool`` /
                                           ``out_of_scope`` / ``human_fallback``，
                                           降级时带后缀（如 ``tool:degraded``）
============  ==========================  ====================================

保留字段名而不是重命名，是为了不让一次内部改造把 API 契约和前端一起打碎；
但含义必须在代码里写清楚——否则下一个读代码的人会以为还有一套规则路由。
"""
from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # --- 输入 ---
    user_query: str
    user_id: str                       # 用于隔离长期记忆
    session_id: str                    # 会话标识
    chat_history: List[Dict[str, str]]
    # 来源白名单：调用方（API/鉴权中间件）按用户部门/权限注入，
    # 检索时只返回 source 在该集合内的片段，实现按来源/部门的知识隔离。
    # None 表示不过滤（默认）；空列表 [] 表示全部拒绝。
    allowed_sources: Optional[List[str]]

    # --- 记忆 ---
    memory_context: str                # 长期记忆注入内容（进子 Agent 系统提示）

    # --- 路由（入口层：意图识别 + 边界管控）---
    #: 五个场景之一：smalltalk / simple_rag / complex_rag / tool / out_of_scope。
    #: **唯一驱动图分支的字段**（见 app/graph/edges.py::scene_route_edge）。
    scene: str
    scene_reason: str                  # 判定理由（人话，供前端面板与排障）
    #: 模型自评置信度，**不参与路由**。本地漏斗判定时恒为 0——那不是"很不确定"，
    #: 而是这个数本来就不适用（没有模型自评），区分两者要看 scene_source。
    scene_confidence: float
    #: router（模型判定）/ router:local（本地漏斗零模型判定）/ router:fallback（规则兜底）
    scene_source: str
    #: 复杂 RAG 实际用于检索的查询（含原问题）。拆歪了是静默错误，故必须可见。
    sub_queries: List[str]
    #: 工具 Agent 报告"我干不了活"（模型不支持 function calling）。
    #: 由 ``tool_route_edge`` 据此改道简单 RAG —— 降级决策属于**边**的职责。
    tool_degraded: bool

    # --- 子 Agent 决策 ---
    intent_type: Optional[str]         # direct / knowledge / tool（实际走出的出口）
    intent_capability: Optional[str]   # 实际执行成功的工具名（逗号分隔）
    intent_source: str                 # 产出答案的 Agent（见模块 docstring）
    model_tier: Optional[str]          # 已废弃的档位字段，恒为 None，仅为兼容前端保留
    route_decision: Dict[str, Any]     # 本轮决策摘要（场景 + 判定理由 + 各环节降级）
    # 子 Agent 的执行步骤（工具调用 / 检索轮次），供前端瀑布图与排查用。
    agent_steps: List[Dict[str, Any]]

    # --- 证据 ---
    retrieve_docs: List[Dict[str, Any]]
    tool_result: Optional[str]
    # 检索置信度（0~1，来自 generator.estimate_confidence）。
    #
    # 为什么必须进 state：它同时是**拒答判定的输入**与**对外展示的字段**，
    # 两者必须用同一个数——分别算一次就会出现"面板显示有依据、答案却说没找到"。
    retrieval_confidence: Optional[float]

    # --- 产出 ---
    answer: Optional[str]
    citations: List[Dict[str, Any]]    # 引用溯源清单
    confidence: float                  # 生成置信度（0~1）
    refused: bool                      # 是否触发优雅拒答
    # 生成为何而止：`stop` 正常写完 / `length` 撞上 max_tokens 被截断。
    # 只统计字数**无法区分**「写了 900 字后被截断」与「正常写完 900 字」，
    # 静默截断正是「回答不完整」的元凶。
    # 写入点：graph/nodes.py::generate_answer_node（读 result.stats）；
    # 读取点：app/api/chat.py 响应体 `finish_reason` / `truncated`。
    # **两端缺一不可**——只写不读等于没观测。
    finish_reason: Optional[str]
    truncated: bool

    # --- 观测 ---
    error_msg: Optional[str]
    need_human: bool
    # 可降级故障的**只读观测通道**（不参与路由）。
    # 存在的理由见 docs/history/project-assessment.md P0-1：检索抖动这类可恢复故障
    # 曾被写进 error_msg，导致条件边在答案生成之后把它覆盖成「已转人工」。
    # 凡是「不影响本轮能否给出答案」的异常都记这里，路由只看 need_human。
    #
    # 写入点有三处（缺一不可，否则会退化成"只读不写"）：
    #   graph/nodes.py::simple_rag_node / complex_rag_node —— 检索抛异常已降级；
    #   graph/nodes.py::tool_node —— 工具步骤 error/rejected + 正文回捞。
    # 读取点：app/api/chat.py::collect_soft_warnings（落日志 + 响应体透出计数）。
    # 这类故障不转人工、不丢答案，恰恰因此最容易静默累积到检索彻底失效才被发现。
    soft_warnings: List[str]
    trace: List[Dict[str, Any]]        # 节点执行轨迹，供前端工作流面板可视化


def create_initial_state(
    user_query: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None,
    user_id: str = "default",
    session_id: str = "",
    allowed_sources: Optional[List[str]] = None,
) -> GraphState:
    """创建初始状态（节点入口可用）。"""
    return GraphState(
        user_query=user_query,
        user_id=user_id or "default",
        session_id=session_id or "",
        chat_history=chat_history or [],
        allowed_sources=allowed_sources,
        memory_context="",
        scene="",
        scene_reason="",
        scene_confidence=0.0,
        scene_source="",
        sub_queries=[],
        tool_degraded=False,
        intent_type=None,
        intent_capability=None,
        intent_source="",
        model_tier=None,
        route_decision={},
        agent_steps=[],
        retrieve_docs=[],
        tool_result=None,
        retrieval_confidence=None,
        answer=None,
        citations=[],
        confidence=0.0,
        refused=False,
        finish_reason=None,
        truncated=False,
        error_msg=None,
        need_human=False,
        soft_warnings=[],
        trace=[],
    )
