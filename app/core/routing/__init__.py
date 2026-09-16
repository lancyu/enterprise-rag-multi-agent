"""混合意图路由（四层漏斗）—— 本包是"意图从哪来"的唯一事实来源。

设计文档：``docs/intent-routing-hybrid-design.md``

对外只需要记住三样东西：

- :func:`match_intent` —— 判一句话走哪个通道，**永不抛异常**；
- :class:`RoutingDecision` —— 决策对象，``channel`` 是唯一驱动图分支的字段，
  其余（``source`` / ``degraded`` / ``gray_reason`` / 候选表 / 阈值 / 耗时）
  全部是**给人看的**，用于观测与复盘；
- :data:`CHANNELS` / :data:`CHANNEL_TARGETS` —— 通道闭集与它到图节点的映射。

⚠️ **import 本包时会执行一次目录自洽校验**，失败即 ``raise``。
这是刻意的：一条拼错的规则不会报错，只会表现为"这条规则从来不生效"，
而没有任何一处能看出这件事。这类失败必须发生在**启动期**，
而不是三个月后的一次排障（借 Haystack ``_validate_routes`` 与
LangGraph ``set(agent_names) - set(handoff_destinations)`` 的做法）。

Phase 0 期间本包还被``app/api/routing.py`` 的预演接口使用，
**生产链路仍走 ``router_agent.route_query``**——两条路并存、互不影响，
等预演数据把阈值标定出来再切换（见文档 §8.1）。
"""
from app.core.routing import (
    anchors,
    arbitration,
    catalog,
    fusion,
    gating,
    router,
    signals,
    vocabulary,
)
from app.core.routing.catalog import (
    CHANNELS,
    CHANNEL_TARGETS,
    DEFAULT_SCENE,
    OUT_OF_SCOPE_ANSWER,
    SCENE_COMPLEX_RAG,
    SCENE_OUT_OF_SCOPE,
    SCENE_SIMPLE_RAG,
    SCENE_SMALLTALK,
    SCENE_TOOL,
    CatalogError,
    IntentSpec,
)
from app.core.routing.router import (
    SOURCES,
    SOURCE_ANCHOR,
    SOURCE_ARBITRATION,
    SOURCE_FALLBACK,
    SOURCE_FUSED,
    SOURCE_LEXICAL,
    RoutingDecision,
    describe_catalog,
    match_intent,
)

# ---------------------------------------------------------------------------
# 启动期自洽校验（见模块 docstring）。**必须在所有子模块都导入之后执行。**
# ---------------------------------------------------------------------------
catalog.validate_catalog(
    catalog.all_specs(),
    resolvable=anchors.resolvable,
    guards=signals.GUARDS,
)

__all__ = [
    "CHANNELS",
    "CHANNEL_TARGETS",
    "CatalogError",
    "DEFAULT_SCENE",
    "IntentSpec",
    "OUT_OF_SCOPE_ANSWER",
    "RoutingDecision",
    "SCENE_COMPLEX_RAG",
    "SCENE_OUT_OF_SCOPE",
    "SCENE_SIMPLE_RAG",
    "SCENE_SMALLTALK",
    "SCENE_TOOL",
    "SOURCES",
    "SOURCE_ANCHOR",
    "SOURCE_ARBITRATION",
    "SOURCE_FALLBACK",
    "SOURCE_FUSED",
    "SOURCE_LEXICAL",
    "anchors",
    "arbitration",
    "catalog",
    "describe_catalog",
    "fusion",
    "gating",
    "match_intent",
    "router",
    "signals",
    "vocabulary",
]
