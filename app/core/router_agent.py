"""路由 Agent —— 多 Agent 协作架构的**唯一入口**。

它做什么
--------
一次模型调用，把用户这句话判给五个去向之一，并在入口处**一次性**拦掉越界问题::

                        ┌──────────── 用户提问 ────────────┐
                        │                                  │
                   ┌────▼─────┐                            │
                   │  路由 Agent │  ← 本模块：意图识别 + 边界管控
                   └────┬─────┘                            │
        ┌───────────┬───┴────┬───────────┬─────────────┐   │
        ▼           ▼        ▼           ▼             ▼   │
    smalltalk   simple_rag  complex_rag  tool     out_of_scope
    （闲聊）     （单文档）   （多文档）   （工具）   （越界·就地拦截）
                                                        │
                                                不进入任何子 Agent

为什么边界管控要集中在入口层
----------------------------
这是本架构与「每个子 Agent 各自写一套边界判断」最本质的差别。分散写法的失效方式
是**静默的**：五个子 Agent 各写五份"什么不该答"的判断，它们必然互相漂移；某次
只改了四份，第五份就悄悄放行了本该拦掉的问题，而**没有任何一处能看出这件事**。

集中之后有三件事变成了可验证的：① 边界规则只有一份，改一处即全局生效；
② 越界问题在**花掉检索、工具调用、L4 生成的预算之前**就被拦下（省钱只是副作用，
主因是它根本不该进那条链路）；③ "本轮为什么走了这条路" 有唯一来源
（``RouteDecision``），可以逐条回归（见 ``tests/test_multi_agent.py``）。

为什么是 5 个场景而不是"直接/知识/工具"3 个出口
------------------------------------------------
出口（``intent_type``，见 ``app/graph/state.py``）回答的是"**答案是怎么来的**"，
它必须由实际发生了什么决定；场景（``scene``）回答的是"**这件事该由谁干**"，
必须由入口判断决定。两者是两个问题，强行合并会得到"路由要预判执行结果"的悖论。
例：工具 Agent 拿到证据后让 L4 组织成话 → 场景是 ``tool``，出口是 ``tool``；
但同一个 Agent 在参数不全时反问用户 → 场景仍是 ``tool``，出口却是 ``direct``。

降级策略：确定性规则兜底，且**保守**
------------------------------------
模型不可用（未配 Key / 调用失败 / 输出不是合法 JSON）时不转人工，而是走
``_fallback_route`` 的确定性规则。理由：路由是一次可判错的**粗分类**，
转人工的代价（用户拿不到任何回答）远大于分错路的代价（多检索一次）。

规则刻意**粗粒度**——只覆盖「明确是寒暄」与「其余一律按制度查询处理」两档：

- 不猜 ``tool``：把「年假有多少天」误判进工具 Agent，会让模型去编一个工号或反问
  "请提供员工编号"，体验远差于多检索一次；
- 不猜 ``out_of_scope``：误拦一个真业务问题 = 用户彻底拿不到答案，代价不可逆。
  宁可漏拦（交给 L4 用"知识库中没有相关信息"收尾），不可错拦。

代价，诚实地记一笔
------------------
新增这一层让每个请求多一次模型调用（闲聊路径除外）。换来的是：越界与闲聊
**不再进入检索/工具/L4**——闲聊路径的模型调用反而从 1 次降到 0 次，
两条相抵；真正变贵的是工具链路（+1 次分类调用）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app import config
from app.core.prompts import render as render_prompt
from app.core.routing.catalog import (
    DEFAULT_SCENE,
    OUT_OF_SCOPE_ANSWER,
    SCENE_COMPLEX_RAG,
    SCENE_OUT_OF_SCOPE,
    SCENE_SIMPLE_RAG,
    SCENE_SMALLTALK,
    SCENE_TOOL,
    SCENES,
)
from app.core.tracing import span
from app.utils.logger import logger

__all__ = [
    "DEFAULT_SCENE",
    "OUT_OF_SCOPE_ANSWER",
    "RouteDecision",
    "SCENES",
    "SCENE_COMPLEX_RAG",
    "SCENE_OUT_OF_SCOPE",
    "SCENE_SIMPLE_RAG",
    "SCENE_SMALLTALK",
    "SCENE_TOOL",
    "route_query",
]

# ---------------------------------------------------------------------------
# 场景常量与越界话术：**唯一定义处已搬到 app/core/routing/catalog.py**，
# 本模块只做 re-export。
#
# 为什么"搬走 + re-export"而不是"各留一份"：它们是**合规话术**与**图分支契约**，
# 两份拷贝迟早会漂移，而漂移是静默的（越界话术少一句提示，没人会收到报错）。
# 项目里有一条测试专门按文本扫描、只允许话术出现在一个文件里。
#
# 为什么必须继续从这里导出：``app/graph/edges.py`` 与 ``app/graph/nodes.py``
# 都 import 本模块的 ``SCENES`` / ``SCENE_*``，既有测试也从这里取。
# 门面保留 = 图拓扑、条件边、既有测试**一行不改**。
# ---------------------------------------------------------------------------

# 明显寒暄的**整句锚定**正则：宁可漏判（落到 simple_rag，L4 会诚实说没找到），
# 不可错判（把业务问题当闲聊打发掉）。故一律要求 ^...$ 且长度受限——
# 少了尾锚 `$`，「好像这个制度不太清楚」会被 `^你好` 的前缀匹配吞掉。
_GREETING_RE = re.compile(
    r"^(你|您)?(好|好呀|好啊|好哇|早|早上好|早安|中午好|下午好|晚上好)[\s!！。.~～，,]*$"
    r"|^(hi|hello|hey|yo|哈喽|嗨|哈啰)[\s!！。.~～，,]*$",
    re.I,
)
_THANKS_RE = re.compile(r"^(谢谢|多谢|感谢|非常感谢|辛苦了|thanks|thank you|thx|3q)[\s!！。.~～，,]*$", re.I)
_BYE_RE = re.compile(r"^(再见|拜拜|bye|goodbye|see you|先这样|回头聊|下次聊)[\s!！。.~～，,]*$", re.I)
_IDENTITY_RE = re.compile(r"^(你是谁|你是什么|你是做什么的|你叫什么|介绍一下你|你能做什么|你能帮我做什么|你会什么|你有什么功能)[\s?？!！。]*$", re.I)

#: 确定性兜底只认「短句 + 整句匹配」，避免长句里夹着"你好"被误判。
_FALLBACK_MAX_CHARS = 12


@dataclass
class RouteDecision:
    """路由 Agent 的产出 —— **本轮为什么走这条路**的唯一记录。

    ``scene`` 是唯一驱动图分支的字段；其余都是给人看的（日志、前端面板、排障）。
    """

    scene: str = DEFAULT_SCENE
    reason: str = ""
    #: 模型对本次分类的自评置信度（0~1）。**不参与任何路由判断**——
    #: 它只用于观测：长期偏低说明提示词里的边界规则没写清楚。
    confidence: float = 0.0
    #: ``router``（模型判定）/ ``router:fallback``（模型不可用，确定性规则兜底）
    source: str = "router"
    #: 本条链路是否发生了降级（模型不可用 / 输出不可解析）。
    degraded: bool = False
    #: 越界场景的对外话术。由 ``out_of_scope_node`` 写入答案。
    out_of_scope_answer: Optional[str] = None
    #: 模型原始输出，仅用于排障（不进日志正文、只截断入 trace）。
    raw: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "reason": self.reason,
            "confidence": round(float(self.confidence or 0.0), 3),
            "source": self.source,
            "degraded": self.degraded,
        }


def route_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_context: str = "",
    model: Optional[Any] = None,
) -> RouteDecision:
    """判定本轮该由哪个 Agent 处理。**永不抛异常。**

    Args:
        query: 用户本轮提问原文。
        chat_history: 已裁剪的对话历史（用于消解代词，不改变分类标准）。
        memory_context: 长期记忆。当前**不注入**路由提示词——路由只看
            "这句话本身要干什么"，把历史记忆塞进来只会增加误判面。
            保留参数是为了与其它 Agent 的签名形状一致（调用方无需按 Agent 分类）。
        model: 注入用模型（测试传假模型）。默认取 ``get_chat_model()``。

    Returns:
        RouteDecision。任何异常都被收敛成确定性规则的结果 —— 路由**不能**是
        本轮失败的原因，它是所有路径的入口。
    """
    if not (query or "").strip():
        return _fallback_route(query, "空提问")

    # 离线（未配 Key）时不浪费一次必然失败的模型调用：直接走确定性规则。
    # 这不是"省一次调用"，而是避免 Mock 模型返回一段与分类无关的文本后
    # 还要走一遍解析失败的分支——行为可预期比"多试一次"更重要。
    if model is None and not config.USE_REAL_LLM:
        return _fallback_route(query, "未配置真实模型（离线模式）")

    try:
        chat = model or _default_model()
        prompt = render_prompt(
            "router",
            chat_history=_format_history(chat_history),
            user_query=query,
        )
        with span("router") as s:
            reply = chat.invoke(prompt)
            raw = _content_of(reply)
            s.attrs["raw_chars"] = len(raw)
        parsed = _parse_route(raw)
        if parsed is None:
            logger.warning("路由输出无法解析为合法场景，回落确定性规则：%r", raw[:120])
            return _fallback_route(query, "路由输出不可解析", raw=raw)

        scene, reason, confidence = parsed
        decision = RouteDecision(
            scene=scene,
            reason=reason,
            confidence=confidence,
            source="router",
            out_of_scope_answer=OUT_OF_SCOPE_ANSWER if scene == SCENE_OUT_OF_SCOPE else None,
            raw=raw,
        )
        logger.info(
            "路由判定：scene=%s(%.2f) 理由=%s", scene, confidence, reason or "-"
        )
        return decision
    except Exception as exc:  # noqa: BLE001
        # 路由失败**绝不**升级为本轮失败：它有一条确定性的降级路径，
        # 而且它是所有路径的入口——入口抛异常等于整轮无回答。
        logger.warning("路由 Agent 调用失败，回落确定性规则：%s", exc)
        return _fallback_route(query, f"路由调用失败：{type(exc).__name__}", error=str(exc))


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------
#: 模型偶尔会把 JSON 包在 markdown 代码块或解释文字里，这里做一次尽力提取。
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _parse_route(raw: str):
    """把模型输出解析成 ``(scene, reason, confidence)``；失败返回 ``None``。

    逐层校验而不是"取到 scene 就用"：模型编出一个不在闭集里的场景名时，
    LangGraph 的条件边会直接抛 ``KeyError``——那个异常会被误读成图配置问题，
    而真实原因是模型输出越界。在这里挡住，错误就发生在它该发生的地方。
    """
    if not raw:
        return None
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    scene = str(payload.get("route") or payload.get("scene") or "").strip().lower()
    if scene not in SCENES:
        return None
    reason = str(payload.get("reason") or "").strip()
    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return scene, reason, min(max(confidence, 0.0), 1.0)


def _format_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 4) -> str:
    """把历史压成若干行纯文本。**只取最近几轮**：路由只需消解代词。"""
    lines: List[str] = []
    for msg in (chat_history or [])[-max_turns * 2:]:
        role = "用户" if (msg or {}).get("role") == "user" else "助手"
        content = str((msg or {}).get("content") or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{role}：{content[:200]}")
    return "\n".join(lines) or "（无）"


def _content_of(message: Any) -> str:
    """取出消息的纯文本内容（``content`` 可能是分块列表，需拼接）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return ""


# ---------------------------------------------------------------------------
# 确定性兜底
# ---------------------------------------------------------------------------
def _fallback_route(
    query: str,
    reason: str,
    raw: str = "",
    error: Optional[str] = None,
) -> RouteDecision:
    """保守的确定性路由：只认「明确是寒暄」，其余一律 ``simple_rag``。

    为什么不做更聪明的规则（关键词命中就送 tool / 命中某种词就判越界）：
    项目为此付过学费——八模块规则路由时代每个关键词表都在互相牵制，修 A 坏 B。
    这里只保留一条规则，且它的错误方向是**安全的**（最多多检索一次）。
    """
    text = (query or "").strip()
    short = len(text) <= _FALLBACK_MAX_CHARS
    if short and (
        _GREETING_RE.match(text) or _THANKS_RE.match(text) or _BYE_RE.match(text)
    ):
        scene = SCENE_SMALLTALK
    elif short and _IDENTITY_RE.match(text):
        scene = SCENE_SMALLTALK
    else:
        scene = DEFAULT_SCENE

    logger.info("路由兜底规则：scene=%s（%s）", scene, reason)
    return RouteDecision(
        scene=scene,
        reason=f"确定性规则：{reason}",
        confidence=0.0,
        source="router:fallback",
        degraded=True,
        raw=raw,
        error=error,
    )


def _default_model():
    from app.providers.llm import get_chat_model

    return get_chat_model()
