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

代价与回收
----------
新增这一层让每个请求多一次模型调用（闲聊路径除外）。换来的是：越界与闲聊
**不再进入检索/工具/L4**——闲聊路径的模型调用反而从 1 次降到 0 次，
两条相抵；真正变贵的是工具链路（+1 次分类调用）。

这次调用后来被**本地快通道**大幅回收（见 :func:`_local_route`）：句式固定的提问
（寒暄 / 身份 / 年假 / 查人 / 工单号）由零模型的锚定层与词面层直接判定，
实测亚毫秒返回；只有判不了的才落到模型。代价是判据从"模型读懂语义"
退化为"句式与词面匹配"，故只采信漏斗里**零成本**的两层，其余一律放行给模型。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app import config
from app.core.llm_access import content_of, default_model
from app.core.prompts import render as render_prompt
from app.core.routing import match_intent
from app.core.routing.anchors import (
    _ANCHOR_MAX_CHARS,
    _BYE_RE,
    _GREETING_RE,
    _IDENTITY_RE,
    _THANKS_RE,
)
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
from app.core.routing.router import SOURCE_ANCHOR, SOURCE_LEXICAL
from app.core.tracing import span
from app.utils.logger import logger

__all__ = [
    "DEFAULT_SCENE",
    "OUT_OF_SCOPE_ANSWER",
    "SOURCE_LOCAL",
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

# ---------------------------------------------------------------------------
# 寒暄 / 致谢 / 致别 / 身份的**整句锚定**正则 + 长度上限：
# **定义处是 ``app/core/routing/anchors.py``**（层①「开口第一层」也用它），
# 本模块在顶部 import，这里只剩门面（理由与上面的 SCENES 相同）。
#
# 原先此处另有一份**逐字相同**的拷贝，靠一条断言把两份钉在一起。它的失效方式是
# 静默的：这里服务"模型失败后的兜底"，那里服务"开口第一层"，同一句话在两条
# 路径上给出不同判断时不会报错，只表现为"有时像寒暄、有时又像业务问题"。
#
# 宁可漏判（落到 simple_rag，L4 会诚实说没找到），不可错判（把业务问题当闲聊
# 打发掉）。故一律要求 ^...$ 且长度受限——少了尾锚 `$`，
# 「好像这个制度不太清楚」会被 `^你好` 的前缀匹配吞掉。
#
# ⚠️ 别把 ``app/core/sub_agents.py`` 里那组同名正则也并过来：它们**故意不同**
# （子串匹配、不锚定），回答的是"已经在闲聊了，挑哪句回复更像话"，
# 与"要不要把这个句子划进闲聊"是两个问题。共用会让一方被另一方的约束绑住。
# ---------------------------------------------------------------------------
#: 确定性兜底只认「短句 + 整句匹配」，避免长句里夹着"你好"被误判。
_FALLBACK_MAX_CHARS = _ANCHOR_MAX_CHARS


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
    #: ``router``（模型判定）/ ``router:local``（本地漏斗零模型判定）
    #: / ``router:fallback``（模型不可用，确定性规则兜底）
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


# ---------------------------------------------------------------------------
# 本地快通道：零模型判定
# ---------------------------------------------------------------------------
#: ``source`` 新增取值：本地漏斗直接拍的板。与 ``router``（模型判定）、
#: ``router:fallback``（模型不可用走规则）并列，让"这次是谁拍的板"一眼可见。
SOURCE_LOCAL = "router:local"

#: 漏斗里**值得直接采信**的层级。只有这两层是零 IO、零模型的纯本地计算。
#: 其余两层（``fused`` 要一次 embedding、``arbitration`` 要一次模型调用）
#: 与模型路由同一量级——采信它们省不下时间，却丢掉了模型路由手里更完整的
#: 上下文（对话历史与越界规则的完整表述）。**省不下来就不做。**
_LOCAL_TRUSTED_SOURCES = (SOURCE_ANCHOR, SOURCE_LEXICAL)

#: 交给漏斗的时间预算：**0 表示"只允许零成本步骤"**。
#: 这不是绕过机制的技巧，而正是漏斗自己设计的预算语义（见 ``routing/router.py``
#: 的模块 docstring）：每一步动手前先算剩余，剩余不足就**不启动**这一步。
#: 层①（锚定）与层②a（词面）在预算检查之前执行，因此照样会跑；
#: 层②b（语义）与层④（仲裁）各自在动手前查剩余，为 0 时不会启动，
#: 于是既不会多花一次 embedding，也不会多花一次仲裁模型调用。
_LOCAL_BUDGET_MS = 0


def _local_route(query: str) -> Optional[RouteDecision]:
    """用本地漏斗做一次**零模型**判定；判不了返回 ``None``（交由模型路由）。

    为什么值得插这一刀
    ------------------
    路由 Agent 每次要花一次模型调用（实测 2000~2700ms），而它**不产出任何
    用户可见的内容**——这整段时间是纯粹的闸门。企业知识库的提问里有相当一部分
    是"你好""我还剩几天年假""查一下张三的座机""T123 什么状态"这类**句式固定**
    的问法，漏斗的锚定层（正则）与词面层（字符 bigram）就能判准，实测亚毫秒级。

    为什么通道可以直接当场景用
    --------------------------
    ``catalog.CHANNELS is catalog.SCENES``——两者是同一套常量，中间不存在映射表，
    也就不存在"映射表与常量悄悄漂移"这一类缺陷。
    """
    if not (query or "").strip():
        return None
    try:
        decision = match_intent(query, budget_ms=_LOCAL_BUDGET_MS)
    except Exception as exc:  # noqa: BLE001
        # 漏斗文档承诺"永不抛异常"，但它是**新加的一道闸门**：异常若漏出来
        # 会打断整轮回答。这里再兜一层，判不了就当它没说话。
        logger.warning("本地漏斗异常，改由模型路由：%s", exc)
        return None
    if decision.source not in _LOCAL_TRUSTED_SOURCES:
        return None

    logger.info(
        "本地快通道命中：scene=%s（%s）%s，本轮未调用路由模型",
        decision.channel, decision.source, decision.reason,
    )
    return RouteDecision(
        scene=decision.channel,
        reason=f"本地漏斗（{decision.source}）：{decision.reason}",
        # 本地判定**没有**模型自评，如实记 0。这不是"很不确定"，而是
        # "这个数本来就不适用"——区分这两者靠 ``source`` 字段，不靠这个数。
        confidence=0.0,
        source=SOURCE_LOCAL,
        degraded=False,
        # 锚定/词面层判不出越界（那要仲裁层），故这里恒为 None；
        # 仍然按同一条件写，是为了将来若放开采信层级时不必回头补这一处。
        out_of_scope_answer=(
            OUT_OF_SCOPE_ANSWER if decision.channel == SCENE_OUT_OF_SCOPE else None
        ),
    )


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

    # ── 本地快通道（零模型）──────────────────────────────────────────────
    # 两个前置条件缺一不可：
    #
    # ① ``model is None`` —— 注入 model 的语义是"这次路由交给这个模型判"，
    #    测试正是靠它隔离外部依赖。若在这里也插一脚，注入的假模型就永远
    #    轮不到：那不是隔离，是掩蔽。
    # ② ``USE_REAL_LLM`` —— 离线（未配 Key）走的是 Mock 模型 + 确定性兜底，
    #    那是一条刻意设计的降级路径。本次改动只为降延迟，不该顺手改其行为。
    if model is None and config.USE_REAL_LLM:
        local = _local_route(query)
        if local is not None:
            return local

    # 离线（未配 Key）时不浪费一次必然失败的模型调用：直接走确定性规则。
    # 这不是"省一次调用"，而是避免 Mock 模型返回一段与分类无关的文本后
    # 还要走一遍解析失败的分支——行为可预期比"多试一次"更重要。
    if model is None and not config.USE_REAL_LLM:
        return _fallback_route(query, "未配置真实模型（离线模式）")

    try:
        chat = model or default_model()
        prompt = render_prompt(
            "router",
            chat_history=_format_history(chat_history),
            user_query=query,
        )
        with span("router") as s:
            reply = chat.invoke(prompt)
            raw = content_of(reply)
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


#: 喂给路由模型的历史轮数。
#:
#: 路由看历史**只为一件事**：消解代词与指代（「他的邮箱是多少」里的"他"）。
#: 故刻意**比生成层少**——生成层按 ``config.SHORT_TERM_WINDOW`` 取，
#: 路由只要够认出上文主语即可。两者不是同一个量，别顺手合并成同一个配置。
_ROUTER_HISTORY_TURNS = 4


def _format_history(
    chat_history: Optional[List[Dict[str, str]]],
    max_turns: int = _ROUTER_HISTORY_TURNS,
) -> str:
    """把历史压成若干行纯文本。**只取最近几轮**：路由只需消解代词。

    ⚠️ 本函数**只做"取最近几轮"这一件事**，裁剪与格式化分别委托给唯一实现：
    裁剪交给 :func:`app.memory.short_term.build_window`（项目里裁剪点只许有一处，
    见其 docstring 记的那次事故），格式化交给
    :func:`app.memory.short_term.format_history`。

    这里曾经是一份**独立实现**：自己切片、自己拼行、还顺手加了每行 200 字的截断。
    它与 ``short_term`` 那份的差别只在"截断点不同"，于是同一条历史在两处得到
    不同文本——路由与生成看到的历史不一致时**不会报错**，只表现为代词偶尔消解错。

    import 写在函数体内，与 ``app.rag.generator._format_history`` 一致：
    这样 ``tests/test_short_term_symmetry.py`` 才有办法**观测**"确实走了那一份"
    （打在模块属性上的替身，只有模块内每轮重新取名字才拦得住）。
    """
    from app.memory.short_term import build_window, format_history

    window = build_window(chat_history or [], max_turns=max_turns)
    return format_history(window)


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
