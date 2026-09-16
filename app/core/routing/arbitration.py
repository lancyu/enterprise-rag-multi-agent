"""层④ 灰区仲裁 —— 全漏斗**唯一**的模型调用点，也只在灰区发生。

它存在的理由
------------
前面三层的产出只有两种：要么"我很确定"，要么"我拿不准"。**拿不准时必须有
一个兜底，而不是硬选一个**——硬选等于把一次 50% 的猜测伪装成确定性结论，
而且没有任何一处能看出它其实是猜的。用户给的前提第三条正是这个意思：
"意图识别不出时兜底用 LLM 去判断"。

三条减少"模型自由的发挥空间"的设计（都来自调研到的开源实现）
------------------------------------------------------------
1. **候选清单从目录渲染**（D6）。写死在提示词里，第 1 条原则
   "意图即数据"就白做了：加一种意图要改两处。
2. **让模型只回编号**。借 LlamaIndex ``_build_choices_text`` 的做法，
   从结构上消灭"名字拼错 / 大小写不同 / 尾随空格"这一类**静默失败**——
   它们不会报错，只会表现为"模型选了个不存在的候选"。编号是闭集，
   越界一眼可见。
3. **解析时长名优先**。``policy_compare`` 与 ``policy_single`` 有公共子串，
   短名先匹配会误命中；先匹配长名可以避免这个坑（编号解析失败时的兜底路径）。

永不抛异常
----------
任何失败（模型不可达 / 超时 / 输出越界 / 解析不了）一律返回 ``None``，
由 :mod:`app.core.routing.router` 走保守兜底。**路由不能是本轮失败的原因**——
它是所有路径的入口，入口抛异常等于整轮无回答。
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from app import config
from app.core.prompts import render as render_prompt
from app.core.routing import catalog
from app.core.routing.fusion import Candidate, call_with_timeout, soft_warn
from app.utils.logger import logger

#: 从模型输出里取第一个整数。模型常把编号写成"2"、"2."、"2）"、"编号2"。
_INT_RE = re.compile(r"\d+")


def render_choices(candidates: Sequence[Candidate]) -> str:
    """把候选渲染成"编号 + 能力名 + 何时该用我"的清单。

    ``description`` 必须带上——它是目录里唯一写"何时该用我"的地方，
    不带的话模型只能靠名字猜，而这正是灰区仲裁最不该发生的事。
    """
    lines = []
    for idx, cand in enumerate(candidates, start=1):
        spec = catalog.spec_by_name(cand.name)
        desc = (spec.description if spec else "") or ""
        lines.append(f"{idx}. {cand.name}（{cand.channel}）—— {desc}")
    return "\n".join(lines)


def parse_choice(raw: str, candidates: Sequence[Candidate]) -> Optional[str]:
    """把模型输出解析成**能力名**。解析失败返回 ``None``。

    编号优先；编号不可用时退回"在候选名里做**长名优先**的子串匹配"。
    """
    if not raw:
        return None
    names = [c.name for c in candidates]

    match = _INT_RE.search(raw)
    if match:
        idx = int(match.group(0))
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1].name

    text = raw.strip()
    for name in sorted(names, key=len, reverse=True):
        if name in text:
            return name
    return None


def choose(
    query: str,
    candidates: Sequence[Candidate],
    *,
    model: Optional[Any] = None,
    timeout_ms: Optional[int] = None,
) -> Optional[str]:
    """让模型从灰区候选里挑一个，返回能力名；任何失败返回 ``None``。

    Args:
        query: 用户原话。
        candidates: 灰区候选（通道层归并后的），顺序即渲染顺序。
        model: 注入用模型（测试传假模型）。默认取 ``get_chat_model()``。
        timeout_ms: 等待上限，默认 ``config.ROUTE_ARBITRATION_TIMEOUT_MS``。
    """
    if not candidates:
        return None

    # 离线（未配 Key）时不浪费一次必然失败的调用。这不是"省一次"，
    # 而是避免 Mock 模型返回一段与判断无关的文本后还要走一遍解析失败的分支——
    # 行为可预期，比"多试一次"更重要。
    if model is None and not config.USE_REAL_LLM:
        return None

    limit = config.ROUTE_ARBITRATION_TIMEOUT_MS if timeout_ms is None else timeout_ms
    try:
        chat = model or _default_model()
        prompt = render_prompt(
            "route_arbitration",
            choices=render_choices(candidates),
            user_query=query,
        )
        reply = call_with_timeout(lambda: chat.invoke(prompt), limit)
        raw = _content_of(reply)
        picked = parse_choice(raw, candidates)
        if picked is None:
            logger.warning("灰区仲裁输出无法解析，走保守兜底：%r", raw[:120])
            return None
        return picked
    except Exception as exc:  # noqa: BLE001
        # 降级要留痕：一次仲裁服务故障若只表现为"怎么突然都走兜底了"，
        # 在日志里与"灰区恰好变多"完全同形，不可查。
        soft_warn(f"灰区仲裁失败，走保守兜底：{type(exc).__name__}: {exc}")
        return None


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


def _default_model():
    from app.providers.llm import get_chat_model

    return get_chat_model()
