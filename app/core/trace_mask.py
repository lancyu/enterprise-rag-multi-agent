"""trace 落盘前的脱敏层 —— 把「正文与凭据不进日志」交给机制，而不是交给记性。

先澄清一个被写错的判断
----------------------
`docs/project-review-and-improvement-plan.md` 的 P0-4 写的证据是
「`trace.jsonl` 会记 query 原文与检索片段」。**实测不成立**：把本地 1188 条历史记录
整份扫一遍，全部 span attr 都是长度 / 计数 / 布尔 / 枚举
（``hits``、``degraded``、``scene="policy"``…），没有一条含 query 原文或片段正文。
唯一出现过的个人数据是 2026-09-15 那批记录里的 ``employee: "E1001"``（工号，
且当前代码已不再写这个字段）。

那这一层还要不要做？**要，而且理由比原判断更硬**：
``span(name, **attrs)`` 与 ``span.attrs[k] = v`` 都是**开放字段**，任何一处将来写下
``span("retrieve", query=query)`` 就会立刻把原文落进磁盘，而且**不会有任何机制察觉**。
「现在没人这么写」不是一条可维护的保证 —— 它靠的是每个人每一次都记得。
所以这一层守的是**将来**：把「偶然没泄露」变成「结构上不会泄露」。

分类判据只有一条：长度
----------------------
本模块**不认键名**来判正文（只有凭据类是例外，见下）。理由：键名会漂移
（``query`` / ``user_query`` / ``question`` / ``text``…），而**枚举天生短、正文天生长**。
用长度区分二者不依赖任何一份需要维护的名单。

- 长度 ≤ :data:`app.config.TRACE_MASK_MAX_CHARS`（默认 64）：原样保留。
  实测现有全部 trace 字符串最长 41 字，故**默认档不改变任何既有字段**——
  这一条有回归测试守着（`tests/test_trace_mask.py` 会从 AST 抽出现有 attr 字面量、
  并在本地有真实 ``trace.jsonl`` 时整份过一遍，断言逐字节不变）。
- 更长：截为 ``前 N 字…<共 M 字>``（M 是**替换前**的原始长度，因为要回答的问题是
  「这段正文本来有多长」）。
  要彻底不留正文，把 ``TRACE_MASK_MAX_CHARS`` 设为 **0**：所有字符串只留长度。

**有意保留的缺口（不是待修的缺陷）**：长度 ≤ 上限的**短自由文本会原样保留**，
例如一句 10 字的提问。因为「短」与「枚举」在形态上不可区分，任何试图区分二者的
判据都是猜。真正要防的两件事都已被覆盖：**凭据与结构化 PII 走形态规则**
（与长度无关，短文本里照样替换），**正文走长度规则**。要连短正文也不留，
就把上限设成 0。

与 ``config.mask_secret`` 的关系：不复用，也不重复
--------------------------------------------------
``mask_secret`` 做的是「**已知**某个值是密钥 → 打印时前 6 后 4」，前提是已经知道
那是密钥；本模块面对的是**不知道哪一段是密钥**的自由文本，判据只能是形态。
两者的输入前提相反，合并会让两边都不成立。
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

#: 替换后的统一占位。键名命中的整值也用它。
_REDACTED = "<REDACTED>"

#: 默认上限。取值依据：实测现有 trace 里最长的字符串值 41 字
#: （``tools = 'find_employee_by_name,query_leave_balance'``），64 留了约 1.5 倍余量。
_DEFAULT_MAX_CHARS = 64

#: 键名命中即整值抹掉 —— **这是唯一按名字判定的地方**。
#: 为什么这里可以按名字、而正文不能：凭据的值可能毫无形态特征（一个随机串和一个
#: 普通短标识符长得一样），形态判据在这里天然无效；而键名是写入方自己起的，
#: 起名时就知道那是凭据。
#: ⚠️ ``token`` 取的是裸子串，会连带命中 ``token_count`` 这类正常指标。**这是有意的**：
#: 宁可少一个指标，不可多一次泄露。现有 trace 的键名没有任何一个命中本表
#: （`tests/test_trace_mask.py` 从 AST 派生键名清单守着这一条）。
_SENSITIVE_KEY_PARTS: Tuple[str, ...] = (
    "authorization",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "cookie",
    "credential",
    "private_key",
)

#: 值里的形态规则。**列表顺序即优先级**，且顺序有两个真实约束：
#: 1. 身份证必须排在手机号之前 —— 18 位身份证的前 11 位可能形似手机号，
#:    先做手机号会把长串切碎，剩下半个身份证号反而更难辨认；
#: 2. 通用 ``key=value`` 规则排在具体密钥前缀之后，避免把 ``sk-xxx`` 只抹成 ``<REDACTED>``
#:    而丢掉「这里原本是个 sk- 前缀密钥」这个信息。
#:
#: ⚠️ 手机号与身份证的边界用的是**字母数字边界**（``(?<![0-9A-Za-z])``），不是单纯的
#: ``(?<!\d)``。这一条是实测踩出来的：``trace_id`` 是 16 位十六进制串，
#: ``b8c7b17477754875`` 里从下标 4 起恰好是一段合法的 11 位手机号形态，
#: 用 ``(?<!\d)`` 判据会把它**整段替换成 <PHONE>**，把一个正常的 trace_id 打坏。
#: 「能被更长的字母数字串包住」意味着它是标识符的一部分，而不是一个独立号码。
_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # 身份证：18 位，地区码 + 出生日期 + 顺序码 + 校验位
    (
        re.compile(
            r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
            r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9A-Za-z])"
        ),
        "<ID>",
    ),
    # 中国大陆手机号：11 位、1 开头、第二位 3-9，两侧不能是字母数字
    (re.compile(r"(?<![0-9A-Za-z])1[3-9]\d{9}(?![0-9A-Za-z])"), "<PHONE>"),
    # 邮箱
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+\b"), "<EMAIL>"),
    # OpenAI / Anthropic 风格的 sk- 密钥
    (re.compile(r"\bsk-[A-Za-z0-9\-_]{12,}\b"), "<REDACTED>"),
    # 显式 Bearer 头
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]{8,}=*"), r"\1 <REDACTED>"),
    # 自由文本里的 key=value / key: value
    (
        re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password)\b\s*[=:]\s*\S+"),
        r"\1=<REDACTED>",
    ),
]

#: 「上游开关关掉了」这条告警每个进程只打一次 —— 每次请求都打会把日志刷爆，
#: 而它表达的是「整段时间内的运行姿态」，不是逐次事件。
_warned_disabled = False
#: 非法上限值（负数）的告警，同理只打一次。
_warned_bad_limit = False


# ---------------------------------------------------------------------------
# 配置读取（**运行时读**，不在 import 期取快照）
# ---------------------------------------------------------------------------
# 为什么运行时读：测试要能 monkeypatch，而 .env 在 import 期就定死了。
# 取快照会让「配了却测不出来」，正是伪配置的老毛病。
def is_enabled() -> bool:
    """脱敏总开关。默认开（见 `app/config.py` 的取值理由）。"""
    from app import config

    return bool(getattr(config, "TRACE_MASK_ENABLED", True))


def _max_chars() -> int:
    """单个字符串值的原文保留上限。契约：返回值 ≥ 0。

    非法值（负数）钳到 **0**，即**更严**的那一档：配错了应当导致多脱敏，
    而不是少脱敏 —— 后者会静默地把正文写进磁盘。

    注意这条钳制本身**不可观测**：``mask_text`` 里 ``max_chars <= 0`` 是同一个分支，
    所以 -5 与 0 的行为完全一致。被测试钉住的是另外两件可观测的事 ——
    ①**负数走严格档、而不是退回宽松的默认 64**；②这条降级会打一条 WARNING。
    这里写成 0 只是让返回值域可读；不要为它单独加用例，那种用例只能靠读实现，
    必然是恒真的（第一版反向验证就踩了这个：改成 `return -99` 后仍然全绿）。
    """
    global _warned_bad_limit

    from app import config

    raw = getattr(config, "TRACE_MASK_MAX_CHARS", _DEFAULT_MAX_CHARS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_MAX_CHARS
    if value < 0:
        if not _warned_bad_limit:
            _warned_bad_limit = True
            from app.utils.logger import logger

            logger.warning(
                "TRACE_MASK_MAX_CHARS=%r 非法（需 ≥ 0），按更严的 0 处理：trace 只留长度", raw
            )
        return 0
    return value


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


# ---------------------------------------------------------------------------
# 脱敏本体
# ---------------------------------------------------------------------------
def mask_text(value: str, max_chars: int) -> str:
    """按「先形态、后长度」两步处理一个字符串。

    顺序不能反：先截断会把一个邮箱切成两半，后半段不再匹配邮箱规则，
    于是「截断反而让敏感串活下来」—— 一个只在长文本上复现的漏洞。
    """
    original_len = len(value)
    for pattern, repl in _PATTERNS:
        value = pattern.sub(repl, value)

    if original_len <= max_chars:
        return value
    if max_chars <= 0:
        return f"<文本 {original_len} 字>"
    return f"{value[:max_chars]}…<共 {original_len} 字>"


def mask_value(value: Any, max_chars: int) -> Any:
    """递归脱敏任意 JSON 形状的值（dict / list / tuple / str / 标量）。"""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _is_sensitive_key(k) else mask_value(v, max_chars))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [mask_value(item, max_chars) for item in value]
    if isinstance(value, str):
        return mask_text(value, max_chars)
    # 数字 / 布尔 / None：不含自由文本，原样返回（也避免了把数字转成字符串）
    return value


def mask_tree(value: Any) -> Any:
    """trace 数据的**唯一**脱敏入口 —— 落盘与对外下发都走它。

    总开关关掉时原样返回，并打一次 WARNING：关闭意味着正文与凭据会原样写盘，
    这种运行姿态必须留下痕迹，不能靠人去读 `.env`。
    """
    global _warned_disabled

    if not is_enabled():
        if not _warned_disabled:
            _warned_disabled = True
            from app.utils.logger import logger

            logger.warning(
                "TRACE_MASK_ENABLED=false —— trace 原样落盘，正文与凭据不做任何脱敏"
            )
        return value

    return mask_value(value, _max_chars())


def reset_warnings() -> None:
    """仅供测试复位「每进程一次」的告警标记。"""
    global _warned_disabled, _warned_bad_limit

    _warned_disabled = False
    _warned_bad_limit = False
