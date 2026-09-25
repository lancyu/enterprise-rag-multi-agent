"""工具 Agent —— 多 Agent 协作架构里**唯一**负责 function calling 的角色。

它在五个 Agent 中的位置
------------------------
::

    用户提问
      └─ 路由 Agent（意图识别 + 边界管控）
           ├─ 闲聊 Agent        （不检索、不调工具）
           ├─ 简单 RAG Agent    （单文档可答的制度查询）
           ├─ 复杂 RAG Agent    （多文档对比 / 综合推理）
           ├─ 工具 Agent  ←—— 本模块
           └─ 越界 → 入口直接拒答，不进任何子 Agent

工具 Agent 的职责边界（**刻意很窄**）
-------------------------------------
只做一件事：**把用户的问题翻译成对业务系统的查询，并把结果如实带回**。

- 抽取参数、缺失时追问、链式调用（查 A → 用 A 的结果查 B）——全部由模型在
  function calling 循环里完成；
- 它**不**负责检索制度文档（那是两个 RAG Agent 的活），也**不**负责寒暄
  （那是闲聊 Agent 的活）；
- 它**不**负责组织最终话术——见下方"为什么还要过 L4"。

这个窄边界是多 Agent 架构的主要收益：每个 Agent 的失败模式是**局部**的。
工具 Agent 出问题只会影响"查数据"这一路，不会波及制度问答或闲聊。

工具清单（3 个，全部只读，全部走 SQLite）
-----------------------------------------
``find_employee_by_name`` / ``query_employee_info`` / ``query_leave_balance``
—— 定义与 JSON Schema 见 ``app/tools/sqlite_tools.py``。

**不做收窄**：三个工具一次性全部交给模型。旧实现按路由判出的能力名只给一个
工具，等于把「模型读工具描述后自愈」这条通道堵死——路由把"张三的部门"误判成
假期查询时，模型本来能读描述发现 ``query_employee_info`` 才对得上，
收窄后它没有这个机会。候选集本身就是白名单，模型不可能调出清单外的工具。

两个出口（本模块最重要的一个决定）
----------------------------------
- **直答出口**：模型一个工具都没调 → 它自己的话就是答案。此时置信度记 1.0、
  无引用。**在新架构下这条路径很少走**——闲聊已由路由 Agent 分流，
  走到工具 Agent 的问题基本都是要查数据的。保留它是为了兜住
  「用户只是问了句"你能干什么"」这类边角情形。
- **证据出口**：调用了工具 → 结果进请求作用域 → 交 L4 受控生成。

为什么最终答案要再过一次 L4，而不是直接用工具循环里的输出
----------------------------------------------------------
工具循环里的模型输出是自由文本，而工具**刻意返回结构化 JSON 而非成句的话**
（见 ``tools/sqlite_tools.py``），模型要把它复述成话必然会改写措辞——
而"改写"正是幻觉的入口（数字抄错、字段张冠李戴，且看不出来）。

交回 L4 的收益是具体的：业务数据以**原文**进入受控生成的上下文，与检索片段
一起经过同一套约束（只依据给定内容、不足则直说），并且**只在那里**产生
引用编号。代价是多一次模型调用——在模型不限流、且这条路径本来就要查数据
（已经付了一次调用）的当下，这笔交换是划算的。

护栏（两条，都只针对「会造成静默错答」的失败）
----------------------------------------------
1. **参数落地校验**：登记在 ``GROUNDED_ARGS`` 的参数必须能在用户原句里
   定位到。防的是模型**生成**而非**抄写**参数——用户问「王小明的部门」，
   模型抽出「王五」会返回另一个人的信息，而用户看不出来哪里错了。
   只登记确有必要的那一个槽位，不做无差别的全参数校验。
2. **Mock 模式降级**：未配置 Key 时用的是本地规则模型，它不实现 ``bind_tools``
   （刻意如此，见 ``providers/llm.py``）。此时工具 Agent 无法工作，
   由 ``graph/nodes.py`` 决定降级到哪条路——**不会假装 function calling
   验证过了**。

关于「我的年假还剩几天」
-----------------------
服务端**没有登录态**，工具 Agent 拿不到"当前用户是哪名员工"这个事实。
因此遇到「**我的**年假」这类问法，模型的正确行为是**向用户索要姓名或工号**
（提示词第 1 条：缺参数就追问，禁止编造），而不是猜一个工号去查库。

这是刻意的取舍：**没有身份就不假装有身份**。若要恢复这条路，正确做法是接真实
SSO 并把员工工号重新注入为服务端事实，而不是让模型去猜。

被本模块取代的历史机制
----------------------
本次多 Agent 重构之前，这里曾是"单 Agent 一把抓"的实现：模型同时决定
"要不要检索"与"要不要调工具"。当时删掉了一整套为**当时的限频账号（RPM=3）**
而造的自造路由与档位机制（清单见 ``_archive/removed-selfbuilt-routing-20260915-1314/``）。

多 Agent 架构把那部分职责重新**显式化**了，但方式不同：不再是"用规则猜该走
哪条链"，而是"用一次模型调用判意图，再由各自独立的 Agent 承担"。
两者的关键差别是**职责可见、失败局部**——而不是省调用。
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app import config
# `content_of` / `default_model` 的唯一实现在 `app.core.llm_access`。
# **刻意不加 `as _xxx` 别名**：死代码门禁要求「真有调用点」，而别名导入它看不穿
# （`ast.ImportFrom` 的 asname 不产生 `Name(Load)`），会把这俩判成死代码。
from app.core.llm_access import content_of, default_model
from app.core.prompts import get as get_prompt
from app.core.request_ctx import (
    add_tool_result,
    get_retrieved_docs,
    get_soft_warnings,
    get_tool_results,
)
from app.core.tracing import span
from app.tools.sqlite_tools import TOOL_AGENT_TOOLS
from app.utils.logger import logger

#: 交给模型的完整工具清单（工具 Agent 的全部能力）。
#: 定义在 ``app/tools/sqlite_tools.py``，那里同时是 JSON Schema 的来源。
AGENT_TOOLS: tuple = tuple(TOOL_AGENT_TOOLS)

_TOOLS_BY_NAME: Dict[str, BaseTool] = {t.name: t for t in AGENT_TOOLS}

#: 必须能在用户原句里定位到的参数（见模块 docstring 的护栏 1）。
#: 只登记「抽错会泄露他人信息 / 返回错误记录」的槽位，不做无差别的全参数校验。
GROUNDED_ARGS: Dict[str, tuple] = {
    # 姓名抽错 → 查到另一位同事的部门；而姓名是自由文本，模型最容易"顺手编"。
    "find_employee_by_name": ("name",),
    # 刻意**不**登记 employee_id：它是「谁」的标识，用户往往不会把工号念一遍
    # （「张三的年假」里只有姓名）。模型应当先用 find_employee_by_name 换取工号，
    # 而不是编一个——这正是「必填参数缺失时向用户追问，禁止编造」那条规格。
}

SYSTEM_PROMPT = get_prompt("agent")


# ---------------------------------------------------------------------------
# 文本形式的工具调用回捞
# ---------------------------------------------------------------------------
# 为什么需要这一段（实测，不是防御性编程）
# ----------------------------------------
# 部分 OpenAI 兼容端点会**间歇性地**不把工具调用放进 ``tool_calls`` 字段，
# 而是按模型自己的对话模板渲染进正文。实测 volcengine 的
# ``doubao-seed-2-1-turbo``（本机 .env 当前配置）对同一套 tools：
#
#     问句                         tool_calls              正文
#     ---------------------------  ----------------------  --------------------------
#     报销标准是什么                （识别为工具调用）        （空）
#     工单 T20240101 现在什么状态    （识别为工具调用）        （空）
#     年假有多少天                  **空**                  {"name":"...","parameters":{...}}
#     你好                          （空）                   正常问候语
#
# 注：本表是**当时的实测记录**，其中若干问句对应随后移除的工具（如工单查询）。
# 这条现象与"工具有哪些"无关，只与端点的输出格式有关，故工具清单换代后
# 本机制照常有效。
#
# 注意第三行：**同一个模型、同一轮工具清单**，只因问句不同就退化成文本形式。
# 这不是采样抖动（连续两次复现），而是模型/端点侧的格式不稳定。
#
# 若不回捞，后果是静默且严重的：``tool_calls`` 为空会被判定为「模型认为无需
# 工具」而走直答出口，把这段 JSON 原样当成答案返回给用户——
# 用户看到 ``{"name":"...","parameters":{...}}``，
# 而且它**不会报错**，只会被记成一次正常的「直答」。
#
# 因此这里做一次尽力回捞：识别已知的包装格式与裸 JSON，校验工具名与参数后
# 当作正常的 tool_call 处理。护栏与真实 tool_call 完全一致（同一个候选集校验、
# 同一套落地校验），**不从文本路径放松任何一条**。
_TEXT_CALL_WRAPPERS = (
    re.compile(r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|>", re.S),
    re.compile(r"<tool_call>(.*?)</tool_call>", re.S),
    re.compile(r"<function_call>(.*?)</function_call>", re.S),
)

#: JSON 对象里可能承载工具名的字段（OpenAI 用 name / arguments，豆包系用 name / parameters）
_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("parameters", "arguments", "args", "input")


def _iter_json_objects(text: str) -> List[Any]:
    """从文本里扫描出所有可解析的 JSON 对象 / 数组（按大括号配对，忽略字符串内的括号）。"""
    found: List[Any] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            if depth == 0:
                start = index
            depth += 1
        elif char in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        found.append(json.loads(text[start:index + 1]))
                    except ValueError:
                        pass
                    start = -1
    return found


def _as_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    """把参数部分规整成 dict；字符串形态（模型常把 JSON 再字符串化一层）也接受。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """从正文里尽力回捞工具调用，返回 ``[{"name":..., "args": {...}}, ...]``。

    只做识别与结构校验（**工具名是否在候选集、参数是否为 dict**），
    语义层面的落地校验（参数能否在原句定位）仍由 ``execute_tool_calls``
    统一负责——两处校验会让"哪一条在守"变得说不清。
    """
    if not content:
        return []
    text = content.strip()
    for wrapper in _TEXT_CALL_WRAPPERS:
        match = wrapper.search(text)
        if match:
            text = match.group(1).strip()

    candidates: List[Any] = []
    try:
        parsed = json.loads(text)
        candidates = parsed if isinstance(parsed, list) else [parsed]
    except ValueError:
        candidates = _iter_json_objects(text)

    calls: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = next((item.get(k) for k in _NAME_KEYS if isinstance(item.get(k), str)), None)
        if name not in _TOOLS_BY_NAME:
            continue
        args = next((_as_arguments(item.get(k)) for k in _ARG_KEYS if k in item), None)
        if args is None:
            continue
        calls.append({"name": name, "args": args})
    return calls


# ---------------------------------------------------------------------------
# 「计划词」——用模型自己的计划，省掉一整轮决策
# ---------------------------------------------------------------------------
# 问题（2026-09-24 实测，14 条真实链路 + 真实模型复现）
# --------------------------------------------------
# 工具 Agent 跑两轮。第 2 轮里有一部分并不是"链式调用的第二跳"，而只是
# "模型看过结果，回一句『够了』"——**而它在同一句话里已经把答案写出来了**，
# 随后被 L4 覆盖重写。真实链路里这个形状占 4/14；真实模型逐条复现时，
# 「E1001 的年假还剩几天」「张三的部门是什么」两问的第 2 轮正文就分别是
# 「剩余年假 5.0 天，剩余调休 2.0 天」「张三的部门是技术部」——**答案已经在手里，
# 却被丢掉再生成一遍**。
#
# 为什么不能本地算出来「这一轮够不够」
# ------------------------------------
# 从工具结果反推"够不够回答用户问的那件事"需要语义比对，本地不是能不能写的问题，
# 而是写出来必定是又一份与模型抢活的规则。所以反过来：**让模型在出发前声明计划。**
#
# 这与"让模型判断自己的证据对不对"是两件事，别混：后者是**循环论证**
# （看不出错才会答错，见 verifier 的说明）；前者是**声明意图**——模型当然知道
# 自己打算查一步还是两步，这个判断不需要看到结果。
#
# 实测（真实端点，7 条问句 × 2 次，计划词 100% 命中且两次完全一致）
# ---------------------------------------------------------------
#   问句                          计划词        该做的
#   E1001 的年假还剩几天           PLAN_DONE     一次够          ← 省一轮
#   张三的部门是什么               PLAN_DONE     一次够（find 已含部门）← 省一轮
#   张三的年假还剩几天             PLAN_MORE     需要解析工号
#   李四是什么时候入职的           PLAN_MORE     需要解析工号
#   王五的直属领导是谁             PLAN_MORE     重名，需追问
#   王五的部门是什么               PLAN_DONE     但结果 ambiguous=true → 守卫拦下
#
# 唯一那次判断偏差（重名那条）**被本地守卫救回**：结果不干净时不信 PLAN_DONE。
#
# ⚠️ 顺带证伪了一条更激进的方案：让模型一轮发出整条链、第二跳参数写占位符
#     `{{0.employee_id}}`。实测模型**根本不写占位符**，而且那段说明会干扰计划词
#     （把"入职时间"误判成 PLAN_DONE）。不做。
PLAN_DONE = "PLAN_DONE"
PLAN_MORE = "PLAN_MORE"
_PLAN_TOKENS = (PLAN_DONE, PLAN_MORE)


def _parse_plan(content: str) -> Optional[str]:
    """从正文里取出计划词；取不到返回 ``None``。

    只看**第一个非空行**，且必须是两个词之一打头。刻意不做模糊匹配：
    认不出来就按"没声明"处理，退回原有两轮路径——**默认保守**。
    """
    for line in (content or "").splitlines():
        token = line.strip().strip("`*_ ").upper()
        if not token:
            continue
        for plan in _PLAN_TOKENS:
            if token.startswith(plan):
                return plan
        return None
    return None


def _strip_plan_token(content: str) -> str:
    """把正文里的计划词去掉，返回剩下的部分。

    计划词是**给机器看的**，不该出现在任何面向用户的文本里。直答出口会把
    ``content`` 原样交给用户，所以那条路径必须先过一遍这个函数。

    整段正文**只有**计划词时保留原文：宁可让用户看到一行怪字符串，也不要把
    回答静默变成空串——后者会和"模型什么都没说"在观测上同形。
    """
    lines = (content or "").splitlines()
    for idx, line in enumerate(lines):
        token = line.strip().strip("`*_ ").upper()
        if not token:
            continue
        if any(token.startswith(plan) for plan in _PLAN_TOKENS):
            rest = "\n".join(lines[idx + 1:]).strip()
            return rest or (content or "").strip()
        return (content or "").strip()
    return (content or "").strip()


def _payload_is_usable(raw: Any) -> bool:
    """工具返回的这份数据，**能不能拿去回答**。本判据的唯一产生点。

    两条都要满足：``ok`` 为真，且**不是重名歧义**。歧义时工具给的是"候选人列表"，
    需要的是向用户追问，不是拿去作答——把候选人当成答案写出去，等于替用户挑了一个。

    解析不出来一律按**不可用**处理：这是"要不要少花一轮"的判据，拿不准就不省。
    """
    try:
        payload = json.loads(str(raw))
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return False
    data = payload.get("data")
    return not (isinstance(data, dict) and data.get("ambiguous") is True)


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------
@dataclass
class ToolStep:
    """一次工具调用的执行记录（供前端瀑布图与排查使用）。"""

    index: int
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"          # ok | rejected | error
    detail: str = ""
    elapsed_ms: int = 0
    #: 这次调用拿回来的数据**能不能拿去回答**（``_payload_is_usable``）。
    #:
    #: 它与 ``status`` 是两件事，不能合并：``status == "ok"`` 只说明"工具跑通了"，
    #: 而 ``{"ok": false, "error": "not_found"}``（查无此人）与
    #: ``{"ok": true, "data": {"ambiguous": true, ...}}``（重名，给的是候选人列表）
    #: 都会记成 ``ok``，却都不是能直接作答的数据。
    #:
    #: 默认 ``False``：**拿不准就不算可用**。这个字段只用来做"要不要少花一轮"的
    #: 判断，保守的默认值等于"不省"，与旧行为一致。
    usable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "tool": self.tool, "status": self.status,
            "detail": self.detail, "elapsed_ms": self.elapsed_ms,
            "usable": self.usable,
        }


@dataclass
class ToolDecision:
    """工具决策阶段的产出。

    ``direct_answer`` 非空表示走到了**直答出口**：本轮**一次都没成功取到证据**
    （模型压根没提工具，或提了但全被护栏/异常挡下），它自己的话就是答案。
    典型场景是必填参数缺失时模型主动向用户追问。

    ``direct_answer`` 为空且 ``used_tools`` 非空表示走到了**证据出口**：
    调用方应带着 ``tool_results`` 进入 L4 受控生成（引用编号与置信度只在 L4 产生）。

    注意判据是 ``used_tools``（**成功**执行过）而不是 ``attempted``（提出过）：
    「提过但被拒绝」时若把模型的话丢掉，用户拿到的是 L4 的「知识库没找到」，
    而他实际只缺一个参数——见 ``_decide`` 里的说明。
    """

    direct_answer: Optional[str] = None
    used_tools: List[str] = field(default_factory=list)
    steps: List[ToolStep] = field(default_factory=list)
    attempted: bool = False            # 模型是否**提出过**工具调用（含被护栏拒绝的）
    degraded: bool = False             # 模型不支持 function calling，已降级为无条件检索
    #: 从正文文本里回捞出来的工具调用数量（见 ``parse_text_tool_calls``）。
    #: 非 0 说明当前模型/端点的 function calling 格式不稳定——这是**换模型前
    #: 必须先看的指标**，因为它不报错，只是把工具调用降级成了正文。
    recovered_calls: int = 0
    error: Optional[str] = None
    #: 本轮取回的证据，由 ``run_tool_agent`` 从请求作用域读回后**显式带出**。
    #:
    #: 为什么要显式带，而不是让调用方自己再读一次作用域：作用域是
    #: contextvars 承载的，一旦跨越了 Runnable / 图的节点边界就可能读到另一个
    #: 对象（见 ``app/core/request_ctx.py`` 的说明）。把结果挂在返回值上，
    #: 调用方就与「当前 context 是哪一个」彻底解耦。
    #: 工具 Agent 不产检索片段，``docs`` 恒为空——保留它是为了让"证据带出"
    #: 这条约定在所有 Agent 上形状一致，调用方不必按 Agent 分类处理。
    docs: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    #: 工具**内部**吞掉的可降级故障。与 ``docs`` 同理显式带出。
    soft_warnings: List[str] = field(default_factory=list)
    #: 是否因为模型声明「一次调用就够」而**省掉了后面那轮决策**。
    #:
    #: 这个值必须能被观测：省掉的那一轮是一次真实的模型往返（1.8~2.9s），
    #: 不看它就没法判断"省了多少"与"省错了几次"。写进 ``tool`` span 的
    #: ``plan_shortcut``，与 ``discarded_chars`` 是同一类埋点。
    plan_shortcut: bool = False

    @property
    def is_direct(self) -> bool:
        return self.direct_answer is not None

    def tool_names(self) -> str:
        """用于日志与前端展示的工具名摘要。"""
        return ",".join(self.used_tools) if self.used_tools else "-"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direct": self.is_direct,
            "attempted": self.attempted,
            "degraded": self.degraded,
            "recovered_calls": self.recovered_calls,
            "used_tools": list(self.used_tools),
            "plan_shortcut": self.plan_shortcut,
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# 消息组装
# ---------------------------------------------------------------------------
def build_messages(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_context: str = "",
) -> List[BaseMessage]:
    """组装对话消息：系统提示（含长期记忆）+ 历史 + 本轮提问。

    长期记忆放在**系统提示**里而不是拼进用户消息：它是跨会话累积的背景认知，
    与「用户这一句说了什么」是两种性质的信息，混在一起会让模型把记忆
    当成用户刚说过的话（进而当作可引用的事实依据）。

    ⚠️ 这里**不注入「当前用户是谁」**。服务端没有登录态，就没有任何可靠来源
    能证明「我」对应哪名员工；把它猜出来填给模型，等于让模型拿一个编造的
    身份去查库。正确的行为由工具提示词承载：**缺参数就问用户，不要编造**。
    """
    system = SYSTEM_PROMPT.format(
        memory_context=memory_context or "（无长期记忆）",
    )
    messages: List[BaseMessage] = [SystemMessage(content=system)]
    for msg in chat_history or []:
        role = (msg or {}).get("role")
        content = str((msg or {}).get("content") or "")
        if not content:
            continue
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=query))
    return messages


# ---------------------------------------------------------------------------
# 工具执行
# ---------------------------------------------------------------------------
def _is_grounded(value: Any, query: str) -> bool:
    """槽位值必须能在用户原句中定位到（大小写不敏感）。"""
    text = str(value or "").strip().lower()
    return bool(text) and text in (query or "").lower()


def _grounding_violation(tool_name: str, args: Dict[str, Any], query: str) -> Optional[str]:
    """检查受护栏保护的参数；违规返回原因，通过返回 None。"""
    for key in GROUNDED_ARGS.get(tool_name, ()):
        if not _is_grounded(args.get(key), query):
            return f"参数 {key}={args.get(key)!r} 无法在用户原句中定位"
    return None


def _schema_hint(exc: ValidationError) -> str:
    """把 pydantic 的校验错误压成一句**给模型看**的说明。

    为什么不用 ``str(exc)``：pydantic 的完整报错含 URL、错误类型代号、内部字段
    路径，几十行且中英混杂。它会被塞进 ToolMessage 回给模型，既浪费 token，
    也让模型难以定位到底哪个参数错了。这里只保留「字段名 + 人话」。
    """
    parts: List[str] = []
    for error in exc.errors()[:3]:
        field = ".".join(str(x) for x in error.get("loc") or ()) or "参数"
        parts.append(f"{field}：{error.get('msg', '格式不正确')}")
    return "；".join(parts)


def _tool_message(call: Dict[str, Any], content: str) -> ToolMessage:
    """构造工具返回消息。

    ``tool_call_id`` 必须原样回填：OpenAI 兼容协议要求每条 tool 消息都能对上
    一次 assistant 的 tool_call，对不上时接口会直接 400。
    """
    return ToolMessage(content=content, tool_call_id=call.get("id") or call.get("name") or "tool")


def execute_tool_calls(
    calls: Sequence[Dict[str, Any]], query: str, start_index: int = 0
) -> List[tuple]:
    """执行模型产出的工具调用，返回 [(ToolStep, ToolMessage), ...]。

    三条不可退让的约定：

    - **候选集校验**：模型可能编出清单外的工具名（罕见但确实发生）。这里再查
      一次工具表是纵深防御——"路径上任何一处放松校验，这里都兜得住"比"相信上游"便宜。
    - **落地校验失败不执行**：宁可不查，也不能拿着错的人名去查对的属性。
      被拒绝的调用仍然回一条 ToolMessage 说明原因，让模型（若还有轮次）能自我纠正，
      同时保证对话结构合法。
    - **工具异常不抛**：工具自身约定永不抛（见三个 tools 模块），这里再兜一层。
      一次工具抖动不该让整轮对话失败——上游会按「无证据」给出诚实回答。
    """
    results: List[tuple] = []
    for offset, call in enumerate(calls):
        name = call.get("name") or ""
        raw_args = call.get("args") or {}
        args = raw_args if isinstance(raw_args, dict) else {}
        step = ToolStep(index=start_index + offset, tool=name, args=args)
        started = time.perf_counter()

        tool = _TOOLS_BY_NAME.get(name)
        if tool is None:
            logger.warning("模型选择了候选集外的工具 %r，已忽略", name)
            step.status, step.detail = "rejected", "工具不在候选集内"
            results.append((step, _tool_message(call, f"没有名为 {name} 的工具可用。")))
            continue

        violation = _grounding_violation(name, args, query)
        if violation:
            logger.warning("工具参数未通过落地校验，拒绝执行：%s（%s）", name, violation)
            step.status, step.detail = "rejected", violation
            results.append((
                step,
                _tool_message(call, f"该调用的参数无法在用户原话中确认，已拒绝执行（{violation}）。"),
            ))
            continue

        try:
            content = tool.invoke(args)
            step.status = "ok"
            step.detail = (content or "")[:80].replace("\n", " ")
        except ValidationError as exc:
            # 模型给的参数**不符合 schema**（如工号不匹配 `^E\d{3,}$`）。
            #
            # 必须与"工具执行失败"分开，判据是**责任在谁**：
            #   - 参数不合 schema → 模型侧问题。回一条说明让它在下一轮自我纠正，
            #     用户完全不必知道。若记成 error，业务工具会触发转人工——
            #     用户只是把工号打错一位，却被告知"已转接人工"，这是误伤。
            #   - 工具抛异常 → 基础设施问题。那才可能确实答不了
            #     （见 graph/nodes.py::_failed_business_tools）。
            #
            # 这也是"为什么保留 pattern 而不是全靠工具内校验"的答案：
            # pattern 作为**提示**显著降低模型把整句问话当编号的概率
            # （见 tools/sqlite_tools.py 的说明），而代价是非法值会在此抛错——
            # 把它归到 rejected 而不是 error，代价就被吸收了。
            logger.warning("工具参数不符合 schema：%s（args=%s）", name, args)
            step.status = "rejected"
            step.detail = _schema_hint(exc)
            step.elapsed_ms = int((time.perf_counter() - started) * 1000)
            results.append((
                step,
                _tool_message(call, f"参数不符合要求，已拒绝执行。{step.detail}"),
            ))
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("工具执行异常：%s", name)
            step.status, step.detail = "error", str(exc)[:80]
            content = f"工具执行失败：{type(exc).__name__}"
        step.elapsed_ms = int((time.perf_counter() - started) * 1000)

        # 工具结果要记进请求上下文，否则 L4 生成时拿不到它——
        # 表现为「模型查到了员工信息，最终答案却说知识库里没有」。
        #
        # 新架构下这里**不再排除检索类工具**：工具 Agent 的 3 个工具全部是业务
        # 查询（制度检索已交给 RAG Agent），所以"每次成功的调用都要记"。
        # （旧实现里 search_knowledge 也在这个列表里，它的产物已以**片段**形式
        # 进证据集，再记一遍会让 L4 上下文里同一段内容出现两次。）
        if step.status == "ok":
            # 跑通 ≠ 可用：查无此人（ok=false）与重名（ambiguous）都会走到这里。
            step.usable = _payload_is_usable(content)
            add_tool_result(str(content))

        results.append((step, _tool_message(call, str(content))))
    return results


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_tool_agent(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_context: str = "",
    model: Optional[Any] = None,
    max_steps: Optional[int] = None,
) -> ToolDecision:
    """执行工具决策阶段，**永不抛异常**。

    Args:
        query: 用户本轮提问原文。
        chat_history: 已裁剪的对话历史（轮数/字符裁剪由 memory 层负责）。
        memory_context: 长期记忆文本，注入系统提示。
        model: 注入用模型（测试传假模型，避免打真实模型）。默认取 ``get_chat_model()``。
        max_steps: 最多几轮模型决策。默认取 ``config.TOOL_AGENT_MAX_STEPS``。

    Returns:
        ToolDecision。``error`` 非空表示模型调用本身失败，调用方应转人工；
        ``degraded`` 为真表示模型不支持 function calling，**调用方需自行决定
        降级到哪条路**（见下）。

    关于 ``degraded``：为什么本模块不自己降级
    ----------------------------------------
    旧实现在这里直接"无条件检索一次"作为兜底。多 Agent 架构下这条兜底**不该
    由工具 Agent 决定**——它是一个**路由决策**（"工具路走不通，该改走哪条"），
    与"意图识别"是同一类判断，属于路由 Agent / 图节点的职责。

    工具 Agent 只如实报告"我干不了活"（``degraded=True``），由
    ``graph/edges.py::tool_route_edge`` 决定改走简单 RAG。这样职责单一：
    本模块永远只关心"怎么用工具"，不关心"用不了工具时怎么办"。

    为什么落在**边**上而不是节点里：``tool_node`` 只能返回状态、不能指定下一个
    节点；若在节点内部直接调用检索函数，读 ``workflow_graph.py`` 的人就看不到
    这条改道——而它每天都在生效。拓扑必须完整地留在拓扑里。

    调用前置条件（重要）
    -------------------
    **调用方必须先在与本函数相同的 context 里建立请求作用域**
    （``reset_request_context()`` + ``set_allowed_sources(...)``，`tool_node` 已做）。

    原因见 ``app/core/request_ctx.py``：LangChain 的 ``Runnable.invoke`` 会在
    **复制的 context** 里执行工具，若作用域是在工具内部才第一次创建，
    它建在那个副本里，工具写进去的证据**调用方读不到**——表现为「工具日志显示
    查询成功，生成层却拿到 0 条数据并拒答」。提前在调用方 context 里建好，
    工具拿到的就是同一个对象引用，就地修改两边都可见。

    关于 ``max_steps``（默认 2）——为什么是"两轮"，以及第 2 轮到底在干什么
    ---------------------------------------------------------------------
    ``TOOL_AGENT_MAX_STEPS`` 默认 2，用于覆盖规格要求的三种情形：
    ① 一次信息收集（可并行发多个调用）；② 链式调用（先 ``find_employee_by_name``
    拿到工号，再 ``query_leave_balance``）；③ 参数不合 schema 被拒绝后
    在同一轮内自我纠正。

    **多留一轮"确认"是净亏损，故只留两轮。** 下一轮存在的唯一理由是让模型看到
    上一轮的工具结果后再决定；一旦它不再发出工具调用，本轮立即收口。而
    **已取到证据时那一轮的文本会被 L4 覆盖**（最终答案必须由受控生成产出、
    带引用编号），它唯一的产出只是"没有更多调用"这个消息，却要付一次完整的
    模型调用（实测 1.2~2.9s）。统计真实工具链路，需要第 3 轮工具调用的为 0。

    第 2 轮的真实形态（2026-09-24 离线核对全部含 ``agent_step_*`` 的真实链路）
    ---------------------------------------------------------------------------
    此前本段把第 2 轮描述成"确认轮"，那**只对其中一部分链路成立**。按埋点实际
    分布，14 条真实工具链路里：

    - 第 1 轮**全部**只发 1 个调用（``attrs.tool_calls == 1``），且**全部**是
      ``find_employee_by_name`` —— 即"姓名 → 工号"这一跳解析。这是工具设计的
      必然：另外两个工具都只收 ``employee_id``，没有它就无法开始。
    - 第 2 轮有 **10 条**发出的是**真正的第二跳**（``query_employee_info`` /
      ``query_leave_balance``），只有 **4 条**是收口（``tool_calls == 0``）。
    - 两种形状的耗时也不同：``steps=1``（1 跳 + 收口）的 tool 阶段中位 6417ms，
      ``steps=2``（2 跳）中位 3915ms。**收口那一轮并不便宜**——它要生成一段
      注定被丢弃的正文（``discarded_chars`` 52 / 71 / 99 字），中位比只发调用的
      那一轮还高。

    结论因此要改写：**"两轮"不是冗余，而是链式调用的结构性下限**。要在两跳之间
    做取舍，模型必须先看见第一跳的结果，这一轮往返省不掉；能省的只有"结果里已经
    有答案时还要再查一遍"那种**多余的第三跳**（受 ``max_steps`` 封顶，且已由
    提示词第 2 条要求"够了就不再调用工具"）。要再往下压一轮，只能改结构
    （计划式链式调用 / 本地充分性判据），那不是提示词能解决的事。
    """
    try:
        decision = _decide(query, chat_history, memory_context, model, max_steps)
    except Exception as exc:  # noqa: BLE001
        logger.exception("工具 Agent 决策阶段异常")
        decision = ToolDecision(error=str(exc))
    # 证据显式带出（而不是让调用方自己再读一次作用域）：见 ToolDecision.docs 的说明。
    decision.docs = get_retrieved_docs()
    decision.tool_results = get_tool_results()
    decision.soft_warnings = get_soft_warnings()
    return decision


def _decide(
    query: str,
    chat_history: Optional[List[Dict[str, str]]],
    memory_context: str,
    model: Optional[Any],
    max_steps: Optional[int],
) -> ToolDecision:
    """决策主循环。与 ``run_tool_agent`` 分层的唯一目的是让异常出口只有一个。"""
    steps_limit = max_steps if max_steps is not None else config.TOOL_AGENT_MAX_STEPS
    decision = ToolDecision()

    chat = model or default_model()
    try:
        bound = chat.bind_tools(list(AGENT_TOOLS))
    except NotImplementedError:
        # Mock 模型不实现 bind_tools（刻意如此，见 providers/llm.py）。
        # 如实报告"干不了"，把降级决策留给调用方——不假装 function calling 通过了。
        logger.info("当前模型不支持 function calling，工具 Agent 无法工作")
        return ToolDecision(degraded=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("绑定工具失败，工具 Agent 无法工作：%s", exc)
        return ToolDecision(degraded=True)

    messages = build_messages(query, chat_history, memory_context)

    for round_index in range(max(1, steps_limit)):
        # 本轮模型自己声明的计划（``PLAN_DONE`` / ``PLAN_MORE``）；认不出来就是
        # ``None``——**默认保守**，照旧走下一轮，与这个机制不存在时行为一致。
        plan: Optional[str] = None
        with span(f"agent_step_{round_index}") as s:
            reply = bound.invoke(messages)
            calls = list(getattr(reply, "tool_calls", None) or [])
            # 正文只在这里取一次：下面的回捞、废弃计数、直答出口用的都是它。
            # （回捞命中时 reply 会被重建成空正文，但那条路径下 calls 必非空，
            #  正文不会再被消费，故此处的取值与重建之后再取等价。）
            content = content_of(reply)
            if not calls:
                # 模型把工具调用写成了正文 → 回捞（见 parse_text_tool_calls
                # 的说明：这是实测到的模型/端点行为，不处理会静默降级成直答）。
                recovered = parse_text_tool_calls(content)
                if recovered:
                    decision.recovered_calls += len(recovered)
                    logger.warning(
                        "模型把 %d 个工具调用写进了正文而非 tool_calls，已回捞：%s",
                        len(recovered),
                        ",".join(c["name"] for c in recovered),
                    )
                    # 重建一条结构合法的 AIMessage：后续 messages 里
                    # tool_calls 与 tool 消息必须成对，否则下次 invoke 会 400。
                    reply = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": c["name"], "args": c["args"],
                                "id": f"textcall_{round_index}_{i}", "type": "tool_call",
                            }
                            for i, c in enumerate(recovered)
                        ],
                    )
                    calls = list(reply.tool_calls)
            s.attrs["tool_calls"] = len(calls)
            s.attrs["recovered"] = decision.recovered_calls
            # 收口轮的正文会被丢弃（见下方 else 分支），记下它有多长。
            # 少了这个观测点就只能靠猜——2026-09-15 实测 2571ms，占工具阶段 5097ms 一半。
            #
            # ⚠️ 2026-09-17 复测（提示词加了「已经取到数据之后，不要再撰写回答」之后）：
            #     单次调用链路 agent_step_1 = 4105ms，discarded_chars = 52 —— 模型照旧写了。
            #     更要紧的是**正文从来不是成本**：52 字 ÷ 吐字 ~94 字/s ≈ 0.5s，
            #     剩下 ~3.6s 是这一轮的模型往返本身。所以「让模型别写」最多省 0.5s、
            #     且省不掉这一轮 —— 真正的浪费是**这一轮该不该发生**。
            #     别再往提示词里加同义句，那对付不了它（实测无效）。
            #
            # ⚠️ 2026-09-24 更正：这个观测点只覆盖**收口形状**的链路。按埋点全员核对，
            #     14 条真实工具链路里只有 4 条会走到这里；另外 10 条的"第 2 轮"是真的
            #     在发第二跳（姓名→工号之后按工号取字段），那一轮是结构性必需的，
            #     不是浪费。别再拿 `discarded_chars` 的条数去推断"第 2 轮该不该有"。
            if not calls and decision.used_tools:
                s.attrs["discarded_chars"] = len(content)
            # 计划词只在**真的要发工具调用**时才算数：没有调用就没有"这一次够不够"
            # 这个问题，而正文此时是给用户看的答案（直答出口），不能当计划读。
            if calls:
                plan = _parse_plan(content)
                s.attrs["plan"] = plan or ""
            logger.info(
                "Agent 第 %d 轮决策：tool_calls=%d（%s）%s",
                round_index + 1, len(calls),
                ",".join(c.get("name", "?") for c in calls) or "无",
                f" plan={plan}" if plan else "",
            )

        if not calls:
            if not decision.used_tools:
                # 直答出口。判据是「**本轮一次都没成功取到证据**」，而不是
                # 「一次工具都没提过」——两者的差别正是一条真实存在的链路：
                #
                #   用户：「我的年假还剩几天」
                #     → 模型不知道"我"是谁，却硬填了一个名字
                #       `find_employee_by_name(name="张三")`
                #     → 参数落地校验发现「张三」不在用户原话里，**拒绝执行**
                #     → 模型改口：「请告诉我你的姓名或工号，我来帮你查。」
                #
                # 按旧判据（`attempted` 为真）这句话会被**丢掉**，本轮退化成
                # 「零证据 → L4 拒答」，用户看到「知识库中没有找到相关信息」——
                # 而他实际只缺一个姓名。这与规格第 3 条
                # （「必填参数缺失时，模型主动向用户追问」）正面冲突。
                # 提过 ≠ 拿到：只有 `used_tools` 非空才说明证据在手，
                # 那时才该把成文权交回 L4（引用编号、置信度、拒答只在 L4 产生）。
                #
                # 过一遍 `_strip_plan_token`：模型可能在正文开头带了计划词
                # （约定上它只在调工具时写，但直答路径不该把那个词给用户看）。
                decision.direct_answer = _strip_plan_token(content)
                logger.info(
                    "Agent 直答 %d 字（本轮未取到业务证据：提出过 %d 次调用）",
                    len(decision.direct_answer or ""), len(decision.steps),
                )
            else:
                # 已收集过证据、模型主动收口 → 交回 L4 生成带引用的答案。
                #
                # ⚠️ content 在这里**被丢弃**，这不是疏漏而是架构决定的结果：
                # 这一轮唯一的有效产出是「没有更多工具调用」，而它已经由 calls
                # 为空表达出来了；正文没有任何消费者（最终答案必须由受控生成
                # 产出，引用编号与置信度只在那一层产生）。
                # 代价是实打实的：提示词已要求模型此时别再写回答，长度记进
                # discarded_chars，供离线核对它到底听没听。
                logger.info(
                    "Agent 证据收集结束，转入受控生成（本轮正文 %d 字被丢弃）",
                    len(content),
                )
            return decision

        decision.attempted = True
        executed = execute_tool_calls(calls, query, start_index=len(decision.steps))
        for step, message in executed:
            decision.steps.append(step)
            # 注意 `used_tools` 的判据仍然是 `status == "ok"`（工具**跑通**了），
            # 而不是 `usable`：查无此人（ok=false）也算"提过并执行过"，此时把成文权
            # 交回 L4 让它说"没查到"是对的行为，改成 `usable` 会把一句"没查到"
            # 变成模型的自由发挥。两者的区别见 `ToolStep.usable`。
            if step.status == "ok":
                decision.used_tools.append(step.tool)

        # 「一次调用就够」→ 后面那轮决策**整个省掉**（见 `PLAN_DONE` 的说明）。
        #
        # 四道同时满足才省，缺任何一道都照旧走下一轮：
        #   ① 开关打开；
        #   ② 模型声明 PLAN_DONE（认不出来就不算）；
        #   ③ **后面确实还有一轮可省**——最后一轮本来就要退出，谈不上省；
        #   ④ 本轮每个调用都取到了**可用数据**（`usable`：ok=true 且非重名歧义）。
        #
        # ④ 是安全底线，也是实测里唯一救回判断偏差的那一道：真实模型对
        # 「王五的部门是什么」也吐了 PLAN_DONE，但工具返回 ambiguous=true，
        # 该做的是追问是哪一位——此时不能省，必须让模型看到候选人。
        #
        # 省错的代价要说清楚：模型若"自认为一次够、实际不够"，用户拿到的是
        # L4 的"资料不足"，**是诚实拒答而不是错答**；且退路是现成的
        # （`TOOL_AGENT_PLAN_SHORTCUT=false` 回到旧行为）。
        if (
            config.TOOL_AGENT_PLAN_SHORTCUT
            and plan == PLAN_DONE
            and round_index < steps_limit - 1
            and executed
            and all(step.usable for step, _ in executed)
        ):
            decision.plan_shortcut = True
            logger.info(
                "Agent 第 %d 轮声明 PLAN_DONE 且本轮数据可用，省掉后续决策轮（工具 %s）",
                round_index + 1, ",".join(step.tool for step, _ in executed),
            )
            return decision

        # 对话结构必须完整：assistant 的 tool_calls 与随后的 tool 消息成对出现。
        # 少一条，下一次 invoke 会被接口以 400 拒绝。
        messages.append(reply)
        messages.extend(message for _, message in executed)

    logger.info("Agent 达到最大决策轮数 %d，转入受控生成", steps_limit)
    return decision
