"""证据校验 Agent —— 只回答一个问题：**取回的证据，真的能回答用户这个问题吗**。

它补的是哪一个缺口
------------------
五 Agent 拓扑是按「失败模式可分离性」切的（见 ``app/graph/workflow_graph.py``），
每类失败都有归属：寒暄走模板、越界在入口拦、检索失败降级为"没找到"、业务数据
查不到转人工。但有一类失败**至今没有归属**：

    证据**取回了**，而且结构完好、字段齐全、置信度不低，语义上却答非所问。

最典型的是工具链路：用户问「李四的年假还剩几天」，模型把"李四"解成了同名的
另一个人，工具返回 ``{"ok": true, "annual_leave_days": 3}`` —— 参数合法、字段
完整、没有任何异常。而 ``generate_answer`` 的置信度取自**检索分数**，它衡量的是
"片段与查询像不像"，看不出"这份数据根本不是这个人的"。于是 3 天被写进一句
流畅的回答里，**没有任何一处会报错**。

职责边界（为什么不让 generate_answer 顺手做了）
---------------------------------------------
``generate_answer`` 已经承担「组织成话 + 引用编号 + 置信度拒答 + 截断观测」四件事，
这四项**必须只有一个产生点**（理由见 workflow_graph 模块 docstring）。再让它判断
"证据对不对"，就得到一个"既生成又裁判"的节点，而裁判**必须发生在生成之前**才有
意义 —— 生成完再判，要么白生成一次，要么拒绝一份已经写好的答案（那是把可恢复的
错误变成不可恢复的）。两件事的时点不同、失败模式不同，合成一个节点只会让两者
都不可观测。

所以它是一个**独立节点**（``app/graph/nodes.py::verifier_node``）：输入是
"问题 + 证据"，输出是一个判定，**不生成任何用户可见的文本**。

判定失败时一律**放行**（fail-open）
-----------------------------------
三层降级全部指向"对齐"：

- 未启用 / 没有证据 / 离线模式 → 直接跳过，不花任何代价；
- 调用异常、超时、输出不可解析 → 记 ``degraded``，但仍按"对齐"放行。

理由是本项目的既有口径：**校验是一次增益，不该成为本轮的失败点。**
判错放行的后果是"用户拿到的答案与没有这个 Agent 时一样"；判错拦截的后果是
"这一轮白跑 + 一次重路由的开销"，最坏情况下问题永远答不出来。两者代价不对称，
所以默认放行。（与 ``router_agent`` 的「路由分错路的代价 < 转人工的代价」同一条推理。）

什么时候动手：按需触发，而不是每轮都查
---------------------------------------
它最初的设计是"每个证据链路都校验一次"。上线后这个代价付得不划算：20 次采样
里 **0 次**判出 MISMATCH，而它稳定占掉首 token 路径上 1.2~3.0 秒。**绝大多数
检索是对的** —— 逐轮花钱去复核一个多数时候正确的结论，等于为小概率事件每天
付固定成本。

所以它改成一个**修复手段**：默认不动手，只在"这一轮有理由怀疑证据不对"时才
启动。判据全部取自前置阶段**已经算出来**的结构性事实，判定本身是纳秒级
（见 ``should_verify``）：

- **本轮有可降级故障**（``soft_warnings`` 非空）—— 检索失败 / 路由降级 /
  工具不可用而改道都会进这里。链路自己已经报了"质量已下降"。**改道那条尤其
  值得留痕**：工具 Agent 干不了活、改走简单 RAG 时，这一轮的检索本来就带着
  "退而求其次"的背景。
- **证据含软回退片段**（``fallback`` 为真）—— 软回退的含义就是"Top1 分数不
  达标、硬凑了几条回来"，检索自己已经说了这批证据是凑的。
- **路由摇摆**（``route_gray_reason == tight_margin``）—— 漏斗在几个通道之间
  咬得很紧，说明**这次判定本身就不稳**；判定不稳，检索偏的概率就跟着上去。

工具链路**也进校验**（2026-09-24 两轮定案，方向刚好相反）
--------------------------------------------------------
这一节是**一个方向性错误的标本**，三段都记下来 —— 只看最后一段的人，
下次会把中间那一段再犯一遍。

**① 起点：``if tool_result:``（"本轮有业务数据 → 校验"）。**
理由是"RAG 有引用与置信度两道兜底，工具链路没有第二道防线"。这条判据是
**代理信号**：它讲的是"这轮走了工具链路"，而风险是"证据可能答非所问"。
工具结果是我们自己 SQLite 里的一行结构化记录，不存在让检索偏掉的那些机制
（片段被切碎、Top1 分数不达标、多路融合选错），**结构上不可能"硬凑"**。
后果是每轮工具请求都白付一次：``verifier_node`` 中位 **1551ms**，占整轮请求
17%~25%（简单问题"要等很久"，这一项就在里面），而全部历史判定里 ``aligned``
只出现过 ``True`` 与"没跑"两种取值，``False`` **一次都没有**（54 次判定、
MISMATCH = 0），``verifier_route_edge`` 的回边同样从未触发。

**② 中间：把判据删掉，顺手把 ``tool → verifier`` 这条边也摘了。**
这是**过度治疗**。删判据解决的是"每轮都付"；边解决的是"**出问题时有没有人
复核**"—— 两件事被一起丢掉了。于是工具侧真正可疑的信号（调用被护栏拒绝 /
执行抛异常 / 正文回捞）虽然照旧写进 ``soft_warnings`` + trace + 前端面板，
却**再也没有复核环节**：*留痕不等于兜底*。当时的论证（"信号仍在留痕，所以
删掉是零新增机制"）只证明了**观测**不受影响，没有证明**兜底**不受影响。

**③ 现状：边接回来，判据保持按需。**
``tool_route_edge`` 的出口回到 ``"verifier"``，``tool_result`` 重新成为本模块的
入参（**必须**，理由见下），但那条代理信号**没有**跟着回来 —— 所以 ① 的
"每轮都付"不会重现：

- **干净的工具轮**（调用正常、拿到数据、无降级痕迹）→ ``should_verify`` 在
  ``soft_warnings`` / 软回退 / 路由摇摆上都不命中 → **零代价跳过**；
- **出过状况的轮次** → 复核一次；判为不符且还有预算时由 ``verifier_route_edge``
  **退回路由重判**（图里唯一那条回边，有上界）—— 这就是兜底。

**``tool_result`` 必须一起传进来，否则这条边是白接的**：工具链路的 ``docs``
是空的（``tool_node`` 写的是 ``tool_result``），证据**全在** ``tool_result`` 里。
只接边不传入参，``should_verify`` 会在"没有证据可校验"那一条上短路，
复核永远不会发生 —— 而且**表现与"跳过"一模一样**，不会有任何报错。

**覆盖面如实记下**：调用被拒 / 执行异常 / 正文回捞 / 路由摇摆这些轮次会被复核。
**仍然兜不住的一类**："工具调用成功、数据可用，但工具本身选错了"（用户问部门，
模型查了年假；参数合法、字段齐全，只有语义比对能看出来）。它**没有本地信号**，
语义比对是唯一手段，要抓到它只能每轮都判 —— 把 ``VERIFIER_MODE`` 切到
``always``（现在对**三条链路**都生效），代价即 ① 里的那 1551ms/轮。

工具侧的**可疑信号本身也照旧留痕**：调用被护栏拒绝、执行抛异常、正文回捞仍然
全部写进 ``soft_warnings``（见 ``graph/nodes.py::_collect_degradations``）并进
trace 与前端面板。**留痕与复核是两件事，两件都要**：留痕负责"事后查得出来"，
复核负责"当场兜住"。执行失败另有出口 —— 转人工。

"路由不是本地产的板"为什么**不**算一条理由
------------------------------------------
这一条曾经写作 ``scene_source != router:local``，看着合理，实测是个哑信号。
漏斗的地板 + 边际双门槛很紧，27 条正常提问里只有 4 条被本地采信，于是这条
判据把 **85%** 的正常轮次都拉回来复核。而那 23 条里，"在通道之间摇摆"的
（``tight_margin``）是 **0 条**，全是 ``low_floor``——*分不够*，不是*拿不准*。

两者的区别是本质的：分不够只说明"这句话与例句不像"，而模型路由读的是
**能力描述**，本来就是给这类句子准备的；拿不准才是"路由可能判错"。
把这两件事合并成一个"漏斗没判"，就只剩一个无差别的大桶，只能一律按可疑
处理——按需触发也就退化成了每轮都跑，正是它要摆脱的那个成本。

于是判据读的是 :data:`app.core.routing.gating.GRAY_TIGHT_MARGIN`，
而不是"路由由谁拍板"（见 ``should_verify``）。

**为什么"让模型自己决定要不要校验"这条路走不通**
------------------------------------------------
一个自然的想法是：让生成模型读到证据后自己喊一句"这份证据不对，去校验一下"。
它的问题不是浪费，而是**循环论证**：``generate_answer`` 拿着错误证据写出流畅
的答案，**正是因为它看不出**证据与问题不匹配（问员工甲的年假、证据是员工乙的，
字段齐全、句式对口）。把"该不该校验"交给它，等于让考生自己判断自己有没有答错
题 —— 判得出来的那些，本来就不会答错。

三条路都堵死：

- **生成之前**问它：模型还没读到证据，没有判断力；
- **生成之中**问它：裁判必须在生成之前才有意义（理由见上一节）；生成到一半
  发现不对，要么把半截答案丢掉重来，要么让它把错的说完；
- **生成之后**问它：同上，而且这一轮的 prefill 已经付过了。

所以"该不该校验"靠的**不是模型的自省，而是链路上已经留下的痕迹**。这也是
``auto`` 模式里不引入任何新调用的原因 —— 它读的全是本轮已有的字段。

跳过时**如实记"没校验"**，不记"对齐"
------------------------------------
``VerifyResult.aligned`` 因此是 ``Optional[bool]``：跳过写 ``None``，而不是
默认的 ``True``。写 ``True`` 会让"我们没查"与"查过、没问题"在观测上完全同形
—— 前端面板会显示"证据对齐"，而实际上这件事本轮没人管。这与项目既有的
"``degraded`` 与 ``gray_reason`` 不能混"是同一条口径：**失败兜底不得谎报状态**。
（行为上两者等价：``verifier_route_edge`` 的判据是 ``is False``，``None`` 放行。）

输出为什么是一个判定词而不是 JSON
---------------------------------
与 ``route_arbitration`` 同一取舍：让模型只吐一个词，解析就不需要"从散文里捞
JSON"那一套，也就不会再多出**第三份** JSON 提取实现（前两份分别在
``router_agent`` 与 ``memory/dream.py``）。顺带输出 token 更少 —— 校验环节是
纯延迟开销，越短越好。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app import config
from app.core.llm_access import content_of, default_model
from app.core.prompts import render as render_prompt
from app.core.routing.fusion import call_with_timeout
from app.core.routing.gating import GRAY_TIGHT_MARGIN
from app.core.tracing import span
from app.utils.logger import logger

#: 判定结论的来源，沿用 ``scene_source`` 的命名习惯：冒号后缀表示降级。
SOURCE_MODEL = "verifier"
SOURCE_SKIPPED = "verifier:skipped"
SOURCE_DEGRADED = "verifier:degraded"

#: 判定词。**只有它**会被当成"不符"的结论，其余输出一律落进"不可解析"。
MISMATCH_TOKEN = "MISMATCH"

# ---------------------------------------------------------------------------
# 触发判据（``auto`` 模式下的"要不要动手"）
# ---------------------------------------------------------------------------
#: 判定结论之外的**第二组事实**：这一轮为什么校验（或为什么不校验）。
#: 它与 ``source`` 正交 —— ``source`` 说的是"谁来判的"，``trigger`` 说的是
#: "为什么决定要判"。两者都要留痕：只记 ``source`` 时，"跳过"是一个无差别的
#: 大桶，看不出是"证据没问题"还是"判据写漏了"，也就没法回头收紧判据。
TRIGGER_DISABLED = "disabled"
TRIGGER_ALWAYS = "always"
TRIGGER_DEGRADED = "degraded"
TRIGGER_WEAK_EVIDENCE = "weak_evidence"
TRIGGER_TIGHT_ROUTE = "tight_route"
TRIGGER_NOT_NEEDED = "not_needed"

#: 送进校验提示词的证据上限（字符）。
#:
#: 校验看的是"对不对"，不是"全不全"：给全量片段只会让提示词变长、模型更容易
#: 被无关片段带偏，而它是纯延迟开销。
_EVIDENCE_CHARS = 1200


@dataclass
class TriggerDecision:
    """**这一轮要不要校验**的结论，以及理由。

    ``needed=False`` 时 ``trigger`` 是 ``not_needed`` / ``disabled`` ——
    两者要分开：前者是"查了、没有可疑迹象"，后者是"这个机制被关掉了"。
    合成一个的话，"关掉校验"就会伪装成"证据都没问题"。
    """

    needed: bool = False
    trigger: str = TRIGGER_NOT_NEEDED
    detail: str = ""


@dataclass
class VerifyResult:
    """校验产出。``aligned=False`` 是**唯一**会让图改道的取值。

    ``aligned`` 是 ``Optional[bool]``：``None`` = **这一轮没做校验**
    （关闭 / 按需判据说不需要 / 没有证据可校验）。它不能默认成 ``True`` ——
    那会把"我们没查"写成"查过、没问题"（理由见模块 docstring）。
    """

    aligned: Optional[bool] = None
    reason: str = ""
    source: str = SOURCE_MODEL
    degraded: bool = False
    #: 触发原因，见 ``TRIGGER_*`` 闭集。判定与跳过都会带上它。
    trigger: str = TRIGGER_NOT_NEEDED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aligned": self.aligned,
            "reason": self.reason,
            "source": self.source,
            "degraded": self.degraded,
            "trigger": self.trigger,
        }


def _evidence_text(
    docs: Optional[List[Dict[str, Any]]] = None,
    tool_result: Optional[str] = None,
) -> str:
    """把证据压成一段纯文本 —— **两个载体都要收**。

    编号格式与 L4 的 ``[n] 来源：xxx`` 保持一致 —— 两条链路对同一份证据的呈现
    应当一样，否则模型在"校验"与"生成"时会看到两种不同的证据形态。

    为什么要收 ``tool_result``：工具链路的 ``docs`` **恒为空**
    （``ToolDecision.docs`` 的说明：工具 Agent 不产检索片段），这一轮的全部证据
    都在 ``tool_result`` 里。只收 ``docs`` 的话，``should_verify`` 会在
    "没有证据可校验"那一条上短路 —— 表现与"干净轮次跳过"**完全一样**，
    边接回来了也不会有人复核。这在 2026-09-24 是这条边"白接"的唯一原因。

    两者"同时非空"在实践中不会发生（工具链路 ``docs`` 为空；``tool_degraded``
    改道 ``simple_rag`` 的轮次在写 ``tool_result`` 之前就返回了），所以这里
    不做预算分配，按序拼接后统一截断。
    """
    parts: List[str] = []
    for idx, doc in enumerate(docs or [], start=1):
        content = str((doc or {}).get("content", "")).strip()
        if not content:
            continue
        parts.append(f"[{idx}] 来源：{(doc or {}).get('source', '')}\n{content}")
    tool_text = str(tool_result or "").strip()
    if tool_text:
        parts.append(f"[工具查询结果]\n{tool_text}")
    return "\n\n".join(parts)[:_EVIDENCE_CHARS]


def _has_soft_fallback(docs: Optional[List[Dict[str, Any]]]) -> bool:
    """证据里是否含软回退片段。

    这里刻意**不**自己算一个分数阈值，只读检索层已经判好的事实：「软回退」
    的含义就是 Top1 分数低于 ``effective_fallback_min()``、于是"硬凑几条回来
    总比返回空好"。再定一个门槛就成了同一个判断的第二份实现。
    """
    return any(bool((doc or {}).get("fallback")) for doc in (docs or []))


def _verifier_mode() -> str:
    """归一化后的触发方式；未知取值降级到默认值并告警。

    ``config`` 是叶子模块（不 import 任何 logger），所以"取值非法"的处置放在
    这里，与 ``app/rag/indexer`` 处置 ``DOCSTORE_STRATEGY`` 的位置一致。
    """
    mode = config.VERIFIER_MODE
    if mode in config.VERIFIER_MODE_CHOICES:
        return mode
    logger.warning(
        "VERIFIER_MODE 取值未知：%r，回退 %s", mode, config.VERIFIER_MODE_DEFAULT
    )
    return config.VERIFIER_MODE_DEFAULT


def should_verify(
    docs: Optional[List[Dict[str, Any]]] = None,
    tool_result: Optional[str] = None,
    route_gray_reason: str = "",
    soft_warnings: Optional[List[str]] = None,
) -> TriggerDecision:
    """**这一轮要不要做证据校验。** 纯函数、零 IO、永不抛异常。

    它是"要不要校验"的唯一产生点（``verify_evidence`` 也走它，不另写一遍）。
    判据与理由见模块 docstring。下面按优先级排列，命中的**第一条**会被记下来：

    1. 关闭 → 不动手；
    2. **证据为空** → 不动手。这条与模式无关：证据为空时"是否对齐"这个问题
       根本不成立，开成 ``always`` 也变不出证据来。判空必须**同时看两个载体**
       （``docs`` 与 ``tool_result``）—— 工具链路的证据只在后者里，少看一个
       就会让工具轮永远停在这一步，且症状与"干净轮次跳过"完全同形；
    3. ``always`` → 动手；
    4. 可降级故障 / 软回退片段 / 路由摇摆 → 动手；
    5. 其余 → 不动手。**这里只判"本轮有没有可疑迹象"，不判"这是哪条链路"。**
       "走了工具链路"曾经是第 4 步的一条判据，但它与风险不是同一件事
       （工具结果是自己库里的结构化记录，没有让检索偏掉的机制），属于**代理
       信号**，已删除 —— 它描述的风险由第 4 步那几条真正命中，而它带来的
       "每轮白付 1551ms"随之消失。取舍的完整过程见模块 docstring。

    第 4 步的每一条判据读的都是**本轮已有的字段**，判定本身不发起任何调用 ——
    这正是"按需"能省下钱的原因：省的是"动手"那一步，不是"判断"那一步。

    ``soft_warnings`` 里有一条来自工具侧：工具不可用**而改道简单 RAG** 时，
    ``tool_node`` 会写"已改道简单 RAG"。那一轮走的是检索、证据是检索来的，
    带着"退而求其次"的背景，正是该复核的对象（判据是"这轮降级过"，不是
    "这轮碰过工具"）。同理，``tool_result`` 入参只用来回答"**有没有证据**"，
    不参与"**要不要**校验"的判断 —— 这两件事混淆就是那条代理信号本身。

    为什么最后一条读的是 "漏斗**为什么**没判" 而不是 "谁拍的板"
    ---------------------------------------------------------
    ``route_gray_reason`` 是漏斗自己给出的灰区原因（取值域见
    :mod:`app.core.routing.gating`），它把两件事分开了：

    - ``tight_margin`` —— 顶两名之间咬得很紧，**这次判定本身就不稳**，算理由；
    - ``low_floor`` / ``no_candidate`` —— 分不够 / 没有候选，只说明这句话与
      例句不像。模型路由读能力描述本来就擅长这类，**不算理由**；
    - ``budget_exceeded`` —— 预算耗尽，属降级，已由 ``soft_warnings`` 那条覆盖。

    用 ``scene_source``（"是不是本地快通道拍的板"）当判据时，这个区分是没有的：
    凡是漏斗没采信的轮次一律算可疑。实测那等于把 85% 的正常提问拉回来复核，
    而其中"摇摆"的是 0 条 —— 按需触发退化成了每轮都跑。
    """
    mode = _verifier_mode()
    if mode == "off":
        return TriggerDecision(False, TRIGGER_DISABLED, "校验已关闭（VERIFIER_MODE=off）")

    if not _evidence_text(docs, tool_result).strip():
        return TriggerDecision(False, TRIGGER_NOT_NEEDED, "没有证据可校验")

    if mode == "always":
        return TriggerDecision(True, TRIGGER_ALWAYS, "VERIFIER_MODE=always：每轮都校验")

    if soft_warnings:
        return TriggerDecision(
            True,
            TRIGGER_DEGRADED,
            f"本轮存在可降级故障：{str(soft_warnings[0])[:60]}",
        )
    if _has_soft_fallback(docs):
        return TriggerDecision(
            True, TRIGGER_WEAK_EVIDENCE, "证据含软回退片段（检索硬凑）"
        )
    if route_gray_reason == GRAY_TIGHT_MARGIN:
        return TriggerDecision(
            True, TRIGGER_TIGHT_ROUTE, "路由在通道间咬得很紧（判定不稳）"
        )
    return TriggerDecision(False, TRIGGER_NOT_NEEDED, "证据侧没有可疑迹象")


def _parse_verdict(raw: str) -> Optional[bool]:
    """把模型输出解析成 ``aligned``；解析不出返回 ``None``。

    只认第一个非空行里的判定词（大小写不敏感）。刻意**不做**模糊匹配：
    「看着有点不像 MISMATCH，但也不像 ALIGNED」这类输出应当落进"不可解析"，
    由调用方按 fail-open 放行，而不是替模型猜一个。
    """
    for line in (raw or "").splitlines():
        token = line.strip().upper()
        if not token:
            continue
        if token.startswith(MISMATCH_TOKEN):
            return False
        if token.startswith("ALIGNED"):
            return True
        return None
    return None


def _reason_line(raw: str) -> str:
    """取判定词之后的第一个非空行作为理由（可缺省）。截断是为了它要进日志与响应体。"""
    for line in [ln.strip() for ln in (raw or "").splitlines()][1:]:
        if line:
            return line[:80]
    return ""


def verify_evidence(
    query: str,
    docs: Optional[List[Dict[str, Any]]] = None,
    tool_result: Optional[str] = None,
    model: Optional[Any] = None,
    route_gray_reason: str = "",
    soft_warnings: Optional[List[str]] = None,
) -> VerifyResult:
    """判断证据是否回答了 ``query``。**永不抛异常。**

    第一件事是问 ``should_verify`` 这一轮该不该动手 —— 它是唯一产生点，
    本函数不再自己判一遍"要不要"（否则同一个策略就有了两份实现）。

    两个证据载体都要收，且**都要传下去**：``docs`` 是检索片段，
    ``tool_result`` 是工具查询结果（工具链路 ``docs`` 恒为空，证据全在这里）。
    只收 ``docs`` 的话，工具轮会停在 ``should_verify`` 的"没有证据可校验"上 ——
    静默跳过，与"干净轮次不需要校验"长得一模一样（见模块 docstring 的 ③）。

    Args:
        query: 用户本轮提问原文。
        docs: 检索类 Agent 取回的片段。
        tool_result: 工具 Agent 取回的业务数据（文本形态）；无工具调用时为
            ``None``。它只用于"有没有证据可校验"与"证据说了什么"，
            **不参与"要不要校验"的判断**（那正是被删掉的那条代理信号）。
        model: 注入用模型（测试传假模型）。默认取 ``get_chat_model()``。
        route_gray_reason: 本地漏斗**为什么**没判（``tight_margin`` 等），
            参与触发判据。只有 ``tight_margin`` 会触发复核。
        soft_warnings: 本轮的可降级故障，参与触发判据。

    Returns:
        VerifyResult。没轮到它动手时 ``aligned=None``（**不是** ``True``）；
        动手之后任何失败才收敛成"对齐且放行"（``aligned=True``）。
    """
    trigger = should_verify(
        docs=docs,
        tool_result=tool_result,
        route_gray_reason=route_gray_reason,
        soft_warnings=soft_warnings,
    )
    if not trigger.needed:
        return VerifyResult(
            aligned=None,
            reason=trigger.detail,
            source=SOURCE_SKIPPED,
            trigger=trigger.trigger,
        )

    evidence = _evidence_text(docs, tool_result)
    if model is None and not config.USE_REAL_LLM:
        return VerifyResult(
            reason="离线模式跳过校验",
            source=SOURCE_DEGRADED,
            degraded=True,
            trigger=trigger.trigger,
        )

    try:
        chat = model or default_model()
        prompt = render_prompt("verify_evidence", user_query=query, evidence=evidence)
        with span("verifier") as s:
            raw = content_of(
                call_with_timeout(lambda: chat.invoke(prompt), config.VERIFIER_TIMEOUT_MS)
            )
            s.attrs["raw_chars"] = len(raw)
            # 触发原因跟着 span 走：只看 source 的话，"跳过"是一个无差别的大桶，
            # 分不清是"证据没问题"还是"判据写漏了"，也就没法回头收紧判据。
            s.attrs["trigger"] = trigger.trigger
        verdict = _parse_verdict(raw)
        if verdict is None:
            logger.warning("证据校验输出不可解析，按对齐放行：%r", raw[:120])
            return VerifyResult(
                reason="校验输出不可解析（已放行）",
                source=SOURCE_DEGRADED,
                degraded=True,
                trigger=trigger.trigger,
            )
        return VerifyResult(
            aligned=verdict,
            reason=_reason_line(raw),
            source=SOURCE_MODEL,
            trigger=trigger.trigger,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("证据校验失败，按对齐放行：%s", exc)
        return VerifyResult(
            reason=f"校验失败已放行：{type(exc).__name__}",
            source=SOURCE_DEGRADED,
            degraded=True,
            trigger=trigger.trigger,
        )
