# 意图路由加固方案：判对了意图，却答不对问题

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> **一句话结论**：意图路由**判对了**（`employee_lookup`），问题出在**执行层**——
> 参数抽取失败后 fail-open 回退知识检索，把一次**确定性事实查询**降级成
> **概率性文档检索**，于是用户拿到的是一段与之无关的制度说明。
>
> **本文是待确认方案，尚未改动任何代码。** 文末 §7 有三个需要你拍板的决策点。

---

## 一、现象复盘：这句话到底走了哪条路

### 1.1 实测（`enterprise_workflow.invoke` 离线跑真实链路）

问句 `四月在哪个部门`：

```
intent_type        = knowledge          ← 被改写了
intent_capability  = employee_lookup    ← 但能力名还是它，自相矛盾
intent_source      = rule+tool_fallback
tool_result        = None
retrieve_docs      = 5 条
answer             = 知识库中没有相关信息。建议：您可通过以下方式查询：
                     - 联系人力资源部咨询：zhaoliu@company.com，分机 8009[2]
                     - 通过OA系统查找内部通讯录
                     - 询问该员工的直属主管或同事
```

对照问句 `张三在哪个部门`（同一链路、同一代码）：

```
intent_type        = tool
tool_result        = 姓名：张三 | 部门：技术部 | 职位：高级工程师 | 邮箱：… | 分机：8021
retrieve_docs      = 0 条
answer             = 张三在技术部[1]。 信息来源：业务查询结果。
```

**同一个句式，只因为名字在不在那张表里，走了两条完全不同的路。**

### 1.2 逐层拆解

| 层 | 代码位置 | 对 `四月在哪个部门` 的行为 | 评价 |
|---|---|---|---|
| L1 意图路由 | `app/core/intent_catalog.py:150-268`（`employee_lookup` 是其中一条声明） | 命中 `detector=employee` → `employee_lookup` / tool 通道，置信度 0.90 | ✅ **判对了** |
| L2 工具节点 | `app/graph/nodes.py:158-212`（`tool_invoke_node`） | `looks_like_employee_query(q)` → `True`，进入员工查询分支 | ✅ 判对了 |
| L3 参数抽取 | `app/tools/user_tool.py:25-60`（`extract_identifier`） | `extract_identifier("四月在哪个部门")` → `''`（空） | ❌ **断在这里** |
| L4 制度查询 | 同上 | `query_enterprise_rule` 未命中 → `"未找到…"` | 正常 |
| L5 兜底 | `app/graph/nodes.py:215-248`（`_fallback_to_knowledge`） | **回退知识检索** | ❌ **错在这里** |
| L6 生成 | L4 生成节点 | 拿着 5 条无关片段作答 | 结果错 |

关键几行（`extract_identifier` 的末段）：

```python
m = _ALIAS_RE.search(text)      # 只认 _ALIASES 里的 5 个姓名
if m: return m.group(0)
m = _LATIN_RE.search(text)      # 或拉丁账号
if m: return m.group(0)
return ""                       # ← 「四月」「张伟」「李明」都落到这里
```

`_ALIASES` 只有 5 条：`张三/李四/王五/赵六/孙七`（`app/tools/user_tool.py:17`）。
再多测两个名字即可复现同一现象：

```
extract_identifier("四月在哪个部门") = ''
extract_identifier("张伟在哪个部门") = ''      ← 同样回退检索
extract_identifier("张三在哪个部门") = '张三'   ← 正常
```

而工具本身**完全有能力给出正确答案**——直接把名字喂给它：

```
query_user_info.invoke("四月") → "未查询到 四月 的员工信息，请核对账号后重试。"
```

**工具是对的，链路是不通的。**

### 1.3 三个彼此独立的缺陷

**D1 · 实体抽取是闭集。**
`_ALIASES` 必须与员工库严格同步，任何漂移都**静默失效**。它的 docstring 写着
「新增员工只需维护上面的 `_ALIASES`」——这句话本身就是问题：**加一个员工要改代码**，
与设计原则「新增能力只加一条声明」直接冲突。而且它连"表外人名"这种最普通的输入都接不住。

**D2 · 失败出口不区分语义。**
`nodes.py` 里「句式命中但抽不出标识符」与「根本不是员工查询」**共用一个出口**
（`_fallback_to_knowledge`）。但这两件事的正确答案完全不同：

- 前者：用户已经明确说清了「我要查 X 的部门」，缺的只是系统不认识那个 X → **应当澄清或明确答"查无此人"**
- 后者：这压根不是工具能回答的问题 → **应当检索**

把两者合并，等于让"我不认识这个名字"冒充"这是一道知识题"。

**D3 · 状态改写掩盖了事实。**
回退时把 `intent_type` 从 `tool` 改写成 `knowledge`（`app/graph/nodes.py:230`），
但 `intent_capability` 仍是 `employee_lookup`。后果有两层：

- **观测口径分叉**：这条请求同时计入「工具意图」和「知识检索」，指标失去意义
- **用户被误导**：答案里那句"联系人力资源部咨询"并不是真在回答，而是因为检索**恰好**
  命中了 HR 文档。它**看起来像"系统查过了"**，实际什么也没查到——这比直接说"没查到"更危险

---

## 二、「回落知识检索」这个行为的得与失

### 2.1 它救过场（当初为什么这么写）

`nodes.py:158-169` 的注释记录了这段历史：旧规则里「邮箱」裸词把
「怎么申请邮箱扩容」误判成工具调用，工具又抽不出参数，于是回一句
「未识别到有效业务参数，无法查询」——用户看到的东西和拒答没差别，
**而这本该是一次知识检索就能回答的问题**。

所以 fail-open 的出发点是对的：**宁可多检索一次，也不要掐断用户**。
它是一个正确的默认，只是被用在了错误的边界上。

### 2.2 代价（五点，全部实测）

| # | 代价 | 具体表现 |
|---|---|---|
| 1 | **答非所问** | 员工属性是确定性事实，知识库没有员工名录，检索必然空转 |
| 2 | **用户得不到确定结论** | 既没查到，也没说"查无此人"，只给了一段制度说明 + 建议 |
| 3 | **看起来像查过了**（最危险） | 「联系人力资源部」是检索碰巧命中的 HR 文档，**是巧合不是答案**，用户会误以为系统真的执行过查询 |
| 4 | **成本** | 一次无谓的 embedding + 检索 + L4 生成（真实模型下是 LLM 调用） |
| 5 | **缺陷永不暴露** | 日志里这只是一次普通的 knowledge 路由，唯一线索是 `rule+tool_fallback`。没有告警、没有指标，不上报就永远没人知道 |

### 2.3 边界在哪

判断标准不是「检索有没有用」，而是**「这次查询有没有确定性答案源」**：

| 情形 | 有确定性答案源？ | 正确动作 |
|---|---|---|
| `张三在哪个部门` | ✅ 员工库，且能定位到人 | 查库返回 |
| `四月在哪个部门` | ✅ 员工库（虽然查无此人） | **查库 → 明确回"未查询到"** |
| `怎么申请邮箱扩容` | ❌ 只能靠制度文档 | 检索 |
| `哪个部门负责报销` | ❌ 只能靠制度文档 | 检索 |

**第 2 行就是本案**：它有一个确定性答案源，而系统却去检索了。

---

## 三、外部参照：`ai-career-copilot` 的意图路由

仓库：<https://github.com/Programmergyt/ai-career-copilot>（已浅克隆精读，61 个 Python 文件）

### 3.1 它是怎么实现的

```
POST /api/chat
  └─ workflow/graph.py  LangGraph 单图，9 个节点，入口固定 planner
       └─ agents/planner.py:32-43  ① 每轮都调一次 LLM 做意图分类
            └─ prompts/intent_classification.py  7 个闭集意图 + 当前会话状态
       └─ agents/planner.py:46-55  ② intent → 节点链，查 _INTENT_PLAN 表
       └─ workflow/graph.py:28-33  ③ _route_after_planner 取 plan[0] 决定下一个节点
       └─ 后续节点各自 _route_after_xxx：`if "content_agent" in plan: ...`
```

核心数据结构（`agents/planner.py:21-29`）：

```python
_INTENT_PLAN = {
    "upload_jd":      ["jd_agent", "gap_agent", "content_agent", "render_agent", "interview_agent"],
    "upload_profile": ["profile_agent", "content_agent", "render_agent", "interview_agent"],
    "gap_analysis":   ["gap_agent"],
    "content_edit":   ["content_agent", "render_agent"],
    "render_edit":    ["render_agent"],
    "export":         [],
    "ask_question":   ["question_agent"],
}
```

### 3.2 值得借鉴的四点

**① 把「当前会话状态」显式注入分类提示词** —— 这是它最有价值的做法。
`prompts/intent_classification.py:14-17` 把 `has_job / has_profile / has_resume`
一起喂给模型。**同一句话在不同状态下意图不同**：简历还没生成时问「导出」，
和处理到一半时问「导出」，正确的后续动作并不一样。

> 我们的路由是**纯 query 的**（只看这一句话）。这带来可缓存、可离线预演的好处，
> 但也确实丢失了状态维度。见 §4.5 P2-1 的取舍分析。

**② 失败时兜底到一个「能直接回答」的节点**，而不是硬猜。
`planner.py:63-79` 在分类彻底失败时把 `current_intent` 落到 `ask_question`，
由 `question_agent` 基于 state 组织回答。**兜底目标是"给用户一个答复"，
而不是"给用户一个分支"。** 这个取向是对的。

**③ 意图分类的输出带 `reason` 字段**（`json_contracts.py:10`）并写进
`workflow_trace`（`planner.py:99-103`）。**可解释性内建于数据结构**，
排查时不用去翻文本日志。

**④ 执行计划是「有序节点列表」而不是散落的 if-else**
（我们已在 `app/core/intent_router.py` 做得更彻底——通道注册表 + 目录驱动）。

### 3.3 不足（六条，其中第 3 条是**实证**的）

**① 每轮必调一次大模型。**
`planner.py:64` 无条件调用 `_classify_intent_async`。默认路径零 LLM 调用不成立。
在 RPM 只有个位数的账号下，等于每问一句吃掉一小半配额。

**② 意图是硬编码闭集，加一种要改四处。**
提示词（`prompts/intent_classification.py`）、`_INTENT_PLAN`（`planner.py:21`）、
条件边映射（`graph.py:158-167`）、以及节点注册（`graph.py:144-152`）——
四处都要改，漏一处不会报错。

**③ `intent` 字段没有枚举校验 → 未知意图静默失败。**
`json_contracts.py:8-10` 是：

```python
class IntentClassificationOutput(BaseModel):
    intent: str = "ask_question"     # ← 纯 str，无 Literal / Enum
    reason: str = ""
```

**实测：以下全部被 Pydantic 接受——**

```
'question'        -> 接受, intent='question'
'ASK_QUESTION'    -> 接受, intent='ASK_QUESTION'
'查询部门'         -> 接受, intent='查询部门'
'ask_question '   -> 接受, intent='ask_question '
```

（大小写、中文、**尾随空格**——最后这个尤其容易由模型输出产生。）

而下游是 `_INTENT_PLAN.get(intent, [])`：**取不到就返回空列表**
→ `_route_after_planner` 看到空计划 → 走 `respond`
→ 用户收到的是 `_build_trace_reply` 生成的**「已完成本轮处理 + 执行过程」模板文本**，
**不是答案，也不报错**。（该链路为静态确认；只有 Pydantic 接受性是实跑的。）

> 这正是用户在意的那个现象的**另一种形态**：助手没有回答，而是给了别的东西。

**④ 没有置信度、没有中间态。**
输出只有 `{intent, reason}`，没有 confidence。拿不准时只能二选一（硬猜或兜底），
**没有「两个都像，我不确定」这个合法的中间态**。

**⑤ 没有预演接口。**
排查误判时只有一行 `Intent classified: %s (reason: %s)` 日志，
看不到第二名是谁、差多少分——而 `reason` 还是自然语言，无法聚合统计。

**⑥ 失败兜底会**谎报**意图。**
`planner.py:67-79` 分类失败时把 `current_intent` 写成 `"ask_question"`，
而 `system_design.md:225` 的建表语句里 `intent VARCHAR(32) NOT NULL` 会把它持久化。
**数据库里存的是"用户想提问"，真实情况是"分类器挂了"。**
观测数据从此不可信——与我们项目的 D3 是同一类病（写状态时图省事，丢掉事实）。

另外它的 `docs/agent_architecture_optimization_report.md` §2.1–2.2 自己也承认
「Planner 职责过重」「intent→固定链路映射不够灵活」——但给出的解法是
拆 Plan 对象 / Phase 1-3 大重构，**没有一条针对上面这些"静默失败"的洞**。

### 3.4 对照表

| 维度 | ai-career-copilot | 本项目 | 判定 |
|---|---|---|---|
| 意图建模 | 闭集枚举硬编码 | **目录即数据**（`intent_catalog.py`） | 本项目的更好 |
| 加一种意图的成本 | 改 4 处 | 加 1 条声明 | 本项目更好 |
| 默认路径 LLM 调用 | **每轮 1 次** | 0 次（确定性 + 词面短路，`llm_free_rate` 有出口） | 本项目更好 |
| 会话状态参与路由 | ✅ 注入 `has_job/has_profile/has_resume` | ❌ 纯 query | **它更好，值得借鉴** |
| 未知意图的处理 | ❌ 静默落空计划 | ✅ `_parse_llm_intent` 白名单校验，解析不出返回 `None` 并保守兜底 | 本项目更好 |
| 拿不准时 | 只能二选一 | 灰区 + 边际门控（绝对地板 + top1−top2） | 本项目更好 |
| 失败时是否谎报状态 | ❌ 写成 `ask_question` | ⚠️ D3：改写成 `knowledge` | **两边都有病，本项目轻一些** |
| 可观测 | `reason` 文本 | `preview_intent` 全候选明细 + 分层来源 | 本项目更好 |
| 兜底语义 | 能直接回答的节点 | **能确定回答的查询 → 却去检索** | **本项目更差（即本案）** |

**综合判断**：本项目在路由**架构**上领先，ACC 唯一实质领先的是"状态参与路由"。
但 ACC 的三类静默失败（未知意图、谎报意图、无置信度）恰好是**我们已经在做对**的部分——
真正的教训是**它把"拿不准"和"失败"这两件事都处理成了"假装正常"**。
而我们的 D1–D3 是同一族病的另一个变种：**把"抽取失败"处理成了"换个分支"**。

---

## 四、修改方案

### 4.0 总览

| 档 | 项 | 一句话 | 触及文件 | 破坏性 |
|---|---|---|---|---|
| **P0** | P0-1 | 抽取：从「闭集别名」改为「句式槽位」 | `query_signals.py` / `user_tool.py` | 低（只增能力） |
| **P0** | P0-2 | 出口分离：抽不出对象 → **澄清**，不回退检索 | `graph/nodes.py` | **中（行为变更，见 §7 决策点）** |
| **P0** | P0-3 | 观测：不再改写 `intent_type`；区分 `tool_clarify` / `tool_fallback` | `graph/nodes.py` | 低 |
| **P1** | P1-1 | 抽取能力与员工库的一致性自检 | `tools/user_tool.py` + 启动自检 | 低 |
| **P1** | P1-2 | 「查无此人」与「未接入人事库」的话术分级 | `app/config.py` + `tools/user_tool.py` | 低 |
| **P1** | P1-3 | 回归测试 + 反向验证 | `tests/test_biz_correctness.py` 等 | 无 |
| **P2** | P2-1 | （可选）会话状态参与路由 | 待定 | 待评估 |
| **P2** | P2-2 | （可选）`employee_lookup` 语义锚点补真实姓名 | `intent_catalog.py` | 无 |

**实施状态（2026-09-14 22:58）**

| 项 | 状态 | 说明 |
|---|---|---|
| P0-1 抽取同源 | ✅ **已实施** | 走 **§7 决策点① 的 A 方案**（判定派生自抽取）。见下 |
| P0-2 出口分离 | ⏸ **待拍板** | 阻塞于 §7 决策点②。属用户可见行为变更 |
| P0-3 观测修正 | ⏸ **待拍板** | 阻塞于 §7 决策点③。会改变对外 `intent_type` 契约 |
| P1-1 / P1-2 | ⏸ 未开始 | P1-2 依赖决策点③ |
| P1-3 测试 | 🚧 部分 | P0-1 的 32 条已落（`tests/test_employee_slot.py`） |

**P0-1 实际落点**（与本节原设计一致，未新增第三份关键词表）：

- `app/core/query_signals.py` 新增 `extract_employee_slot(query) -> Optional[str]`，
  **三态返回值**：`None` = 不是员工查询 / `''` = 是但缺对象 / `"四月"` = 有对象。
- **判定改为其派生**：`looks_like_employee_query(query) = extract_employee_slot(query) is not None`。
  判定语义**逐条保持不变**（既有 9 条断言一行未改即通过）——因为
  `extract_employee_slot` 返回 `None` 当且仅当 `_EMPLOYEE_QUERY_PATTERN` 不命中。
- `app/tools/user_tool.py` 的 `extract_identifier` 增加第 3 级（别名 → 拉丁账号 → 句式槽位），
  返回类型不变，调用方无需改动。
- **实测效果**：`四月在哪个部门` / `张伟在哪个部门` / `王小明在哪个部门` 从
  「回退检索、拿 5 条无关片段作答」变为「走工具通道、输出『未查询到 X 的员工信息』」，
  `intent_type` 保持 `tool`，回退检索条数 **0**。

**仍未解决**：`谁的邮箱` / `我的邮箱是多少` 仍是「判定通过 → 抽取为空 → 回退检索」，
因为这两句**句子里根本没有对象**（第 ③ 类槽位），修它们属于 P0-2 的范围。

### 4.1 P0-1 · 抽取：从「闭集别名」到「句式槽位」

**问题重述**：判定层（`looks_like_employee_query`）与抽取层（`extract_identifier`）
是**两套独立规则**，各自维护。判定说"这是员工查询"，抽取却说"我认不出这个人"——
两条规则之间没有任何约束，这就是分叉的土壤。

> 这已经是本项目第三次遇到同一族病：**同一语义在两处各定义一遍**
> （第一次是索引键、第二次是 metadata 字段、这次是人名抽取）。
> 所以修法也要遵循同一个原则：**先收敛成单一事实来源，再谈准确性。**

**做法**：把 `app/core/query_signals.py:62-73` 的 4 个句式分支
从「整句匹配」升级为「**带捕获组的槽位匹配**」，并暴露一个新函数：

```python
def extract_employee_slot(query: str) -> Optional[str]:
    """返回句式里被索取的属性主体。

    - None  → 这不是员工属性取值查询（"怎么申请邮箱扩容"）
    - ''    → 是员工查询，但没指明具体对象（"谁的邮箱"、"我的邮箱"）
    - "四月" → 是员工查询，且给出了对象
    """
```

配套把判定函数收敛为它的派生：

```python
def looks_like_employee_query(query: str) -> bool:
    return extract_employee_slot(query) is not None
```

**收益**：判定与抽取**不可能再漂移**——它们已经是同一段正则。这是本项最重要的部分，
比"能抽出四月"更重要。

**槽位里怎么过滤噪声**：槽位捕获后仍需一道黑名单，否则「查一下**哪个**部门的流程」
会把「哪个」当人名。黑名单只需覆盖三类词：疑问代词（哪个/什么/谁/哪一位）、
属性词（部门/邮箱/工号…）、how-to 词。**这些清单都已存在于 `query_signals.py`**，
直接复用，不新增第三份。

**`user_tool.extract_identifier` 的改动**：保持它是**唯一的对抽取入口**，
内部改为三级（顺序即优先级）：

```python
1. _ALIAS_RE   —— 员工库已知别名（工具的领域知识，命中最准）
2. _LATIN_RE   —— 拉丁账号
3. extract_employee_slot(text) —— 表外人名（句式知识）
```
失败仍返回 `""`，**返回类型不变**，调用方无需改。

> 抽取准确率天然达不到 100%（中文人名与产品名/项目名可能同形：「四月」也可能是个项目代号）。
> 但这**不影响本方案成立**——因为 P0-2 把"抽取失败"的后果从"去检索"改成了"明确澄清"，
> 即使抽错，用户看到的也只是一个可以立刻纠正的追问，而不是一段无关的制度说明。

### 4.2 P0-2 · 出口分离：澄清 ≠ 检索

`app/graph/nodes.py:158-212`（`tool_invoke_node`）目前是：

```python
if looks_like_employee_query(query):
    identifier = extract_identifier(query)
    if identifier:
        new_state["tool_result"] = query_user_info.invoke(identifier)
        return new_state
# ← 抽不出就继续往下走，最后落到 _fallback_to_knowledge
```

改为**三分支**：

```python
if looks_like_employee_query(query):
    identifier = extract_identifier(query)
    if identifier:
        → 查库返回（不变）
    else:
        → 澄清回复，直接返回（新增，不再往下走）
```

澄清话术分两种（由 §4.4 P1-2 的配置开关决定），**核心是让用户能立刻纠正**：

- 已接入人事库：`「没能识别出您要查询的同事。请提供姓名或工号，例如"张三的部门"。」`
- 未接入人事库：`「员工信息查询功能尚未接入。您可以通过 OA 通讯录查询，或联系人力资源部。」`

**为什么不是"猜一个人名去查"**：猜错会返回另一个人的部门，比不返回更糟。

**为什么"谁的邮箱"这类无具体对象也应澄清而不是检索**：
它同样有确定性答案源（员工库），只是缺对象。检索制度文档永远答不了它。
这条属于**行为变更**，见 §7 决策点②。

**必须保住的对照组**：`怎么申请邮箱扩容` / `哪个部门负责报销` 这类
**根本没有进入员工分支**（被 `howto` guard 挡在 L1），它们的检索路径**一行都不动**。

### 4.3 P0-3 · 观测修正

| 字段 | 现状 | 改为 |
|---|---|---|
| `intent_type` | 回退时改写成 `knowledge` | **保持 `tool`**（忠实反映路由判了什么） |
| `intent_source` | 统一 `rule+tool_fallback` | `rule+tool_clarify`（澄清）/ `rule+tool_fallback`（真降级，仅制度查询未命中时） |
| `soft_warnings` | 无 | 澄清时追加 `"员工查询未识别到对象：<query>"`，进入既有观测通道 |

**为什么要区分**：`tool_clarify` 的比例就是**抽取质量的直接度量**。
没有这个数，P0-1 的效果无法验证，也不知道该往锚点里补哪些真实姓名（对应 §4.5 P2-2）。

### 4.4 P1 · 加固

**P1-1 一致性自检。**
`_ALIASES` 的**值域必须等于** `_MOCK_USERS` 的键集合（当前是 5=5）。
加一条模块级/启动期校验，不一致打 WARNING（严格模式拒绝启动）。
接入既有的 `_check_routing_config` 自检项。

> 注意：P0-1 之后别名表不再是唯一路径，漂移不再致命——但它仍是**最准的那一级**，
> 且这类"两份数据必须相等"的约束不写下来就一定会漂。成本几行，值得。

**P1-2 话术分级配置。**
新增 `EMPLOYEE_DIRECTORY_ENABLED`（默认 `false`）。
当前 `_MOCK_USERS` 只有 5 人，本质是**演示数据**。此时回「未查询到 四月 的员工信息」
是**不诚实的**——它暗示系统查过一个真实的人事库。默认 `false` 时应当直说
"功能未接入"，接入真实 HR/LDAP 后再置 `true`，届时话术自动切换为"未查询到 X"。

**P1-3 回归测试 + 反向验证** —— 见 §五。

### 4.5 P2 · 可选项（需你决定是否纳入本轮）

**P2-1 会话状态参与路由（借鉴 ACC 的亮点）。**
把 `has_*` 之类的状态维度注入路由决策。**倾向不做**，理由：
我们路由是纯 query 的——**可缓存、可离线预演、可复现**，这三条价值很高；
引入状态会让 `preview_intent` 失去复现性。若确有需要，建议走**后置修正**而非主路由：
例如「连续两次同一问句未识别出对象 → 直接给固定答复」，不动主链路。

**P2-2 `employee_lookup` 的 utterances 补真实姓名。**
`docs/history/intent-routing-redesign.md` §6.1 已提过：语义锚点是手写的，没校准。
补 20~30 条带真实姓名/工号形态的模板句是**零代码**动作，能直接降低灰区率。

### 4.6 明确不做

- **不删除 fail-open 机制**。它是对的，只是被用在了错误的边界（§2.3）。
  `_fallback_to_knowledge` 保留，继续服务"制度查询未命中"这一类。
- **不做中文 NER / 引入分词模型**。为一个确定性槽位引入模型依赖不划算，
  且会破坏"离线层可跑"这一约束。
- **不照搬 ACC 的 Planner 大重构**（Plan 对象 / DAG）。
  `docs/history/intent-routing-redesign.md` 已经是更合适的形态；ACC 自己那份重构方案
  也没解决它的静默失败问题。

---

## 五、验证计划

### 5.1 新增测试

| # | 用例 | 断言 | 类别 |
|---|---|---|---|
| 1 | `四月在哪个部门` | 走 **tool** 分支、**不触发检索**（`retrieve_docs` 为空）、回答含"未识别/未查询到" | 故障回归 |
| 2 | `张伟在哪个部门` | 同上（换一个表外人名，证明不是特判"四月"） | 故障回归 |
| 3 | `张三在哪个部门` | 仍返回 `技术部`，`tool_result` 非空 | **对照组** |
| 4 | `怎么申请邮箱扩容` | 仍走 knowledge、**仍触发检索** | **对照组（防改死 fail-open）** |
| 5 | `哪个部门负责报销` | 仍走 knowledge、**不触发工具** | **对照组** |
| 6 | `looks_like_employee_query is 派生自 extract_employee_slot` | 断言两者同源（源码级） | **同源护栏** |
| 7 | `extract_identifier` 三级顺序 | 别名命中不被人名槽位抢走 | 单元 |
| 8 | `intent_type` 在澄清路径上仍是 `tool` | 不被改写成 `knowledge` | 观测口径 |

### 5.2 反向验证（护栏不是空转）

改完必须证明测试**真的会红**，否则等于没加：

1. 把 `extract_identifier` 退回「只认别名表」→ 用例 1、2 必须**红**
2. 把澄清分支改回 `_fallback_to_knowledge` → 用例 1、2 必须**红**
3. 把 `looks_like_employee_query` 改回独立正则 → 用例 6 必须**红**

### 5.3 门禁（每次改动后全跑）

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/ -q
./.venv/bin/ruff check app scripts tests --no-cache --output-format concise
./.venv/bin/python scripts/verify_doc_linenos.py
```

改动会让文档行号漂移 → 用 `scripts/fix_doc_linenos.py --write` 回填（只写回校验器已算出的值）。
另需同步 `docs/project-introduction.md` §6.2 表格里的**自我指涉数字**（测试条数、声明条数）——
那类数字不被校验器覆盖，只能手工核。

---

## 六、风险与回滚

| 风险 | 可能性 | 缓解 |
|---|---|---|
| 澄清话术让用户觉得"系统变笨了"（以前至少会给一段制度） | 中 | 话术里给出**可执行的下一步**（提供姓名/工号、或明确说功能未接入），而不是干巴巴一句"没识别到" |
| 人名槽位误抽（把产品名/项目名当人名） | 中 | 后果已被降级为"一次可纠正的追问"；且别名表优先级最高，库内的人不会被误判 |
| `looks_like_employee_query` 语义变更波及既有 9 条断言 | 中 | §5.1 用例 6 守住同源；既有 `tests/test_biz_correctness.py:72-87` 全部保留不改，作为对照 |
| 澄清路径新增 `soft_warnings` 污染既有断言 | 低 | 先跑一遍全量测试确认 |

**回滚**：改动集中在 3 个文件、无数据迁移、无配置强依赖。
`git` 尚未初始化——**建议本次改动前先完成 P0-1（`git init` + 首次提交）**，
让回滚有据可依（这正是 `docs/history/production-readiness-review.md` 里仍挂着的 P0 项）。

---

## 七、待你确认的三个决策点

### ① 抽取层要不要动「单一事实来源」这一刀？

> **已按 A 实施（2026-09-14）**，但有一处与原设计不同，记录如下。
> 原设计说要「把 4 个句式分支升级为带捕获组」，即改动 `_EMP_QUERY_BODY` 本身，
> 「影响面稍大」。实施时发现**不必动它**：保留 `_EMPLOYEE_QUERY_PATTERN` 原样作为判定，
> 另建一组**独立的捕获组正则** `_SLOT_PATTERNS`（复用同一个 `_EMP_ATTR` 与 `_HOWTO_MARK`），
> 由 `extract_employee_slot` 依次尝试。这样：
> ① 判定语义**逐条不变**（既有 9 条断言一行未改即通过）；
> ② 同源由**代码结构**保证（`looks_like_employee_query` 的 `co_names` 里必须有
> `extract_employee_slot`，且有测试守住），而非靠两段正则长得像。
> **换句话说：A 的目标（不可漂移）达到了，A 的代价（动主干正则）绕开了。**

- **A（推荐）**：把判定与抽取收敛成同一段正则（`looks_like_employee_query`
  派生自 `extract_employee_slot`）。**彻底消除分叉土壤**，但要改
  `query_signals.py` 的 4 个句式分支，影响面稍大。
- **B（保守）**：只加一个独立的「表外人名抽取」函数，不动现有判定。
  改动更小，但判定与抽取**仍是两套规则**，下次还会漂。

### ② 抽不出对象时，行为怎么定？

- **A（推荐）**：**一律澄清**，不回退检索。含「谁的邮箱」「我的邮箱」这类
  无具体对象的句子。
- **B（更保守）**：只对"看起来给了名字但不在库里"的句子澄清；
  「谁的邮箱」这类保留现有回退检索行为。

> 差异只在少数边界句。A 更一致，B 更不容易让人感到"变笨"。

### ③ 「员工信息查询」的对外语义怎么定？

- **A（推荐）**：新增 `EMPLOYEE_DIRECTORY_ENABLED`（默认 `false`），
  未接入时明确回"功能未接入，请走 OA 通讯录"。**不假装查过。**
- **B**：不区分，统一回"未查询到 X 的员工信息"。
  改动更小，但在只有 5 条演示数据时**对用户不诚实**。

---

### 附：本文引用的所有代码坐标（2026-09-14 快照）

| 位置 | 内容 |
|---|---|
| `app/core/intent_catalog.py:150-268` | `INTENT_CATALOG`（`employee_lookup` 为其中一条声明） |
| `app/core/query_signals.py:62-73` | 员工查询的 4 个句式分支 |
| `app/core/query_signals.py:169-181` | `looks_like_employee_query` |
| `app/graph/nodes.py:158-212` | `tool_invoke_node` |
| `app/graph/nodes.py:215-248` | `_fallback_to_knowledge` |
| `app/tools/user_tool.py:17` | `_ALIASES`（5 条） |
| `app/tools/user_tool.py:25-60` | `extract_identifier` |
| `app/tools/user_tool.py:69-86` | `query_user_info` |
| `app/core/intent_router.py:469-484` | `_parse_llm_intent`（我们做对的那处枚举校验） |
| `tests/test_biz_correctness.py:72-96` | 既有的员工查询断言（本次不改） |
