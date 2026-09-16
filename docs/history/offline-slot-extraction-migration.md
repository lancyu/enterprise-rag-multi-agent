# 离线槽位抽取：历史动机、现存代价与迁移方案

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> 状态：**地基已落地并验证**（`bind_tools` 委托 + 工具 JSON Schema），
> **真正删除规则抽取待拍板**（见 §五）。
> 关联：`docs/history/tool-calling-enablement-plan.md`（前置能力）、
> `docs/history/intent-routing-hardening-plan.md`（P0-1 抽取层修复）
> 验证脚本：`artifacts/diag_tool_schema.py`（零配额、不发请求）

---

## 一、为什么要写「离线规则抽取」

这不是一个拍脑袋的决定，是四层约束叠出来的结果。理解它，才能判断今天
该删哪一部分、该留哪一部分——**四层里只有一层真的失效了**。

### 1.1 物理层：当年的 function calling 根本调不通

项目的模型调用统一经过自造的限流重试包装器 `_RateLimitRetryModel`
（`app/providers/llm.py:213-320`）。它只实现了 `_generate` / `_stream`，
**没有委托 `bind_tools`**，于是落到 `BaseChatModel` 的默认实现：

```python
get_chat_model().bind_tools([query_user_info])   # → NotImplementedError
```

底层 `ChatOpenAI` 是支持的——能力被自己加的壳挡住了。
**在调用不通的年代，"在调用点写规则"不是一个选择，而是唯一能走的路。**

配套的还有配额：免费档账号 RPM=3，一次对话本身就要 1~2 次调用。
在那个预算下，"每次工具查询再花一次调用去抽参数"是付不起的。

### 1.2 设计层：路由必须零成本，这是项目的核心卖点

`intent_recognize` 的设计是「规则优先 + 模型兜底」，规则层拦截了约 80% 的输入
（`app/graph/nodes.py:19`）。这条设计的价值主张很明确：**大部分请求在路由环节
一次模型都不调**。

而"这句话要不要走工具通道"正是路由的职责。它一旦改成模型调用，就意味着
**每一次请求**（包括那 80% 的知识问答）都要先付一次模型调用——这与项目的
核心卖点直接相反。所以路由判定必须是规则，这是留的。

### 1.3 传导层：判定零成本 → 抽取被顺带写成规则

这是最容易被忽略的一步滑坡。看 `app/core/query_signals.py:62-77`：
为了判定「四月在哪个部门」像不像在问某人的属性，正则已经把这句话
**逐段解析过一遍**了——查询动词、对象、属性词、句尾疑问词。
既然对象那一段已经在手里，**顺手捕获出来是零边际成本的**。

于是"抽取"跟着"判定"一起变成了正则。这不是谁偷懒，而是**同一句话
在同一处被解析两遍是浪费**这个直觉的必然结果。

### 1.4 补丁层：抽取失败之后，只剩"自己造一个抽取器"这一条路

当 `extract_identifier` 抽不出表外人名（「四月」「张伟」）时，P0-1 的修法是
**再加一级句式槽位**（`extract_employee_slot`）。方向没错——判定与抽取同源
确实是好的——但它把"抽取"这件事更牢地焊在了规则层：
每多一种说法，就要多一条正则；正则之间还会互相牵制
（`_HOWTO_MARK` 里加一个词，可能吞掉一整类正常问句，见 `query_signals.py:19-35`）。

---

## 二、离线抽取现在长什么样

三个抽取点，散在两层、三种风格：

| # | 位置 | 抽取器 | 抽什么 | 风格 |
|---|---|---|---|---|
| 1 | `app/core/query_signals.py:203-207` | `find_ticket_ids` | 工单号 `T\d{3,}` | 格式正则 |
| 2 | `app/tools/user_tool.py:53-88` | `extract_identifier` | 人名 / 拼音账号 | 别名表 → 拉丁正则 → 句式槽位（三级） |
| 3 | `app/graph/nodes.py:196` | **无抽取器** | 制度查询 | 整句直接传给工具 |

三者的**耦合方式也不同**：`find_ticket_ids` 被路由与工具共用（好），
`extract_identifier` 只被工具用而判定另有一套（坏，P0-1 修过），
制度查询干脆不抽（中性）。

**关键观察：三者的"抽取难度性质"其实完全不同**——这决定了该不该删：

- **工单号是「格式」**：`T` + 数字，值域封闭、可验证。正则对格式是
  100% 精确、零成本、零幻觉的。**换模型反而更差**（模型可能编造单号）。
- **人名是「语义」**：中文姓名无边界（「九月」是月份还是人名？「四月」是
  项目代号还是员工？），枚举永远不全。**这才是该交给模型的**。
- **制度主题是「子串匹配」**：工具内部靠 `k in lowered` 找条目，传原文最稳，
  多一层改写就多一次走样。**不需要抽，需要的是"选不选这个工具"的判断**。

---

## 三、代价：为什么现在要动它

### 3.1 同一语义多处定义——本项目已经犯了四次

| 次 | 语义 | 两处定义 |
|---|---|---|
| 1 | 文档索引键 | 两路召回各自算 key，RRF 融合时把不同片段当成一条去重 |
| 2 | metadata 字段 | 写入侧与读取侧字段名不一致 |
| 3 | 人名抽取 | 判定用句式正则、抽取用 5 条别名表（P0-1） |
| 4 | 证据强度底线 | R2 新增常量 vs `retriever.py` 里已有的一道死闸门 |

抽取散在两层三风格，是第 3 次的同类土壤：**任何一处改了，另一处不会跟着改**。

### 3.2 覆盖率 = 枚举长度

`extract_identifier` 第 1、2 级是闭集（5 条别名表 / 拉丁正则），
第 3 级才由句式槽位兜底。而第 3 级本身也是正则，只是把墙往后挪了一格：
换一种句式（「四月的部门是？」「我想知道四月的部门」），仍可能落空。

### 3.3 维护成本与语料漂移

每加一种问法 = 改正则 + 加回归用例 + 评估是否误伤既有分支。
P0-1 的文档里已经记过：为修 A 而收紧的正则，把 B 一起挡在了外面。

---

## 四、迁移方案

### 4.1 已落地（本轮完成，零行为变化）

**① 打开 function calling 的物理通道**
`_RateLimitRetryModel.bind_tools`（`app/providers/llm.py:268-279`）+
`_RetryBoundTools`（`app/providers/llm.py:189-210`）。

关键约束：工具绑定路径**必须复用**同一套限流重试，否则"限流怎么办"就有了
两套实现（第一次数教训）。做法是把 `_generate` 的退避循环抽成
`_call_with_rate_limit_retry`（`app/providers/llm.py:237-261`），两条路径共用。

实测（`artifacts/diag_tool_schema.py`）：
- `bind_tools` 不再抛 `NotImplementedError`，返回 `_RetryBoundTools`；
- 假 bound 连抛两次 429 → 调用 3 次后成功，**证明重试真的走了同源实现**。

**② 把参数契约写成显式 JSON Schema**

| 工具 | schema 类 | 补了什么 |
|---|---|---|
| `query_user_info` | `UserInfoArgs`（`app/tools/user_tool.py:26-35`） | 值形态、必须来自原话、正例 |
| `query_enterprise_ticket` | `TicketArgs`（`app/tools/ticket_tool.py:18-28`） | `pattern` 格式约束、禁止编造编号 |
| `query_enterprise_rule` | `RuleQueryArgs`（`app/tools/rule_tool.py:15-23`） | 可用主题清单（供工具选择） |

写 schema 时实测抓到两个**会静默劣化抽参质量**的坑：

1. **pydantic 类 docstring 会被渲染进 JSON Schema 的顶层 `description`**，
   并被 LangChain 原样发给模型。也就是说，写"为什么要这样设计"的开发注释，
   会让模型读到一段与任务无关的中文说明。**三个 Args 类因此都不写类 docstring**，
   说明改用 `#` 注释（`artifacts/diag_tool_schema.py` 里有断言钉住）。
2. **`convert_to_openai_tool` 会丢掉 pydantic 的 `examples` 字段**。
   所以示例必须写进 `description` 文本，写进 `examples=` 是无效的。

### 4.2 待落地（真正的删改）

```
tool_invoke_node（app/graph/nodes.py:158-212）
  现状：
    1. find_ticket_ids(query)              → 正则抽工单号
    2. looks_like_employee_query(query)     → extract_identifier(query)
    3. query_enterprise_rule.invoke(query)  → 整句
  目标：
    1. 候选工具由路由能力给出（employee_lookup → query_user_info，…）
    2. model.bind_tools(候选).invoke([HumanMessage(query)]) → tool_calls
    3. 有 tool_call → 执行；无 → 见 §五 决策①
    4. 制度查询保持整句直传（本就不需要抽取）
  删除：
    · extract_identifier + _ALIAS_RE + _LATIN_RE（user_tool.py:53-88）
    · extract_employee_slot 的槽位捕获部分（query_signals.py:126-166 里的 _SLOT_PATTERNS 等）
    · looks_like_employee_query 改回直接使用 _EMPLOYEE_QUERY_PATTERN（判定层保留）
```

### 4.3 一条改不掉的物理边界

上一轮实测（`artifacts/diag_function_calling2.py`）：

| 问句 | 模型输出 |
|---|---|
| 四月在哪个部门 | `query_user_info({"user_identifier": "四月"})` ✅ |
| 我的分机号是多少 | **无 tool_calls**，反问"请提供员工工号/姓名/邮箱前缀" |

第二行说明：**值不在句子里时，换 function calling 一样抽不出来**。
模型会**主动澄清**，这比现在硬编码的 `_fallback_to_knowledge` 更正确——
但它是**行为变更**，正是 §五 决策① 的内容。

---

## 五、必须拍板的三个决策点

### 决策① Mock / 无配额时，抽不出参数怎么办？

这是**最关键**的一处。`MockChatModel` 同样没有 `bind_tools`，
所以 CI 与无 Key 环境下模型抽参必然不可用。

| 选项 | 行为 | 代价 |
|---|---|---|
| **A（推荐）** 澄清出口 | 反问「您想查询哪位员工？」 | 用户可见行为变更；需重写约 10 条测试 |
| B 回退知识检索 | 保持现有 `_fallback_to_knowledge` | **功能倒退**：回到 P0-1 之前的「判对了却答不对」 |
| C 规则降级路径 | 保留规则抽取作为兜底 | 没真正删掉，只是变成了第二条路径 |

**推荐 A**：它诚实，且不需要任何规则——"我抽不出"就应该说"我抽不出"，
而不是拿一段无关的制度说明冒充答案。B 会把 P0-1 修好的东西退回去。

### 决策② 工单号正则 `find_ticket_ids` 删不删？

**建议不删。** 理由是 §二 的性质区分：工单号是**格式**不是语义，
正则对格式是精确解；而且路由判定 `looks_like_ticket_query` 依赖同一个正则，
删掉会连带拆掉路由。给它的 JSON Schema 加 `pattern` 已经足够。

### 决策③ 路由判定 `looks_like_employee_query` / `looks_like_ticket_query` 删不删？

**建议不删。** 见 §1.2：它们是"要不要花钱调模型"的闸门，是项目零成本路由的
地基。删掉它们 = 每次请求都先调一次模型判意图，与核心设计相反。
**要删的是"抽取"，不是"判定"。**

---

## 六、验收方式（实施后）

1. `bind_tools` 不再抛 `NotImplementedError` —— 已通过（§4.1）。
2. **反向验证**：把 `_RetryBoundTools.invoke` 里的 `_call_with_rate_limit_retry`
   换成直接调用 `self.bound.invoke(...)`，§4.1 的 429 场景必须**立即变红**。
3. 抽参路径的测试用**假模型**（返回固定 `tool_calls`）覆盖，不打配额。
4. 既有 369 条测试全绿；ruff 全绿；死代码扫描通过。
5. **限流语义未分叉**：全仓 `_is_rate_limit` 仍只有一处定义、两处调用。
