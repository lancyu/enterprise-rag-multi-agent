# 多 Agent 协作架构：从「一个 Agent 一把抓」到「五个 Agent 各司其职」

> 触发问题：「把单一 Agent 架构改造为多 Agent 协作架构，包含五个分工明确的 Agent：
> **路由 / 闲聊 / 简单 RAG / 复杂 RAG / 工具**。路由 Agent 作为全局入口负责意图识别
> 与边界管控，把幻觉管控集中在入口层；闲聊 Agent 不访问知识库、不调用任何工具，
> 避免闲聊误触发 RAG 或 FunctionCall 而浪费 token。」
>
> 本文回答四件事：**整体架构设计**、**各 Agent 的输入输出**、**路由分发规则**、
> **重构方案**。
>
> 代码位置：`app/core/router_agent.py`（路由）、`app/core/sub_agents.py`（闲聊 +
> 两个 RAG）、`app/core/tool_agent.py`（工具）、`app/graph/{nodes,edges,workflow_graph,state}.py`
> （编排）。工具与 schema 见 `docs/tool-json-schema.md`。
> 回归用例：`tests/test_multi_agent.py`（76 条）。

---

## 一、为什么拆：单 Agent 的病症不是"慢"，而是**失败模式不可分**

改造前的拓扑只有三个节点：

```
memory_load ──► agent（一次 function calling 决策）──► generate_answer
```

一个 Agent 在一次决策里既决定"要不要检索"、又决定"要不要查业务数据"、还要决定
"要不要直接回答"。合并带来的问题不是性能，而是**同一种现象对应多种互不相干的
原因**：

| 用户看到的现象 | 可能是 | 日志里长得一样吗 |
|---|---|---|
| 回答"知识库中没有找到" | ① 模型判定是寒暄、决定不调工具 ② 该调工具但忘了调 ③ 调了工具但参数没给全被拒 ④ 真的检索不到 | **一模一样** |
| 回答"查不到这位员工" | ① 库挂了 ② 姓名确实不在库里 ③ 模型把姓名抽错了 | **一模一样** |

第一张表里 ②③④ 的处置方式完全不同（补提示词 / 向用户追问 / 诚实作答），但排障时
只能靠人品猜。这正是"分工"要解决的问题——不是把一次调用拆成五次，而是让**每一类
失败有归属**。

拆完之后：

| 现象 | 现在的归属 | 处理 |
|---|---|---|
| 寒暄 | 闲聊 Agent（模板直出） | 不可能失败 |
| 越界问题 | 路由 Agent 在入口拦下 | 不消耗任何下游预算 |
| 制度问题 | 简单 / 复杂 RAG Agent | 检索失败 → 降级为"没找到"（诚实） |
| 业务数据 | 工具 Agent | 执行失败 → 转人工（不能拿"知识库没找到"糊弄） |
| 参数没给全 | 工具 Agent 的直答出口 | 直接向用户追问 |

---

## 二、整体架构

### 2.1 三层结构

```
┌──────────────────────────────────────────────────────────────────────┐
│ 编排层  app/graph/  ——  9 节点 / 3 条件边的 LangGraph 有向图           │
│   只做"谁在什么时候被调用"，不含任何业务判断（业务判断都在 Agent 里）      │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
┌──────────────────────────────────────────────────────────────────────┐
│ Agent 层  app/core/  ——  五个 Agent + 两个确定性兜底                    │
│   router_agent   入口判定 + 边界管控（唯一入口，永不失败）               │
│   sub_agents     smalltalk（模板直出）/ simple_rag / complex_rag       │
│   tool_agent     3 个只读工具的 function calling、参数护栏、追问         │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
┌──────────────────────────────────────────────────────────────────────┐
│ 能力层  app/rag（检索 + L4 生成）/ app/db（SQLite 只读）/ app/tools     │
│   能力本身不含"什么时候用"的判断——由 Agent 层决定                        │
└──────────────────────────────────────────────────────────────────────┘
```

**分层的判据**是"这段逻辑属于谁的决定"：检索阈值属于能力，检索失败该不该转人工属于
Agent，走哪条链路属于编排。三者混在一起时，改一个阈值可能悄悄改变一条路由。

### 2.2 运行拓扑

```
                         用户提问
                            │
                            ▼
              ┌──────────────────────────┐
              │ 0. memory_load           │  身份解析（user_id → 工号）+ 长期记忆
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │ 1. router  路由 Agent     │  一次模型调用：意图识别 + 边界管控
              └────────────┬─────────────┘
                           │ scene（五选一，闭集）
        ┌──────────┬───────┴────────┬───────────┬─────────────┐
        ▼          ▼                ▼           ▼             ▼
   smalltalk   simple_rag      complex_rag     tool      out_of_scope
   闲聊·模板    单次检索        拆解→多次检索   function    常量话术
        │          │                │         calling        │
        │          │                │           │            │
        │          │                │      ┌────┴─────┐      │
        │          │                │      │ 决策失败  │──────┼──► human_fallback
        │          │                │      │ 反问用户  │──────┼──► END
        │          │                │      │ 不支持FC  │──┐   │
        │          │                │      │ 取到证据  │  │   │
        │          ▼                ▼      ▼           │  │   │
        │     ┌──────────────────────────────┐        │  │   │
        │     │ 3. generate_answer  L4 受控生成 │◄───────┘  │   │
        │     │    引用编号 · 置信度 · 拒答      │           │   │
        │     └───────────┬──────────────────┘           │   │
        │                 │ 生成失败                       │   │
        ▼                 ▼                              ▼   ▼
       END          human_fallback ◄─────────────────── simple_rag（改道）
                          │
                          ▼
                         END
```

节点清单（`app/graph/workflow_graph.py::NODE_NAMES` 是唯一来源）：`memory_load`、
`router`、`smalltalk`、`out_of_scope`、`simple_rag`、`complex_rag`、`tool`、
`generate_answer`、`human_fallback`。

### 2.3 两个编译产物

| 链路 | 编译产物 | 差别 |
|---|---|---|
| `/chat/ask`（非流式） | `enterprise_workflow`（9 节点） | 证据出口 → `generate_answer` |
| `/chat/ask/stream`（SSE） | `pre_generation_workflow`（8 节点） | 证据出口 → `END`，生成由端点逐 token 推 |

两者由**同一个装配函数**（`_wire(graph, *, generation_target)`）编译，唯一差别是
"证据出口通向哪"这一个参数。

> 这条约束是被一个真实缺陷逼出来的：旧实现在流式端点里重抄了整条前置链路，结果
> 工具抛异常时图会把答案换成「已转接人工」，流式链路却照样去调模型生成——同一次
> 提问在两条链路上给出不一致的回答。现在节点函数、条件边函数、分支映射全部是同一份
> 代码，**漂移在结构上不可能发生**（`test_full_and_pre_generation_graphs_share_every_node`）。

---

## 三、五个 Agent 的输入输出

### 3.0 统一产出形状

子 Agent 只负责**取回证据或给出直答**，不负责组织最终话术。

```python
# app/core/sub_agents.py
@dataclass
class AgentAnswer:
    text: Optional[str] = None          # 非 None → 直答出口，本轮不进 L4
    docs: List[Dict[str, Any]] = field(default_factory=list)      # 非空 → 证据出口
    tool_results: List[str] = field(default_factory=list)         # 业务工具的 JSON 信封
    sub_queries: List[str] = field(default_factory=list)          # 复杂 RAG 实际检索过的查询
    steps: List[Dict[str, Any]] = field(default_factory=list)     # 前端工作流面板
    soft_warnings: List[str] = field(default_factory=list)        # 可降级故障（只观测）
    degraded: bool = False
    error: Optional[str] = None         # 非空 → 本轮无法自动处理，调用方转人工
```

工具 Agent 用同形状的 `ToolDecision`（`direct_answer` / `used_tools` / `steps` /
`docs` / `tool_results` / `soft_warnings` / `degraded` / `error`）。**形状一致**是
刻意的：调用方（图节点）不必按 Agent 分类处理。

三个字段的语义刻意不重叠：

- `text=""` 与 `text=None` **不是一回事**。前者是"说了一句话但内容是空的"，后者是
  "没打算说话"。把两者混成一个真值判断，空回答会被当成正常直答。
- `error` 只在"这个 Agent 本身不可用"时才非空。判据与全项目一致：
  **能不能从其他来源得到答案**。

### 3.1 路由 Agent —— `app/core/router_agent.py`

| 项 | 内容 |
|---|---|
| **输入** | `query`（本轮原话）、`chat_history`（最近 4 轮，仅用于消解代词）、注入的 `model`（测试用） |
| **输出** | `RouteDecision(scene, reason, confidence, source, degraded, out_of_scope_answer, raw, error)` |
| **模型调用** | 1 次（纯文本分类，不绑工具） |
| **失败行为** | **永不抛异常**。空提问 / 离线 / 输出不可解析 / 调用异常 → `_fallback_route` |
| **关键约束** | 输出的 `scene` 必须落在 `SCENES` 闭集内，否则回落兜底（挡住模型编造的场景名） |

`memory_context` 参数接收但**不注入**提示词：路由只看"这句话本身要干什么"，
把长期记忆塞进来只会增加误判面。保留参数是为了与其它 Agent 的签名形状一致。

### 3.2 闲聊 Agent —— `app/core/sub_agents.py::run_smalltalk_agent`

| 项 | 内容 |
|---|---|
| **输入** | `query` |
| **输出** | `AgentAnswer(text=<模板>, steps=[{kind: "template", detail: <模板名>}])` |
| **模型调用** | **0 次** |
| **检索 / 工具** | **0 次** |
| **模板** | identity / greeting / thanks / bye / default 五张，正则优先级挑选 |

**为什么闲聊不调模型**（三个理由按重要性排序）：

1. **零幻觉**：模板里不出现任何业务事实，也就不可能编造事实。用一次模型调用去
   生成"你好呀，有什么可以帮你"是拿幻觉风险换措辞变化，不划算。
2. **离线一致**：未配 Key 时 `MockChatModel` 对闲聊会返回一段与 Context 有关的
   文本——用户会看到「你好」被回以「未在知识库中检索到与您问题相关的内容」。
   荒唐，而且难以排查（离线链路看起来"正常运行"）。
3. **可枚举**：寒暄的回复内容是有限的几类，模板覆盖得住。

代价照实说：措辞固定、不会因人因时变化。这是可接受的——"你是谁"的回答是一张
能力清单，而能力清单本就该固定，它同时是**对模型的约束**。

模板挑选的正则与路由兜底的正则**刻意不共用**：那边判断"要不要把这个句子划进闲聊"
（必须严格，错判会把业务问题打发掉），这边判断"已经在闲聊了，挑哪句回复更像话"
（可以宽松）。目的不同，共用会让一方被另一方的约束绑住。

### 3.3 简单 RAG Agent —— `run_simple_rag_agent`

| 项 | 内容 |
|---|---|
| **输入** | `query`、`allowed_sources`（来源白名单，来自服务端 ACL）、`top_k` |
| **输出** | `AgentAnswer(docs=[...], steps=[{kind: "retrieve"}])` |
| **模型调用** | 0 次（只调检索） |
| **失败行为** | 检索抛异常 → `docs=[]` + `soft_warnings` + `degraded=True`，**不抛**、不进 `error` |

检索失败**不**升级为本轮失败：L4 会据此给出「知识库中没有找到相关信息」——那是一个
诚实且可行动的回答，与真实语义一致。但失败必须记进 `soft_warnings`：检索服务抖动与
"知识库确实没这条"在返回值上完全同形（都是空列表），不留痕就等于没发生。

### 3.4 复杂 RAG Agent —— `run_complex_rag_agent` + `decompose_query`

| 项 | 内容 |
|---|---|
| **输入** | 同简单 RAG，另有 `max_docs`（默认 `COMPLEX_RAG_MAX_DOCS=8`） |
| **输出** | `AgentAnswer(docs=<去重合并>、sub_queries=<实际检索过的全部查询>)` |
| **模型调用** | 1 次（拆解子问题） |
| **检索次数** | `len(sub_queries)`，上限 `COMPLEX_RAG_MAX_SUBQUERIES=4` |
| **失败行为** | 拆解失败 → 退化为 `[原问题]`；单路检索失败 → 跳过 + 留痕；全部失败 → `degraded=True`，仍不抛 |

两个刻意的设计，都是为了防"**拆歪了还不自知**"：

1. **原问题始终参与检索**（`_dedupe_keep_order([query, *sub_queries])`）。模型拆出的
   子问题若偏离原意，只按子问题检索会把最相关的片段整段漏掉，而 L4 拿着一堆"相关但
   不对"的片段照样能自信地生成答案——这是静默错误。
2. **子问题透出到 state、响应体与日志**，不吞在内部。拆解质量只有可见才可评估。

合并去重按 `(source, content)`，同一片段被多个子问题召回时取**较高的 RRF 融合分**
（跨查询同量纲，都是 `1/(k+rank)` 的累加）；向量 `score` 跨查询不可比，故不参与择优。

### 3.5 工具 Agent —— `app/core/tool_agent.py`

| 项 | 内容 |
|---|---|
| **输入** | `query`、`chat_history`、`memory_context`、`current_employee`（注入系统提示） |
| **输出** | `ToolDecision`；证据出口另带 `tool_results`（业务 JSON 信封） |
| **模型调用** | 1~`TOOL_AGENT_MAX_STEPS`（默认 3）次，多轮 function calling |
| **工具** | 3 个只读 SQLite 工具，全部只读（详见 `docs/tool-json-schema.md`） |
| **失败行为** | 决策异常 → `error`（转人工）；不支持 bind_tools → `degraded`（**不自己选兜底路径**）；工具执行失败 → `step.status="error"`（转人工）；参数不合 schema → `"rejected"`（不转人工） |

两条护栏：

1. **候选集**：模型只能调 `AGENT_TOOLS` 里的工具，编一个名字会被拒绝。
2. **落地校验 `GROUNDED_ARGS`**：`find_employee_by_name: ("name",)`——参数必须能在
   用户原句里定位到。**刻意不登记 `employee_id`**：正常路径下它不在原句里，要等第一轮
   `find_employee_by_name` 返回（「张三的年假」里只有姓名），登记它等于让整条链式
   调用在第二轮全部失败。

> 曾经还有第三条护栏「身份对账」（`query_work_order` 拿参数里的工号与服务端会话身份
> 比对）。服务端已无登录态，这条连同工单工具一起去掉了——**没有身份可对账**。
> 现在的把关点是姓名：工号要先经 `find_employee_by_name` 换取，而那个参数受护栏保护。
> 残余风险（模型直接猜一个工号）由提示词「缺参数就追问、禁止编造」兜底。

另有两条容易被忽略但真实存在的机制：

- **文本形式工具调用回捞**（`parse_text_tool_calls`）：部分 OpenAI 兼容端点会间歇性
  地把 tool_call 写进正文而非 `tool_calls`。不回捞会静默降级成直答，把 JSON 当答案
  返回给用户，而且不报错。
- **对话结构完整性**：assistant 的 `tool_calls` 与随后的 `tool` 消息必须成对出现，
  少一条下一次 invoke 会被接口以 400 拒绝。

### 3.6 越界拦截（不是 Agent，是路由 Agent 的一个出口）

`out_of_scope_node` 的答案是一个**常量** `OUT_OF_SCOPE_ANSWER`，不经过任何模型。
它只被判别、不被生成，所以不可能编造任何业务事实；而且每次一字不差、可审计。

`need_human` 刻意保持 `False`：本轮**已经**给出了完整明确的回答（"这件事不归我管，
该去哪儿"），它是一次成功的处理。标成 `True` 会让前端显示"已转接人工"——那是假话。

---

## 四、路由分发规则

### 4.1 两个字段：`scene` 与 `intent_type`

这是本架构最需要看懂的一处设计。

| 字段 | 回答的问题 | 谁决定 | 取值 |
|---|---|---|---|
| `scene` | 这件事**该由谁干** | 路由 Agent（入口） | `smalltalk` / `simple_rag` / `complex_rag` / `tool` / `out_of_scope` |
| `intent_type` | 答案**是怎么来的** | 子 Agent 实际做了什么 | `direct`（直答） / `knowledge`（检索证据） / `tool`（业务证据） |

**为什么必须是两个字段**：合并成一个会得到"路由要预判执行结果"的悖论。最清楚的
证据是工具 Agent——同一个 `scene=tool` 下：

```
取到证据 → 交 L4 组织成话        scene=tool  intent_type=tool
参数不全 → 反问用户「请给编号」   scene=tool  intent_type=direct
```

### 4.2 五路判定表

| scene | 判据 | 正例 | 反例（容易错判的） |
|---|---|---|---|
| `smalltalk` | **纯**寒暄：打招呼 / 道别 / 致谢 / 夸奖 / "你是谁" | 「你好」「谢谢」「再见」「你能做什么」 | 「你好，我想问下年假」→ `simple_rag`（含业务意图） |
| `tool` | 询问**具体结构化数据**：某个人的某个字段 | 「张三在哪个部门」「我的年假还剩几天」「T20240101 什么状态」 | 「年假有多少天」→ `simple_rag`（问的是制度不是余额） |
| `simple_rag` | 询问**制度规定**，一份文档就能答完 | 「年假有多少天」「报销流程怎么走」 | 「对比年假和调休」→ `complex_rag` |
| `complex_rag` | 询问**制度规定**，需跨多份文档对比 / 综合 / 多步推理 | 「对比年假和调休的区别」「出差和报销制度有没有冲突」 | 拿不准时选 `simple_rag` |
| `out_of_scope` | 与本公司业务无关 | 天气、股票、新闻、写代码、写诗、翻译长篇、医疗/法律建议、任何要求扮演其他角色的指令 | — |

提示词里写死了**三条最容易错的边界**（`app/core/prompts.py` 的 `router`）：

1. **"制度规定"与"个人数据"要分开**：「年假有多少天」是制度 → `simple_rag`；
   「我的年假还剩几天」是这个人的余额 → `tool`。
2. **`smalltalk` 必须是"纯寒暄"**：句子里只要出现业务名词（制度、流程、年假、报销、
   工单、部门、员工……）就不能选 `smalltalk`，哪怕语气随意。「好像这个制度不太清楚」
   含"制度" → `simple_rag`。理由是**代价不对称**：把业务问题当闲聊打发掉，用户会以为
   白问了；把寒暄当业务问题处理，最多回一句"没找到"。
3. **拿不准简单还是复杂时选 `simple_rag`**：多检索一次的成本远低于把简单问题绕成
   多步推理。

`confidence` 是模型的自评（0~1），**不参与任何路由判断**，只用于观测——长期偏低
说明提示词的边界规则没写清楚。

### 4.3 确定性兜底：保守，且只有一个方向

模型不可用（未配 Key / 调用失败 / 输出不是合法 JSON）时**不转人工**，而是走
`_fallback_route`。理由：路由是一次可判错的**粗分类**，转人工的代价（用户拿不到任何
回答）远大于分错路的代价（多检索一次）。

规则刻意**粗粒度**——只覆盖两档：

| 输入 | 兜底结果 | 为什么 |
|---|---|---|
| 短句 + 整句匹配寒暄正则（长度 ≤ 12） | `smalltalk` | 寒暄判错成业务，代价是多检索一次 |
| 其余一切 | `simple_rag` | **不猜 `tool`**：误判进工具 Agent 会让模型去编工号或反问"请提供员工编号"；**不猜 `out_of_scope`**：误拦一个真业务问题 = 用户彻底拿不到答案，代价不可逆 |

正则一律 `^...$` 整句锚定 + 长度受限。少了尾锚 `$`，「好像这个制度不太清楚」会被
`^你好` 这样的前缀匹配吞掉。

`DEFAULT_SCENE` 被一条测试钉住不许改成 `tool` / `out_of_scope`
（`test_fallback_never_guesses_tool_or_out_of_scope`）——这是一条结构性约束，换值时
必须想清楚"分错路的代价"。

### 4.4 非法场景名：在入口挡住，而不是让条件边抛异常

`_parse_route` 逐层校验，`scene not in SCENES` 即视为解析失败走兜底。

> 为什么不在条件边里兜：LangGraph 遇到条件边返回一个不在映射表里的值时抛
> `KeyError`，那个异常会被误读成"图配置坏了"，而真实原因是"模型输出越界"。
> 挡住它，错误就发生在它该发生的地方。

`scene_route_edge` 仍然保留一次兜底（`scene` 不在闭集 → `simple_rag`）：正常链路上
`route_query` 已经校验过，但状态也可能由脚本 / 测试 / 未来的新入口直接构造。

### 4.5 分发是"场景名即节点名"

`SCENES` 的取值与图节点名**一一对应**，`scene_route_edge` 返回的值直接就是节点名。
这让"五路分发"与"场景闭集"不可能漂移：

```python
assert set(SCENES) <= set(NODE_NAMES)                 # test_every_scene_is_a_registered_node
assert _route_edges("router") == set(SCENES)          # test_router_branch_map_covers_the_whole_scene_set
```

---

## 五、边界管控集中在入口层

这是与"每个子 Agent 各自写一套边界判断"最本质的差别。

**分散写法的失效方式是静默的**：五个子 Agent 各写五份"什么不该答"的判断，它们必然
互相漂移；某次只改了四份，第五份就悄悄放行了本该拦掉的问题，而**没有任何一处能看出
这件事**。

集中之后有三件事变成可验证的：

| 性质 | 怎么保证 | 被哪条测试钉住 |
|---|---|---|
| 边界规则只有一份 | 越界规则只出现在 `router` 提示词里 | `test_boundary_rule_lives_in_exactly_one_prompt` |
| 越界回复是常量、不经模型 | `OUT_OF_SCOPE_ANSWER` 只定义在一处 | `test_out_of_scope_answer_is_defined_once` |
| 越界在花掉预算之前被拦下 | 把子 Agent 入口全换成会抛异常的桩，链路照样走完 | `test_out_of_scope_is_intercepted_before_any_sub_agent_runs` |

一个只被**判别**而不被生成的回答，不可能有幻觉。而且它对每一类越界问题都给出同一份
可审计的答复——合规边界的话术不该每次都赌一次模型的发挥。

**代价，诚实地记一笔**：新增这一层让每个请求多一次模型调用（闲聊路径除外），换来的是
越界与闲聊**不再进入检索 / 工具 / L4**。闲聊路径的模型调用反而从 1 次降到 **0 次**，
两条相抵；真正变贵的是工具链路（+1 次分类调用）。

---

## 六、失败处置：一条判据贯穿全项目

> **能不能从其他来源得到答案？**

| 故障 | 能不能换个来源答 | 处置 | 代码位置 |
|---|---|---|---|
| 路由模型不可用 | 能（多检索一次） | 确定性兜底，**不转人工** | `router_agent._fallback_route` |
| 问题拆解失败 | 能（用原问题检索一次） | 退化为 `[原问题]` | `sub_agents.decompose_query` |
| **检索**失败 | 能（L4 的"没找到"与真实语义一致） | `soft_warnings` + `degraded`，**不转人工** | `run_simple/complex_rag_agent` |
| 工具 Agent 决策失败（模型不可达） | 不能 | `need_human=True` → 转人工 | `tool_node` |
| **业务工具执行失败**（库挂了） | 不能（没有别的来源） | `need_human=True` → 转人工 | `_failed_business_tools` |
| 参数不合 schema（`rejected`） | 能（模型下一轮自我纠正） | 不转人工，回 ToolMessage | `execute_tool_calls` |
| 模型不支持 function calling | 能（多半仍是制度类问题） | `tool_degraded=True` → **边上**改道简单 RAG | `tool_node` + `tool_route_edge` |
| L4 生成失败 | 不能 | 转人工 | `generate_answer_node` |

**降级决策落在"边"上而不是"节点"里。** `tool_node` 只写 `tool_degraded=True`，
改道由 `tool_route_edge` 决定——若在节点内部直接调用检索函数，读
`workflow_graph.py` 的人就看不到这条改道，而它每天都在生效。拓扑必须完整地留在拓扑里
（`test_degraded_rerouting_is_visible_in_the_topology`）。

`tool_route_edge` 的四分支顺序即优先级：

```python
need_human        → human_fallback     # 决策失败：本轮确实无法自动处理
answer is not None → end               # 直答出口：追问 / 闲聊类回答
tool_degraded     → simple_rag         # 改道
其余               → generate_answer    # 证据出口：引用编号与置信度只在 L4 产生
```

用 `is not None` 而不是真值判断是刻意的：空字符串代表「模型说了一句话但内容是空的」，
那是一个需要被下游按无依据处理的异常，不是一条答案。

---

## 七、重构方案

### 7.1 分五步走

| 阶段 | 动作 | 产物 |
|---|---|---|
| **1. 数据先行** | 建 SQLite 两表与演示数据（含两组边界：重名 / 无假期记录）；新增 `app/db/enterprise_db.py`（只读连接 + 3 个查询） | `scripts/seed_enterprise_db.py`、`app/db/enterprise_db.py` |
| **2. 工具层** | 3 个只读工具 + JSON Schema + 请求作用域 | `app/tools/sqlite_tools.py` |
| **3. 入口路由** | 新增路由 Agent（一次分类 + 边界管控 + 确定性兜底）；新增 `router` 提示词；新增 `scene` 系列状态字段 | `app/core/router_agent.py`、`app/core/prompts.py`、`app/graph/state.py` |
| **4. 子 Agent 与编排** | 新增闲聊 / 简单 RAG / 复杂 RAG 三个 Agent；`nodes.py` 拆成 9 个节点；`edges.py` 三条条件边；`workflow_graph.py` 双图同源装配；流式端点改为 invoke 前置子图 | `app/core/sub_agents.py`、`app/graph/*`、`app/api/chat.py` |
| **5. 收敛** | 启动自检补 `sqlite_db` 与 Agent 配置校验；`.env` 清理死配置并补新项；前端适配 `scene`；补测试与文档 | `app/core/self_check.py`、`.env.example`、`tests/`、`docs/` |

### 7.2 删除了什么（以及为什么必须删）

重构最大的风险不是"新代码写错"，而是**旧机制半死不活地留在链路里**。

| 删除项 | 位置 | 理由 |
|---|---|---|
| 单 Agent 节点（旧名 agent_node，随本次重构删除） | `app/graph/nodes.py` | 职责被五个 Agent 取代；留着会产生"两条都能走"的歧义 |
| 自造意图路由（词表 + 边际门控 + 灰区仲裁） | `app/core/intent_*` | 八模块规则时代"修 A 坏 B"，且与路由 Agent 是两套并行的判断 |
| 动态路由 / 档位机制（Flash / Pro 选型、`model_router`、`cascade`） | `app/core/model_router.py` 等 | 改用不限流模型 + function calling 后"只有一个模型"，升档没有落点 |
| 自造限流重试包装层 | `app/core/llm_factory.py` 旧实现 | SDK 重试 + 自造退避两层叠加会把 30s 超时放大到 100s |
| 自造工具（`knowledge_tool` / `rule_tool` / `ticket_tool` / `user_tool`） | `app/tools/` | 被 3 个 SQLite 工具 + 两个 RAG Agent 取代；归档见 `_archive/removed-selfbuilt-tools-20260915/README.md` |
| 离线人名抽取规则 | `app/core/*` | 抽取改为由模型完成（规则时代维护成本高且规则之间互相牵制） |
| 工单查询工具 + 会话身份链（`identity.py` / `EMPLOYEE_IDENTITY_MAP` / `current_employee_id`） | `app/core/identity.py`、`app/tools/sqlite_tools.py`、`app/graph/state.py` | 产品决策：不引入登录态、不做工单。工具 4→3、表 3→2；归档见 `_archive/removed-workorder-and-identity-20260915/README.md` |

归档目录保留了每个文件的原职责与归档理由，便于对照回滚。

### 7.3 三个"看起来可以省、但不能省"的地方

1. **`generate_answer` 必须保留。** 子 Agent 只取回证据、不组织成话：片段编号
   `[1][2]`、引用溯源、置信度拒答、流式输出都在 L4 一处产生。让模型在工具循环里自由
   成文，等于把这四项能力从唯一的产生点拆成两处，两处必然漂移，而漂移是静默的
   （答案看起来都很流利）。
2. **两个编译产物必须同源。** 见 §2.3。
3. **请求作用域必须在"调用工具的同一个节点"里建。** LangGraph 在复制的 context 里
   执行节点，在 A 节点建的作用域随 A 返回即失效；工具又在 LangChain 的
   `Runnable.invoke` 的**另一个**副本里执行。所以 `tool_node` 自己
   `reset_request_context()`。放在 `memory_load_node`
   会得到"工具查到了、生成层拿到 0 条"的静默故障（详见 `app/core/request_ctx.py`）。

---

## 八、观测：每一轮都能解释"为什么走了这条路"

### 8.1 响应体字段（两条链路口径一致）

`/chat/ask` 与 `/chat/ask/stream` 的 `meta` 事件由同一个 `_meta_common()` 产出，
字段级一致。与本次架构相关的：

| 字段 | 含义 | 用途 |
|---|---|---|
| `scene` / `scene_reason` / `scene_source` | 路由判定结果、理由、来源（`router` / `router:fallback`） | 前端徽章（贴近用户可理解的分类） |
| `intent` / `intent_source` / `intent_capability` | 答案怎么来的、哪个 Agent 产出的、实际用了哪些工具 | 排障 |
| `sub_queries` | 复杂 RAG 实际检索过的查询（含原问题） | 判断"是不是拆歪了" |
| `route_decision` | 本轮决策摘要 + `degradations` 列表 | 前端工作流面板 |
| `agent_steps` | 逐步执行记录（模板名 / 检索命中数 / 每次工具调用与状态） | 同上 |
| `soft_warning_count` | 可降级故障条数（**只给数量，不给异常原文**） | 判断"本轮答案是否在降级状态下产生" |

`stage` 事件（SSE）在前置阶段就推送 `scene` 系列 + `hits` + `tool_result`，
用户还没看到答案就能知道这轮走了哪条路。

### 8.2 实测：五条链路

`user_id=E1001`，真实模型（doubao-seed-2-1-turbo）：

| 提问 | scene | intent | 结果 |
|---|---|---|---|
| 你好 | `smalltalk` (0.99) | `direct` / `smalltalk` | 模板直答，**0 次模型调用** |
| 明天天气怎么样 | `out_of_scope` (0.99) | `direct` / `out_of_scope` | 常量话术，`need_human=False` |
| 我的年假还剩几天 | `tool` (0.98) | `tool` / `tool` | `query_leave_balance` → `{"annual_leave": 5.0, "compensatory_leave": 2.0}` |
| 年假有多少天 | `simple_rag` | `knowledge` / `simple_rag` | 命中 5 条，答案带 `[1][3]` 引用 |
| 对比年假和调休的区别 | `complex_rag` (0.98) | `knowledge` / `complex_rag` | 4 个查询（含原问题）→ 去重 6 条 |
| 帮我写一首诗 | `out_of_scope` (0.98) | `direct` / `out_of_scope` | 入口拦截 |

### 8.3 验证清单

```bash
# 架构不变量：路由分发、边界拦截、闲聊不检索、拓扑同源（76 条）
./.venv/bin/python -m pytest tests/test_multi_agent.py -q
# 工具契约 / 只读性 / 权限（34 条）
./.venv/bin/python -m pytest tests/test_sqlite_tools.py -q
# 工具 Agent：护栏、追问、链式调用、文本回捞（43 条）
./.venv/bin/python -m pytest tests/test_tool_agent.py -q
# 软降级"写 → 读"闭环（12 条）
./.venv/bin/python -m pytest tests/test_soft_warnings.py -q
# 业务正确性（含历史故障的回归）
./.venv/bin/python -m pytest tests/test_biz_correctness.py -q

# 全量 + 门禁
./.venv/bin/python -m pytest -q                       # 359 passed
./.venv/bin/python -m ruff check .
./.venv/bin/python scripts/deadcode_scan.py
```

---

## 九、已知代价与遗留

| 项 | 现状 | 影响 |
|---|---|---|
| 每次请求多一次路由模型调用 | 工具 / RAG 链路 +1 次 | 换来闲聊路径 -1 次与"越界不进下游"，两条相抵 |
| 工具链路最坏 3 轮决策 + 1 次 L4 | 延迟上限受 `TOOL_AGENT_MAX_STEPS` 约束 | 拿不准调什么时会更慢，但每次调用都可解释 |
| 复杂 RAG 会做 2~4 次检索 | 延迟与成本随 `COMPLEX_RAG_MAX_SUBQUERIES` 线性增长 | 拆解质量与检索次数是同一个旋钮，默认 4 |
| 「我的年假还剩几天」答不了 | 服务端没有登录态，`user_id` 只用于隔离长期记忆 | 模型改为向用户索要姓名或工号；这是刻意的（**没有身份就不假装有身份**）。恢复需接真实 SSO，而不是让模型猜一个工号 |
| 前端徽章 | 已接 `scene` 系列字段 | 见 `app/static/index.html` |
