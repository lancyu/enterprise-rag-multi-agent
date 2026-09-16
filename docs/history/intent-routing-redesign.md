# 意图路由重构：从「写死的四选一」到「目录驱动的四层决策」

> ⚠️ **本文提出的「目录驱动四层决策」已整体删除，仅作历史记录保留。**
> 收录方：`docs/multi-agent-architecture.md`（单 Agent → 五 Agent 协作重构）。
> 现在的做法：路由 Agent **一次模型调用**在五个场景（`smalltalk` / `out_of_scope`
> / `simple_rag` / `complex_rag` / `tool`）里选一个，外加确定性正则兜底。
> 「意图在 `intent_catalog.py` 里是数据、加意图不改代码」这个目标被**换了一种解法**：
> 场景名即节点名，新增一个 Agent 才需要动拓扑，而改判定口径只需改提示词一处。
> 代码存档：`_archive/removed-selfbuilt-routing-20260915-1314/app/core/intent_catalog.py`
> 与 `intent_router.py`。
>
> 保留它的理由：**第二章「问题定位」的故障链条完全有效** ——
> "张三在哪个部门" 被当成知识问答，今天仍然是 `scene` 分类必须处理的那类边界
> （人 vs 制度），只是判定者从自研四层决策换成了路由 Agent。
>
> ---
>
> 以下为原文（写作时）：
>
> 触发问题：「意图路由过于单一，模型无法准确理解真实意图。例如查询
> **「张三在哪个部门」却被路由到知识问答**。希望意图路由的设计更宽泛、更灵活，
> 而不是把意图写死。」
>
> 本文分两部分：**一、调研**高质量项目怎么做意图路由，提炼可借鉴的做法；
> **二、改造**本项目的实现，并给出验证与操作手册。

---

## 一、问题定位：那句话到底是怎么被路由错的

先把故障链条查清楚，否则改造就是盲改。追下来是**两个缺陷叠加**：

### 缺陷 1：how-to 标志词表里混进了「在哪」

`app/core/query_signals.py` 的前身（原 `intent_router.py`）里有一张 how-to 标志表：

```python
_HOWTO_MARK = r"(?:怎么|如何|怎样|怎么办|咋办|在哪|在哪里|哪儿|哪里|为什么|...)"
```

它的用途是排除「提到属性词但其实在问操作步骤」的句子，例如
「我的邮箱**怎么**配置」——这是对的。但「在**哪**里」被一起放进来之后：

```
「张三在哪个部门」
  → 含「在哪」→ 被 _HOWTO_MARK 整句否决，不再算员工查询
```

### 缺陷 2：知识规则里的「在哪」把它捞走了

`_KNOWLEDGE_RULES` 有一条「流程咨询」规则：

```python
(re.compile(r"(怎么办|如何|怎样|怎么申请|如何申请|在哪|哪里)"), "knowledge", 0.78, "流程咨询")
```

于是这句话在**工具组失败之后**被知识组命中，最终 `intent=knowledge, 0.78`。
用户问的是某个人的部门，拿到的却是一段制度说明。

### 根因不在这一条规则，而在结构

就算把「在哪」从 how-to 表里删掉、把这条修好，结构性问题依然在：

1. **加一种意图要改四处代码**：正则表、`_RULE_GROUPS`、`edges.py` 的 if-else、
   提示词里的四行说明。漏改任何一处都是**静默不一致**（不报错，只是行为不对）。
2. **判断依据单一**：只看「句子里出现了什么词」。同一个正则的两半会互相牵制——
   为了修 A 收紧某个词，会顺手把 B 一起挡掉，正是本次故障的成因。
3. **非此即彼**：规则没命中就必须二选一（调 LLM 或兜底），没有「两个都像，
   我不确定」这个中间状态。而现实中大量输入恰恰是中间状态。

---

## 二、调研：高质量项目怎么设计意图路由

调研对象与源码快照（均为 `main` 分支，2026-09 抓取）：

| 项目 | 关键文件 | 定位 |
|---|---|---|
| semantic-router（aurelio-labs） | `semantic_router/routers/base.py`(1859 行)、`route.py`、`routers/hybrid.py`、`encoders/bm25.py` | 专做语义路由的库 |
| LlamaIndex | `query_engine/router_query_engine.py`、`selectors/{llm,pydantic,embedding}_selectors.py` | 工具/引擎选择器 |
| Haystack | `components/routers/llm_messages_router.py`、`conditional_router.py` | 组件的路由分支 |
| LangGraph | `types.py` 的 `Command`、`prebuilt/chat_agent_executor.py` | Agent 工具派发 |

### 2.1 semantic-router：意图是**数据**，不是枚举

这是四个项目里最直接可借鉴的一个。一条「意图」的全部定义就是
`semantic_router/route.py:69-77` 的七个字段：

```python
name: str
utterances: Union[List[str], List[Any]]
description: Optional[str] = None
function_schemas: Optional[List[Dict[str, Any]]] = None
llm: Optional[BaseLLM] = None
score_threshold: Optional[float] = None
metadata: Optional[Dict[str, Any]] = {}
```

**只有 `utterances` 进打分链路**（`routers/base.py:258-277` 把每条示例句展平成
`{route, utterance, metadata}` 逐条向量化入索引）。多写一条示例句 = 多一个向量，
**零代码改动**。配置还能整体外部化（`base.py:106-153` 支持从 JSON/YAML 读），
并在初始化时把路由阈值下发到每一条（`base.py:432-435`），判定处就是一行比较
（`base.py:645-651`）：

```python
passed = total_score >= current_threshold
```

没命中怎么办？返回一个**空对象**，不硬猜（`base.py:700-701`、`787-788`）：

```python
# if no route passes threshold, return empty route choice
return RouteChoice()
```

多路融合用的是**恒定加权**而不是「按 query 选一路」（`routers/hybrid.py:31-32`、
`536-543`）：稠密向量乘 `alpha`、稀疏乘 `(1-alpha)`，`alpha` 默认 0.3。
稀疏侧是自研 BM25，公式原文（`encoders/bm25.py:200-208`）：

```python
idf = np.log((N + 1) / (df + 0.5))
tf_normed = tf / (k1 * (1.0 - b*b*(tf_sum[..., None] / self._avg_doc_len)) + tf)
```

它还提供了**用 LLM 从 description/函数签名反向生成 utterances**的离线工具
（`route.py:213-265`），prompt 里明确要求「用真实值而不是占位符」，
解析时用 `<config></config>` 标签 + 必填键校验四道闸防跑偏。

> **需要更正一个流传的说法**：这个快照里**并没有** `dynamic_threshold`
> 这个机制（全目录 grep 零命中），也没有任何 LLM 仲裁在路由决策链路上。
> 唯一的「自动」是 `base.py:1846-1853` 的**离线**随机搜索标定阈值
> （`np.linspace` 跑 100 个候选、按人工标注集选最优），它需要 X/y 标注。
> 所以「自适应阈值」这件事，本项目得自己补——见 §3.3。

**可借鉴**：① 意图配置化 + 示例句即向量；② 「空对象 = 未命中」而不是硬猜；
③ BM25 词面路对专名（工单号、VPN、人名）远比向量可靠，且**完全本地零调用**；
④ 反向生成示例句是离线的一次性动作，与线上限流无关。

### 2.2 LlamaIndex：能力自带描述，选择器只认下标

`RouterQueryEngine` 初始化时把工具拆成两个**下标对齐**的平行数组
（`router_query_engine.py:121-122`）：

```python
self._query_engines = [x.query_engine for x in query_engine_tools]
self._metadatas = [x.metadata for x in query_engine_tools]
```

`ToolMetadata` 只有 `description`（必填）+ `name`（可选）。而选择器拼提示词时
**只用了 description**（`llm_selectors.py:26-33`）：

```python
text = " ".join(choice.description.splitlines())
text = f"({ind + 1}) {text}"
```

关键是**新增能力 = 注册一个 QueryEngineTool，路由代码零改动**。
而且它把「为什么选它」也返回出来（`base_selector.py:51`、`router_query_engine.py:188/199`）：

```python
SingleSelection{index: int, reason: str}
...
final_response.metadata["selector_result"] = result
```

选择器是可插拔的（`base_selector.py:79-96` 只要求实现一个 `_select`），三种实现取舍明确：

| 选择器 | 成本 | 输出约束 | 精度 |
|---|---|---|---|
| `LLMSingleSelector` | 1 次 LLM | 自由文本 + 解析 | 中 |
| `PydanticSingleSelector` | 1 次 LLM | function calling / JSON schema | 高 |
| `EmbeddingSingleSelector` | 0 次 LLM | 无 | 低（但最便宜） |

**它的两个缺口恰好是本项目必须补的**：`EmbeddingSingleSelector`（`embedding_selectors.py:51-71`）
**没有阈值、没有「都不选」分支**，相似度 0.05 也照样返回 top1；而且越界时直接
`raise ValueError("Failed to select query engine")`（`router_query_engine.py:192-193`），
没有兜底。

**可借鉴**：① 描述写成「**何时该用我**」；② 选择理由回传做回归评测；
③ 选择器可插拔；④ 阈值与未命中分支必须自己补，不能照抄。

### 2.3 Haystack：两条路，各有取舍

**LLM 路由**（`llm_messages_router.py`）不是「把候选描述给模型选」，而是
「模型自由输出 + 正则匹配」：

```python
output_names: list[str]     # 候选出口
output_patterns: list[str]  # 每个出口对应的正则
```

按顺序匹配，第一个命中即 `break`；**全不命中走 `unmatched` 出口**（`:169-170`），
并把模型的原始文本单独透出便于调试（`_RESERVED_OUTPUT_NAMES`，`:16`）。
它比闭集 `unknown` 强的地方就在这个 `unmatched`：语义是「模型说了，但不在我的
预期集合里」，而不是「模型没说」。

**条件路由**（`conditional_router.py`）走的是另一条路——规则是**纯数据**：

```python
class Route(TypedDict):
    condition: str            # Jinja2 布尔表达式
    output: str | list[str]
```

按序渲染取第一个真值（`:423-430`），沙箱执行（`:280`），
并且**启动时校验模板变量**（`:283`、`486`）。无命中直接抛
`NoRouteSelectedException`（`:484`）——异常而非静默兜底。

**可借鉴**：① `unmatched` 语义 + 原始输出外露；② 规则数据化之后**必须配一道
启动期校验**，否则写错字段名只表现为「这条规则从来不生效」；
③ 兜底策略要显式二选一（要么静默走默认，要么抛异常），不要模棱两可。

### 2.4 LangGraph：跳转目标应由声明推导

`Command`（`types.py:799-824`）把「更新状态」和「跳到哪个节点」打包成一个返回值：

```python
class Command:
    update: Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

`create_react_agent` 则是把工具直接 `bind_tools` 给模型，由模型的 tool_call
决定调用哪个——**没有意图枚举**，新增工具就是新增一个带 schema 的函数
（`prebuilt/chat_agent_executor.py:173-215`）。

**可借鉴**：节点跳转不该散落成一串 `if intent == ...`，而应由声明/注册表推导。
（`create_react_agent` 那条「全交给模型」的路子本项目**不能抄**——见下面的约束。）

### 2.5 调研结论：本项目的硬约束决定了取舍

本项目上游账号是**推理模型 + RPM=3**（连续两次调用要隔 20 秒）。
这条约束直接砍掉了两个选项：

- ❌ **不能**照 `create_react_agent` 把决策全交给模型：每次提问先花一次配额做意图
  分类，用户每问两句就撞限流。旧实现在这条路上已经吃过苦头。
- ⚠️ **可以**保留 LLM，但必须降到「罕见路径」：只在真正拿不准时才用。

于是取舍如下（**默认全程零大模型调用**）：

| 借鉴来的做法 | 来源 | 在本项目的落法 |
|---|---|---|
| 意图是数据、示例句即锚点 | semantic-router | `app/core/intent_catalog.py` 的 `IntentSpec` |
| 能力自带「何时该用我」的描述 | LlamaIndex | `IntentSpec.description`，同时作向量锚点与仲裁材料 |
| 选择理由回传 | LlamaIndex | `IntentRoute.evidence` + `preview_intent()` |
| 词面（BM25）+ 向量双路 | semantic-router | 关键词（免费）+ 缓存的 embedding |
| 未命中不要硬猜 | semantic-router | 边际门控 + 灰区 + 保守兜底 |
| `unmatched` 语义 + 原始输出外露 | Haystack | 灰区原因 `gray_reason` + `raw=` 进 evidence |
| 规则数据化配启动校验 | Haystack | `test_catalog_entries_are_self_consistent` |
| 跳转目标由注册表推导 | LangGraph | `intent_catalog.ROUTE_TARGETS` |
| **自己补**：自适应阈值 | （无人可抄） | 通道级边际门控，见 §3.3 |

---

## 三、改造：目录驱动的四层决策

### 3.1 新增一层「查询信号」，把句式判定与意图解耦

新增 `app/core/query_signals.py`：只回答「这句话长什么样」，不关心意图。

```
intent_catalog（意图目录） ──引用──> query_signals（查询信号）
```

依赖是**单向**的，判定可以随意图增删而稳定存在。两层修复：

1. **移除** how-to 标志里的「在哪 / 在哪里 / 哪儿 / 哪里」。判定「哪个部门」
   是取值查询还是操作问题，真正的依据是**属性词是否处于被索取的位置**
   （句式 B2 要求「哪个/什么 + 属性词」**落在句尾**），而不是句中出现「在哪」二字。
2. **补入**「审批 / 办理 / 负责」——这些词出现时问的确实是「该找谁办、归谁管」，
   属于流程咨询（「哪个部门负责报销」即此类）。

### 3.2 意图目录：意图是数据

`app/core/intent_catalog.py` 里一条能力就是一条声明：

```python
IntentSpec(
    name="employee_lookup",            # 能力名（开集，可自由新增）
    channel=CHANNEL_TOOL,              # 路由通道（闭集，稳定契约）
    description="查询某位具体员工的属性取值：工号、邮箱、所在部门、分机号…",
    utterances=("张三的邮箱是多少", "张三在哪个部门", ...),   # 语义锚点
    keywords=("工号", "邮箱", "部门", ...),                  # 词面锚点
    guard="howto",                     # 命中即该能力清零
    detector="employee",               # 确定性检测器，命中即短路
)
```

**为什么 `channel` 保持闭集、只开放 `name`？** 这是刻意的分工：

- `channel` 决定**去哪个图节点**，是对外契约（API 的 `intent` 字段、前端的
  `intent == "smalltalk"` 都依赖它）。做成开集就必须让 `edges` 在运行期动态解析
  节点名，图拓扑不可静态审查，拼错一个字母只会在线上暴露。
- `name` 决定**叫什么能力**，只用于观测、评测与以后的按域过滤。它是开集，
  加多少都不影响路由安全。

一句话：**通道闭集保证路由安全，能力开集保证覆盖灵活。**

当前目录 7 条能力 / 3 个通道：

| 能力 | 通道 | 说明 |
|---|---|---|
| `smalltalk` | smalltalk | 纯寒暄，整句锚定 |
| `employee_lookup` | tool | 某个人的属性取值 |
| `ticket_lookup` | tool | 工单/报修单状态 |
| `hr_policy` | knowledge | 年假、考勤、加班、社保 |
| `finance_policy` | knowledge | 报销、差旅、发票 |
| `it_support` | knowledge | VPN、密码、邮箱配置、打印机 |
| `company_policy` | knowledge | 员工手册、通用制度与审批流程 |

> 知识通道细分到业务域，不只是为了好看：它让词面与语义锚点更**精确**
> （VPN 问题命中 `it_support` 而不是一坨通用「knowledge」），也是以后做
> **按域过滤检索来源**（`allowed_sources`）的挂载点。

### 3.3 门控做在**通道**层级（本轮最关键的一个设计决定）

一开始我按「能力」比较边际，结果被一个真实反例打回：

```
「报销流程怎么走」
  finance_policy 命中「报销」→ 词面 0.5
  company_policy 命中「流程」→ 词面 0.5
  → 打平 → 判定「胶着」→ 白白升级到灰区（多一次 LLM 调用）
```

可它**根本没有风险**：两个能力同属 `knowledge` 通道，无论选哪个，图都走
`knowledge_retrieve`、检索都用同一套语料。用户完全无感。

所以正确的问题是「**去哪个分支**存不存在歧义」，而不是「叫哪个能力名」。
于是门控分两步：

```python
# 1. 按通道归并（候选已按融合分降序，首次出现即该通道最优）
best_by_channel = {}
for cand in cands:
    best_by_channel.setdefault(cand.spec.channel, cand)
ranked = list(best_by_channel.values())

# 2. 地板与边际都在「通道排名」上判定
if sem_used and top.sem_raw < floor:            → 灰区(low_floor)
if top.fused - ranked[1].fused < margin:        → 灰区(tight_margin)
```

**为什么不能只用一个固定阈值？** 固定阈值分不清两种截然不同的情境：

| 情境 | 第一名的绝对分 | 该怎么做 |
|---|---|---|
| 全体都很低（谁都不像） | 0.3 | 老实兜底，别硬塞 |
| 前两名咬得很紧（两个都像） | 0.3 | 也别硬猜 |

两者的「第一名绝对分」可能完全一样，真正有信息量的是**第一名与第二名的差距**。
这一招是从 RouteLLM / 级联分类的「弃权带」（abstain band）搬来的——本项目在动态
模型路由里已经用过一次（`model_router.py` 的 `ROUTING_ABSTAIN_BAND`）。

### 3.4 四层结构

```
① 确定性锚定   寒暄整句锚定 / 员工属性句式 / 工单号        零成本，精度最高，命中即返回
② 融合打分     词面（关键词，免费）+ 语义（向量，走缓存）   fused = (w_lex·lex + w_sem·sem)/(w_lex+w_sem)
③ 边际门控     通道级地板 + 通道级边际                     纯计算
④ 灰区仲裁     只有「两个通道咬得很紧」才问大模型          一次 LLM（罕见）
   兜底        仲裁不可用 → 保守回落知识通道（不是掐断）
```

两个实现细节值得记：

- **语义分用「相对分」做融合、用「绝对分」做地板。** 绝对余弦在
  「本地哈希向量」与「真实神经向量」两种模式下量纲差一个数量级，
  所以融合前先按候选做 min-max 归一化（跨模式可比），而地板判定仍看绝对余弦
  （毕竟「谁都不像」是个绝对判断）——地板值按 embedding 模式自适应
  （`config.effective_intent_floor()`）。
- **语义不可用时按剩余信号重新归一权重**（`_Cand.fused(sem_used)`），
  否则 embedding 一挂，所有分数会被腰斩、全都掉进灰区。

### 3.5 灰区仲裁：候选由目录**动态渲染**

提示词里的候选清单不再是写死的四个词，而是从目录生成：

```python
INTENT_PROMPT.format(catalog=catalog_prompt_block(), user_query=query)
```

`catalog_prompt_block()` 遍历目录输出「能力名（通道）：description」。
**新增一条 `IntentSpec`，仲裁提示词自动认识它，一个字都不用改。**

解析时**长名优先**——`hr_policy` 与 `company_policy` 有公共子串
`_policy`，短名先匹配会误命中。

### 3.6 降级链（任何一层挂了都能跑）

| 挂掉的东西 | 行为 |
|---|---|
| embedding 不可用 | 退化为「确定性 + 词面」两层，仍返回合法结果 |
| 语义层抛异常 | 异常被吞掉并记 warning，同上 |
| 大模型仲裁失败 | 保守回落 `knowledge` 通道（宁可多检索一次） |
| `INTENT_LLM_ARBITRATION=false` | 灰区直接兜底，**一次模型都不调** |
| `INTENT_SEMANTIC_ENABLED=false` | 纯离线路由，**完全断网可跑** |

### 3.7 图边不再写死

`app/graph/edges.py` 从注册表取跳转目标：

```python
def intent_route_edge(state):
    return route_target(state.get("intent_type") or "")
```

新增能力不需要碰这个文件。

---

## 四、操作手册：怎么加一种新意图

**只需要改一个文件**：`app/core/intent_catalog.py` 的 `INTENT_CATALOG`。

```python
IntentSpec(
    name="device_borrow",
    channel=CHANNEL_KNOWLEDGE,              # 必须是已登记的 4 个通道之一
    description="借用公司设备：笔记本、显示器、转接线等资产的借用规定。",  # 写「何时该用我」
    utterances=("怎么借显示器", "笔记本可以借吗"),   # 用户可能怎么问，多写几条
    keywords=("借用", "借显示器", "借笔记本"),        # 词面锚点，专名尤其有用
)
```

加完跑一次测试即可（有一条用例专门守这个承诺）：
`tests/test_intent_routing.py::test_new_capability_only_needs_a_catalog_entry`。

**要新建一个通道**（很少见）才需要动 `ROUTE_TARGETS` + 图里加节点。

调参（`.env`）：

| 变量 | 默认 | 什么时候动 |
|---|---|---|
| `INTENT_MARGIN_MIN` | 0.06 | 灰区太频繁 → 调小；路由太激进 → 调大 |
| `INTENT_FLOOR_REAL` / `_LOCAL` | 0.32 / 0.05 | 大量「谁都不像」被硬塞进某个能力时调大 |
| `INTENT_LLM_ARBITRATION` | true | 限流吃紧时关掉，灰区直接兜底 |
| `INTENT_SEMANTIC_ENABLED` | true | 需要完全离线运行时关掉 |

排查「这句话为什么路由到这里」：

```bash
curl -s --noproxy '*' -X POST http://127.0.0.1:8001/routing/intent-preview \
  -H 'Content-Type: application/json' \
  -d '{"query":"张三在哪个部门","use_embedding":true}'
```

返回里带**完整候选得分表**（词面 / 语义 / 融合）与灰区原因——旧实现只有一行日志，
看不到第二名是谁、差多少分，无法判断是规则问题、阈值问题还是数据问题。

---

## 五、验证

### 5.1 故障回归

| 问句 | 旧结果 | 新结果 |
|---|---|---|
| **张三在哪个部门** | `knowledge 0.78`（流程咨询） | **`tool` / `employee_lookup`**（确定性锚定） |
| 张三属于哪个部门 | `knowledge` | `tool` / `employee_lookup` |
| 怎么申请邮箱扩容 | `knowledge` | `knowledge` / `it_support`（未被误伤） |
| 哪个部门负责报销 | `knowledge` | `knowledge` / `finance_policy`（未被误伤） |
| 报销流程怎么走 | `knowledge` | `knowledge` / `finance_policy`（同通道打平不再进灰区） |

真正决定成本的是**零大模型率**：确定性锚定 + 词面能覆盖的输入，
一次模型调用都不产生（`get_route_stats()["llm_free_rate"]`）。

### 5.2 测试

新增 `tests/test_intent_routing.py`（32 项），覆盖四类约束：

1. **故障回归**：员工属性取值查询不再被 how-to 词误杀；对照组保证 how-to 不被放开；
2. **目录即数据**：新增能力只加一条声明，路由与图边自动认识它；目录自洽校验
   （通道已登记、名称唯一、检测器/guard 名可解析）；提示词与路由器契约一致；
3. **通道级门控**：同通道打平不算歧义；跨通道打平才进灰区；无信号走兜底；
4. **降级**：语义层异常/返回 None、仲裁失败/被开关关掉，四种情况都返回合法结果。

### 5.3 门禁

```
pytest 246 passed（原 214，新增 32）
ruff   All checks passed
deadcode_scan  默认模式 3 项 = 豁免清单 3 项（无新增、无僵尸）
deadcode_scan --strict  9 项存量（人工巡检用，非门禁）
```

---

## 六、遗留与未做

1. **语义锚点没有用真实语料校准**。`employee_lookup` 的 utterances 是手写的，
   semantic-router 那条「用 LLM 离线反向生成示例句」的路子**没走**。如果现场
   发现某类问法老是落进灰区，最有效的动作是往对应能力的 `utterances` 里补
   20~30 条**带真实姓名/系统名**的模板句——这是零代码、零线上成本的动作。
2. **`INTENT_MARGIN_MIN` 和地板值是拍的，没有标定**。本项目已经在动态模型路由
   里有 `calibrate_threshold` 的成熟做法（按期望占比在真实样本上反解），
   意图路由这一层还没接。样本够了应当补。
3. **能力名已对外暴露（2026-09-12 补齐）**。`intent_capability` 走通了四处：
   `GraphState` + `create_initial_state`（LangGraph 用 TypedDict 做 schema，
   **未声明的键会被静默丢弃**，这两处必须成对改）→ `intent_recognize_node` 写入 →
   `/chat/ask`、`/chat/ask/stream`（`stage` 与 `meta` 两个事件）、`/workflow/execute`
   四个响应点透出 → 前端展示。
   对外契约不动：`intent` 仍是**通道**（闭集），`intent_capability` 是**开集**的
   细粒度能力名，新增意图时前端与日志立刻可见，无需改契约。
4. **知识通道细分到业务域后，尚未用于过滤检索来源**。现在 4 个知识能力都走
   同一个 `retrieve_knowledge_docs`，`channel` 之外的 `name` 只是标签。
   下一步可以按能力给 `allowed_sources` 加域约束，减少跨域噪声召回。
