# 模型路由重设计：从「规则命中即返回」到「证据加权 + 默认小模型」

> ⚠️ **提示：文中提到的上游账号配额（`RPM=3` / 「限频账号」/「免费档」）是写作当时的事实，
> 现已不成立**，请勿当作现状依据——说明见 [`docs/README.md`](../README.md) 第二节的告示。

> ⚠️ **本文所述档位路由机制已删除，仅作历史记录保留。**
> 收录方：`docs/multi-agent-architecture.md`（单 Agent → 五 Agent 协作重构）。
> 删除理由：改用不限流模型 + 原生 function calling 后，"按复杂度选 Flash/Pro"
> 的前提（免费档 RPM=3 的配额稀缺）不复存在；`model_tier` 字段恒为 `default`
> 只为不打破前端契约。代码存档：`_archive/removed-selfbuilt-routing-20260915-1314/`
> （`model_router.py` / `complexity_scorer.py` / `query_signals.py` / `cascade.py`）。
>
> 保留它的理由：**证伪过程本身可复用**。本文完整记录了一次"先建规则、
> 再用真实语料打脸、最后做减法"的闭环，以及"分类器到底要不要用"的判定标准——
> 这套判据与档位机制是否存活无关。
>
> **以下为原文**（写作于档位机制尚存时）：本文是对 `app/core/model_router.py` 的
> **整体重设计**，不是打补丁。
> 上一轮分析（问题定位）见本文第二章的实测基线；本文给出判断维度、路由策略、
> 默认与升级规则、边界处理，以及**分类器到底要不要用**的判定标准。
> 外部参照与逐条取舍见第三章。

---

## 〇、一句话结论

当前实现的根因是：**把「规则」当成了「决策」**——一条正则命中就返回，于是

- 88% 的请求被规则短路，分类器只看得到剩下 12%；
- 44% 的请求由「≤14 字 → Flash」这条伪信号决定；
- 真正的复杂短问句（"如果试用期请假呢"）永远升不上去；
- 而 `source` 字段还把它们记成 `classifier`，导致"该不该留分类器"的统计依据是错的。

新设计的核心只有三条：

1. **规则不再决定档位，只调制分数**——所有维度都参与打分，加权求和，不做首命中短路。
2. **默认永远是小模型**，升级必须拿出**正证据**：分数过线 **且** 至少一个维度给出决定性证据；
   两个独立维度同时给出决定性证据时，无需再看分数。
3. **分类器默认关闭**，只在"灰区"打开，且要先满足三个准入条件（见 4.8）。

原型在 22 条简单 + 16 条复杂问句上实测：**误升率 14% → 0%，升级召回 81% → 100%**（见第五章）。

---

## 一、设计目标与不变量

### 1.1 目标

| 目标 | 具体含义 |
|---|---|
| 默认省钱 | 默认档 = 小模型（Flash）。不升级是常态，升级是例外 |
| 有据升级 | 升级必须能回答"**是哪一条证据**让它升的"，可解释、可复现 |
| 零外部依赖可用 | 判断维度全部是本地的确定性规则，断网、限流、embedding 挂掉都能跑 |
| 永不阻断 | 路由任何环节失败都退化为"默认档继续生成"，绝不抛异常 |
| 可标定 | 阈值由真实分布反解，不是魔数；升级占比与预算挂钩 |
| 可观测 | 每次决策记录各维度贡献明细，能直接回答"为什么是这一档" |

### 1.2 必须保持的不变量（改坏了就是回归）

1. `get_chat_model(tier)` 的调用契约不变（`app/providers/llm.py`）。
2. `LLM_TIER_FLASH_MODEL` / `LLM_TIER_PRO_MODEL` **留空时**，两档回落同一模型，
   路由只产出观测数据、不改变实际行为——这是"先观测再启用"的安全态，必须保留。
3. `GraphState` 的 `model_tier`、`route_decision` 字段语义不变（前端与 trace 在读）。
4. `/routing/stats`、`/routing/preview`、`/routing/calibrate` 三个端点保持可用
   （字段可增，不可删）。
5. 路由关闭（`ROUTING_ENABLED=false`）时，行为与引入路由前**逐字节一致**。

### 1.3 一个前提性判断：为什么默认档必须是小的

设 P(升级) = 升到 Pro 的请求占比，k = Pro 相对 Flash 的成本倍数。

- **误升**（不该升却升）→ 立刻多花 `k-1` 倍钱，**不可回收**；
- **漏升**（该升却没升）→ 答案质量下降，但可由级联（生成后重试）兜底，**可回收**。

两者代价不对称，且只有前者不可回收。所以正确策略是
**默认下压、有据才升、用级联兜住漏升**。
这也是本文与"按难度二选一"式路由的一个关键区别。

---

## 二、当前实现的实测基线（为什么必须重写）

以下数字全部在项目真实语料（`intent_catalog` 的 56 条 utterances + description）上复现。

### 2.1 规则层吃掉了绝大部分流量

```
规则层短路：49/56 = 88%     分类器只看得到剩下 7 条
规则分布：  {'超短 query': 44, '打招呼': 2, '工单号查询': 2, '道别': 1}
档位分布：  {'flash': 49}   ← 全部走 Flash
```

`_DIFFICULTY_RULES` 是**自上而下、命中即返回**的。`^[\s\S]{1,14}$`（超短 query）
排在最后，但因为目录里的问句本来就短，它反过来吃掉了 44 条。
"四层漏斗逐层兜底"的设计意图，实际是**第一层吞掉了 79% 的流量**。

### 2.2 三个结构性缺陷

| # | 缺陷 | 证据 |
|---|---|---|
| 1 | **长度闸门是伪难度信号，且方向存疑** | `≤14 字 → Flash`。但"试用期年假怎么算""竞业协议有哪些例外"都是短而难。模块 docstring 自己写着"追问通常比首问更难"，而短追问恰好被这条强制降档，且因短路没有翻案机会 |
| 2 | **`source` 无条件写成 `classifier`** | `model_router.py:641-644` 的 `else` 分支不判断 `dominant`。分类器完全不参与时仍记为 `classifier` → `by_classifier` 虚高 → **用它判断"分类器值不值得留"会得出错误结论** |
| 3 | **同一份置信度挂三个互不协调的阈值** | `REFUSE_THRESHOLD=0.25`、`CASCADE_MIN_CONFIDENCE=0.25`、`ROUTING_THRESHOLD=0.5`。置信度落在 `[0.25, 0.50)` 时：系统判定"证据不够好、不值得升档"，却**已经按 Pro 计费**，级联还会再拒绝升档一次 |

### 2.3 规则层的具体误判

`要求详尽` 规则含裸词 `系统`，在企业 IT 语境里它是**名词**：

```
IT 系统怎么登录              -> pro  规则=要求详尽     ← 简单问题被顶成最贵档
报销系统里的发票怎么上传        -> pro  规则=要求详尽
系统登录不上                 -> pro  规则=要求详尽
```

### 2.4 同标注集上的对照：当前 vs 新设计

| 分组 | 当前实现升 Pro | 新设计升 Pro |
|---|---|---|
| 简单（22 条，不该升） | 3/22 = **14%**（全是"系统"误触发） | 0/22 = **0%** |
| 复杂（16 条，该升） | 13/16 = **81%** | 16/16 = **100%** |

当前漏判的 3 条正说明长度/关键词路线的天花板：
"项目组既要满足客户交付时间，又要控制人力成本，这种情况下加班费怎么算"、
"如果公司政策调整导致年假缩水，员工可以主张什么"、
"在同时触发竞业限制和服务期约定的情况下，离职时应该怎么处理"。

### 2.5 生死代码

| 问题 | 证据 |
|---|---|
| `_MIN_RULE_CONFIDENCE = 0.6` 是死分支 | 规则置信度最低 0.70，过滤条件永不成立 |
| `dominant` 第三个子句恒真 | `score == retrieval_difficulty` 已蕴含它（5 组取值验证结果一致） |
| 三条寒暄规则不可达 | 寒暄由 `detect_smalltalk` 先接走 → `smalltalk_reply` → END，到不了 `model_route` |
| `estimate_confidence` 每请求算两次 | `model_route_node` 一次 + `generate_answer` 一次；根因是置信度没写进 state |
| 线程池超时不取消任务 | `ThreadPoolExecutor(max_workers=2)` + `result(timeout=)`，超时后 worker 仍在跑；embedding 连续挂起两次即**永久占满**，之后所有打分必然超时 |
| `ROUTING_DEFAULT_TIER` 无取值校验 | 写错会静默按 flash 跑（对比 `VECTOR_DB_TYPE` 已有类型校验，属双标） |

---

## 三、调研：五个高质量项目怎么做模型路由

> 全部来自源码精读（`raw.githubusercontent.com` 直取），非二手转述。

### 3.1 RouteLLM —— 单一标量 + 分位标定 + 默认走弱

`routellm/routers/routers.py` 的抽象极简：

```python
class Router(abc.ABC):
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt): ...      # 返回 0~1 的"强模型胜率"

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak                       # ← 默认走弱模型
```

三个可抄点：

1. **打分与决策分离**——路由器只产出一个标量，阈值是外部参数。
2. **阈值靠分位标定**，不拍脑袋：
   ```python
   threshold = thresholds_df[router].quantile(q=1 - args.strong_model_pct)
   ```
   即"想让 30% 的请求走强模型，就在真实分布上取 70 分位"。
3. **默认走弱**——这正是本项目的目标形态。
4. 细节：`CausalLLMRouter` 在输出非法时 `return 1`（**升到强模型**），
   即"出错时倒向质量"。这与我们"出错时倒向默认档"相反，因为我们要控成本——
   但值得注意的是**它的失败方向是显式选择过的**，而不是随手写的。

**不抄**：它的 router 是按"某个特定强弱模型对"训练的（如 GPT-4 vs Mixtral），
中文企业语料上方向性未经验证；`bert_gpt4_augmented` 训练集 81% 英文、中文仅 3.1%。
`BertScorer` 的 docstring 里已经写了这个警告，这是对的。

### 3.2 litellm `complexity_router` —— 最接近本题的工业实现

`litellm/router_strategy/complexity_router/`。它的做法与当前实现**几乎相反**：

| 维度 | 权重 | 说明 |
|---|---|---|
| `tokenCount` | 0.10 | 短=简单，长=复杂 |
| `codePresence` | 0.30 | |
| `reasoningMarkers` | 0.25 | |
| `technicalTerms` | 0.25 | |
| `simpleIndicators` | 0.05 | **负权重**（"what is"、"define"） |
| `multiStepPatterns` | 0.03 | |
| `questionComplexity` | 0.02 | 多个问号 |

**加权求和 → 4 档**（SIMPLE <0.15 ≤ MEDIUM <0.35 ≤ COMPLEX <0.60 ≤ REASONING）。
零 API 调用、亚毫秒。

值得逐条抄的六条：

1. **维度化 + 权重可配**，而不是正则表命中即返回。
2. **负权重维度**：明确"简单指示词"应该**减分**。当前实现完全没有减分项，
   只有"降档规则"，而降档规则又会被后面的规则抢先。
3. **区分「没证据」与「证据表明简单」**（原文）：
   > A prompt where no dimension fires scores exactly 0.0 … the score to tier mapping
   > calls it SIMPLE **by default rather than by evidence**. Around half of general traffic
   > scores that way. Those requests reach the classifier instead.
   这是本设计里"默认档"与"低分档"必须分开记录的直接依据。
4. **`reasoning_override`（单点决定性证据的升档通道）**：命中 ≥2 个推理标志就升到
   REASONING，**但要求分数也达到 `reasoning_override_min_score`**（默认等于最低档边界），
   以免"在无关紧要的问题上说了句'一步一步来'就买到最贵的档"。→ 我们的 `strong_count` 规则。
5. **系统提示词里的推理标志不触发 override**，否则一句"Think step by step"会把全部流量顶到高档。
6. **`heuristic_first` / `hybrid` 两种"要不要叫分类器"的模式**：
   - `heuristic_first`：本地打分落在廉价档就直接用，其余交给 LLM 分类器；
   - `hybrid`：本地分**离任一档边界超过 `margin`** 就直接用，落在边界带内才叫分类器。
   并明确警告：分数是**离散权重组合出来的"块状"分布**，margin 必须对着真实分数分布挑。
7. **分类器熔断**：单次硬超时 → 进程级熔断 → 冷却 30s → 单请求探测 → 成功才闭合。
   期间所有请求走**本地打分器**（不是走某个固定档）。
8. **`stall_escalation`**：agentic 场景下，最新的工具调用重复/报错 ≥N 次就升一档。
   锚定"最新一次调用"以免用陈旧的证据升级已经恢复的任务。
9. **modality 门控**：分类器只看文本，带图请求可能被判到不支持视觉的模型 →
   向上找到最近的支持视觉的档，**只升不降**。

**不抄**：4 档 + LLM 分类器 + 各种企业特性（会话 pin、租户、SLO）对
"两个档位、RPM=3"的场景是过度设计。我们抄它的**打分结构与两条护栏**，不抄它的规模。

### 3.3 vllm-project/semantic-router —— 信号 / 决策 / 选择算法三层声明式

配置即架构（`config/fragments/`）：

```yaml
# signal：声明"存在哪些可判断的事实"
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.10
        hard:  ["solve this step by step", "compare multiple tradeoffs", "analyze the root cause"]
        easy:  ["answer briefly", "quick summary", "simple rewrite"]

# decision：用 AND/OR/NOT 组合信号，带 priority
  decisions:
    - name: priority_safe_escalation_route
      priority: 160
      rules:
        operator: AND
        conditions:
          - {type: domain,    name: business}
          - operator: OR
            conditions:
              - {type: keyword,     name: urgent_keywords}
              - {type: complexity,  name: needs_reasoning:hard}
          - operator: NOT
            conditions:
              - {type: jailbreak, name: prompt_injection}
      modelRefs: [{model: qwen2.5:3b, use_reasoning: true}]

# algorithm：在候选模型里选，带 SLO 约束
algorithm:
  type: multi_factor
  multi_factor:
    weights: {quality: 0.4, latency: 0.2, cost: 0.2, load: 0.2}
    slo: {max_tpot_ms: 200, max_ttft_ms: 800, max_cost_per_1m: 5.0, max_inflight: 50}
    on_no_candidates: cheapest
```

可抄四条：

1. **信号是命名的、可独立测试的、声明式的**——不是一张巨大的正则表。
   故障定位粒度从"哪条规则"细化到"哪个信号"。
2. **`hard` / `easy` 候选句 + threshold**：正是当前 `PrototypeScorer` 的思路，
   但它是**配置**而非硬编码，且阈值可调。
3. **`reask` / `dissatisfaction` 信号**（本项目直接可用）：
   当前轮的 user 消息与上一轮（`lookback_turns: 1`）或前两轮（`lookback_turns: 2`）
   相似度 ≥ 0.8 → `likely_dissatisfied` / `persistently_dissatisfied`。
   这比我们"追问更难"的假设精确得多：**追问不等于难，重复追问才等于没答好**。
4. **`on_no_candidates: cheapest`**：无候选时的兜底是**显式声明的策略**，不是默认行为。
5. **`shadow-dispatch` 插件**：影子流量。这是"先观测再启用"的正确工程形态——
   比"把两档模型配成同一个"更彻底（可以拿真实分布去评估另一个模型）。

**不抄**：`signal → decision → plugin` 的完整框架需要配置中心与校验体系，
对一个单体 Python 服务是杀鸡用牛刀。我们抄它的**信号命名法**与 **reask 信号**。

### 3.4 aurelio-labs/semantic-router —— 阈值 + 允许"无决策"

`semantic_router/routers/base.py`：

```python
if current_threshold := (route.score_threshold or self.score_threshold):
    passed = total_score >= current_threshold
else:
    passed = True                    # 没设阈值就一律通过
...
return passed_routes                  # 可能为空列表 = 没有决策
```

并强调 `RouteChoice.similarity_score` 要**外露**，让调用方自己判断。
→ 抄"**允许返回没有决策**"和"**分数必须外露**"这两条。

### 3.5 Arch-Router —— 决策与模型映射解耦

`katanemo/Arch-Router-1.5B`（arXiv:2506.16655）：1.5B 模型，输入 query +
一组"领域-动作"策略描述，**输出策略名**，再由映射函数把策略映射到模型。

关键设计：**策略在 prompt 里传入，因此新增/替换模型只改配置、不重训**。
另外它显式支持 `{"route": "other"}`——**"都不匹配"是一个合法输出**，
且提示词里明确要求"用户意图已满足或不相关时返回 other"。

→ 抄"**策略与模型映射解耦**"（我们的"维度 → 档位"应是配置，不是硬编码 if）与
"**other 是合法输出**"（我们不硬猜）。

### 3.6 litellm `adaptive_router` —— 在线反馈闭环（现在不做）

Bandit：按 `request_type` 分桶，对 `(request_type, model)` 做 Thompson 采样，
`quality_weight·sample + cost_weight·normalized_cost` 取 argmax。
后置 hook 用正则 + 工具调用信号给"上一轮出答案的模型"记功过，
每 ~10s 批量落库。

它的 README 里有一份诚实的"Known v0 limitations"：延迟没进打分、样本硬上限 200
（超出静默丢弃）、信号只有正则且偏英文、`_compute_bandit_delta` 是"v0 猜测，跑够
1000 个会话后请重调"。

→ **现在不做**，但这份"把局限写在 README 里"的做法值得学。
我们要先有**可信的观测**（当前 `by_source` 是错的），才谈得上闭环。

### 3.7 横向对比：我们该抄什么

| 项目 | 核心机制 | 抄 | 不抄 |
|---|---|---|---|
| RouteLLM | 标量 + 分位标定 + 默认走弱 | 分位标定；默认走弱；失败方向要显式选择 | 按特定模型对训练的 router |
| litellm complexity_router | 多维加权 + 4 档 + 边界带 | **维度加权**；**负权重**；**没证据≠简单**；**override 需分数托底**；**熔断回落到本地打分** | 4 档、LLM 分类器、企业特性 |
| vllm semantic-router | signal → decision → algorithm | **信号命名化**；**reask 信号**；**兜底策略显式声明**；影子流量 | 完整插件框架 |
| aurelio semantic-router | 阈值 + 无决策 | **允许没有决策**；**分数外露** | 以意图为目标（我们要的是难度） |
| Arch-Router | 策略与模型映射解耦 | **映射是配置**；**"都不匹配"合法** | 1.5B 生成式 router（RPM=3 装不下） |
| litellm adaptive_router | 在线 bandit | 局限坦白写进文档 | 现在不做闭环（观测还不可信） |

**一句话**：当前实现是"参照物驱动"的产物——每条机制单独看都有出处，
但没人回头问"合起来是不是过度"。本次重设计的方向相反：**先定问题（默认小模型、
有据升级），再从参照里挑最少的机制去满足它**。

---

## 四、新设计

### 4.1 总体结构

```
                    ┌──────────────────────────────────────────┐
   用户 query ──────▶│ ① 信号层（纯本地、确定性、零网络）        │
   + 会话状态        │   7 个 query 维度 + 2 个上下文维度        │
   + 检索证据        └───────────────┬──────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ ② 融合层                                 │
                    │   score = Σ(wᵢ × sᵢ) + 上下文修正项       │
                    │   同时产出 strong_count（决定性证据个数）  │
                    └───────────────┬──────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ ③ 决策层（在 (score, strong) 平面上分区）  │
                    │   强证据×2 → 升 / 强证据×1 且过线 → 升     │
                    │   灰区 → 默认档（分类器开着才问它）        │
                    │   其余 → 默认档                           │
                    └───────────────┬──────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ ④ 执行层  get_chat_model(tier)            │
                    │   + 会话 pin（升级锁，闲置 TTL）           │
                    │   + 生成后级联兜底（漏升的第二道闸）       │
                    └───────────────┬──────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │ ⑤ 反馈层  决策明细落 trace + 指标端点      │
                    │   维度贡献 / strong / 灰区原因 / pin 命中  │
                    └──────────────────────────────────────────┘
```

与当前实现的四个**结构性差异**：

| | 当前 | 新设计 |
|---|---|---|
| 规则的地位 | **决策**：命中即返回 | **证据**：只贡献分数，不单独决定档位 |
| 融合方式 | `max(分类器, 1-置信度)` | 加权和 + `strong_count` 双轴 |
| 默认档 | 多路各自回落，语义不清 | **唯一的默认 = 小模型**，且记录"是没证据还是证据表明简单" |
| 置信度信号 | 单调 `1 - confidence` | **三段式**（干净 / 零散 / 不足），不是越不相干越该升档 |

### 4.2 判断维度

用户提的三类复杂度——**跨领域、跨时段、多条件推理**——对应 D1/D5/D3/D2/D4。
每个维度产出 `sᵢ ∈ {0, 0.5, 1.0}`：`0` 未命中，`0.5` 弱证据（单侧提示），
`1.0` 决定性证据。同一维度内多个模式命中时取最大强度，第二个命中 +0.25（封顶 1.0）。

#### Query 侧七维（权重和为 1.0）

| # | 维度 | 权重 | 决定性证据（1.0） | 弱证据（0.5） |
|---|---|---|---|---|
| D1 | `reasoning` 推理深度 | 0.20 | `为什么/原因/依据/原理/凭什么`；`权衡/取舍/怎么选/建议/方案` | `分析/评估/判断/推断` |
| D2 | `multi_subject` 多实体 | 0.18 | `对比/比较/区别/差异/异同/优劣`；`分别/各自` | `汇总/归纳/总结/梳理/列举`；`A 和 B` 结构 |
| D3 | `multi_constraint` 多条件 | 0.15 | `同时(符合/满足/适用/触发)`；`既要…又要…`；`在…情况下…怎么` | `且/并且/同时满足` |
| D4 | `hypothetical_boundary` 假设与边界 | 0.14 | `如果/假如/假设/若/一旦…(会/是否/能不能/如何/该/可以/哪些/什么)`；`例外/除外/不适用/边界` | `前提/条件/适用范围` |
| D5 | `temporal_span` 跨时段 | 0.13 | `相比/较 + 去年/上月/同期`；`趋势/变化/同比/环比` | `去年/本季度/上季度`；`近 N 个月` |
| D6 | `output_demand` 输出负担 | 0.12 | `详细/完整/全面/逐条/逐步/展开/分点/列表`；**`系统地`/`全流程`/`端到端`** | `流程/步骤/清单` |
| D7 | `length` 篇幅 | 0.08 | `≥200 字` | `≥60 字` |

**三处关键修正**：

1. **D6 用 `系统地` 而不是裸词 `系统`**——直接消灭 2.3 节的三个误判。
   裸词 `系统` 在企业语料里是名词（报销系统、IT 系统），不是程度副词。
2. **D7 权重从"事实上的主判据"降到 0.08，阈值从 14 字提到 60/200 字**。
   长度只是弱信号，不足以单独决定档位。
3. **不设"降档规则"**。降档由"默认档"承担，而不是靠一条会和后面规则抢顺序的
   `→ flash` 规则。这是消除 2.1 节 88% 短路的根本办法。

#### 上下文侧两维（独立于 query，可加可减）

**C1 `evidence` 证据充分度**（来自检索/工具）：

| 检索置信度 | 修正 | 理由 |
|---|---|---|
| `≥ ROUTING_EVIDENCE_HIGH`（0.75） | **−0.05** | 证据干净、多半单跳可答，不需要更强模型 |
| `[LOW, HIGH)`（0.25 ~ 0.75） | **+0.10** | **升档甜区**：材料在库里，但零散/有歧义，需要模型做整合 |
| `< ROUTING_EVIDENCE_LOW`（0.25） | **0**（且**压制升级**） | 库里没有的东西，换 Pro 也变不出来。此时该走拒答/人工兜底，而不是烧钱 |
| 调用方未提供置信度 | **0**（不压制） | 离线标定 / 单条预演没有检索上下文，不能当成"证据不足" |

> **实现期的修正（与初稿的差异）**：初稿把"甜区"下界写成 `0.45`，即需要
> `ROUTING_EVIDENCE_SWEET` 这第三个常数。落地时改用 `ROUTING_EVIDENCE_LOW`
> 作为甜区下界 —— 因为它和第 3 行是**同一个语义**（"证据是否足以支撑回答"），
> 拆成两个数只会让"到底该调哪个"变成新的困惑。
> 代价是甜区更宽（0.25 起而非 0.45 起），收益是基准线从三个数收敛成一个。

最后一行是与当前实现最本质的分歧：现在的 `1 - confidence` 是**单调**的——
置信度越低越该升档——这在低置信度区间是**错的**：
"知识库里没有这条制度"和"这条制度有三份文件需要综合"是两回事，
前者升档是纯浪费。同时这也把三个互相打架的阈值统一了：

```
REFUSE_THRESHOLD = CASCADE_MIN_CONFIDENCE = ROUTING_EVIDENCE_LOW   （默认同源，0.25）
```
一个数、一个方向。（实现上由 `ROUTING_EVIDENCE_LOW` 派生出另外两个的**默认值**，
仍允许单独覆盖，但会立刻在自检里被看见。）

**C2 `conversation` 对话负担**：

| 条件 | 修正 | 理由 |
|---|---|---|
| 上一轮是 Pro 且 pin 未过期 | **+0.12** | 同一复杂话题的延续 |
| 本轮与上一轮用户消息的 bigram Jaccard `≥ 0.75`（近乎重复） | **+0.15** | 用户重复提问 ⇒ 上一轮没答好。借 vllm 的 `dissatisfaction` 信号 |
| 首轮、无历史 | 0 | 无信息 |

> 注意：**"是追问"本身不加分**，只有"重复追问"才加分。
> 当前实现假设"追问通常比首问更难"，这个假设太粗——"那试用期呢"确实更难，
> 但"再说一遍"只是没看到。用 Jaccard 把它俩分开。

**压制闸门 `suppress`（只在 `strong_count == 0` 时生效）**：

| 命中 | 处理 |
|---|---|
| `\bT\d{3,}\b` 工单号直查 | 分数封顶 0.20 |
| 整句寒暄 | 分数封顶 0.20 |

**为什么是"封顶"而不是"返回"**：这正是当前实现 88% 短路的根因。
新语义是——"工单号本身不代表难，但**如果同一句话里还有复杂维度**（如
`T123 和 T456 的报销差异`），照样能升"。命中即返回会让规则之间抢顺序，
封顶则不会。

### 4.3 融合：为什么是加权和 + strong_count，而不是 max

当前实现用 `max` 的理由是"单一强信号就足以升档，不该被平均掉"。这个理由本身没错，
但它在**多维**场景下会失控：任一维度给出 0.5 的弱证据就足以顶到阈值。

反过来，纯加权和也有反面风险：一个**决定性**信号（"有哪些例外情况"）
因为只有一个维度命中，加权后可能刚不到阈值。

所以两条都要，分工明确：

```
base  = Σ (wᵢ × sᵢ)                              # 证据的"总量"
strong = count(sᵢ == 1.0)                        # 证据的"质量"
score = clamp(base + ctx_evidence + ctx_conversation, 0, 1)
```

- **加权和**回答"证据够不够多"——多个独立维度共同指向复杂，才算复杂；
- **`strong_count`** 回答"有没有一锤定音的证据"——有一个决定性的，就直接升。

两条路径互为补集，把 max 的失控和纯加权和的迟钝同时堵住。

`strong_count` 的设计直接借自 litellm 的 `reasoning_override`：
"命中 ≥2 个推理标志就升档，**但要求分数也达到最低边界**"，
防止"随口说一句'详细点'就买到最贵的档"。

### 4.4 决策规则

在 `(score, strong_count)` 平面上的分区：

| 区域 | 条件 | 档位 | `source` |
|---|---|---|---|
| **A 强证据重叠** | `strong ≥ 2` | **Pro** | `evidence_multi` |
| **B 单点决定性 + 过线** | `strong == 1` 且 `score ≥ T` | **Pro** | `evidence_single` |
| **C 灰区** | `\|score − T\| < band` 且未落入 A/B | 默认档 | `gray`（分类器开启时才问它） |
| **D 其余** | — | 默认档 | `default` |
| **E 被压制** | `suppress` 命中且 `strong == 0` | 默认档 | `suppressed` |
| **F 降级** | 打分器异常 / 配置非法 | 默认档 | `degraded` |

补充规则：

- **灰区（C）不是"不敢定档"**。当前的 `abstain` 语义是"票数接近所以弃权"，
  在新设计里灰区的含义是**"只看 query 判不出来，该问就问"**——
  分类器关闭时它就是默认档，开启时它才是一次真正的查询机会。
  这样灰区率成了一个**可直接测量的数字**，用来决定分类器值不值得留（见 4.8）。
- **`score = 0` 且 `strong = 0`**（没有任何维度命中）→ 默认档，`source=default`，
  并在决策里记 `no_signal: true`。**绝不记为"证据表明简单"**——
  litellm 那段注释（3.2 节第 3 条）就是为这条写的。
- **决策永不抛异常**，任何环节异常 → F 区，默认档继续。

### 4.5 默认与升级规则

**默认规则（三条，按优先级）**

1. `ROUTING_ENABLED=false` → 全局默认模型，行为与引入路由前完全一致。
2. 档位模型未配置（Flash/Pro 同名）→ 路由照常产出决策与观测，但**实际调用同一个模型**。
3. 其余所有情况 → **Flash**。不升级是常态。

**升级规则（必须同时满足）**

1. 分数过线：`score ≥ ROUTING_ESCALATE_THRESHOLD`；
2. **正证据**：至少一个维度给出决定性证据（`sᵢ = 1.0`），
   或者两个维度同时给出决定性证据（此时免检第 1 条）。

用一张表总结：

| 场景 | 判定 |
|---|---|
| 单一事实直问（"年假多少天"） | 无维度命中 → Flash |
| 简单但要求"详细说明" | 只有 D6 命中（0.12）→ 低于阈值 → Flash |
| 对比两个制度（"差旅费和招待费的区别"） | D2 决定性（0.18），strong=1 → **Pro** |
| 假设 + 例外（"试用期请假怎么算，有哪些例外"） | D4 决定性（0.14）→ 看阈值；若同时有 D3 则 strong=2 → **Pro** |
| 工资单上的"系统登录不上" | D6 不再误命中（已改为"系统地"）→ Flash |
| 工单号直查 | 压制闸门封顶 → Flash |
| 工单号 + 对比 | 压制闸门因 strong≥1 失效 → D2 生效 → Pro |

**会话 pin（升级沿用）**

- pin 只保存**更高档位**，写在决策之后；**永不把 Flash 钉住**。
- **闲置 TTL**（默认 900s，每次复用刷新），不是会话总时长。
  借 litellm 的 `session_affinity_ttl_seconds` 语义。
- 话题切换（与上一轮 Jaccard `< ROUTING_TOPIC_SWITCH_JACCARD`）时 pin 不生效。
- 多 Worker 部署时 pin 表退化为"无 pin"，**只损失粘性、不影响正确性**——
  与当前实现同样的降级承诺，但要把这个降级写进注释，而不是靠"进程内字典"默认发生。

> **实现期的修正（与初稿的差异）**：初稿写的是"pin 期内**不降级**"，等于让 pin
> 直接定档。落地时改为 **pin 只贡献 `+0.12` 分、不强制锁档**。
> 原因：本设计的不变量是「升级必须拿出正证据」。一旦允许 pin 直接定档，
> 一个 Pro 会话里后续所有提问（包括"年假多少天"）都会留在 Pro 上，
> 等于绕过这条不变量，长会话成本会失控。
> 实测确认：pin 生效时"今年的报销有什么变化"（临界难度）能升上去，
> 而"年假有多少天"（零证据）**仍然留在 Flash**。

**级联（第二道闸，兜住漏升）**

- 前置路由从"问题长什么样"猜难度；有一类失败在问题侧毫无信号——
  **证据明明检索到了，Flash 却软拒答**。这类只能在生成后兜底。
- 保留现有 `app/core/cascade.py`，但把它的门槛与 C1 统一：
  `CASCADE_MIN_CONFIDENCE` 与 `REFUSE_THRESHOLD` 共用 `ROUTING_EVIDENCE_LOW`。
- **流式链路不适用**（token 已推送无法回收）。因此：
  **流式链路要把升级阈值下调一档**（`T_stream = T − 0.03`），
  因为那里没有第二道闸。这是本次设计里唯一一处"同一份逻辑、两条参数"，
  必须在代码里显式注释，否则将来会被当成不一致的 bug 改掉。

### 4.6 阈值标定：不拍魔数

沿用 RouteLLM 的分位法，但做两处项目化修正：

```python
# 1) 在真实 query 分布上取分位，而不是拍一个数
threshold = quantile(scores, 1 - target_pro_pct)

# 2) target_pro_pct 由预算反推，而不是随手写 0.3
#    Pro 相对 Flash 贵 k 倍时，预算只允许总成本上涨 x%：
#    pro_pct ≤ x / (k - 1)
```

**两处修正（当前实现的两个错误）**：

1. **标定必须走与线上完全相同的打分路径**。
   当前 `calibrate_threshold` 调 `route_model(q, use_classifier=..., record=False)`
   **从不传 `retrieval_confidence`**（恒 `None`），线上却会传——
   标定拟合的是"只有分类器"的分布，线上是"分类器 + 检索"的融合分布。
   新设计里标定只标定 **query 侧分数**（C1/C2 是动态的，不进标定），
   并且复用 `score_query()` 同一个函数，杜绝口径漂移。
2. **样本必须贴近线上分布**。用 `intent_catalog` 的 utterances 只代表"典型问法"，
   不代表线上真实分布（真实分布里短问句更多、口语更多）。
   标定样本应取自线上 query 日志，或至少按真实占比加权。

**配套指标（标定是否可信的判据）**：

- 需要一份**带标注的评测集**：每条 query 标 `expected_min_tier`。
  没有它，阈值只能"看起来合理"，无法证明。
- 评测指标：**升级召回率**（复杂问句被升的比例）、**误升率**（简单问句被升的比例）、
  以及**在给定误升率上界下的最大召回**。当前基线见第五章。

### 4.7 边界情况

逐条给处理策略。**每一条都对应一个测试**。

| # | 边界 | 处理 |
|---|---|---|
| 1 | query 为空 / 纯空白 | 返回默认档，`source=default`，`reason="empty"`，不进统计的 pro 分母 |
| 2 | 超短（"嗯""继续""好的"） | 无维度命中 → 默认档。**不因短而降档**（已经没有降档规则了）。若有会话 pin 且 pin 为 Pro，则沿用 |
| 3 | 超长（>2000 字，API 上限） | API 层已截断；D7 给 1.0，但不单独构成升级（权重 0.08 < 阈值） |
| 4 | 无检索结果 / 工具返回空 | `has_docs=False` → C1 不参与，且**压制升级**；交拒答或人工兜底 |
| 5 | 已拒答（`refused=True`） | 不再升级。这是"证据不足"的决策，不是"模型没用好" |
| 6 | 工具通道（`tool_result`） | 置信度**只算一次**并写入 `state["retrieval_confidence"]`，路由与生成共享。修掉当前"算了又丢" |
| 7 | 无 `session_id` / 首轮 | C2 不参与，pin 不生效。功能降级，决策仍正确 |
| 8 | 重复提问 / 不满 | C2 的 Jaccard ≥ 0.75 给 +0.15。**注意不要把"真的想问两遍"误判成不满**，所以只加分、不强制升档 |
| 9 | 话题切换 | 与上一轮 Jaccard < 0.2 且本轮无强证据 → 解除 pin，重新判定 |
| 10 | 分类器超时 / 异常 / 未安装 | 回落到**本地维度打分的结果**（不是固定档），并开熔断（1 次硬超时 → 冷却 30s → 单请求探测） |
| 11 | 分类器线程池耗尽 | 新设计**默认不用线程池**（本地规则是纯计算，微秒级）。若开启分类器，改用 `asyncio.wait_for` 或带超时的信号量，且超时要能**取消**任务 |
| 12 | 两档模型同名 / 未配置 | 只观测不改变行为；`/routing/stats` 显式标注 `tier_configured: false` |
| 13 | 档位模型名写错 / 不可用 | `get_chat_model` 已有降级；路由侧不做额外处理，但启动自检要能发现（列进 `self_check`） |
| 14 | 多 Worker 部署 | pin 表进程内 → 退化为无 pin。**只失去粘性，不影响正确性**，注释写明；要真粘性再换 Redis |
| 15 | 流式链路 | 无级联兜底 → 升级阈值 `T_stream = T − 0.03`，代码内注释理由 |
| 16 | 冷启动（无历史分布） | 用 `ROUTING_ESCALATE_THRESHOLD_DEFAULT`（0.14，见第五章实测），并在指标里标 `calibrated=false` |
| 17 | 非中文 / 中英混 | 维度规则按"命中即得分"，英文关键词（`comparison`/`trade-off`）作为补充；**不追求覆盖**，未命中的走默认档即可 |
| 18 | 多模态输入（图片） | 本文不涉及，但要留门：若路由目标模型不支持该模态，**只向上找支持该模态的档**，绝不向下（借 litellm 的 modality gate） |
| 19 | 会话预算耗尽 | pin + 级联合计每会话升档次数上限（默认 1），超出记 `budget_exhausted` |
| 20 | 路由本身异常 | 决策 F 区，默认档继续生成，`route_decision.source="error"`，并计入 `degraded` 指标 |

### 4.8 分类器：到底要不要（本节回答"根据实际情况看是否用得到"）

**结论：默认关闭（`ROUTING_CLASSIFIER=none`），只在灰区打开，且必须先满足三个准入条件。**

理由：

1. **它的作用域只有灰区**。A/B/E 区由强证据直接决定，D 区没有证据，
   分类器在这两处**没有话语权**——它唯一能起作用的是 C 区（`|score − T| < band`）。
   所以问题不是"分类器准不准"，而是"**灰区有多大**"。灰区率 3% 时，
   引入一个网络往返 + 一套依赖 + 一份缓存，换 3% 的流量改善，不划算。
2. **RPM=3 是硬约束**。LLM 分类器（Arch-Router 那类）在这个配额下等于
   每问一句吃掉一小半配额，**直接排除**。可选形态只剩 `prototype`（复用检索的
   embedding，零额外模型）或本地小模型。
3. **`prototype` 的真实短板**：余弦 margin 经 sigmoid 压成"概率"，
   温度 `0.08` 是拍的；margin 的实际分布随 **embedding 供应商**漂移
   （本地哈希向量与真实神经向量的量纲差一个数量级）。
   它不是不能用，而是**必须先标定**——和阈值一样。

**准入条件（三个都要满足才打开）**

| # | 条件 | 怎么测 | 不满足的后果 |
|---|---|---|---|
| 1 | 灰区率 ≥ 10% | `/routing/stats` 的 `gray_rate` | 收益上限太小，不值得 |
| 2 | 有 ≥200 条带标注的真实问句（标 `expected_min_tier`） | 离线评测集 | 无法证明它比"灰区一律走默认档"更好 |
| 3 | 在灰区上的**升级精确率 ≥ 0.8** 且召回相对"一律默认档"提升 ≥ 20% | `/routing/eval` 跑评测集 | 它只会增加成本，不增加正确率 |

**形态选择顺序**：`none` →（满足准入后）`prototype` 且仅灰区 →（RPM 放宽后）本地小模型 →（最后）LLM 分类器。

**工程约束（无论选哪种）**

- **失败回落必须是"本地维度打分的结果"，不是某个固定档**（借 litellm 熔断语义）。
- **熔断**：单次硬超时 → 进程级冷却 30s → 单请求探测 → 成功才闭合。
- **打分必须可取消**（修掉当前线程池超时后 worker 仍在跑的问题）。
- **统计必须分类计数**：`by_source` 必须由 `dominant` 决定，
  且新增 `by_retrieval`。修掉 2.2 节的 `source` 误标注——
  **在这条修好之前，任何"用数据决定砍不砍分类器"的结论都建立在错误数字上。**

**判定流程（可执行的决策树）**

```
灰区率 < 10% ────────────────────────────► 保持 none
   │ ≥10%
   ▼
有 ≥200 条标注问句？ ── 否 ───────────────► 先建评测集，保持 none
   │ 是
   ▼
prototype 在灰区上：升级精确率 ≥0.8 且召回提升 ≥20%？
   ├─ 否 ────────────────────────────────► 保持 none（灰区交默认档，靠级联兜底）
   └─ 是 ────────────────────────────────► 开启，且只触发于灰区（gray-only）
```

### 4.9 观测与可解释性

当前 `route_decision.reason` 是一句字符串，只带 `dominant`；新设计里决策对象
必须自带**完整证据明细**：

```python
{
  "tier": "pro",
  "source": "evidence_multi",          # evidence_multi|evidence_single|gray|default|suppressed|degraded|error
  "score": 0.31,
  "threshold": 0.14,
  "strong_count": 2,
  "no_signal": false,
  "dimensions": {                       # ← 关键：每一维的贡献都可查
    "multi_subject":  {"s": 1.0, "w": 0.18, "contrib": 0.180, "hits": ["相比"]},
    "temporal_span":  {"s": 0.5, "w": 0.13, "contrib": 0.065, "hits": ["今年"]},
    "reasoning":      {"s": 0.5, "w": 0.20, "contrib": 0.100, "hits": ["分析"]}
  },
  "context": {"evidence": 0.10, "evidence_band": "scattered",
              "conversation": 0.0, "reask_jaccard": 0.12},
  "suppressed": null,
  "pin": {"active": false, "prev_tier": "flash"},
  "elapsed_ms": 1
}
```

**指标（挂在现有 `/routing/stats` 上，不新建体系）**

| 指标 | 用途 |
|---|---|
| `pro_rate` | 成本；与 `target_pro_pct` 对比 |
| `dimension_hits` | **哪些维度在真正驱动升级**——用来砍掉从不生效的维度（当前实现从没做过这件事） |
| `strong_count_dist` | 决策主要落在哪个区 |
| `gray_rate` | 分类器准入条件 1 |
| `suppressed_rate` / `degraded_rate` | 压制闸门与降级是否异常高 |
| `no_signal_rate` | "没证据"的占比。若很高 → 维度覆盖不足，规则要补 |
| `pin_hit_rate` | 粘性是否真的在起作用 |
| `cascade_trigger_rate` / `cascade_lift` | 第二道闸的收益 |

配套新增 `/routing/eval`：传入带标注的评测集，直接返回升级召回率、误升率、
以及在给定误升率上界下的最大召回，并给出建议阈值。

### 4.10 配置项对照

| 新配置 | 默认 | 由谁演进而来说明 |
|---|---|---|
| `ROUTING_ENABLED` | true | 不变 |
| `ROUTING_DEFAULT_TIER` | `flash` | 不变，**但要加取值校验**（只允许 tier 集合内） |
| `ROUTING_ESCALATE_THRESHOLD` | `None`（→ 用 `_DEFAULT`） | 取代 `ROUTING_THRESHOLD` |
| `ROUTING_ESCALATE_THRESHOLD_DEFAULT` | `0.14` | 取代 `ROUTING_THRESHOLD_DEFAULT=0.5`（0.5 在新标度上等于永不升级） |
| `ROUTING_TARGET_PRO_PCT` | `0.15` | 取代 `ROUTING_STRONG_PCT=0.3`，且应与预算挂钩 |
| `ROUTING_STREAM_THRESHOLD_DELTA` | `0.03` | 新增：流式无级联兜底，阈值下调一档 |
| `ROUTING_GRAY_BAND` | `0.04` | 取代 `ROUTING_ABSTAIN_BAND=0.08`（新标度上 0.08 过宽） |
| `ROUTING_MIN_STRONG_EVIDENCE` | `1` | 新增 |
| `ROUTING_OVERRIDE_STRONG_COUNT` | `2` | 新增 |
| `ROUTING_DIMENSION_WEIGHTS` | `""`（用代码内权重） | 新增（JSON，允许只覆盖部分维度） |
| `ROUTING_EVIDENCE_LOW` | `0.25` | 新增，**统一** `REFUSE_THRESHOLD` 与 `CASCADE_MIN_CONFIDENCE` 的默认值 |
| `ROUTING_EVIDENCE_HIGH` | `0.75` | 新增 |
| `ROUTING_PIN_ENABLED` | true | 取代 `ROUTING_STICKY_TIER` |
| `ROUTING_PIN_TTL_SECONDS` | `900` | 新增 |
| `ROUTING_TOPIC_SWITCH_JACCARD` | `0.2` | 新增：话题切换判定，命中则 pin 不生效 |
| `ROUTING_CLASSIFIER` | **`none`** | 默认从 `prototype` 改为 `none` |
| `ROUTING_CLASSIFIER_TRIGGER` | `gray` | 新增：`off` / `gray` / `always` |
| `ROUTING_CLASSIFIER_TIMEOUT_MS` | `3000` | 不变 |
| `ROUTING_CLASSIFIER_COOLDOWN_S` | `30` | 新增（熔断） |
| `LLM_TIER_FLASH_MODEL` / `LLM_TIER_PRO_MODEL` | `""` | 不变 |

`ROUTING_EVIDENCE_LOW` 的默认值最终定为 **0.25**（不是初稿的 0.45）——
它同时派生 `REFUSE_THRESHOLD` 与 `CASCADE_MIN_CONFIDENCE` 的默认值，
取 0.45 会把线上拒答门槛从 0.25 抬到 0.45，是**另一个独立的、会影响答案可得性的行为变更**，
不该夹在路由重设计里一起上线。要抬门槛就单独抬，并在评测集上量它带来的拒答增量。

**废弃**：`ROUTING_USE_RETRIEVAL_SIGNAL`（C1 成为常设维度，不再需要开关）、
`ROUTING_ABSTAIN_BAND`（被 `ROUTING_GRAY_BAND` 取代）、
`ROUTING_THRESHOLD` / `ROUTING_THRESHOLD_DEFAULT` / `ROUTING_STRONG_PCT` /
`ROUTING_STICKY_TIER`（改名）、`_MIN_RULE_CONFIDENCE`（死分支，删）、
三条寒暄/致谢/道别规则（不可达，删）。

**旧变量名不做静默兼容**：`validate_routing_config()` 会检测它们并发出警告，
但**不读取旧值**。理由：旧名背后的语义都变了 —— 尤其
`ROUTING_THRESHOLD_DEFAULT=0.5` 在新标度上等于"永不升级"，
静默沿用会得到"看起来一切正常、实际永不升档"的结果，比直接报错难排查得多。

---

## 五、验证：原型实测

原型（维度打分器 + 决策规则）在 **22 条简单 + 16 条复杂**的标注集上跑出的结果：

### 5.1 分数分布——分得开

```
easy（22 条, 不该升）  min=0.00  p50=0.00  p90=0.00  max=0.00
hard（16 条, 该升）    min=0.14  p50=0.26  p90=0.35  max=0.45
strong_count           easy 全为 0        hard ∈ [1, 2]
```

**简单组一个都没命中任何维度**——这正是"多维证据"相对"关键词命中"的价值：
`年假有多少天`、`报销需要什么材料`、`IT 系统怎么登录` 全部得 0，
而当前实现会把后两个顶成 Pro。

### 5.2 阈值扫描

| 阈值 T | 升级召回 | 误升率 | 准确率 |
|---|---|---|---|
| 0.12 | 100% | 0% | 100% |
| **0.14** | **100%** | **0%** | **100%** |
| 0.16 | 81% | 0% | 92% |
| 0.20 | 75% | 0% | 89% |
| 0.30 | 38% | 0% | 74% |

`0.14` 是这份标注集上的拐点，故取作 `ROUTING_ESCALATE_THRESHOLD_DEFAULT`。
**注意**：分数不是概率，是"加权证据和"，标度本身没有绝对意义——
换一套维度权重就要重新标定。这正是 4.6 节强调"阈值必须由分位反解"的原因。

### 5.3 真实语料上的升级占比

在 `intent_catalog` 的 56 条上，新设计升级 **0 条（0%）**。
这与直觉一致：目录里的问法都是单一事实查询（"年假有多少天"）。
**当前实现在同一份语料上也是 0 条 Pro——但原因完全不同**：
当前是靠"≤14 字"这条伪信号把 44 条一起按到 Flash，
新设计是因为它们**确实没有任何复杂维度**。
前者在遇到"短而难"的问题时会漏，后者不会。

### 5.4 与当前实现的直接对照

| 分组 | 当前实现 | 新设计 |
|---|---|---|
| 简单（22 条） | 3/22 = 14% 误升 | **0/22 = 0%** |
| 复杂（16 条） | 13/16 = 81% 召回 | **16/16 = 100%** |

**这些数字的诚实边界**：标注集只有 38 条、由本次分析手工构造，
只能证明"设计方向成立"，**不能替代真实评测集**。
上线前必须用线上 query 抽样 + 人工标注（≥200 条）复跑一遍（见 4.8 准入条件 2）。

---

## 六、迁移路径

分三步，每步**独立可验收、可回滚**。

### P0 —— 先让观测变可信（不改任何路由行为）

1. 修 `source` 判定：由 `dominant` 决定，新增 `by_retrieval`。
2. 删除死代码：`_MIN_RULE_CONFIDENCE`、`dominant` 的第三个恒真子句、
   三条不可达的寒暄规则。
3. 置信度写入 `state`，消除重复计算与工具通道"算了又丢"。
4. `ROUTING_DEFAULT_TIER` 加取值校验。
5. 新增 `/routing/eval` 与带标注的评测集（≥200 条）。

**验收**：`by_source` 三路加起来等于 `total`；
`/routing/eval` 能给出当前实现的召回/误升率（预计约 81% / 14%，2.4 节）。

> ⚠️ **这一步必须先于 P1**。当前 `by_source` 是错的，
> 任何"用数据决定砍不砍分类器"的结论都会建立在错误数字上。

### P1 —— 换掉打分器（行为改变的唯一一步）

1. 新增 `app/core/complexity_scorer.py`：七维 + 两上下文维，纯函数，无 IO。
2. `model_router.route_model` 改为：**先跑全部维度 → 融合 → (score, strong) 决策**。
   删除规则短路、删除长度降档、把压制闸门改为"封顶"。
3. 保留 `RoutingDecision` 结构，**增加 `dimensions` / `strong_count` / `no_signal` 字段**。
4. 用 P0 的评测集标定阈值，写入 `ROUTING_ESCALATE_THRESHOLD`。

**验收**：评测集上误升率 ≤ 2%、召回 ≥ 90%；
`/routing/stats` 的 `dimension_hits` 能看出是哪些维度在驱动升级；
关掉 `ROUTING_ENABLED` 或两档不配置时，行为与 P0 完全一致。

### P2 —— 按数据决定分类器与粘性

1. 按 4.8 的决策树评估分类器：不满足就保持 `none`（这是**预期结果**，不是失败）。
2. 若开启：只在灰区触发，熔断 + 回落到本地打分结果。
3. 上会话 pin 的 TTL 语义，替换当前的 `ROUTING_STICKY_TIER` 布尔开关。
4. 统一 `REFUSE_THRESHOLD` / `CASCADE_MIN_CONFIDENCE` / `ROUTING_EVIDENCE_LOW`。

**验收**：`gray_rate`、`pin_hit_rate`、`cascade_lift` 可读且稳定。

---

## 六之二、实现状态（本次落地了什么）

**已完成（P0 + P1 + P2 的大部分）**，测试从 271 → 322 条全绿：

| 文件 | 改动 |
|---|---|
| `app/core/complexity_scorer.py` | **新增**：七维 + 两上下文维，纯函数无 IO；`suppress` 是封顶不是返回 |
| `app/core/model_router.py` | 重写：删规则短路与长度闸门；`(score, strong_count)` 双轴决策；分类器默认关、仅灰区、带熔断；`source` 按决策来源计数；pin 改 TTL + 只加分 |
| `app/core/self_check.py` | 新增 `routing_config` 快检项（配置写错在启动时暴露） |
| `app/config.py` | 新增 12 项、废弃 6 项；`validate_routing_config()`；`REFUSE_THRESHOLD` / `CASCADE_MIN_CONFIDENCE` 默认值统一到 `ROUTING_EVIDENCE_LOW` |
| `app/graph/nodes.py` | `retrieval_confidence` 恒定计算并写入 state；`model_route_node(state, streaming=)` |
| `app/graph/state.py` | 新增 `retrieval_confidence` 字段（一份事实、三个消费者共用） |
| `app/rag/generator.py` | `prepare_generation` / `generate_answer` 支持复用上游置信度，消除每请求算两遍 |
| `app/api/routing.py` | 新增 `/routing/eval`；`/routing/stats` 增加维度命中、强证据分布、灰区率等 |
| `tests/test_model_routing.py` | **新增** 51 条：评测集指标 + 误判回归 + 结构性断言 + 降级承诺 |

**未做（有意留白，见第七章）**：在线 bandit 闭环、LLM 分类器、影子流量、4 档。

**上线前必须补的一步**：用线上真实 query 抽样标注 **≥200 条**（每条标 `expected_min_tier`），
跑 `POST /routing/eval` 复验。当前 38 条集子只能证明方向成立——
它由人工构造，不代表真实分布（真实分布里短问句更多、口语更多）。

---

## 七、不做的事

| 不做 | 为什么 |
|---|---|
| 在线 bandit 闭环（litellm `adaptive_router`） | 前置条件是有可信的观测与标注。当前 `by_source` 就是错的 |
| LLM 分类器 | RPM=3 下不可行；形态上是"每问一句吃掉一小半配额" |
| 扩展到 4 档 | 两个模型都没有稳定的中间档；加档位只增加配置面，不增加信息 |
| 插件 / 规则表达式 DSL（vllm 那套） | 单体服务，维度是代码里的纯函数更易测、易读 |
| 引入 `routellm` pip 包 | 它的预训练 router 面向英文 Arena 语料，中文企业场景方向性未验证；我们的维度是**领域相关**的，训练数据也拿不到 |
| 影子流量 | 值得做，但依赖 P0 的评测集先建好；列为 P2 之后 |

---

## 附录 A：与现有代码的差异清单

| 文件 | 改动 |
|---|---|
| `app/core/model_router.py` | 重写 `route_model` 的规则层与融合层；删 `_DIFFICULTY_RULES`（迁到 `complexity_scorer`）；改 `_effective_threshold`；扩 `_RouteStats`（`by_retrieval`、`dimension_hits`、`gray`）；修 `source`；删线程池 |
| `app/core/complexity_scorer.py` | **新增**：七维 + 两上下文维，纯函数 |
| `app/core/request_ctx.py` | 不变（query 向量复用保留，这个设计是对的） |
| `app/graph/nodes.py` | `model_route_node` 写入 `retrieval_confidence` 到 state；生成节点改为读 state |
| `app/core/model_router.py::calibrate_threshold` | 改为标定 query 侧分数；复用同一个 `score_query()` |
| `app/api/routing.py` | 新增 `/routing/eval`；`/routing/stats` 增加维度与灰区字段 |
| `app/config.py` | 按 4.10 增删；`ROUTING_DEFAULT_TIER` 加校验 |
| `app/core/cascade.py` | 置信度门槛改用 `ROUTING_EVIDENCE_LOW` |
| `tests/test_model_routing.py` | **新增**：38 条标注集 + 维度单测 + 决策分区 + 降级用例 |
| `docs/history/dynamic-routing-design.md` | 标注为"已被本文取代"，保留作为历史记录 |

## 附录 B：必须写的测试

```python
# 1. 维度单测：每个维度的决定性/弱证据各一条
@pytest.mark.parametrize("q,dim", [
    ("对比差旅费和招待费的区别", "multi_subject"),
    ("如果试用期请假，年假怎么算", "hypothetical_boundary"),
    ("今年的报销相比去年有什么变化", "temporal_span"),
])
def test_dimension_fires(q, dim): ...

# 2. 误判回归：这次修掉的三条，必须永远钉住
@pytest.mark.parametrize("q", ["IT 系统怎么登录", "报销系统里的发票怎么上传", "系统登录不上"])
def test_bare_system_word_no_longer_escalates(q):
    assert route_model(q).tier == TIER_FLASH

# 3. 对照组：同样重要——别把简单问题修好、把复杂问题弄坏
@pytest.mark.parametrize("q", [
    "对比正式员工和外包员工的年假差异，并说明各自的审批流程",
    "如果员工在试用期请假，年假该怎么算？有哪些例外情况",
])
def test_complex_still_escalates(q):
    assert route_model(q).tier == TIER_PRO

# 4. 结构承诺：规则不再短路
def test_all_dimensions_always_evaluated():
    """命中一条降档规则后，复杂维度仍必须被评估。"""
    d = route_model("T123 和 T456 的报销标准有什么区别")
    assert "multi_subject" in d.dimensions     # 实现里明细字段叫 dimensions
    assert d.tier == TIER_PRO                  # 压制闸门不吞掉复杂维度

# 5. 统计可信：source 必须与主导信号一致
def test_source_matches_dominant_signal():
    d = route_model("公司地下停车位的申请条件和排队规则是什么",
                    retrieval_confidence=0.2, has_docs=True, use_classifier=False)
    assert d.source != "classifier"

# 6. 降级：打分器异常/分类器超时都要返回合法档位
def test_scorer_failure_degrades_to_default(monkeypatch): ...
def test_classifier_timeout_falls_back_to_local_score(monkeypatch): ...

# 7. 阈值标定口径一致
def test_calibration_uses_same_scoring_path_as_online():
    """标定与线上必须走同一个 score_query，否则阈值必然漂移。"""
```
