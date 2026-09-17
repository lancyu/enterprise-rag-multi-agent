# 动态路由设计方案（Flash / Pro 自动选型）

> ⚠️ **提示：文中提到的上游账号配额（`RPM=3` / 「限频账号」/「免费档」）是写作当时的事实，
> 现已不成立**，请勿当作现状依据——说明见 [`docs/README.md`](../README.md) 第二节的告示。

> ⚠️ **本文所述机制已全部删除，仅作历史记录保留。**
> 最终归宿：`docs/history/model-routing-redesign.md`（重设计）→ `docs/multi-agent-architecture.md`
> （多 Agent 重构，档位机制整体删除，`model_tier` 字段恒为 `default` 只为不打破前端契约）。
> 代码存档：`_archive/removed-selfbuilt-routing-20260915-1314/`。
>
> 保留它的理由：本文记录的**问题现场**与**调研过程**仍然有效
> （限流 RPM=3 的硬约束、依赖 "张三在哪个部门" 那次真实故障、
> 每个机制抄自哪个开源实现的逐条出处）。
> 但它提出的**方案本身已被实测证伪**：核心机制是"规则命中即返回"，
> 实测在项目真实语料上短路了 88% 的请求，其中 44 条由
> `^[\s\S]{1,14}$`（超短 query → Flash）这条伪信号决定。
> 新设计把它改成"规则只调制分数、不决定档位"，对照数据见新文档第 2.4 / 5.4 节。
>
> 本文的教训值得单列：**每条机制单独看都有出处、都合理，
> 合起来却过度** —— 典型的"参照物驱动"而非"问题驱动"的设计。

---

## 一、项目现状核对：方案的接入点与四处不符

### 1.1 现状（读代码所得）

| 关注点 | 现状 | 位置 |
|---|---|---|
| 模型调用 | 单模型。`config.LLM_MODEL_NAME` 一个，经 `get_chat_model()` 返回**全局单例**（`_chat_model` 缓存） | `app/core/llm_factory.py:265` |
| 已有路由 | 「规则优先 + LLM 兜底 + 保守默认」三层，但路由的是**意图**（knowledge/tool/unknown），**不是模型档位** | `app/core/intent_router.py:196` |
| 生成链路 | `prepare_generation()` → `stream_answer_tokens(inputs)` → `ANSWER_PROMPT \| get_chat_model()` | `app/rag/generator.py:168,219,247` |
| 已有置信度 | `estimate_confidence()`：RRF 融合分归一化（跨 embedding 供应商可比），`REFUSE_THRESHOLD=0.25` 触发拒答 | `app/rag/generator.py:121` |
| 已有观测 | 节点 `trace`（含 elapsed_ms）+ `get_route_stats()` 规则命中率 + `/health` 的 `dump_config()` | `app/graph/nodes.py:36`、`app/core/intent_router.py:135` |
| 容错 | `_RateLimitRetryModel`：429 指数退避 + temperature 自愈；无 Key 时 `MockChatModel` 兜底 | `app/core/llm_factory.py:145` |
| 依赖 | venv 223MB，**无 torch / transformers** | `requirements.txt` |
| 回归基线 | `tests/test_service.py` 15/15 全过 | — |

### 1.2 方案中四处与现状不符（需要修正）

**① Layer 1 的关键词白名单与现有意图规则语义冲突。**
`intent_router` 已把「密码」「怎么」判为 knowledge **意图**。同一批关键词再承担「难度」职责会互相污染——一个词既决定"走不走 RAG"又决定"用哪档模型"，后期任何一边调整都会误伤另一边。
→ **难度规则必须独立成表**（`_DIFFICULTY_RULES`），与 `_KNOWLEDGE_RULES` 物理隔离。

**② 「多轮追问沿用上次模型」方向性有风险。**
现有 `GraphState` 是单次请求级，跨轮状态只在会话历史里，`build_short_term_window` 只裁剪文本，没有"上轮用了哪个模型"的字段——需要新增状态并写入会话。
更关键的是语义：追问往往比首问更难（"那如果…呢""还有例外吗"），**沿用 Flash 会把用户锁死在低档**。
→ 改为**只做降级锁、不做升级锁**：上轮用 Pro 且本轮无降档信号 → 沿用 Pro；上轮 Flash 但本轮出现升档信号 → 允许升档。

**③ Layer 2 的「~10ms」不成立。**
`xlm-roberta-base` 是 12 层 / 768 隐层，CPU 上 batch=1 单次 forward 实测 **20~60ms**，加 tokenize 与首次加载（约 1.2GB 常驻内存）。10ms 只在 GPU 上成立。
→ 延迟预算按 CPU 写实，并配 `ROUTING_CLASSIFIER_TIMEOUT_MS` 硬闸门。

**④ Layer 3 的「置信度低就升 Pro」概念错配 + 缺成本闸门。**
- 概念错配：项目现有的 `estimate_confidence()` 是**检索侧**置信度（证据质量），不是**生成侧**置信度。用证据分判断"Flash 答得行不行"抓不到「证据强但 Flash 归纳能力不足」这一类——而这恰恰是升档唯一有价值的场景。
- 成本：级联等于请求做两遍，最坏情况（Flash 全量输出 + Pro 全量输出）**比直接用 Pro 还贵**。
→ 生成侧置信度需另做（见 §4.3），并加三道闸门。

---

## 二、GitHub 参考实现

### A. lm-sys/RouteLLM — 5,442 stars，最后提交 2024-08（Python）

| 文件 | 作用 |
|---|---|
| `routellm/routers/routers.py` | 全部路由器实现，`BERTRouter` 在 L106-125 |
| `routellm/controller.py` | 决策入口 + 参数校验 |
| `routellm/calibrate_threshold.py` | 阈值标定 |
| `routellm/evals/evaluate.py` | APGR 评测 |

**决策接口（最值得抄的一点）**——只实现一个方法，把"打分"和"决策"彻底分离：

```python
class Router(abc.ABC):
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt) -> float:  # 返回 0~1
        ...
    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        return routed_pair.weak
```

**打分（BERTRouter）**——实测 config.json 底模为 `xlm-roberta-base`、3 标签：

```python
inputs  = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
logits  = self.model(**inputs).logits.numpy()[0]
softmax = np.exp(logits - logits.max()); softmax /= softmax.sum()
binary_prob = softmax[-2:].sum()      # P(label1) + P(label2)
return 1 - binary_prob                # 即"强模型胜率"
```

**阈值是标定出来的，不是拍的**——按"期望的强模型调用占比"在目标 query 分布上反解分位数：

```
python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5
→ For 50.0% strong model calls for mf, threshold = 0.11593
```

README 明确警告：标定数据集必须贴近线上 query 分布，否则实际占比会漂移。

**失败处理**：`CausalLLMRouter` 输出无效时 `return 1`——**保守升档**；`Controller` 严格校验 threshold ∈ [0,1] 与 router 名，否则抛 `RoutingError`。

**评测**：APGR（Average Performance Gap Recovered）+ 调用占比曲线，在 MMLU / GSM8K / MT-Bench 上对比 random 基线。

**关于「直接 pip 装就能用」的核实结论**：

| 项 | 核实结果 |
|---|---|
| 底模 | **`xlm-roberta-base`**（config.json 实测，非英文 BERT）→ 多语言含中文，可直接编码中文 query ✅ |
| 标签数 | 3（0/1/2） |
| checkpoint | `routellm/bert_gpt4_augmented` |
| 训练数据 | Chatbot Arena，81% 英文、**中文仅 3.1%**；强/弱对 = GPT-4 vs Mixtral-8x7B ⚠️ |
| 迁移性 | 论文 §4.4 称换模型对后仍有效，但**必须在我们自己的 Flash/Pro 对上重新标定阈值** |
| pip 包 | `routellm` 0.2.0，最后发布 **2024-07-08**（约 2 年未更新）；依赖 torch / transformers / datasets / litellm / openai / sklearn / pandas ⚠️ |
| 标签语义 | 源码注释（`routers.py:121` "label 1 and 2 (tie, tier 2 wins)"）含糊，**方向性必须本地验证**（见 §5 Phase 2 准入条件） |

> **建议：不装 `routellm`。** 真正要用的只有上面 15 行 softmax 数学，装包会把 223MB venv 撑到 3GB+ 并引入一个停更 2 年的依赖树。

### B. vllm-project/semantic-router — 5,491 stars，2026-09-02 仍在更新（Go）

| 文件 | 作用 |
|---|---|
| `src/semantic-router/pkg/classification/complexity_classifier.go` | 复杂度分类（zero-shot 原型对比） |
| `src/semantic-router/pkg/decision/engine.go` | 决策引擎（布尔表达式树 + 置信度 + 兜底策略） |

**打分：不需要任何标注数据的 zero-shot 原型对比。**
每条复杂度规则在 YAML 里配一组 hard 候选 + 一组 easy 候选，各自嵌入并聚成 `prototypeBank`（支持多原型以覆盖"难"的多种形态）。query 嵌入后：

```
margin = sim(query, hard) - sim(query, easy)   → 按 margin 判 hard/easy，输出 Confidence
```

多通道融合取 **`d(t) = max(|d_vis|, |d_sem|)`**——**取 max 而非加权和**：任一通道强烈判定为"难"就足以升档，不该被另一通道平均掉。这个思路可直接搬到我们的"分类器分 vs 检索难度分"融合上。

**决策引擎**：信号匹配 + 布尔表达式树（AND/OR/NOT）；每条 decision 算 `Confidence = 命中信号置信度均值`；多条命中时 `selectBestDecision` 按策略选最优；输出结构化 `DecisionTrace`。

**兜底是一级配置项**：`on_unknown` 可选 `match` / `no-match` / **`fail_request`**，并定义 `DecisionUnresolvedError`——把"信号不可用时怎么办"提升为配置，而不是硬编码 if-else。

**可观测**：`metrics.RecordDecisionMatch(...)`、`RecordDecisionEvaluation(duration)`。

### C. aurelio-labs/semantic-router — 3,862 stars，2026-08-24 仍在更新（Python）

语义路由的 Python 实现：Route（语义簇）由 utterance 向量均值聚类，本地 encoder 可跑 CPU。
关键设计：`Route` 内可设 `score_threshold`，**不达阈值则不路由到任何 route**——即 router 允许"不决策"，交给下游默认值。这提醒我们给 L2 留一个"我不确定"的出口。

---

## 三、方案对比：可取之处与不足

### 3.1 可取

1. **分层（便宜先、贵后）+ 每层可关闭**——方向正确，与 RouteLLM / vllm 的分层哲学一致。
2. **结合检索质量分**——两个参考项目**都没有**这个信号，而本项目**已有现成实现**（RRF 归一化置信度，跨供应商可比）。RAG 场景下"证据是否命中"是难度的最强预测器之一，这是本方案的差异化优势。
3. **保留人工反馈回流**——与项目已有的评估迭代模块天然衔接。

### 3.2 不足（按重要性排序）

| # | 不足 | 依据 / 后果 | 修正 |
|---|---|---|---|
| 1 | **阈值是拍的**（4-5 / 3 / 1-2 无依据） | 无法做连续成本调节 | 抄 RouteLLM 的 `strong-model-pct` 标定法 |
| 2 | **"Score 1-5" 与 "P(strong wins)" 混用** | 离散档无法按成本连续调；两套语义后期难以维护 | 统一为**连续概率 + 单一阈值**，分档只做 UI 展示 |
| 3 | **Layer 3 无成本闸门** | 最坏比全用 Pro 还贵 | 三道闸门（见 §4.3） |
| 4 | **置信度概念混用** | 项目只有检索侧置信度，抓不到"证据强但归纳弱" | 新增生成侧信号（见 §4.3） |
| 5 | **多轮沿用方向性错误** | 追问更难，沿用 Flash 会锁死低档 | 只做降级锁，不做升级锁 |
| 6 | **缺"不决策"出口** | 分类器边缘分数被迫二选一 | 借鉴 aurelio：低置信度区间 → 走 `default_tier` 并标记 `abstain` |
| 7 | **分类器不可用时的路径未定义** | 本项目有 Mock / 限频 429 的真实约束，加载失败不能拖垮服务 | L0 前置闸门 + `degraded` 标记（本沙箱已验证 HF 完全不可达，这是真实风险） |
| 8 | **可观测只有指标名无落点** | 项目已有 trace / stats / `/health` | 直接挂到现有设施（见 §4.5） |

---

## 四、改进设计

### 4.1 关键数据结构

```python
@dataclass
class ModelTier:
    name: str                    # "flash" | "pro"
    model_name: str              # 覆盖 config.LLM_MODEL_NAME
    temperature: float
    max_tokens: int
    timeout: int
    input_price_per_1m: float    # 用于成本核算
    output_price_per_1m: float

@dataclass
class RoutingDecision:
    tier: str                    # flash | pro
    score: float                 # 0~1，越大越该用 Pro（统一为 P(strong wins) 语义）
    threshold: float             # 本次实际生效的阈值
    source: str                  # rule | classifier | cascade | abstain | degraded | default
    reason: str                  # 人类可读：命中哪条规则 / 分类器分多少
    elapsed_ms: int
    rule_name: Optional[str] = None
    degraded: bool = False
    layers_tried: List[str] = field(default_factory=list)
```

### 4.2 四层结构（在方案三层之上补一个 L0）

**L0 前置闸门**（新增）
`ROUTING_ENABLED` / 分类器已加载 / 分类器未超时 —— 任一不满足 → `degraded=True`，直接走 `default_tier`（默认 `flash`）。**保证分类器永远不会拖垮主链路。**

**L1 规则层**（<1ms，独立成表 `_DIFFICULTY_RULES`）

- *硬升档 → pro*：工单号 `T\d{3,}`（工具结果需二次归纳）、多跳句式（`对比|分别|区别|汇总|为什么|如果.+会怎样|例外|前提`）、超长 query（> N 字符）、显式要求（`详细|完整|逐条|分析|总结一下`）。
- *硬降档 → flash*：寒暄 / 致谢（`intent=unknown`）、纯数值直问（"年假多少天"）、Top1 高置信度且单片段命中。
- *会话降级锁*（修正方案 ②）：`last_tier=pro` 且本轮无降档信号 → 沿用 pro；`last_tier=flash` 且本轮有升档信号 → **允许升档**。
- 每条规则带 `confidence`，低于 `_MIN_RULE_CONFIDENCE`(0.6) 视为未命中——与 `intent_router` 现有风格保持一致。

**L2 分类器层**（CPU 20~60ms，可配超时）

统一协议，三种实现可插拔切换：

```python
class ComplexityScorer(Protocol):
    def score(self, query: str, context: dict) -> float: ...   # 0~1，越大越难
```

| 实现 | 依赖 | 说明 |
|---|---|---|
| `prototype` | 现有 embedding API | **冷启动首选**。抄 vllm：YAML 配 hard/easy 候选句，算 margin。零标注、零新依赖 |
| `bert` | transformers + torch | `routellm/bert_gpt4_augmented`（xlm-roberta-base），复刻 BERTRouter 15 行数学 |
| `none` / no-op | 无 | 恒返回 0.5，测试与降级用 |
| ~~轻量 LLM 打分~~ | — | **不推荐**：本项目限频约 3 RPM，会吃掉一半配额 |

**信号融合**——沿用 vllm 的 `max()` 启发：

```python
retrieval_difficulty = 1 - normalized_confidence      # 检索越差，越该升档
final_score = max(classifier_score, retrieval_difficulty)
```

取 max 而非加权和：单一强信号就足以升档，不该被平均掉。

**L3 级联兜底**（带闸门，默认关闭）

- *触发条件（生成侧，非检索侧）*：
  1. 回答出现软拒答词（`没有相关信息|无法回答|不确定`）**且**检索置信度 ≥ `CASCADE_MIN_CONFIDENCE` —— 即"证据有，是模型没用好"，**这是唯一划算的升级场景**；
  2. 引用覆盖率 = 0 但 docs 非空；
  3. 答案异常短（< 20 字且问题 > 30 字）。
- *三道闸门*：`CASCADE_ENABLED`（默认 false）、`CASCADE_MAX_PER_SESSION=1`、`CASCADE_MIN_CONFIDENCE`（低于此值不升级——没证据升了也白升，直接走现有拒答）。
- 升档后**丢弃** Flash 输出，不做拼接（避免"半段 Flash + 半段 Pro"口径不一致）。
- **流式路径默认关闭级联**：token 已推送无法回收。

### 4.3 可配置项（全部环境变量，沿用项目现有风格）

```bash
ROUTING_ENABLED=true
ROUTING_DEFAULT_TIER=flash              # 降级 / 未启用时的档位
ROUTING_STRONG_PCT=0.3                  # 标定目标：Pro 调用占比 → 反解阈值
ROUTING_THRESHOLD=                      # 显式阈值（留空则用标定值）
ROUTING_CLASSIFIER=auto|prototype|bert|none
ROUTING_BERT_CHECKPOINT=routellm/bert_gpt4_augmented
ROUTING_BERT_MAX_LEN=512
ROUTING_CLASSIFIER_TIMEOUT_MS=150       # 超时 → 视为不可用，degraded
ROUTING_USE_RETRIEVAL_SIGNAL=true
ROUTING_STICKY_TIER=true                # 会话降级锁
ROUTING_ABSTAIN_LOW=0.35                # 低于此分 → abstain，走 default_tier
ROUTING_ABSTAIN_HIGH=0.65               # 高于此分 → 直接 pro
CASCADE_ENABLED=false                   # 默认关（成本优先），评估后再开
CASCADE_MAX_PER_SESSION=1
CASCADE_MIN_CONFIDENCE=0.25
LLM_TIER_FLASH_MODEL=                   # 留空回落 LLM_MODEL_NAME
LLM_TIER_PRO_MODEL=
```

### 4.4 失败与超时处理

| 场景 | 行为 |
|---|---|
| 分类器权重下载失败 / 缺包 / HF 不可达 | 记一次 warning，自动降为 `none`，服务照常（**本沙箱已验证 huggingface.co 完全不可达，这是真实风险**） |
| 单次打分超时 | 线程池 + `future.result(timeout=...)` 掐断 → `degraded` 走 default_tier |
| 分类器返回 NaN / 异常 | 视为不可用，同上 |
| Pro 档调用失败（429 / 400） | 现有 `_RateLimitRetryModel` 重试耗尽后降回 flash 再试一次；最终失败走 `human_fallback` |
| 规则层与分类器结论冲突 | 规则优先（可复现、零延迟），但记入 trace 便于复盘 |

所有降级统一标记 `source="degraded"`，暴露在 API 响应与 trace 中。

### 4.5 可观测指标（挂到现有设施，不新建体系）

- **trace**：新增 `model_route` 节点，detail 形如 `tier=pro score=0.72 src=classifier`。
- **路由统计**（仿 `get_route_stats()`）：total / by_rule / by_classifier / by_degraded / by_abstain、**Pro 占比**、规则命中率、分类器 P50/P95 延迟、级联触发率、级联后置信度提升均值。
- **成本**：按 tier 的输入输出 token × 单价累加，输出 `cost_total` 与 `cost_saved_vs_all_pro`。
- **新增 API**：`GET /routing/stats`、`POST /routing/calibrate`（离线标定：输入 query 样本 + 目标占比 → 输出建议阈值）、`POST /routing/eval`（回放标注集 → 输出准确率 / Pro 占比 / 成本）。
- **面板**：现有"智能对话"卡片的路由徽章加一档 tier 标签（flash / pro / ↑级联）。

### 4.6 接入点（严格保持向后兼容）

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `app/core/llm_factory.py` | `get_chat_model(tier: str = None)`，内部按 tier 缓存实例 | **无参调用行为完全不变** → 15/15 回归不受影响 |
| `app/core/model_router.py`（新建） | `route_model(query, state) -> RoutingDecision` | 与 `intent_router.route()` 并列，互不干扰 |
| `app/graph/state.py` | 新增 `model_tier` / `route_decision` 字段 | TypedDict `total=False`，向后兼容 |
| `app/graph/nodes.py` | 新增 `model_route_node`，插在 `intent_recognize` 之后、`generate_answer` 之前 | 图结构只增节点 |
| `app/rag/generator.py` | 三个函数增加可选 `tier` 参数，默认 None → 用全局模型 | 现有调用零改动 |
| `app/api/chat.py` | 响应体新增 `model_tier` 字段 | **只增不改**，前端可忽略 |

---

## 五、推荐实现路径（分阶段，每阶段独立验收）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P0** | 骨架：`ModelTier` / `RoutingDecision`、`get_chat_model(tier=)`、`model_route_node`、`ROUTING_ENABLED=false` 默认关闭、trace 与 stats | 15/15 回归全过，线上行为与现在**完全一致** |
| **P1** | L1 规则层 + 会话降级锁 | 规则覆盖率 ≥ 60%，规则路由延迟 < 1ms |
| **P2** | L2 分类器：**先 `prototype`**（zero-shot，用现有 embedding，**无需下载模型、零新依赖**），跑通评分与阈值标定；**再上 `bert`** 作为可选增强。两者实现同一 `ComplexityScorer` 协议 | **准入门槛**：50 条明显简单 + 50 条明显复杂的中文 query，复杂组得分均值必须显著高于简单组（验证方向性，尤其是 BERT checkpoint 的标签语义） |
| **P3** | 阈值标定 + 评测脚本 + `/routing/*` 接口 | 100 条标注集上给出 Pro 占比 / 质量对比 / 成本对比 |
| **P4**（可选） | L3 级联（默认关闭）+ 面板展示 | 级联触发率 < 10%，级联后置信度确有提升 |

### 关于「pip install routellm」的建议

**不装。** 三条理由：
1. 包最后发布 2024-07-08，约 2 年未更新；
2. 依赖 torch + transformers + datasets + litellm + openai + sklearn + pandas，会让 223MB venv 膨胀到 3GB+；
3. 真正需要的只有 `BERTRouter` 里 15 行 softmax 数学。

替代方案：直接 `transformers` + `torch`（后续可转 ONNX Runtime 瘦身到 ~15MB 运行时 + 550MB 权重），自己写 40 行 loader，行为完全可控、依赖可裁剪。

> ⚠️ **环境限制提示**：本沙箱 `huggingface.co` **完全不可达**（HTTP 000），`pip` 走 SSL 也被证书拦截。BERT 分支的权重下载需要在能连 HF 的环境进行，或用 `hf-mirror.com` 镜像（本次 config.json 就是通过镜像核实到底模为 `xlm-roberta-base` 的）。`prototype` 分支不受此限制——这也是把它排在前面的现实理由。

---

## 六、待你确认的问题

1. **Flash / Pro 具体指哪两个模型？**（当前 `.env` 只配了一个 `LLM_MODEL_NAME`，需要你给出档位对应的模型名与大致单价）
2. **优化目标偏向**：成本优先（Pro 占比 20-30%）还是质量优先（Pro 占比 50%+）？这直接决定 `ROUTING_STRONG_PCT` 的标定目标。
3. **BERT 分支是否纳入首版**？若你的部署环境能连 HF，建议 P2 一并做；否则先只上 `prototype`（零依赖，用现有 embedding 服务即可）。
4. **L3 级联**是否纳入首版（默认关闭，P4 再评估）？
5. **是否需要保留人工反馈回流**（点赞/点踩 → 训练数据落盘）？项目已有评估迭代模块，可复用但需新增存储。
