# 开发日志（整合版）

> **这是什么**：把 `docs/history/` 下 25 篇过程性文档，按时间线与主题压成一份可通读的日志。
>
> **为什么整合**：这些文档原先散在 `docs/` 根目录，其中 19 篇的开头都带着
> 「本文写作于单 Agent 架构时期，所述模块已随多 Agent 重构删除」的过期声明。
> 它们对**当时的决策**有价值，对**理解现在的代码**只有干扰。整合后：
> 一份日志告诉你"为什么是今天这个样子"，原文留档供追溯细节。
>
> **怎么用**：想知道当前架构 → 读 [`../multi-agent-architecture.md`](../multi-agent-architecture.md)
> 与 [`../project-introduction.md`](../project-introduction.md)；
> 想知道某个设计为什么长这样 → 在本日志里搜关键词，再跳原文。

本项目的演进可分五个阶段。**关键转折是第 3 阶段**：单 Agent 拆成五 Agent 后，
前两阶段的大量文档整体失效——本日志的主要价值就是把这批"过期但有用"的东西讲清楚。

---

## 阶段 1 · 切分改造（2026-09-05 ~ 09-08）

当时的核心问题：企业制度文档结构松散，固定长度切分把"第 4.2 条"和它的前提切散，
检索回来的是半句话。

| 文档 | 解决了什么 | 当前状态 |
|---|---|---|
| [`文档切分方案设计.md`](文档切分方案设计.md) | v1.0 方案：从检索链路完整性出发评估切分策略 | 方案已落地为 `structure` 策略 |
| [`切分改造任务清单.md`](切分改造任务清单.md) | 任务拆解，2026-09-07 现场重新核实代码状态 | 已完成 |
| [`chunking-ab-report.md`](chunking-ab-report.md) | **离线影子对比**：A=`recursive` 基线 ｜ D=`structure`+上下文头+元数据增强。跑完索引已重建为 D 组 | 结论即现行默认配置 |
| [`CHANGELOG-chunking.md`](CHANGELOG-chunking.md) | **无 git 环境下的变更追溯替代方案**，配 `scripts/backup.sh <任务号> <文件...>`，回退 `cp -R artifacts/backup/<任务号>/. .` | 已被 git 取代，但 backup.sh 仍在用 |

**遗留影响**：`structure` 切分策略、`docs/chunking-contract.md` 的「冻结项」契约、
以及 `scripts/chunking_ab.py` 都是这一阶段的产物。契约本身仍是**现行有效**文档。

---

## 阶段 2 · 单 Agent 架构的全面评审（约 2026-09-04 ~ 09-11）

一口气做了五份审查 + 两份对标调研。**这一阶段的结论质量最高，
但也最容易被误读为"在说现在的代码"** —— 它们评的是**单 Agent 版本**。

### 2.1 价值与定位

| 文档 | 核心结论 |
|---|---|
| [`feasibility-and-value.md`](feasibility-and-value.md) | **「项目有价值，但价值主张与真实价值点错位。」** 真正的价值在**「可嵌入的 RAG 引擎」**（检索质量、降级链、成本控制三项经得起实测）；对外主张的「打通企业内部业务系统」恰恰是唯一未经验证、且存在合规阻断的一环。所以"有没有价值"不能一句话回答——**换个用途，答案就翻转**。实测环境：`kimi-k2.6` + `BAAI/bge-m3`，12 篇语料 / 176 条片段 |
| [`value-remediation-plan.md`](value-remediation-plan.md) | 把上一条转成**三项可批准动议**，按"能否立即做"排序。例如 `README.md` 写"词面倒排索引 174 个键"、实测 176（P1-1 键同源修复后未同步） |

### 2.2 质量审查

| 文档 | 核心结论 |
|---|---|
| [`project-assessment.md`](project-assessment.md) | 全面评估。其 **P0-1** 成了后续多处代码注释的引用锚点：**可恢复故障不该写进 `error_msg` 从而把抖动升级成人工转接** |
| [`production-readiness-review.md`](production-readiness-review.md) | 五维判定：功能完成度 🟢 / 代码质量 🟢 / 测试覆盖 🟡（**缺真实模型与真实向量库的端到端冒烟**）/ 配置健壮性 🟡 / 可观测性 🔴（**无 metrics 导出、无告警**）。其 **P0-5** 记录了一个很典型的坑：CI 的文档行号 job 实际是红灯（64 处不一致），**首次 push 后必定失败**，且报错会淹没在 64 条输出里 |
| [`redundancy-review.md`](redundancy-review.md) | 冗余与过度特化审查。抓到一条结构性隐患：**流式链路绕过编译图，自己重写了一遍意图分支**（`app/api/chat.py`）——同一套分支逻辑存在两份实现 |
| [`deadcode-governance.md`](deadcode-governance.md) | 死代码治理。关键发现：本项目有**三处框架约定导致的隐式引用**，纯名字统计的通用工具必然大面积误报（vulture 在 60% 档报 50+ 项，约一半是假的）。**结论是自造 AST 扫描器，而不是继续调 vulture 参数** |
| [`module-inventory.md`](module-inventory.md) | 全量模块盘点（2026-09-11 时点快照）。抓到三个 `to_dict` 因"同名在别处出现过"或"写进了 `__all__`"被误判为已使用，**为此新增 `scripts/refgraph_scan.py`（限定名引用分析）** |
| [`可优化.md`](可优化.md) | 针对**重排序 / 可观测性 / 并发**三个维度的补充调研，读 dify 源码作为参考证据 |

### 2.3 对标业界

| 文档 | 核心结论 |
|---|---|
| [`health-check-benchmark.md`](health-check-benchmark.md)（2026-09-04） | 对标 dify / ragflow / LiteLLM / Langflow + Kubernetes 探针规范。**最锋利的一条**：本项目"70 秒启动自检 + `get_chat_model()` 里的 `model.invoke('ping')`"，在四个对标项目里**找不到同类实现**——即这是自己发明的、且代价明确的坏模式 |
| [`rag-architecture-benchmark.md`](rag-architecture-benchmark.md) | RAG 选型对标。选型标准三条：**检索链路完整性**（解析→切分→索引→召回→重排→生成→评测是否闭环）、代码规范性、社区活跃度 |

### 2.4 架构设计的两次探索（均已被取代）

| 文档 | 为什么被取代 |
|---|---|
| [`dynamic-routing-design.md`](dynamic-routing-design.md) | 动态路由（Flash/Pro 自动选型）。**机制已全部删除**，归宿：`model-routing-redesign` → `multi-agent-architecture` |
| [`model-routing-redesign.md`](model-routing-redesign.md) | 档位路由重设计。**核心洞察值得记住**：根因是「**把"规则"当成了"决策"**」——一条正则命中就 `return`，于是 88% 的请求被规则短路、分类器只看得到剩下 12%；而 **44% 的请求竟由「≤14 字 → Flash」这条伪信号决定**。最终档位机制整体删除（改用不限流模型 + 原生 function calling） |

---

## 阶段 3 · 两次线上故障复盘

这两份文档的方法论价值高于结论本身——**在无 git、无 diff 的条件下，
靠日志与实测把因果链查清**。

| 文档 | 症状 | 定位到的根因 |
|---|---|---|
| [`biz-correctness-incident.md`](biz-correctness-incident.md) | ① 正常问答被拒绝回答 ② 工具调用一直加载、长时间无响应 | 无 diff 可比对，结论全部由日志与实测支撑 |
| [`incomplete-slow-answer-incident.md`](incomplete-slow-answer-incident.md) | 回答不完整 + 很慢（与上一条**无重叠**） | 默认模型 `kimi-k2.6` 在正文之前会先产出一段**隐藏的** `reasoning_content`——不计入输出却占用时间 |

**遗留影响**：TTFT 埋点（`StreamStats` 记 `ttft_ms`/`chunks`/`chars`）就是这两轮故障的产物，
用来把"排队/网络慢"与"模型吐字慢"区分开。

---

## 阶段 4 · 意图路由的反复重构

这是全项目**推翻次数最多**的一块，值得单独看它的演进逻辑。

### 4.1 问题从哪来

`intent-routing-redesign.md` 记录的最初故障：故障是**两个缺陷叠加**，
其中一个是一张 how-to 标志表把 **「在哪」当成了 how-to 标志词**，
于是「张三**在哪**个部门」被整句否决、又被知识规则捞走 → 用户问某个人的部门，
拿到的是一段制度说明。

`intent-routing-hardening-plan.md` 记录了它的升级版病症：
问句「四月在哪个部门」出现 `intent_type = knowledge` 但 `intent_capability = employee_lookup`
—— **两个字段自相矛盾**。这就是后来那条铁律的来源：
**判据必须区分"提到某个词"与"要取这个值"，只靠关键词表做不到**。

`tool-invocation-online-vs-offline.md` 与 `offline-slot-extraction-migration.md` 关心的是
下一个问题：判对了意图之后，**工具参数从哪来**。后者有一句很清醒的判断——
离线规则抽取是**四层约束叠出来的**，不是因为"大家觉得该这么做"，
而且**四层里只有一层真的失效了**（所以改造要精准，不该整体推倒）。

### 4.2 工具调用为什么"物理上不可用"

[`tool-calling-enablement-plan.md`](tool-calling-enablement-plan.md) 的结论很典型：

> function calling 在本项目**不是"没选择用"，而是"物理上不可用"**。

根因是自造的限流重试包装器 `_RateLimitRetryModel`（`app/providers/llm.py`）
只实现了 `_llm_type` / `_generate` / `_stream`，**没有实现 `bind_tools`**，
调用直接 `NotImplementedError`。教训：**在 LangChain 上自己包一层，
等于自己承担抽象契约的完整性**。

### 4.3 最终归宿

```
dynamic-routing-design  ─┐
model-routing-redesign  ─┼─→  多 Agent 重构：档位机制整体删除，
intent-routing-redesign ─┘    路由改为「路由 Agent 一次模型调用五选一」
                                        ↓
                            现行：四层混合路由（docs/intent-routing-hybrid-design.md）
```

> 现行设计见 [`../intent-routing-hybrid-design.md`](../intent-routing-hybrid-design.md)，
> 其核心是「**惰性升级**」：词面能判就直接出结论、一次 embedding 都不算；
> 判不了才升语义，语义还判不了才进灰区问模型。**兜底用 LLM，但不轻易用。**

---

## 阶段 5 · P0 落地与单 Agent 档案归位

| 文档 | 说明 |
|---|---|
| [`p0-implementation-design.md`](p0-implementation-design.md) | P0 七项落地方案。方案里参考了 nanobot 的扁平分组（`agent/ bus/ config/ cron/ session/ utils/`） |
| [`项目学习指南.md`](项目学习指南.md) | 965 行的**面试准备文档**——把项目从"能跑通"讲到"能讲清"。与代码无关，纯个人用途 |

### 阶段 5 的架构分水岭

单 Agent → **五 Agent 协作**（路由 / 闲聊 / 简单 RAG / 复杂 RAG / 工具）。
这次重构删掉了大量模块（自建路由 6 个文件、自建工具 3 个、工单与身份模块），
**本目录 19 篇文档的过期声明，指的就是这次删除**。

被删源码存档在仓库根的 `_archive/`（**已加入 `.gitignore`，不随仓库分发**）。

---

## 附：这些文档教给我们的、至今仍适用的事

从上面所有内容里，提炼出跨阶段都成立的几条：

1. **静默失效最危险。** 死代码扫描漏报、路由规则"从来不生效"、字段自相矛盾——
   共同点都是**不报错**。所以关键性质必须有测试主动钉住，而不是靠代码正确。
2. **通用工具会误报，因为框架有隐式引用。** vulture 报 50+ 项约一半是假的；
   正解是补一层 AST 限定名分析，不是调参数。
3. **自造包装层 = 自担抽象契约。** `_RateLimitRetryModel` 没实现 `bind_tools`，
   直接让整条 function calling 链路不可用。
4. **规则是调制器，不是决策器。** "正则命中即返回"会短路掉 88% 的流量，
   还给伪信号（≤14 字）留下了决策权。
5. **文档过期比没有文档更贵。** 19 篇带着过期声明的文档散在 `docs/` 根目录，
   会让接手人把历史当成现状——这正是本次整合的直接原因。
