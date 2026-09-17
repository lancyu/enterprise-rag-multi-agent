# 文档索引

本目录分两层：**根目录只放"描述当前系统"的文档**，`history/` 放过程性记录。

判断规则很简单：**这篇文档描述的是"现在的代码"，还是"当初为什么这么做"？**
前者留根目录，后者进 `history/`。

---

## 一、当前有效（描述现行架构）

按建议阅读顺序排列。

| 文档 | 讲什么 | 什么时候读 |
|---|---|---|
| [`../README.md`](../README.md) | 项目是什么、怎么跑起来 | **从这里开始** |
| [`project-introduction.md`](project-introduction.md) | **带精确行号的代码地图 + 小白原理讲解**。每个模块在哪个文件的第几行 | 要改代码前先查这里定位 |
| [`multi-agent-architecture.md`](multi-agent-architecture.md) | 五 Agent 协作架构（路由 / 闲聊 / 简单 RAG / 复杂 RAG / 工具） | 理解整体编排 |
| [`intent-routing-hybrid-design.md`](intent-routing-hybrid-design.md) | 四层混合路由**设计稿 + 实现后记**：惰性升级、时间预算、不谎报三字段、词表从例句反推（**§十一必读**，见下方告示） | 理解路由判据**为什么**这么定 |
| [`tool-json-schema.md`](tool-json-schema.md) | 工具 Agent 的实现与 Function Calling JSON Schema | 增删工具时 |
| [`deployment.md`](deployment.md) | 部署与运维手册，**含"它不能怎么跑"** | 上线 / 排障 |
| [`chunking-contract.md`](chunking-contract.md) | 切分契约。**违反「冻结项」= 破坏性变更** | 改切分逻辑前必读 |
| [`dify-integration.md`](dify-integration.md) | Dify External Knowledge API 接入契约 | 接 Dify 时 |
| [`project-review-and-improvement-plan.md`](project-review-and-improvement-plan.md) | **全面审查（2026-09-17 时点快照）**：目录 / 职责与重复实现 / 依赖环 / 数据流 / 同类项目对标 / 与标准 RAG 的差距 + P0~P2 改进清单 | 要动结构或做优化前，先看这份找依据 |

> ⚠️ `project-introduction.md` 的行号由 `scripts/verify_doc_linenos.py` 校验，
> **改 `app/` 下任何文件的行数都会让它过期**。流程见该文档自身说明。

> ⚠️ **`intent-routing-hybrid-design.md` 是「实施前」的设计稿 + 「实施后」的复盘，不是现状描述。**
> 它的**漏斗设计**（惰性升级 / 时间预算 / 门控判据）与代码一致，但三处会误导：
> ① 它反复提到的字段 `scene_capability` **最终没有落地**——实现成了 `scene`
> （答"该由谁干"）+ `intent_capability`（答"答案怎么来的"）；
> ② 文中的行号与执行预期是写作当时的快照，`verify_doc_linenos.py` 手动指定它时会报
> **3 处**（都是"非符号区间"，校验器认不出、不是漂移；人工复核过仍指对位置）；
> ③ **别再引用它标着"已修正"的短路径行号当现状**——它没有任何自动机制守着。
>
> 👉 **读它请先读 §十一「实现后记」（2026-09-16）**：那里写清了实现期与原稿的
> 四处偏离、判对率的实测变化（零 LLM 下 9/29 → 20/29）、阈值重标的标定数据，
> 以及一件容易被误读成"已生效"的事——**四层漏斗至今尚未接入生产链路**。
>
> 看现状请看 `multi-agent-architecture.md`；看这份是为了理解**当初为什么这么设计**。

---

## 二、历史记录（`history/`）

**先读这一份**：

| 文档 | 讲什么 |
|---|---|
| [`history/DEVELOPMENT-LOG.md`](history/DEVELOPMENT-LOG.md) | **开发日志（整合版）**——25 篇过程性文档按五个阶段压成一份可通读的日志，含每篇的核心结论与"为什么被取代" |

`history/` 下其余 25 篇是**原文留档**。它们的共同点是：描述的架构已经不是现在的架构。
其中 15 篇开头带这样的声明：

> ⚠️ 本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除。

**它们不该被当成现状来读**，但值得保留——里面记录的是「当时为什么这么选」
「踩了什么坑」「怎么定位的」，这类信息在代码里是找不到的。

### ⚠️ 读 `history/` 时最容易踩的一个坑：把「当时的外部约束」当成现状

这批文档写作时，上游是 Moonshot 的**测试账号**（`RPM=3`、「连续两次调用要隔 20 秒」、
「免费档」）。后来的模型选型、提示词、脚本默认参数里都留过**以它为由**的论证，
连 `scripts/*.py` 的默认休眠 20 秒都曾是照它定的。

**那个账号早已换掉**——现用火山方舟（`ark.cn-beijing.volces.com` + `glm-5-2-260617`），
这个配额不存在了。所以凡是读到 `RPM=3` / 「限频账号」/「每分钟 3 次」/「配额吃紧」，
都请当成**「当时为什么这么设计」的语境**，不要当成现状依据。

还有一处极易混淆：**`RATE_LIMIT_PER_MINUTE=20` 与上游配额毫无关系**。
它是本服务**对自己的入站限流**（按 IP 限用户请求数）——数字 20 也是当年从上游 RPM
倒推出来的，如今同样没有标定依据。

带这类描述的文档开头已加了一行指向本节的提示。**只在 `history/` 内加提示、
不改写原文**——改写历史记录等于篡改历史，那些句子本身是准确的现场记录。
唯一例外是 `data/开发者API接入指南.md`：那句「免费额度每分钟 3 次」是**被检索的语料**，
改了会动 RAG 语义，所以**一个字都不动**。

现行约束请看 [`../README.md`](../README.md) 与 [`deployment.md`](deployment.md)。

按主题分布：

| 主题 | 文档 |
|---|---|
| 切分改造 | `文档切分方案设计.md`、`切分改造任务清单.md`、`chunking-ab-report.md`、`CHANGELOG-chunking.md` |
| 质量审查 | `project-assessment.md`、`production-readiness-review.md`、`redundancy-review.md`、`deadcode-governance.md`、`module-inventory.md`、`可优化.md` |
| 价值评估 | `feasibility-and-value.md`、`value-remediation-plan.md` |
| 故障复盘 | `biz-correctness-incident.md`、`incomplete-slow-answer-incident.md` |
| 业界对标 | `health-check-benchmark.md`、`rag-architecture-benchmark.md` |
| 路由演进 | `dynamic-routing-design.md`、`model-routing-redesign.md`、`intent-routing-redesign.md`、`intent-routing-hardening-plan.md` |
| 工具调用 | `tool-calling-enablement-plan.md`、`tool-invocation-online-vs-offline.md`、`offline-slot-extraction-migration.md` |
| 落地设计 | `p0-implementation-design.md` |
| 个人用途 | `项目学习指南.md`（面试准备，965 行） |

---

## 三、还没写、但值得写的

诚实记一笔，避免"文档看起来很全"的错觉：

- **端到端冒烟测试**：目前单元/契约测试充分，但**缺真实模型 + 真实向量库的端到端冒烟**。
- **可观测性**：有 `/health` 与 trace，**无 metrics 导出、无告警**。
- **多实例部署**：会话 pin、路由统计、内存向量库、词面索引**全是进程内状态**；
  多副本下的失效表现已写进 `deployment.md`，但没有外置状态的方案。
