# 生产就绪与开源发布审查报告

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> 审查日期：2026-09-14
> 审查对象：`langgraph-enterprise-bot`（企业级 LangGraph RAG 智能助手）
> 结论：**功能侧接近就绪，工程/运维/开源侧有明显缺口；且仓库尚未初始化 Git，发布工作实际上还没开始。**
> 本文所有数字均为实测值（附复现命令），非估算。

> **2026-09-14 更新：本仓库不按开源项目发布。**
> 首次审查时补齐的开源外壳（`LICENSE` / `NOTICE` / `CONTRIBUTING.md` /
> `SECURITY.md` / `CODE_OF_CONDUCT.md` / `CHANGELOG.md` / `Makefile` /
> `.editorconfig` / `.gitattributes` / `.pre-commit-config.yaml` / `.github/`）
> **已全部移除**，依赖它们的引用已同步修正。
>
> 移除的只是"外壳"，**门禁本体一个未动**——`tests/`、
> `scripts/verify_doc_linenos.py`、`pyproject.toml` 里的 ruff/vulture/deptry
> 配置、`requirements-dev.txt` 原样保留；移除后三门禁复跑结果：
> **337 passed / ruff 全绿 / 550 条行号声明一致**。
> 下文 §六 的相关条目已标注为「已移除」，其余结论不受影响。

---

## 一、总判定

| 维度 | 状态 | 说明 |
|---|---|---|
| 功能完成度 | 🟢 良好 | 36 个接口、五层 RAG + 8 节点工作流闭环可用 |
| 代码质量 | 🟢 良好 | 337 项测试全绿、ruff 零告警、死代码门禁三层 |
| 测试覆盖 | 🟡 可用但有盲区 | 单元/契约充分，**缺真实模型与真实向量库的端到端冒烟** |
| 配置健壮性 | 🟡 有缺口 | 有配置校验与 fail-closed 设计，但 `.env.example` 漏了 5 个安全项 |
| 可观测性 | 🔴 不足 | 有 `/health` 与 trace，**无 metrics 导出、无告警** |
| 部署运维 | 🟢 已补齐 | `docs/deployment.md` 覆盖部署形态 / 验证 / 守护 / 反代 / 备份 / 回滚；多实例限制已写明 |
| 开源合规 | ⚪ 已按决策撤销 | 首次审查补齐的 Apache-2.0 + 协作模板已于 2026-09-14 全部移除（本仓库不开源）；**若日后要对外分发，需重新选定许可证** |
| 版本管理 | 🔴 阻断 | **不是 Git 仓库**，无历史、无协作、无法回滚 |

**能否上线**：单实例、内网、有专人维护 → 补完 P0 即可上。
面向公网 / 多副本 / 交给他人运维 → P0 + P1 都必须清。

---

## 二、现状盘点（实测）

### 2.1 规模

| 项 | 数值 | 复现方式 |
|---|---|---|
| 应用代码 | 13,968 行 / `app/` | `find app -name '*.py' \| xargs wc -l` |
| 测试代码 | 5,861 行 / `tests/` | `find tests -name '*.py' \| xargs wc -l` |
| 设计文档 | 7,405 行 / 17 篇 `docs/` | `wc -l docs/*.md` |
| API 端点 | 36 个 | 见 2.3 |
| 演示语料 | 12 篇 → 176 条向量片段 | `ls data/ \| wc -l`、`vs.count()` |

### 2.2 质量门禁（当前状态）

```
pytest tests/ -q          →  337 passed, 1 warning
ruff check app scripts tests  →  All checks passed!
未使用代码/依赖扫描        →  零发现（tests/test_deadcode.py）
文档行号回验              →  scripts/verify_doc_linenos.py 通过
```

> 唯一告警：`langchain-community` 已进入 sunset 状态，`app/utils/doc_loader.py:215`
> 仍从它导入 `TextLoader`。不阻断，但属于技术债（见 §四-7）。

### 2.3 接口分布

| 模块 | 端点数 | 模块 | 端点数 |
|---|---|---|---|
| `knowledge.py` | 6 | `evaluation.py` | 4 |
| `memory.py` | 6 | `dify.py` | 4 |
| `routing.py` | 6 | `test.py` | 3 |
| `chat.py` | 5 | `workflow.py` | 2 |

---

## 三、上线差距清单（待办）

### P0 — 阻断项，不做完不能上线

> 这 4 项的共同特征：**后果不可逆或不可见**——出问题时没有挽回窗口。

| # | 事项 | 为什么要命 | 验收标准 |
|---|---|---|---|
| **P0-1** | **初始化 Git 仓库** | 当前 `git rev-parse` 返回 `fatal: not a git repository`。没有版本历史 = 无法回滚、无法协作、无法追溯"哪次改动导致线上异常" | `git log` 有首个提交；`.git` 存在于仓库根 |
| **P0-2** | **确认 `.env` 永不入库** | `.env` 含真实 LLM / Embedding / LangSmith 密钥。`.gitignore` 已正确写死（`.env` / `.env.*` / `!.env.example`），但**仓库尚未 init，一次 `git add -A` 就会把密钥写进历史且删不掉** | `git ls-files \| grep '^\.env$'` 无输出；首次 `git add` 前先跑 `git status` 人工确认 |
| **P0-3** | **补部署与运维文档** | `docs/` 下 17 篇全是设计/复盘文档，**没有任何一篇讲"怎么把它跑起来"**。接手人只能翻 README 的快速开始，遇到进程守护、反向代理、备份恢复、升级回滚全部无据可依 | 新增 `docs/deployment.md`，覆盖 §七 全部内容 |
| **P0-4** | **写清多实例部署的限制** | 会话 pin 表、级联预算、路由统计、内存向量库、词面索引**全部是进程内状态**。多 Worker / 多副本下会出现：会话粘性失效、级联预算被放大 N 倍、`/routing/stats` 数字随机漂移。**当前所有文档对此一字未提**——这是最容易让接手人踩坑的一处 | README 与 `docs/deployment.md` 明确写出"默认单进程；多副本需外置状态"及降级后的表现 |

**P0-2 的补充**：即使 `.gitignore` 正确，也建议在 `git init` 后**先提交一次**，再考虑加 pre-commit 钩子（`.pre-commit-config.yaml` 已就位，含 `detect-private-key`）。

---

### P0 补录 — 本次审查中发现并已修复

> 这 2 项是审查过程中实测发现的**发布阻断项**，已在本次一并修好。

| # | 事项 | 发现 | 处置 |
|---|---|---|---|
| **P0-5** | **CI 的文档行号 job 实际是红灯** | `python scripts/verify_doc_linenos.py` 报 **64 处不一致**。原因是上一轮模型路由重构改动了 8 个文件的行数，而 `docs/project-introduction.md` 的 524 条声明没有同步。**即首次 push 后 CI 必定失败**，且失败信息会淹没在 64 条输出里 | 已按 AST 实际值回填全部 64 处，声明数 524 → **550**，现在退出码 0 |
| **P0-6** | **校验器有第 10 类盲区：散文引用只查越界、不查是否仍指向原符号** | 文末「一次请求的生命周期」等处的 24 条 `app/x.py:A-B` 引用中，**9 条已静默漂移**（如节点④还指向 `model_router.py:526-653`，而主入口 `route_model` 早已移到 `709-848`）。旧规则只校验"范围在文件界内"，漂移后**仍然界内**，因此从未报错 | 已修正 9 处引用，并把「精确命中某个符号」补进 `verify_doc_linenos.py` 作为第 10 类检查（已反向验证：改回旧值会被抓住） |

> **这两条印证了同一件事**：门禁的价值取决于它的覆盖面。
> "零不一致"只有在覆盖面被证明完整之后才是一个结论——
> 一个只认得出部分写法的校验器，会把"文档悄悄变错"伪装成"全部通过"。

---

### P1 — 上线前应完成

| # | 事项 | 证据 / 说明 |
|---|---|---|
| **P1-1** | **修词面索引键冲突** ✅ **已修（09-14）** | 向量库 176 条片段，词面索引只有 **174 个键**，2 条被同名键覆盖。**根因**：`chunk_index` 是「每条 Document 各自从 0 编号」，而 PDF 加载器**每页产出一条 Document、`source` 相同**（那份 2 页白皮书 = 0,1 + 0,1）。**实际修法**（与原建议「改用与向量 id 同源的键」不同）：① `chunk_documents` 改为**按 source 全局递增编号**；② 键的生成抽成 `lexical.chunk_key` **单一函数、dense/lexical 两路共用**；③ 加冲突告警（观测点不拦截）。**未采用内容哈希键**的原因：向量库 id 本就是内容哈希，`SearchResult` 不暴露 id，改用同源键需改 3 个向量库实现 + 落库元数据；而位置键只需修编号即唯一，且更省内存、可读。**验收**：重建后向量库 176 == 词面索引 176，原先被顶掉的白皮书第 2 页片段以最高分召回 |
| **P1-2** | **补 `.env.example` 安全配置段** | 290 行的示例配置里，`AUTH_ENABLED` / `AUTH_API_KEY` / `AUTH_EXEMPT_PATHS` / `CORS_ORIGINS` / `RATE_LIMIT_PER_MINUTE` **一项都没写**（grep 计数为 0；`SOURCE_ACL` 是唯一已写的一项，见 `:259-264`）。结果是：使用者根本不知道有鉴权开关，部署到公网也默认 `AUTH_ENABLED=false`、`CORS_ORIGINS=*` |
| **P1-3** | **加 metrics 导出** | 全仓库无 `prometheus` / `/metrics` 引用。有 `/health` 和 trace，但**没有可供告警的时序指标**（QPS、P95 延迟、拒答率、Pro 占比、降级次数）。上线后只能靠人肉看日志 |
| **P1-4** | **给 compose 的 app 服务加 healthcheck** | `docker-compose.yml` 中 redis / etcd / minio / milvus **都有** healthcheck，唯独 `app` **没有**。这会让 `depends_on` 无法正确等待应用就绪 |
| **P1-5** | **修正 README 过期数据** | README:369 写"知识库：3 篇文档 → 30 条向量片段"，实测是 **12 篇 → 176 条**。对外文档里的数字不符会直接削弱可信度 |
| **P1-6** | **归档根目录草稿** | 根目录有 4 个未被任何文档引用的中文草稿，共 2,590 行（`项目学习指南.md` 962 / `文档切分方案设计.md` 822 / `切分改造任务清单.md` 568 / `CHANGELOG-chunking.md` 238）。既不属于源码也不是入口文档，应移入 `docs/` |
| **P1-7** | **建立容量基线** | 无并发压测、无 P95 延迟基线、无内存占用记录。`scripts/` 下 22 个脚本里没有压测工具。"能跑"和"能扛"是两件事 |
| **P1-8** | **补端到端冒烟** | CI 与测试全部是纯单元判定（`USE_REAL_LLM=false` + 本地哈希向量），**从不验证真实模型 + 真实向量库的组合**。这类问题只有上线当天才会暴露 |

---

### P2 — 可延后

| # | 事项 | 说明 |
|---|---|---|
| P2-1 | CI 增加 `concurrency` 与 `permissions` | 当前 ci.yml 未收敛令牌权限、未取消同分支的重复运行 |
| P2-2 | 加 `CODEOWNERS` | 多人协作后自动指派 Reviewer |
| P2-3 | CHANGELOG 自动化 | 现为手工维护（`CHANGELOG.md` 已建） |
| P2-4 | 替换 sunset 依赖 | `app/utils/doc_loader.py:215` 从 `langchain-community` 导入 `TextLoader`，该包已宣布 sunset |
| P2-5 | `artifacts/` 治理 | 168K，含旧版源码副本与本次隔离的 `uv.lock`，已被 gitignore；建议定期清理 |

---

## 四、项目做得不好的地方

按"会造成多大实际损害"排序，不按修起来难不难。

### 1. 🔴 状态全在进程内，却按无状态服务的方式描述自己

`session pin 表`、`级联预算`、`路由统计`、`内存向量库`、`词面索引` —— 这五处都是模块级单例或进程内字典。
**它们全部在单实例下正确，在多实例下静默出错。**

最危险的是"静默"：多副本部署后服务照常返回 200，只是会话粘性失效、统计数字漂移、级联预算被放大 N 倍。**没有任何报错，只有不正确的行为。**

> 这是整份报告里我最想强调的一条。同类项目（RAGFlow / Dify）都把状态外置到 Redis/DB，本项目把 Redis 用在了缓存但没用在会话状态上。

### 2. 🔴 文档与实现已经漂移

| 位置 | 文档说 | 实际是 |
|---|---|---|
| `README.md`（实测结果章节） | 3 篇文档 → 30 条片段 | 12 篇 → 176 条 |
| `docs/history/project-assessment.md` | P0-4 向量库非原子写 | **已修**（`os.replace` + 一致性校验） |
| `docs/history/project-assessment.md` | P1-3 词面检索 O(N) | **已修**（倒排表 + `_avg_len` 缓存） |
| `.dockerignore` 注释 | 引用 `docs/history/production-readiness-review.md` | 本次审查前**不存在** |

旧评估报告 15 项里有 **2 项已修但报告未更新**、**1 项描述已不准确**。
文档没说谎，只是没人回头改——但对外发布的文档里，没改就等于说谎。

### 3. 🟡 词面索引与向量库的键不同源 —— **已于 09-14 修复**

向量库用内容哈希做 id，词面索引用 `source::chunk_index` 做键。
两套键各自维护，**没有任何一致性校验**，导致 176 → 174 的静默丢失（见 P1-1）。
根因是把"片段身份"定义了两遍。

修复后：`chunk_index` 按 source 全局递增（消除撞键的物理条件），键的生成收敛到
`lexical.chunk_key` 单一函数供 dense / lexical 两路共用（消除"两处各写一遍"的
分叉条件），并在写入侧加了冲突告警。

**元数据口径也已同源（同日补修）**：词面路的 `lexical_meta` 原先另立一份 6 字段
白名单，改为全量继承 `chunk.metadata`。这是同一份"分叉病"的第二处——**键是一处，
字段是另一处**。顺带修掉一个真实缺陷：Dify 外部知识库的 `metadata_condition`
只认对外白名单里的字段，按 `doc_version` / `file_type` 等过滤时条件**恒假**
（`_match_conditions` 取不到值 → `is` 比较恒假），记录被静默丢弃、返回空且不报错，
表现为「知识库明明有这份文档，Dify 却永远检索不到」。

需要注意的是，这次改动**不改变任何现有对外响应**（实测 chat / dify 两处出口
0/176 差异）——它的价值不在"修 bug"，而在消除两处各自定义的字段口径，
以及让 Dify 的 metadata 过滤从"配了也用不了"变为可用。

### 4. 🟡 安全能力默认关闭且无人知晓

鉴权（`AUTH_ENABLED`）、来源 ACL（`SOURCE_ACL`）、限流（`RATE_LIMIT_PER_MINUTE=20`）**代码都实现了**，
设计也是正确的（fail-closed：开了鉴权却没配 Key 会拒绝所有请求）。
但 `.env.example` 里一个字都没提 → **等于没做**。

好的安全设计必须配好的"默认值引导"。当前的默认值引导是零。

### 5. 🟡 无运行时指标，只有事后日志

`/health` 告诉你"现在活着"，`logs/trace.jsonl` 告诉你"刚才做了什么"，
但**没有任何东西能回答"过去一小时 P95 延迟涨了没有""拒答率是不是异常了"**。
这意味着上线后无法做主动运维，只能被动救火。

### 6. 🟡 根目录混入 2,590 行内部草稿

`项目学习指南.md` / `文档切分方案设计.md` / `切分改造任务清单.md` / `CHANGELOG-chunking.md`
既不是源码、也不是项目入口，却和 `README.md`、`LICENSE` 平级躺在根目录。
访客打开仓库第一眼看到的就是这堆草稿。

### 7. 🟢 少量技术债（不阻断）

- `langchain-community` 已 sunset，`TextLoader` 需迁到独立包
- `docs/history/dynamic-routing-design.md` 与 `docs/history/model-routing-redesign.md` 并存，前者已被后者取代但未标注
- `docs/history/可优化.md` 文件名不规范（无主题、无日期）

---

## 五、同类项目对比

选取 6 个同类型项目（RAG 问答 / LLM 应用框架），只对比**工程规范维度**，不比功能。

| 项目 | 许可证 | 测试目录 | CI | 部署编排 | 贡献规范 | 部署文档 |
|---|---|---|---|---|---|---|
| **本项目**（09-14 决策后） | ❌ 已撤回 | ✅ 337 项 | ❌ 已移除 | ✅ compose | ❌ 已移除 | ✅ 已补齐 |
| `infiniflow/ragflow` | Apache-2.0 | ✅ | ✅ | ✅ compose+k8s | ✅ | ✅ |
| `langgenius/dify` | 自定义（基于 Apache-2.0 + 限制） | ✅ | ✅ | ✅ compose+k8s | ✅ | ✅ |
| `chatchat-space/Langchain-Chatchat` | Apache-2.0 | ✅ | ✅ | ✅ compose | ✅ | ✅ |
| `Cinnamon/kotaemon` | Apache-2.0 | ✅ | ✅ | ✅ compose | ✅ | ✅ |
| `open-webui/open-webui` | 自定义（BSD-3 + 品牌条款） | ✅ | ✅ | ✅ | ✅ | ✅ |
| `weaviate/Verba` | BSD-3-Clause | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ |
| `langchain-ai/chat-langchain` | MIT | ⚠️ | ✅ | ❌ 示例量级 | ⚠️ | ⚠️ |

### 从对比里读出的三件事

**① 许可证：Apache-2.0 是这类项目的主流选择。**
6 个成熟项目里 4 个用 Apache-2.0（RAGFlow、Langchain-Chatchat、kotaemon、private-gpt）。
它比 MIT 多一条明确的专利授权，企业用户接受度最高 —— 本项目已采用，正确。

**② 唯一的实质性差距是"部署文档"，不是代码。**
所有 6 个项目都有部署文档，本项目当时没有。这是本次审查里**投入产出比最高的一处**——
README 已有 Docker 快速开始，把它扩写成一篇正经的运维文档，成本很低，收益是"别人能自己跑起来"。
**已于 09-14 补齐**（`docs/deployment.md`），见 P0-3 / P0-4。

**③ 更值得注意的差距：本项目在"代码质量门禁"上反而领先。**
死代码三层扫描（AST 扫描器 + ruff + vulture + deptry）、文档行号回退验证、
`docs/history/health-check-benchmark.md` 这类量化复盘 —— 这些在同类项目里都很少见。
**本项目的问题不在代码写得差，而在"工程外壳"没做完。**

> **说明**：本表是**决策前**的对比快照。2026-09-14 作者决定本仓库不开源，
> 开源外壳（许可证 / CI / 贡献规范）已整体移除，故「本项目」一行的对应列
> 改标为「已撤回 / 已移除」。**唯一实质差距「部署文档」已于同日补齐**，
> 至此与同类项目在仓库结构维度上不再有短板；项目在死代码三层扫描、
> 文档行号回退验证上仍保持领先。
>
> 补充：以上对比只覆盖仓库结构维度。star 数、语言构成等易变指标未纳入，
> 因为它们反映的是项目热度而非工程质量。

---

## 六、GitHub 发布准备清单

### ✅ 仍然保留（工程必需，与开源定位无关）

| 文件 | 作用 |
|---|---|
| `.gitignore` | 覆盖密钥、虚拟环境、缓存、运行时数据、artifacts |
| `.dockerignore` | 构建上下文排除（含密钥与开发工具） |
| `artifacts/quarantine/` | 隔离无用的 `uv.lock`（Python 版本声明自相矛盾、无 `[project]` 段、无任何引用） |

### ⚪ 已移除（开源外壳，2026-09-14 作者决定不开源）

| 文件 | 原作用 | 移除影响 |
|---|---|---|
| `LICENSE` | Apache-2.0 官方全文 | 无许可证 = 默认保留全部权利；**日后对外分发前须重新选定** |
| `NOTICE` | Apache-2.0 第 4 条署名声明 | 随许可证一并撤销 |
| `CONTRIBUTING.md` | 开发环境 / 分支模型 / 提交与 PR 流程 | 无外部协作者，无需 |
| `SECURITY.md` | 漏洞报告渠道 + 上线前安全清单 | **其中的安全清单有独立价值**，如需要可迁入 `docs/` |
| `CODE_OF_CONDUCT.md` | 社区行为准则 | 无外部社区，无需 |
| `CHANGELOG.md` | Keep a Changelog 格式 | 变更记录按需落在 `docs/` |
| `Makefile` | `make help/test/lint/run/docker` 入口 | 命令仍可直接执行（见 §2.2 门禁表） |
| `.editorconfig` / `.gitattributes` | 编辑器与换行符一致性 | 纯编辑体验，无功能影响 |
| `.pre-commit-config.yaml` | ruff / 私钥检测 / YAML 校验钩子 | 钩子消失，**ruff 仍可由命令行执行** |
| `.github/workflows/ci.yml` | test / deadcode / docs 三 job | 自动化消失，**门禁本体仍在**，改为本地串行执行 |
| `.github/dependabot.yml` / `ISSUE_TEMPLATE/*` / `PULL_REQUEST_TEMPLATE.md` | 依赖更新与协作模板 | 无外部协作，无需 |

> 一句话：**没有一项功能或质量门禁因这次移除而失效**——移除的都是外壳，
> 门禁的执行命令见 §2.2，实测三门禁全绿。

**同时顺手修掉的 4 项**（均已在下方 ⬜ 清单中划掉）：

| 项 | 改动 |
|---|---|
| **P0-5** | `docs/project-introduction.md`：按 AST 实际值回填 64 处过期行号，声明数 524 → 550 |
| **P0-6** | `scripts/verify_doc_linenos.py`：新增第 10 类检查（散文引用须精确命中符号）；修正文档中 9 处漂移引用 |
| **P1-2** | `.env.example`：补 `AUTH_ENABLED` / `AUTH_API_KEY` / `AUTH_EXEMPT_PATHS` / `CORS_ORIGINS` / `RATE_LIMIT_PER_MINUTE` 五段，290 → 327 行 |
| **P1-5** | `README.md:369`：`3 篇文档 → 30 条片段` 改为 `12 篇文档 → 176 条向量片段` |
| **P1-6** | 4 个根目录草稿移入 `docs/archive/`，根目录 `.md` 从 9 个减到 5 个 |

> 顺带修正的一处文档不准确：原 §四-2 表格写"`.env.example` 未提及 `SOURCE_ACL`"，
> 实测 `SOURCE_ACL` **已经写在** `.env.example:259-264`，缺的是另外 5 项（见 P1-2）。

### ⬜ 发布前仍需完成

- [ ] **P0-1** `git init` + 首次提交（见 §七）
- [ ] **P0-2** 提交前人工核对 `git status`，确认 `.env` 不在列表
- [ ] **P1-3** 加 metrics 导出
- [ ] **P1-4** docker-compose 的 app 服务补 healthcheck
- [ ] **P1-7** 建立容量基线
- [ ] **P1-8** 补真实模型 + 真实向量库的端到端冒烟
- [ ] 若确需远端托管：确定仓库名与账号（本仓库按**私有**定位，不发布开源、不含 CI）

> 已完成：~~P0-3~~、~~P0-4~~、~~P0-5~~、~~P0-6~~、~~P1-1~~、~~P1-2~~、~~P1-5~~、~~P1-6~~、~~P1-9~~、~~P1-10~~
>
> P0-3 / P0-4 的落点：新增 `docs/deployment.md`——部署形态与单实例限制（五处进程内
> 状态及其多副本后果）、环境准备、启动、**四步部署验证**、systemd 与 Nginx 反代
> （含 SSE 关缓冲、反代下限流 IP 两个坑）、备份恢复、升级回滚、故障速查；
> `README.md` 在 Docker Compose 之后补「部署形态：默认单实例」小节并链到前者。
>
> P1-1 的落点：`app/rag/indexer.py`（按 source 全局编号 + 冲突告警）、
> `app/rag/lexical.py`（新增 `chunk_key` 单一键函数）、
> `app/rag/retriever.py`（`_hit_key` 改为复用该函数）、
> 新增 `tests/test_chunk_keys.py`（7 条）。测试 322 → **329**。
>
> P1-9（两条召回路元数据同源）的落点：`app/rag/indexer.py`（词面 `lexical_meta`
> 由 6 字段白名单改为全量继承 `chunk.metadata`）、`app/api/dify.py`（对外字段
> 白名单抽成 `_build_payload_meta` 并补文档级业务字段，修复 metadata 过滤恒假），
> 新增 `tests/test_meta_align.py`（4 条）+ `tests/test_dify_api.py` 回归（4 条）。
>
> P1-10（测试限流隔离）：`tests/conftest.py` 新增 autouse fixture 重置入站限流器。
> 它是**进程内全局单例**，命中记录跨用例累积——套件里打 HTTP 的用例一多就把
> 20 次/分钟的窗口提前打满，后面的用例拿到 429，表现为「单独跑通过、整套跑失败」
> 的顺序依赖。测试 329 → **337**。

---

## 七、可执行的上线流程

### 阶段 0 · 提交前自查（最重要，只做一次）

```bash
cd /Users/hj1993/WorkBuddy/2026-08-31-18-34-30/langgraph-enterprise-bot

# 1) 确认密钥不会入库 —— 必须先看到 .env 出现在 ignored 列表里
git init
git check-ignore -v .env          # 期望输出：.gitignore:N:.env
git status --short | grep -c '^?? \.env'   # 期望输出：0

# 2) 跑一遍全部门禁（原 Makefile 已随开源外壳移除，直接给命令）
PYTHONPATH=. ./.venv/bin/python -m pytest tests/ -q
./.venv/bin/ruff check app scripts tests --no-cache --output-format concise
./.venv/bin/python scripts/verify_doc_linenos.py
```

> ⚠️ **只要 `git check-ignore .env` 没有输出，立刻停止**，
> 先修 `.gitignore` 再 `git add`。密钥一旦进入历史，重写历史也无法保证被彻底清除。

### 阶段 1 · 首次提交与推送

```bash
# 3) 首次提交
git add -A
git status --short | head -40      # 再核对一次，确认没有 .env / .venv / vector_store
git commit -m "chore: 初始化仓库并完成发布前审查修复"

# 4) 推到远端（如为私有托管，先在网页端建空仓库，不要勾选初始化 README/LICENSE）
git branch -M main
git remote add origin https://github.com/<your-account>/langgraph-enterprise-bot.git
git push -u origin main
```

### 阶段 2 · 门禁复跑（无 CI，改为本地执行）

```bash
# 5) 提交后在**干净的工作区**再跑一遍，确认提交进去的代码本身是绿的
PYTHONPATH=. ./.venv/bin/python -m pytest tests/ -q
./.venv/bin/ruff check app scripts tests --no-cache --output-format concise
./.venv/bin/python scripts/verify_doc_linenos.py
```

三道不全绿就不要进阶段 3 —— 说明有测试或门禁在本地环境下没被触发。
（原 `.github/workflows/ci.yml` 已随开源外壳移除，门禁改为本地纪律，勿省。）

### 阶段 3 · 部署（单实例，推荐路线）

```bash
# 6) 准备配置
cp .env.example .env
#    必填：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME
#    公网部署必填：AUTH_ENABLED=true、AUTH_API_KEY=<强随机串>、CORS_ORIGINS=<你的域名>

# 7) 启动（二选一）
docker compose up -d              # 默认只起 app + redis
# 用 Milvus：VECTOR_DB_TYPE=milvus docker compose --profile milvus up -d
# 完整部署步骤（含验证、守护、反代、备份、回滚）见 docs/deployment.md
# 或
PYTHONPATH=. ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

# 8) 验证真的连上了（而不是悄悄降级）
curl --noproxy '*' http://localhost:8001/health
python scripts/check_vector_db.py  # 确认不是内存降级
```

### 阶段 4 · 上线后必做

```bash
# 9) 建索引
curl -X POST http://localhost:8001/knowledge/rebuild

# 10) 跑一次真实模型的端到端冒烟（单测覆盖不到这一段）
curl -X POST http://localhost:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"年假有多少天"}'
```

**回滚预案**（当前尚无，上线前必须准备）：

```bash
git log --oneline -5              # 找到上一个可用提交
git revert <bad-commit>           # 或 git reset --hard <good-commit>（仅在无人依赖时用）
docker compose down && docker compose up -d
```

> 注意：向量库持久化在 `vector_store/`，回滚代码不会回滚索引。
> 若本次改动涉及切分逻辑，需同时重建索引（`POST /knowledge/rebuild`）。

---

## 八、建议的执行顺序

```
第 1 步（今天，1 小时）
  P0-2 确认密钥不外泄  →  P0-1 git init + 首次提交  →  P0-3 写部署文档

第 2 步（本周，半天）—— ✅ 已全部完成
  ~~P1-1 词面键冲突~~  ~~P1-2 .env.example 安全段~~  ~~P1-5 README 数据~~  ~~P1-6 归档草稿~~

第 3 步（上线前）
  P1-3 metrics  →  P1-4 healthcheck  →  P1-8 端到端冒烟  →  P1-7 容量基线

第 4 步（上线后按需）
  P2 全部
```

**如果只能做一件事**：先做 P0-2（确认密钥不入库）。
其余所有问题的代价都是"要返工"，只有这条的代价是"泄漏不可逆"。

---

## 附：本次审查的证据文件

| 文件 | 说明 |
|---|---|
| `docs/history/project-assessment.md` | 2026-09-09 的上一轮评估（部分条目已过期，见 §四-2） |
| `docs/history/deadcode-governance.md` | 死代码治理机制说明 |
| `docs/history/health-check-benchmark.md` | 启动自检耗时基线与优化记录 |
| `docs/history/redundancy-review.md` | 冗余评审记录 |
