# 项目审查与改进清单

> **时点快照：2026-09-17**。本文描述的是**当时的**代码状态，结论与行号都会随代码改动过期。
>
> 审查方式：AST 静态解析（未 import 项目模块）+ 文件系统遍历 + 实跑项目自带脚本 +
> 对标调研（直取同类项目源码，未用 GitHub API）。**未修改任何被测代码。**
>
> ⚠️ 文中的行号引用**未持续纳入门禁 ④**——`scripts/verify_doc_linenos.py` 默认只校验
> `docs/project-introduction.md`。手动传本文也能过，但那只是「当时对得上」，没人守住它。
> 所以：能点符号名的地方就点符号名（**符号名不随代码增删行失效**），行号只作定位参考。
> 反面例子是「文件名后跟一段行号区间」那种写法——它一旦漂移还是界内，看不出错。
> 
> 本文若已完成使命（条目做完或方案被推翻），按 `docs/README.md` 的规则移入 `docs/history/`。

---

## 0. 结论摘要

**先说不是问题的地方**（避免把优点也当债来还）：

- **分层方向是干净的**：`app/api/` 是最外层、`app/tools/` 是最内层，**没有任何一层反向依赖 `app/api/`**（全仓 0 条边）。
- **两条链路没有逻辑分叉**：非流式与流式共用同一个 `_wire()` 装配函数、同一批节点函数、同一个 `build_generation_inputs()`。这是很多项目做不到的。
- **测试与门禁的密度远高于同规模自研项目**：466 个用例 / 8535 行测试、四道本地门禁、文档行号校验器已覆盖 13 类声明 598 条。
- **检索栈的骨架比多数同类开源项目完整**：混合召回 + RRF + 父子块 + 来源 ACL 落地 + 记忆分层 + 意图路由，Dify 在这些点上并不比它强。

**真正的问题集中在三处**：

| # | 问题 | 一句话概括 |
|---|---|---|
| **A** | **同一语义反复定义，且已发生行为漂移** | 13 类重复实现里，至少有 4 类**两份副本的行为已经不同**（布尔解析、后缀白名单、鉴权解析、历史裁剪）——这是项目自己记过「已犯 4 次」的那个缺陷族的**现存实例** |
| **B** | **分层靠约定，不靠约束；质量门禁靠自觉，不靠平台** | 没有 CI、没有 pre-commit、没有分层契约。四道门禁全部依赖「本地记得跑」，而它们在同一个工作区里跑，一次漏跑就等于没有 |
| **C** | **门禁只守「行号对得上」，不守「话是对的」** | 行号校验器把 598 条声明守到全绿的同时，文档里赫然写着**一个不存在的函数名** `answer_route_edge`，并把 3 条条件边说成 4 条——**语义一致性没有任何机制把关** |

---

## 一、文件目录结构

### 1.1 事实

```
langgraph-enterprise-bot/
├── app/            78 个 .py    15934 行   ← 源码
├── tests/          30 个 .py     8535 行   ← 测试（466 个用例）
├── scripts/        12 个 .py     4081 行   ← 一次性/门禁脚本
├── docs/           34 个（30 .md）         ← 文档，含 history/ 22 篇
├── data/           13 个文件（12 语料 + enterprise.db）  ← 有意入库
└── 产物类：artifacts/ logs/ vector_store/ memory_store/ _archive/ .pytest_tmp/
                 ↑ 全部已被 .gitignore 覆盖，git ls-files 为 0 条
```

`git status --porcelain --untracked-files=all` **输出为空**：当前版本库里没有不该入库的东西。

### 1.2 组织是否合理

**合理的地方**：目录名与职责基本一一对应（`api` / `graph` / `rag` / `memory` / `db` / `providers` / `tools` / `utils`），
按「处理阶段」切分（与 RAGFlow、Dify 一致），比按技术类型切（`models/ controllers/ services/`）更适合 RAG 这种流水线式系统。

**不合理的 4 处**（都是「位置」问题，不是「缺失」问题）：

| 问题 | 事实 | 为什么是问题 |
|---|---|---|
| 工具脚本混进应用包 | `app/static/gen_favicon.py`（148 行，PNG/ICO 光栅化） | 既不属 `app/` 的任何职责，也不被任何模块 import（死代码扫描报 `unreferenced_module`）；它应该在 `scripts/` |
| 「兼容 shim」散在三个层 | `app/core/llm_factory.py`(23 行) → `providers/llm`；`app/utils/embedding.py`(22 行) → `providers/embeddings`；`app/rag/rerank.py`(8 行) → `providers/rerank` | **同一类事物（转发层）放在三个不同的包**。其中 `rag/rerank.py` 只有 1 个生产消费者，价值最低 |
| 五层 RAG 有两个出口 | `app/core/rag_engine.py`(109 行) 与 `app/rag/__init__.py`(68 行) 都是五层的对外门面 | 改一个入口要同时想两个；新人不知道该从哪个进 |
| 备份目录无界增长 | `artifacts/backup/doc-linenos-*` 快照，**4 天内 21 份**，`artifacts/` 共 25 MB | 已 gitignore 不影响仓库，但本地全文检索/编辑器索引会被旧版源码副本污染 |

---

## 二、模块职责与重复实现

### 2.1 职责过载（单文件承载多种无关职责）

| 文件 | 行数 | 混在一起的职责 | 证据 |
|---|---|---|---|
| `app/api/chat.py` | 461 | HTTP 端点 + 响应体契约 + 业务编排（invoke 图）+ 持久化 + 记忆整理 + 日志 | `chat_ask` 一个函数里同时出现 `@router.post`、`create_initial_state`、`enterprise_workflow.invoke`、`save_message`、`maybe_consolidate` |
| `app/core/tool_agent.py` | 694 | 提示词 + **手写 JSON 大括号配对扫描器** + 工具 schema 校验 + 多轮决策循环 + 文本形态回捞 + 护栏 | `_iter_json_objects`、`parse_text_tool_calls`、`execute_tool_calls`、`_decide` |
| `app/db/vector_db.py` | 667 | 3 个后端实现 + 原子文件写 + 进程内索引 + **工厂降级决策** + Milvus 代理网络诊断 | `MemoryVectorStore` / `ChromaVectorStore` / `MilvusVectorStore` / `_loopback_proxy_hint` |
| `app/core/self_check.py` | 347 | 自检框架（**三份名字集合**）+ 9 项检查 + 直接 SQL + **硬编码业务探针问句** | `FAST_ITEMS` / `DEEP_ITEMS` / `_collect_checks` 三处各写一遍清单；`_check_retrieval` 里硬编码 `"年假有多少天"` |
| `app/api/dify.py` | 376 | HTTP + **另一套 Bearer 鉴权** + 分数归一化数学 + metadata 白名单 + **10 个运算符的条件过滤引擎** + **75 行手写 OpenAPI** | `_check_auth`、`_normalize_score`、`_match_conditions`、`dify_openapi` |

**反例（职责单一，可作为其他模块的样板）**：`app/graph/edges.py`(95)、`app/core/routing/similarity.py`(138)、`app/rag/reorder.py`(28)、`app/utils/logger.py`(74)。

### 2.2 重复实现清单（13 类）

> 缺陷族「**同一语义在两处各定义一遍**」是项目自己在 `docs/history/` 里记过、并明确写过「已犯 4 次」的问题。
> 下面**全部是复核过的现存实例**。逐字相同已用 sha1 比对函数体确认。

| 编号 | 同一语义 | 副本数 | 位置 | 严重度 |
|---|---|---|---|---|
| **D-1** | 中文虚词表 `_CJK_STOP` + `_cjk_runs()` + 两个字符集 | **2 处逐字相同**（75 字符的集合内容 sha1 一致） | `app/rag/retriever.py:62` / `app/rag/lexical.py:42` | **高** |
| **D-2** | 支持的文件后缀白名单 | **3 处，形态还不同** | `app/utils/doc_loader.py:20`（集合，带点）/ `app/rag/prepare.py:95`（元组，带点）/ `app/api/knowledge.py:85`（集合，**不带点**） | **高** |
| **D-3** | RRF 归一化上界公式 `(w_d+w_l)/(k+1)` | **2 处，各算一遍** | `app/api/dify.py:85` / `app/rag/generator.py:164` | 中 |
| **D-4** | 中文分词与字符集 | **3 套风格** | `lexical.py` / `retriever.py`（剔虚词）vs `app/providers/embeddings.py:30`（不剔）；汉字区间还有 `一-鿿` 与 `\u4e00-\u9fff` 两种写法 | 中 |
| **D-5** | `_content_of(message)` + `_default_model()` | **各 4 份，函数体 sha1 完全相同** | `app/core/tool_agent.py:693` / `app/core/sub_agents.py:336` / `app/core/router_agent.py:357` / `app/core/routing/arbitration.py:180` | **高** |
| **D-6** | 问候/致谢/告别正则 + 长度上限 12 | **2 份逐字相同** | `router_agent._GREETING_RE/_THANKS_RE/_BYE_RE/_IDENTITY_RE` 与 `anchors` 里的同名四条逐字一致（`anchors` 的注释自己写着「与 router_agent.py 逐字一致」），`_FALLBACK_MAX_CHARS` 与 `_MAX_CHARS` 也各写一遍 | 中 |
| **D-7** | `_format_history` | 3 处，默认值不同 | `app/memory/short_term.py:64`（唯一实现）/ `generator.py:191`（已收敛为委托 ✅）/ `router_agent.py:346`（**仍是独立实现**，且自带魔数 `max_turns=4`） | 中 |
| **D-8** | 配置布尔解析 | **1 个函数 + 8 处内联** | `_env_bool`（`app/config.py:25`，**把 `"none"` 当假**）vs 内联版（`config.py:67,85,99,428,440,462,474,482`，**不认 `none`**） | **高** |
| **D-9** | 取文件名（basename） | **8 处** | 只有 `prepare.py:112` 与 `generator.py:126` 处理 `\`；`retriever.py:490`、`evaluator.py:171`、`chat.py:92`、`dify.py:127/245`、`knowledge.py:156` 都不处理 | 中 |
| **D-10** | 文档内容长度上限 50000 | **2 处，语义相反** | `app/utils/validator.py:17`（**超限报错**，硬编码）vs `config.py:618` + `api/knowledge.py:126`（**超限截断**，走配置） | 中 |
| **D-11** | Bearer / API Key 解析 | **2 套规则不同** | `app/main.py:111`（`X-API-Key` 优先 + `startswith("bearer ")`）vs `app/api/dify.py:183`（只认 `Authorization` + `split(None,1)`）。`"Bearer\txxx"` 在两处的判定**相反** | 中 |
| **D-12** | 健康检查端点 | 2 个，鉴权豁免不一致 | `GET /health`（在 `AUTH_EXEMPT_PATHS` 内）vs `GET /test/health`（**不在**） | 低 |
| **D-13** | 魔数 `min(len(history)//2 + 1, 10)` | 2 处 | `app/api/chat.py:240` 与 `app/api/chat.py:403`，无命名常量 | 低 |

**为什么 D-8 / D-10 / D-11 标「高」**：它们不只是重复，而是**两份副本的行为已经不同**。
`MEMORY_ENABLED=none` 与 `CHUNK_CONTEXT_HEADER=none` 结果相反、同一份文档走两个上传
入口一个拒收一个截断、同一句「谢谢」/同一个 token 在两条路径上判定相反——
这类问题**不会报错，只会让行为随机地取决于走了哪条代码路径**，与项目历史上踩过的坑同型。
（D-11 的相反判定已在修复时**实测确认**：`"Bearer\txxx"` 在 Dify 端点放行、在入站中间件 401。）

> **⚠️ 本表的一处自我订正（2026-09-18）**：D-6 原先把 ``app/core/sub_agents.py`` 那组同名正则
> 也算作重复，**这是错的**。该模块 112-115 行自己写明「与路由 Agent 的兜底正则不是同一件事」：
> 那边回答"已经在闲聊了，挑哪句回复更像话"（宽松无害），这边回答"要不要把这个句子划进闲聊"
> （错判会把业务问题打发掉）。**有意的偏离不该被写成缺陷**——真正的重复是
> `router_agent` ↔ `anchors` 那一对（逐字相同）。修 D-6 时只收敛了这一对，
> 并在 `sub_agents` 处补了一句"别顺手并过去"。

---

## 三、依赖关系

### 3.1 包级方向（好的一面）

```
app/api   → core / db / graph / memory / providers / rag / utils        ← 最外层
app/core  → db / graph / memory / providers / rag / tools / utils      ← 唯一的八向汇聚点
app/tools → db                                                          ← 最内层（只有 1 条出边）
任何包    → app/api                                                     ← 0 条边 ✅
```

扇入 Top：`app/config.py`(58 个文件)、`app/utils/logger.py`(42)、`app/utils/doc_loader.py`(13)、`app/rag/prepare.py`(13)。

### 3.2 环与跨层回边

| 类型 | 实例 | 判断 |
|---|---|---|
| **真双向循环** | `app/utils/logger.py:22` ↔ `app/core/tracing.py:103`（logger 要 trace_id 拼前缀，tracing 要 logger 打日志） | 需要拆：把「取 trace_id」下沉到一个无依赖的小模块 |
| **配置中心反向依赖能力层** | `app/config.py:245,254,377,386` 延迟 import `app.utils.embedding.get_embedding_mode`，而 `utils/embedding` → `providers/embeddings` → `config` | **这是 SCC-1 的根**。`config` 本该是最底层，现在依赖了模型提供层 |
| `core` ↔ `rag` 双向 | `core→rag` 1 行（`rag_engine.py:14`）；`rag→core` 7 处（`generator.py:31,300`、`retriever.py:28,142,143`、`evaluator.py:30`） | 五层 RAG 并非对 `core` 无感：L4 生成依赖 `core.prompts` / `core.llm_factory`，L3 检索依赖 `core.request_ctx` |
| `core` → `graph` 回边 | `app/core/self_check.py:14,108` import 编译图与 `create_initial_state`，而 `graph→core` 有 16 处 | 自检的 12 项里只有 2 项需要整张图，其余可以留在 core |
| `rag` → `memory` | `app/rag/generator.py:205`：L4 直接依赖短期记忆包 | 与「五层 RAG 是独立流水线」的说法不符 |
| 包 `__init__` 自环 | `memory/`、`rag/`、`core/routing/` 三处「`__init__` re-export 自己的子模块」 | Python 常见模式，当前能跑通，但**初始化顺序成了隐式契约** |

**关键判断：这些环目前都不致命，但没有一条被机制守住。** 项目对分层的唯一保障是目录名与文档叙述；
下一个人只要 import 一下，方向就破了，而四道门禁**没有一道会因此变红**。

---

## 四、数据流与调用链

### 4.1 四条链路（都读得通，这是加分项）

| 链路 | 入口 | 路径 | 说明 |
|---|---|---|---|
| **主链路** | `POST /chat/ask` | 中间件（鉴权 → 限流 → trace）→ `api/chat.py` → LangGraph 9 节点 → 五层 RAG → 响应 | 清晰，职责边界明确 |
| **流式** | `POST /chat/ask/stream` | 共用前置，在证据出口分叉：跑**前置子图**（8 节点）后由端点自己驱动生成 | `_wire(generation_target=...)` 同源装配，**避免了双链路逻辑分叉** |
| **Dify 兼容** | `POST /retrieval` | `api/dify.py` → **直接进 L3 `retriever.retrieve`** | 跳过图、路由、L1/L2、L4 受控生成 |
| **工具 Agent** | 图内 `tool_node` | `core/tool_agent.py` → `tools/sqlite_tools` → SQLite(只读) | 作用域在**调用工具的同一个节点**里建立（跨 context 读不到证据） |

条件边**只有 3 个**：`scene_route_edge`（五路）、`tool_route_edge`（四去向）、`error_route_edge`（转人工）。

### 4.2 四个不一致点（事实陈述，是否有意需作者判断）

1. **异常信息对外暴露不一致**：非流式刻意脱敏，只给 trace_id（`app/api/chat.py` 的 `chat_ask` 异常分支）；
   流式却把 `str(exc)` 直接拼进 SSE 事件（`app/api/chat.py` 的 `chat_ask_stream`）。**同一服务两种口径**。
2. **Dify 链路绕过了几乎所有中间层**：不写会话历史、不写长期记忆、不受 L4 拒答策略约束，
   且 `/dify/retrieval` **不在限流豁免元组内**——它会被按 IP 限流，而它通常是机器调用。
3. **上传路径与重建路径的 PDF 解析不是同一套**：`app/api/knowledge.py` 的 `knowledge_upload_file` 用 `tempfile` + `PyPDFLoader`（一级）；
   `app/utils/doc_loader.py` 有三级降级（pdfplumber 表格识别 → pypdf → 图片 OCR）。
   **同一个 PDF 走两条路径会得到不同文本**。
4. **`GET /dify/info` 与 `GET /health` 都豁免鉴权，但 `test/health` 不豁免**（见 D-12）。

---

## 五、与同类开源项目的对标

对标对象：RAGFlow、Dify、chat-langchain（**已改版为托管 MDA，不含本地索引**，改用 LangChain 官方
`rag-research-agent-template` / `retrieval-agent-template` 替代）、LlamaIndex、Haystack、Onyx、kotaemon。
（版本为 2026-09-17 抓取的默认分支；未核实项下文已标注。）

### 5.1 最反直觉的一条：**四家主流实现没有一家默认用 RRF**

| 实现 | 融合方式 | 出处 |
|---|---|---|
| RAGFlow | `FusionExpr("weighted_sum")`，term/vector 双权重 | `rag/nlp/search.py` |
| Dify | `WEIGHTED_SCORE`：jieba TF-IDF 余弦 + 向量余弦加权 | `api/core/rag/rerank/weight_rerank.py` |
| Onyx | Vespa `global-phase` 内 `normalize_linear(bm25)` 与 `normalize_linear(closeness)` 按 `query(alpha)` 加权 | `danswer_chunk.sd.jinja` |
| Haystack | 支持 RRF（**k=61**）但**默认 `CONCATENATE`** | `components/joiners/document_joiner.py` |
| LlamaIndex | 4 种模式**默认 `SIMPLE`**，RRF 是可选（k=60） | `core/retrievers/fusion_retriever.py` |
| **本项目** | **RRF（不可切换）** | `app/rag/retriever.py` `rrf_fuse` |

**读法**：RRF 的优势是免调参、跨分值尺度鲁棒——对本项目「词面分是 Dice∈[0,1]、向量分尺度随模型漂移」
这个处境其实是**对的**选择。真正的差距是：**人家把权重做成了运行时参数（可调、可 A/B），本项目写死了。**

### 5.2 关键环节横向对比

| 环节 | 业界做法 | 本项目 | 差距 |
|---|---|---|---|
| 文档解析 | RAGFlow 自建 DeepDoc（OCR + 版面 10 类元素 + 表格结构识别 TSR），PDF 解析器**可插拔 10 种**；LlamaIndex 把深度解析做成独立产品 | 基于 Markdown 结构切分；PDF 三级降级（pdfplumber → pypdf → OCR） | **大**：无版面分析、无表格结构识别 |
| 切分 | RAGFlow **模板制**（15 种按文档类型）+ 切分结果**可视化供人工干预** | 结构感知 + 递归降级 + A/B 脚本 + 指标脚本 | **小**：机制齐全，只是缺可视化 |
| 混合召回 | RRF 或加权；Onyx 备**两个 rank-profile**（语义主导 / 关键词主导），确保两路各自取够候选 | RRF，三路并发召回 | 中 |
| rerank | 分两派：本地混合打分（RAGFlow/Dify）与神经 cross-encoder；**默认多关** | `app/rag/rerank.py` **仅 8 行转发 shim**，实质是空壳 | **大** |
| 多查询 | LangGraph 模板用 `Send` 并行扇出多查询 | 有复杂 RAG 的子问题拆解，但检索侧无多查询 | 中 |
| 父子块 | LlamaIndex `relational/`（Hierarchical / AutoMerging） | 有 `parent_store`（自研） | 小 |
| 评测 | RAGFlow 跑 ndcg@10/map@5/mrr@10，**按指标升序输出最差 query**；LlamaIndex 有完整检索指标库 + **从文档自动造评测集**；Onyx 把 **ACL 身份**纳入评测 | `app/rag/evaluator.py`(383 行) + 离线脚本；**无检索侧指标** | **大** |
| 增量索引 | LlamaIndex `DocstoreStrategy` 三态 + node hash 去重 + ingest cache；RAGFlow 按 doc_id 全量删 + 内容 hash 变更检测 | 有 chunk key 不变量（解决 key 唯一性），**无对账机制** | **大** |
| ACL 粒度 | Onyx 文档级 + 外部身份（邮箱/组），**在检索层过滤**；Dify 数据集级 + 成员授权表 + RBAC；RAGFlow 知识库级 | **来源级**（`app/core/source_acl.py`，48 行，fail-closed） | 中 |
| 可观测性 | 三派：自研埋点+多后端（LlamaIndex/Onyx/Dify）、平台绑定（LangChain 系）、自研 span 树写文件（**本项目独家**） | 自研 span 树 → `logs/trace.jsonl` | 中：**无脱敏层** |
| CI | 主流项目**全部有 CI**；Dify 37 个 workflow；**RAGFlow/Dify/Haystack/Onyx/kotaemon 都有 `.pre-commit-config.yaml`**；RAGFlow/Dify/Haystack/Onyx 都有 `AGENTS.md` + `CLAUDE.md` | **无 CI、无 pre-commit**（四道门禁全靠本地手动串行） | **大** |
| 分层约束 | **Dify 用 `api/.importlinter` 把 4 层架构写成断言**，且 `unmatched_ignore_imports_alerting = error`（豁免只减不增） | 靠目录名与文档叙述 | **大** |

### 5.3 三条可直接落地的具体经验（有公开踩坑记录）

1. **喂给神经 reranker 的必须是自然文本，不能是分词后的文本**（RAGFlow `rag/nlp/search.py` 注释）——
   否则相关度分会整体崩塌，你会被迫把阈值压到荒谬的低值，表现是「rerank 开了但没效果」。
2. **混合检索不要用单路相似度做前置阈值过滤**（Dify issue #35233）——
   会在融合之前丢掉字面完美匹配但向量分低的片段。**阈值只应作用于融合后的最终分**。
3. **rerank 会破坏分页语义**（RAGFlow 把它建模成参数不变量：开 rerank 时强制 `page=1`，并校验候选池
   `rerank_candidates_count > page*page_size`）——本项目检索侧目前没有同类不变量。

---

## 六、与「标准 RAG 流水线」的差距

以工业界标准流水线为标尺逐环节打分（★ 越多越强，满分 5）：

| 环节 | 打分 | 一句话 |
|---|---|---|
| 文档加载 | ★★★☆ | 三级降级齐全，但只对 PDF；上传路径还是另一套 |
| 深度解析（版面/表格/OCR） | ★★☆ | 无版面分析，表格靠 pdfplumber 尽力而为 |
| 切分 | ★★★★ | 结构感知 + 短块合并 + A/B + 指标，**这一环是强项** |
| 索引 | ★★★☆ | 向量 + 词面双写，key 不变量守得住；缺内容指纹 |
| 召回 | ★★★★ | 三路并发 + RRF + ACL 落地 + 父子块，**骨架完整** |
| 精排（rerank） | ★☆ | **空壳** |
| 生成 | ★★★★ | 引用编号 + 置信度拒答 + 截断可见 + 优雅降级 |
| 记忆 | ★★★★ | 短期窗口 + 长期四层 + 蒸馏，自研项目中少见 |
| 路由 | ★★★★ | 多层漏斗 + 本地快通道 + 模型路由，设计有据 |
| 评测 | ★★ | 有生成侧评测，**无检索侧指标、无回归基线** |
| 可观测 | ★★★ | 自研 span 树，轻量可用；无 metrics、无告警、无脱敏 |
| 权限 | ★★★☆ | 入站 fail-closed + 来源 ACL；无文档级、无评测覆盖 |
| 增量与对账 | ★★ | 有重建，无「重跑会不会多一份」的对账 |
| 工程（CI/门禁） | ★★ | 门禁设计精良，**但无平台承载** |

**一句话**：**「检索栈」的完成度接近同类开源项目，「工程闭环」的完成度明显落后。**

---

## 七、改进清单（按优先级）

> 每条给出：**问题 → 证据 → 方向 → 验收标准**。优先级依据：**会不会出事故 / 会不会继续腐化**，
> 不按工作量排。P0 共 5 条，建议**逐条做完再进 P1**（P0-5 例外，可随时做）。

### P0 —— 正确性与安全性

#### P0-1 收敛会漂移的重复实现（先修「两份行为不同」的 4 类）
- **证据**：D-6（正则锚定不一致）、D-8（`_env_bool` vs 8 处内联，`none` 判定相反）、
  D-10（同一上限一份报错一份截断）、D-11（Bearer 解析两套规则）。
  另有 D-1/D-2/D-3/D-5 属「逐字复制」，是下一次漂移的种子。
- **方向**：按「**唯一实现 + 其余全部委托**」收敛。优先级：
  ① `_CJK_STOP`/分词下沉到 `app/rag/lexical.py`（或新建 `app/utils/text.py`）；
  ② `_content_of`/`_default_model` 各留一份（放 `app/providers/`），其余 3 处 import；
  ③ 布尔解析统一走 `_env_bool`（先确认内联的 8 处是否**有意**不认 `none`——若有意，就把它做成 `_env_bool(..., none_is_false=True)` 的显式参数）；
  ④ 后缀白名单、basename、RRF 上界、鉴权解析各留一处。
- **验收**：每类重复**只剩 1 处定义**；新增**反向验证护栏**（把修复退回，对应测试必须变红——
  这是项目已在别处用过的有效手法）；`pytest` 全绿 + `ruff` 全绿。

#### P0-2 统一异常对外口径
- **证据**：`app/api/chat.py` 的 `chat_ask`（脱敏，只给 trace_id）vs `chat_ask_stream`（回显 `str(exc)`）。
- **方向**：流式与非流式共用同一个「对外错误文案」构造；原始异常只进 `logs/`。
- **验收**：护栏断言「SSE 的 error 事件里**不含**异常类名/堆栈/path」。

#### P0-3 上传路径与重建路径的 PDF 解析对齐
- **证据**：`app/api/knowledge.py` 的 `knowledge_upload_file`（一级 `PyPDFLoader`）vs `app/utils/doc_loader.py`（三级降级）。
- **方向**：上传路径改为调用 `doc_loader` 的同一函数；若确实要「上传求快」，把它做成
  **显式的解析档位参数**并写进文档，而不是让两条路径各自实现。
- **验收**：同一个 PDF 经两条路径产出的文本**逐字相同**（写一条测试守它）。

#### P0-4 trace 落盘前加脱敏层
- **证据**：`logs/trace.jsonl` 会记 query 原文与检索片段；入站鉴权已 fail-closed、来源 ACL 已隔离，
  但 trace 本身没有脱敏。参考 Onyx 的 `tracing/masking.py`。
- **方向**：在 `app/core/tracing.py` 的 `_persist()` 前加一层 mask（密钥、邮箱、手机号、
  可选「正文只留长度」开关），并把开关做成配置项。
- **验收**：开关打开时，trace 里搜不到测试用的假密钥/邮箱/手机号。

#### P0-5 修掉文档里「话不对」的地方，并给语义加一道最小守卫
- **证据**（都已核实）：
  - `docs/project-introduction.md:168` 点名了一个**代码里不存在的函数** `answer_route_edge`
    （实际是 `error_route_edge`）；
  - 同一文档 `:50` 写「8 个 + 2 个条件分支」，实际是 **9 节点 / 3 条件边**；
  - `:167` 与 `README.md:216` 都写「四个条件分支」，而 `README.md` **自己的图里只标了 ①②③**，
    `app/graph/edges.py:3` 也自述「三条条件边」。**同一仓库 5 种说法。**
- **方向**：① 先统一口径（建议以代码为准：9 节点 / 3 条件边）；
  ② **给校验器加第 14 类：文档里出现的 `xxx_edge` / `xxx_node` 等符号名，必须在代码里真实存在**
  （这正好补上「行了号对了、名字却不存在」这个语义盲区；实现成本低，因为 AST 索引现成）。
- **验收**：第 14 类上线后，把 `answer_route_edge` 写回文档，校验器必须变红；
  四道门禁全绿，且声明数显著上升（覆盖面无变化则说明没命中）。

### P1 —— 工程闭环与检索质量

#### P1-1 建 CI（**单项收益最大的一条**）
- **证据**：无 `.github/`、无 `.pre-commit-config.yaml`；四道门禁全靠本地手跑。
- **方向**：先上**最小可用版**——一个 workflow 串行跑四道门禁。按 LangGraph 模板的做法，
  **单测与集成测试拆成两个 job**（集成测试起外部依赖，失败原因不同，混在一起会让人对红灯麻木）。
  本地侧同时加 `.pre-commit-config.yaml`（跑 ruff + 快速单测）。
- **验收**：故意制造一次失败（改一个测试断言）→ CI 变红；恢复 → 变绿。

#### P1-2 用 `import-linter` 把分层写成可执行契约
- **证据**：`core ↔ rag` 双向、`config → utils.embedding` 反向、`core → graph` 回边，**当前无一被机制拦住**；
  Dify 的 `api/.importlinter` 是现成范本。
- **方向**：定义 `api → core → {rag, graph, memory, db, providers} → utils` 的层次契约，
  开启 `unmatched_ignore_imports_alerting = error`（豁免只减不增）。
- **验收**：故意在 `app/rag/` 里 import `app.api` → 契约检查变红。

#### P1-3 rerank 从空壳变成真实现（连同三个工程约束一起做）
- **证据**：`app/rag/rerank.py` 8 行；`app/providers/rerank.py` 有懒加载实现但未被启用。
- **方向**：① 输入用**原始 chunk 文本**（不用分词结果）；② **阈值只作用于融合后的最终分**；
  ③ 定义 `rerank_candidates_count` 默认 64，并把它与分页的关系建模成不变量。
- **验收**：对同一批问句做 A/B（项目已有 `scripts/probe_routing.py` 与 chunking A/B 的范式），
  rerank 开/关在 **NDCG 或命中率**上有可复现的差异，且有基线快照。

#### P1-4 补检索侧指标与基线
- **证据**：有 `app/rag/evaluator.py`（生成侧）与 `scripts/baseline_snapshot.py`，
  但**没有 HitRate / MRR / Recall / NDCG**；生成指标差时无法区分「没召回到」与「召回到了没用上」。
- **方向**：在离线脚本里补检索指标；参考 RAGFlow 的**逐 query 按指标升序输出最差样本**——
  看一个平均分远不如看最差的 10 条。
- **验收**：`scripts/eval_generation.py` 或新增脚本能输出「指标 + 最差 N 条 query」，
  并与 `baseline_snapshot.py` 的基线做 diff。

#### P1-5 增量索引的对账
- **证据**：有 `chunk_index` 全局递增不变量（解决 key 唯一性），但**重跑一次会不会多一份、源删了索引里还在不在**没有机制。
- **方向**：引入 `DocstoreStrategy` 三态语义（UPSERTS / DUPLICATES_ONLY / UPSERTS_AND_DELETE）
  + 内容指纹；**配置不自洽时降级并告警**，不静默。
- **验收**：同语料连跑两次 `build_index()`，向量库与词面索引条数**不变**；
  删掉一个源文件后重建，其片段**全部消失**。

#### P1-6 解除配置中心的反向依赖
- **证据**：`app/config.py:245,254,377,386` → `app.utils.embedding.get_embedding_mode`；
  形成 `config → utils.embedding → providers.embeddings → config` 的环。
- **方向**：把「当前 embedding 模式」变成一个由**启动时注入**的普通值（或独立的 `app/runtime_flags.py`，
  零依赖），`config` 只读它、不 import 业务层。
- **验收**：`config.py` 里不再出现任何 `app.` 内部的 import；SCC-1 消失。

#### P1-7 给文档行号校验器补回归测试（**门禁自己没有护栏**）
- **证据**（本次审查期间新发现）：`scripts/verify_doc_linenos.py` 是**唯一**守着 598 条行号声明的机制，
  但 `tests/` 下 26 个 `test_*.py` **没有任何一个** import 或调用它——全仓只有
  `scripts/fix_doc_linenos.py` 引用它。它自己 13 类声明的验证，历来靠**一次性变异脚本**
  （写进 `/tmp/`，跑完即弃），第 11、12、13 类都是这么验的。
- **本轮实测代价**：给第 13 类加「找不到定位标记就报错」这条守卫时，`config_section` 前置条件
  漏写，立刻产生误报——而这**只有手工拿另一份文档去跑才暴露**。没有任何自动化会在那一刻变红。
  换句话说：**能守住别人的机制，自己无人守**，这正是报告开头 C 条同一类问题。
- **方向**：把变异用例固化成 `tests/test_doc_linenos.py`。两个设计要点（都来自本轮踩坑）：
  ① 变异施加在**文档副本**上，真文档一字不动；
  ② 按**标记子串**定位目标行，且载体找不到时 **fail 而不是 skip**——skip 会让用例
  在文档结构变化后静默失效，正是本项目的「护栏恒真」坑。
  该文件还会顺手把「四道门禁」变回三道本地命令（校验器进 pytest）。
- **验收**：把第 13 类的 `config_section` 前置条件删掉（即退回修复）→ 新增用例必须变红；
  这是本项目已在别处用过的反向验证手法。

### P2 —— 结构与体验

| 编号 | 事项 | 证据 / 方向 | 验收 |
|---|---|---|---|
| P2-1 | 工具脚本归位 | `app/static/gen_favicon.py` → `scripts/` | 死代码扫描不再报 `unreferenced_module` |
| P2-2 | shim 归位 | 3 个转发 shim 统一到一处（或直接删除、改调用方） | `app/core/llm_factory.py` / `app/utils/embedding.py` / `app/rag/rerank.py` 至少删掉其中 1 个 |
| P2-3 | 五层 RAG 只留一个出口 | `core/rag_engine.py` 与 `rag/__init__.py` 二选一 | 文档里只出现一个入口名 |
| P2-4 | 多查询并行检索 | LangGraph 原生 `Send` 扇出（同技术栈，改动最小） | 复杂 RAG 在 NDCG 上有提升，且延迟不线性增长 |
| P2-5 | ACL 纳入评测 | 参考 Onyx `evals/eval_cli.py --search-permissions-email`；本项目只需给评测脚本加 `--as-user` | 存在用例「用户 A 检索不到源 B」，且纳入回归 |
| P2-6 | 排序加入反馈/时间因子 | 参考 Onyx 的 `document_boost` × `recency_bias` | 老文档不再长期霸榜（可观测到排序变化） |
| P2-7 | 低相关度显式告警 | 参考 kotaemon「检索结果低相关时给用户告警」 | 低分召回时前端有提示，而不是照常生成 |
| P2-8 | `artifacts/backup/` 保留策略 | 4 天 21 份快照，无界增长 | 只保留最近 N 份 / 按天去重 |
| P2-9 | 核对 RRF 的 k | 项目用 `1/(k+rank)`；Haystack 注释指出论文的 60（1-based）应对应代码里的 **61**（0-based） | 确认 rank 起点后决定是否改为 61，并记录理由 |
| P2-10 | 统一「条件边/节点数」的表述 | 见 P0-5；`docs/history/` 里的旧说法**不改写**，只在 `docs/README.md` 加提示 | 现行文档（README + project-introduction + multi-agent-architecture）口径一致 |

---

## 八、建议的执行批次

| 批次 | 内容 | 为什么这个顺序 |
|---|---|---|
| **第 1 批** | **P1-1 建 CI** + **P1-7 校验器补测试** + **P0-5 语义守卫（第 14 类）** | 这三条是「放大器」：CI 让后续所有改动的验证成本降一个数量级；P1-7 让校验器的每次改动都有回归兜住（否则加第 14 类本身就是又一次「靠手工验」）；第 14 类让文档语义首次被机制守住。先做它们，后面每条都更省力 |
| **第 2 批** | P0-1 ~ P0-4（重复收敛、异常口径、PDF 对齐、trace 脱敏） | 都是一次性、可验收、不带设计争议的修复 |
| **第 3 批** | P1-2 分层契约 → P1-3 rerank → P1-4 检索指标 → P1-5 增量对账 → P1-6 解环 | 前一条为后一条提供验证手段（没有指标就做不了 rerank 的 A/B） |
| **随时** | P2-* | 结构整理与体验优化，可穿插进行 |

### 执行进度（2026-09-18 起追加）

> 本节是**唯一**会随执行更新的部分；上面的条目正文保持在审查时点的原样，
> 便于回看"当时看到的是什么"。

| 项 | 状态 | 说明 |
|---|---|---|
| P0-1 | ✅ 已完成 | 13 类重复实现全部收敛为"唯一实现 + 其余委托"。4 类行为漂移按**更严格的一侧**钉住，每项都补了**反向验证过**的护栏（把修复退回 → 对应用例变红，已逐项实测）。新增两处实现 `app/utils/text.py`、`app/core/llm_access.py`、`app/utils/auth_header.py` |
| P0-2 | ✅ 已完成 | 对外故障文案收敛到 `app/core/errors.py::public_detail`（固定前缀 + trace_id）。**实际修复面比审查时估计的大**：除 SSE 回显 `str(exc)` 外，还堵掉两条绕过异常分支的走私通道 —— `/workflow/execute` 原样透出内部 `error_msg`、`human_fallback_node` 把 `error_msg[:60]` 塞进随响应体下发的 `trace.detail`。新增 `tests/test_error_boundary.py`（11 项），逐条反向验证过；另加两条结构约束（边界层不绑定异常对象、凭证只经 `trace_ref` 取） |
| P0-3 | ⏳ 待做 | |
| P0-4 | ⏳ 待做 | |
| P1-1 | ⚠️ **方向已订正** | 原写"建 CI"，与本仓库 2026-09-14 的**既定决定**（本仓库不开源，不重建 `.github/`、不建 pre-commit）直接冲突。改为**本地一键门禁**：把行号校验器纳入 pytest（四道 → 三道），单独脚本承载配置分区表那一类 |
| P1-2 ~ P1-6 | ⏳ 待做 | 第 3 批 |
| P0-5 / P1-7 | ⏳ 待做 | 第 1 批遗留 |

**顺手记下的一个工具缺口**（修 P0-1 时踩到）：`scripts/fix_doc_linenos.py`
**处理不了 config 分区表**（校验器第 13 类）。那张表 25 行会随 `app/config.py` 的
任何增删整体错位，而回填脚本对它零覆盖 —— 每次只能按校验器的报错**手工整表重写**。
这是"校验器能查、修不了"的孤例，值得给它补一段回填逻辑。

---

## 附录：本次审查**未覆盖**的部分（诚实记账）

1. **未运行完整测试套件之外的端到端验证**：未启动服务、未对真实模型发请求；
   调用链结论来自静态阅读与函数调用点核对。`466 passed` 为实跑结果。
2. **未审阅 `artifacts/`、`_archive/`、`logs/`、`vector_store/`、`memory_store/` 的内容**，只统计了规模与 git 状态。
3. **未核对 `.env` 与 `.env.example` 的键差异**（`.env` 含密钥，已被 gitignore）。
4. **对标调研中明确未核实的项**：Dify 的增量重建粒度与是否有 RAG 质量评测集、
   RAGFlow 的 benchmark 是否进 CI、Onyx 的评测是否进 CI、LlamaIndex/Haystack/kotaemon 的 CI 内容、
   Verba（本次完全未调研）。这些在下文对标表里均未作为结论使用。
5. **依赖环分析对动态 import 不敏感**（`importlib`、字符串模块名），实际耦合可能被低估。
6. **`deadcode_scan.py` 的引用判定按裸名统计**，同形名会漏报——报告项数是**下界**。
