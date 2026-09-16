# 全量模块盘点与无用代码判定（2026-09-11）

> ⚠️ **本文是 2026-09-11 的**时点快照**，不是当前模块清单。**
> 此后项目经历了一次单 Agent → 五 Agent 协作架构的重构
> （见 `docs/multi-agent-architecture.md`），模块集合已变化：
> 本文提到的 `core/intent_router.py` / `model_router.py` / `cascade.py` /
> `smalltalk.py` / `chat_memory.py` 与 `tools/{rule,ticket,user}_tool.py`
> **均已删除**（存档见 `_archive/`），新增了 `core/router_agent.py` /
> `sub_agents.py` / `tool_agent.py`、`tools/sqlite_tools.py`、
> `db/enterprise_db.py`。
>
> 那次重构里短暂存在的 `core/identity.py`（会话身份 → 工号）与 `query_work_order`
> 工具**随后也一并删除**（产品决定：不做登录态、不做工单），存档见
> `_archive/removed-workorder-and-identity-20260915/`。当前只有 3 个只读工具。
>
> **当前模块与行号的权威来源是 `docs/project-introduction.md`**——
> 它由 `scripts/verify_doc_linenos.py` 逐条回验（退出码 0），不会悄悄变错。
> 本文保留的价值在于**判定方法与结论**（五路门禁、限定名引用分析），
> 阅读时请把模块名当作"当时的现场"。
>
> 目的：把项目里「有什么」和「哪些没人用」一次性说清楚。
> 方法：全量模块清单 + 五路门禁 + 新增的限定名引用分析（补扫描器的撞名漏报）。

---

## 0. 摘要

| 项 | 数量 |
|---|---|
| Python 模块总数 | **96**（`app/` 69 + `scripts/` 6 + `tests/` 21） |
| 代码总行数 | **17 201**（app 11 359 / scripts 2 224 / tests 3 618） |
| 五路门禁 | 全绿（pytest 147 / ruff / vulture / deptry / 扫描器 3 项全在 allowlist） |
| **新判定为无用** | **9 项**（本轮新发现，门禁此前看不见） |
| 严格模式遗留待判 | 6 项（上一轮已列出，仍未处理） |

**本轮最重要的发现**：门禁不是没报，而是**报不出来**——`prompts.render`、`ShortTermMemory`、
三个 `to_dict` 全部因为「同名名字在别处出现过」或「名字写进了 `__all__`」被判为已使用。
为此新增了 `scripts/refgraph_scan.py`（限定名引用分析），专门补这类漏报。

---

## 1. 模块总览（按层分组）

### 1.1 入口与配置

| 模块 | 行数 | 职责 |
|---|---:|---|
| `app/main.py` | 214 | FastAPI 入口：生命周期、路由注册、静态面板、trace_id 注入 + 鉴权限流中间件 |
| `app/config.py` | 564 | 全局配置中心；`dump_config()` 供 `/health` 输出脱敏快照 |

### 1.2 API 层（`app/api/`，9 个）

| 模块 | 行数 | 职责 |
|---|---:|---|
| `chat.py` | 417 | 核心问答入口：多轮上下文 + 长期记忆 + SSE 流式 |
| `dify.py` | 336 | Dify 外部知识库兼容接口（检索 / OpenAPI 导入 / 接入自检） |
| `knowledge.py` | 175 | 知识库管理：列表 / 上传 / 删除 / 语义检索 / 重建索引 |
| `memory.py` | 133 | 记忆管理：查看 / 编辑 / 蒸馏 / **`PUT /memory/soul`（带鉴权）** |
| `routing.py` | 117 | 动态路由观测：档位分布、阈值标定、单条预演 |
| `evaluation.py` | 80 | L5 评估：检索评测、系统报告、调优建议、反馈统计 |
| `workflow.py` | 127 | 工作流状态查询与手动触发（含拓扑漂移校验） |
| `test.py` | 33 | 服务自测：全量 7 项 / 快速 3 项 / 轻量健康检查 |

### 1.3 核心层（`app/core/`，16 个）

| 模块 | 行数 | 职责 |
|---|---:|---|
| `model_router.py` | 692 | 动态路由：难度规则 + 原型/BERT 打分器，Flash↔Pro 选型 |
| `cascade.py` | 203 | **L3 级联**：Flash 质量不达标时升 Pro 重生成（三道闸门） |
| `intent_router.py` | 278 | 意图路由：规则优先 + LLM 兜底 |
| `self_check.py` | 231 | 启动自检 7 项 + 轻量健康检查 |
| `smalltalk.py` | 199 | 寒暄识别与模板直出（不进模型） |
| `tracing.py` | 161 | trace_id 贯穿 + 嵌套 span 树 + `trace.jsonl` 持久化 |
| `rag_engine.py` | 109 | 五层架构兼容门面 |
| `prompts.py` | 112 | Prompt 注册表 |
| ~~`chat_memory.py`~~ | — | **已迁出**：2026-09-11 归位为 `app/memory/chat_history.py`（见 §2.5） |
| `rate_limit.py` | 59 | 入站限流（内存滑动窗口，按 IP） |
| `source_acl.py` | 48 | 按用户的来源访问控制 |
| `observability.py` | 39 | LangSmith 接入（幂等设环境变量） |
| `request_ctx.py` | 32 | 请求级上下文（query 向量跨节点复用） |
| `errors.py` | 28 | 统一异常基类 |
| `llm_factory.py` | 25 | **向后兼容 shim** → `app/providers/llm.py` |

### 1.4 RAG 五层（`app/rag/`，11 个）

| 模块 | 行数 | 层 | 职责 |
|---|---:|---|---|
| `prepare.py` | 200 | L1 | 数据准备：归一化、指纹去重、近重复判定、元数据增强 |
| `structure.py` | 334 | L1.5 | 结构解析：章节树、标题检测、按章切分 |
| `indexer.py` | 547 | L2 | 索引构建：切分策略、短片段治理、父子块、确定性 chunk id |
| `retriever.py` | 497 | L3 | 检索优化：三路并发召回 → RRF 融合 → 阈值/去重/软回退 |
| `lexical.py` | 250 | L3 | BM25 词面倒排索引（独立召回路） |
| `parent_store.py` | 139 | L3 | 父子双层索引的父块旁路存储 |
| `reorder.py` | 28 | L3 | 缓解 lost-in-the-middle 的片段重排 |
| `generator.py` | 338 | L4 | 受控生成：上下文组装、置信度、引用抽取、流式 token |
| `evaluator.py` | 382 | L5 | 检索评测、忠实度、反馈统计、系统报告与建议 |
| `rerank.py` | 8 | L3 | **向后兼容 shim** → `app/providers/rerank.py` |

### 1.5 图工作流（`app/graph/`，5 个）

| 模块 | 行数 | 职责 |
|---|---:|---|
| `nodes.py` | 463 | 8 个业务节点（记忆加载→意图→检索/工具→路由→生成→人工兜底） |
| `workflow_graph.py` | 104 | 图组装编译 + Mermaid 拓扑 |
| `state.py` | 85 | `GraphState` 定义（含 `soft_warnings` 读写点注释） |
| `edges.py` | 50 | 条件分支：意图分支 / 异常分支 |

### 1.6 记忆系统（`app/memory/`，7 个）

> 2026-09-11 完成长短期对称化：持久化层 `chat_history.py` 从 `app/core/` 归位，
> 长短期各四层一一对应（详见 §2.5）。

| 模块 | 行数 | 职责 |
|---|---:|---|
| `chat_history.py` | 166 | **短期**持久化：Redis 优先降级内存，带 seq 与 ltrim |
| `__init__.py` | 177 | 门面：长期/短期记忆获取、窗口裁剪、按需归档、自动蒸馏 |
| `store.py` | 331 | **长期**持久化：纯文件 I/O（**原子写**） |
| `long_term.py` | 178 | 长期记忆：事实 / 画像 / 人格 / 归档 |
| `short_term.py` | 199 | 短期记忆裁剪算法 + `ShortTermMemory` 领域对象（2026-09-11 重建） |
| `consolidator.py` | 154 | 对话整理：过期对话压缩归档 |
| `dream.py` | 148 | 记忆蒸馏：归档 → 长期知识 |

### 1.7 模型提供者（`app/providers/`，5 个）

| 模块 | 行数 | 职责 |
|---|---:|---|
| `llm.py` | 318 | LLM：真实模型优先 + Mock 降级 + 限流重试包装 |
| `embeddings.py` | 276 | Embedding：API / 本地哈希双模式 + 两层缓存 |
| `rerank.py` | 102 | Cross-encoder 精排（默认关闭，失败优雅降级） |
| `base.py` | 49 | 抽象接口 `Embedder` / `Reranker` |
| `__init__.py` | 53 | 统一出口 + 迁移说明 |

### 1.8 数据层（`app/db/`，3 个）

| 模块 | 行数 | 职责 |
|---|---:|---|
| `vector_db.py` | 354 | 向量库：Chroma / 内存双后端，**原子落盘** |
| `redis_db.py` | 166 | Redis 连接池 + `MemoryRedis` 降级替身 |

### 1.9 工具与工具函数

| 模块 | 行数 | 职责 |
|---|---:|---|
| `tools/ticket_tool.py` | 72 | 工单进度查询（远端失败回退本地） |
| `tools/rule_tool.py` | 35 | 规章制度查询 |
| `tools/user_tool.py` | 41 | 用户信息查询 |
| `utils/doc_loader.py` | 283 | PDF/MD/TXT 加载，PDF 表格与图片抽取 + 可选 OCR |
| `utils/validator.py` | 137 | 请求校验与清洗（Pydantic 模型） |
| `utils/cache.py` | 110 | Embedding 两层缓存 |
| `utils/logger.py` | 79 | 全链路日志（trace_id 注入） |
| `utils/embedding.py` | 22 | **向后兼容 shim** → `app/providers/embeddings.py` |
| `static/gen_favicon.py` | 148 | 构建期 favicon 生成（手工执行，非运行时模块） |

### 1.10 脚本与测试

| 模块 | 行数 | 职责 |
|---|---:|---|
| `scripts/deadcode_scan.py` | 904 | 未使用代码/依赖扫描器（支持 `--strict`） |
| `scripts/chunking_ab.py` | 326 | 切分 A/B 影子对比（离线） |
| `scripts/chunk_metrics.py` | 171 | 切分结构指标（零 API 成本） |
| `scripts/baseline_snapshot.py` | 133 | 切分基线快照（无 git 环境下的回归护栏） |
| **`scripts/refgraph_scan.py`** | — | **本轮新增**：限定名引用分析，补撞名漏报 |
| **`scripts/module_inventory.py`** | 83 | **本轮新增**：模块清单生成器 |
| `tests/`（21 个） | 3 618 | 147 个用例，覆盖 RAG/记忆/路由/级联/死代码门禁 |

---

## 2. 无用代码判定

### 2.1 判定工具链（五路 + 一补）

| 工具 | 回答的问题 | 本轮结果 |
|---|---|---|
| pytest | 改坏了没 | 147 passed |
| ruff（F/E4/BLE/RUF100） | 未使用 import / 变量 / 僵尸 noqa | All checks passed |
| vulture | 未使用函数与变量（80% 置信） | 零输出 |
| deptry | 声明了却没用的依赖 | Success |
| `deadcode_scan.py` | 未引用定义 / 死配置 / 未用依赖 | 默认 3 项（全在 allowlist）、严格 9 项 |
| **`refgraph_scan.py`** | **上面几路因撞名而看不见的** | **新暴露 12 项 SUSPECT** |

### 2.2 已处置：判定为「无用」并删除（9 项，2026-09-11 落地）

| # | 条目 | 位置 | 判定理由 | 处置 |
|---|---|---|---|---|
| ~~1–4~~ | ~~`ShortTermMemory` 类及 `window`/`format`/`stats`~~ | `short_term.py` | **已改判**：见 §2.5，该对象不是死代码，而是**放错了位置**（持久化层跑出 `app/memory/` 包导致它无处落地）。已归位并重新接线 | **恢复并接线** |
| 5 | `get_short_term_memory()` | `app/memory/__init__.py:63` | 生产与测试**均无人导入**，只在 `__all__` 里出现 | **删除后按对称方案重建**（§2.5） |
| 6 | `clear_cache()` | `app/memory/__init__.py:153` | 同上，零消费 | **删除** |
| 7 | `IntentRoute.to_dict` | `intent_router.py:83` | 消费端 `intent_recognize_node` 只取 `.intent`/`.source`，图状态是扁平的 | **删除** |
| 8 | `SearchResult.to_dict` | `vector_db.py:72` | 只有 3 个字段，而检索管线手写的 dict 有 6 个（多 `source`/`lexical`/`chunk_index`）——是更差的副本 | **删除** |
| 9 | `GenerateResult.to_dict` | `generator.py:75` | 消费端把它摊平成 `answer`/`citations`/`confidence`/`refused`/`error_msg` 五个独立状态字段 | **删除** |

**为什么是删而不是接线**：三个 `to_dict` 都没有落点。图状态（`GraphState`）是**有意设计成扁平**的——
`intent_recognize_node` 只写 `intent_type`/`intent_source`，`generate_answer_node` 把结果摊成五个独立字段。
接线意味着把扁平状态改成嵌套结构，与 LangGraph 的状态用法相悖，且不能带来任何收益。

**为什么门禁此前一条都没报**：

- 1–6：`app/memory/__init__.py` 有 `__all__`，名字一进 `__all__` 就被扫描器当成「被字符串引用」→ 洗白。
- 7–9：`to_dict` 这个名字被 5 个类共用，扫描器按**裸名并集**判定，只要有任何一个类的 `to_dict` 被调用，其余 4 个全部跟着被判为已使用。

**处置后复验**：

```
pytest        147 passed
ruff          All checks passed!（删除后产生 1 处级联孤儿 import：short_term.py 的 typing.Any，已一并清理）
vulture       零输出
扫描器默认     3 项（全在 allowlist，与处置前一致）
refgraph      SUSPECT 12 → 6，剩余 6 项均为 §2.3 判定保留的条目
```

### 2.3 不判为无用、但门禁目前看不见（3 项，建议补豁免理由）

| 条目 | 判定 | 说明 |
|---|---|---|
| `MemoryRedis.ping` | 保留 | 与已豁免的 `ttl` / `dbsize` 同理（降级替身接口完整性）；但门禁只看得见后两个，`ping` 因 `client.ping()` 打在真实 Redis 上而隐形。建议与 ttl/dbsize 统一口径 |
| `MockChatModel._generate` | 保留 | 覆盖 LangChain 外部基类，扫描器无法识别外部基类的重写 |
| `_RateLimitRetryModel._stream` | 保留 | 委托模式 `self._delegate._stream`，静态类型推断不到具体类 |

### 2.5 改判：`ShortTermMemory` 不是死代码，是放错了位置（2026-09-11 后半程）

用户指出「memory 部分是参考 nanobot 写的」。对照 **HKUDS/nanobot** 源码后发现，
先前的删除判据只看到"没人调用"，没看到**为什么没人调用**。

**nanobot 的真实结构**（`nanobot/agent/memory.py` 不存在，短期记忆在 `nanobot/session/manager.py`）：

- `Session`（`:276`）**自身持有 `messages`**，`get_history(max_messages, max_tokens,
  extend_to_user)`（`:344`）直接读自身状态裁剪
- `SessionManager`（`:1644`）是**唯一入口** `get_or_create(key)`（`:1740`），带缓存
- 载体与入口一体，贯穿整个 agent loop

**本项目当初只搬了壳**：`ShortTermMemory` 只持有 `session_id`/`max_turns`/`max_chars`
三个配置字段，**不持有消息**（`window(history)` 要外部传），在 LangGraph 的函数式
数据流里无处安放。

**更根本的问题是不对称**：短期记忆的持久化层 `app/core/chat_memory.py`（Redis）
**跑出了 `app/memory/` 包**。持久化与裁剪一分家，`short_term.py` 就只剩算法，对象化
自然落不了地。长期记忆四层（store / long_term / get_long_term_memory / get_memory_context）
全在包内，所以对象化成立。

**已执行的对称改造**：

| 层 | 长期记忆 | 短期记忆（改造后） |
|---|---|---|
| 持久化 | `memory/store.py`（文件） | `memory/chat_history.py`（Redis，**从 `core/` 归位**） |
| 领域对象 | `LongTermMemory` | `ShortTermMemory` |
| 取实例门面 | `get_long_term_memory(user_id)` | `get_short_term_memory(session_id, history)` |
| 注入 prompt | `get_memory_context(user_id)` | `build_short_term_window(history)` 保留兼容 |

**一个关键设计点**：`history` 由**构造参数注入**，对象不自己读 Redis。因为 API 层
`chat.py:105` 已经 `await get_history(session_id)` 加载过，历史经 `GraphState` 传到生成
节点才裁剪；对象再读一次等于每轮多一次往返。注入式让对象语义上「持有消息」（对齐
nanobot `Session`），又不重复 I/O —— OO 载体与函数式数据流之间的折中点。

**另一个决定：刻意不提供 `format()`**。生产链路严格两步——先 `window` 裁剪，再由
`generator._format_history` 只做格式化。对象若给一个"裁剪 + 格式化"一步到位的方法，
调用方会以为拿到窗口文本，实际触发第二次裁剪，让 `SHORT_TERM_WINDOW` 静默失效。
这正是 `test_format_history_does_not_re_truncate` 守住的坑：**裁剪点必须唯一**。

**改造后不再有死代码**：`ShortTermMemory` 已被 `generate_answer_node` 与流式链路
`chat_ask_stream` 消费，`refgraph` 复扫确认无检出项。

**顺带合并了两份重复的格式化实现**：`short_term.format_history()` 与
`generator._format_history()` 曾几乎逐字相同，唯一差别是前者内部会再裁一次
（而后者刻意不裁）。处理办法不是删掉前者，而是：

- `short_term.format_history(history)` 改为**只格式化不裁剪**（去掉 `max_turns`/`max_chars`）
- `generator._format_history` **委托**给它

一举三得：消除重复、给它真实消费点、顺手去掉二次裁剪的坑。
新增 3 项测试守住（`test_format_history_does_not_crop` /
`test_generator_delegates_to_memory_formatter` / `test_two_formatter_outputs_are_identical`）。

### 2.4 严格模式遗留、仍未处理（6 项）

`CHUNK_MIN_CHARS`、`KnowledgeBaseEmpty`、`reset_smalltalk_stats`、`filter_by_section`、
`SectionTree.path_at`、`detect_structure` —— 已由上一轮列出，本轮用独立方法交叉验证，
**两路结果完全一致**，确认真属「生产代码无人使用」。

---

## 3. 方法论：静态检测的五类盲区（本轮实测）

1. **裸名撞名**：同名名字在别处出现即被判为已使用。→ `to_dict` 5 兄弟、早期的 `prompts.render`。
2. **`__all__` 洗白**：写进 `__all__` 等于被「字符串引用」，死代码立刻隐形。→ `get_short_term_memory` / `clear_cache`。
   **反直觉的点：`__all__` 本是契约声明，在这里却成了盲区制造机。**
3. **外部 / 抽象基类的重写识别不到**：`MockChatModel._generate` 覆盖的是 `langchain` 的类，仓库里没有基类定义。
4. **动态属性读取**：`getattr(store, "get_texts", None)` 让 `get_texts` 活着，但只有 `config` 目标的 getattr 被算作证据。
5. **只写不读的状态字段**：对 dict 赋值，扫描器结构上查不到。→ 上一轮已修的 `soft_warnings` 属此类。

---

## 4. 非死代码，但值得记一笔

| 项 | 说明 |
|---|---|
| **三个向后兼容 shim** | `core/llm_factory.py` / `utils/embedding.py` / `rag/rerank.py` 仍被 **20+ 处**引用，不是死代码。但构成「新旧双入口」（`app/providers/*` 与 shim 并存），是**技术债不是垃圾**，迁移完成后应删除 |
| ~~`.pytest_tmp/` 未加 .gitignore~~ | **误报，已更正**：`.gitignore:24` 早就忽略了它。初次检查用了 `grep "a\|b"`,BSD grep 不认 `\|` 交替且**静默返回 0**，于是误判为"未覆盖" |
| **流式链路是第二套工作流实现** | `docs/history/redundancy-review.md` 已列为最高通用性风险，本轮未处理 |
| **重复的 `_RouteStats` / `_STATS`** | `intent_router.py` 与 `model_router.py` 各有一份同名的统计类，属可合并的重复实现 |

---

## 5. 复现命令

```bash
# 五路门禁
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m deptry .
.venv/bin/python -m vulture app scripts --min-confidence 80
.venv/bin/python scripts/deadcode_scan.py .            # 默认口径
.venv/bin/python scripts/deadcode_scan.py . --strict   # 严格口径（人工巡检用）

# 本轮新增
.venv/bin/python scripts/refgraph_scan.py .            # 限定名引用：DEAD / SUSPECT / TEST-ONLY
.venv/bin/python scripts/refgraph_scan.py . --name render   # 追踪单个名字
.venv/bin/python scripts/module_inventory.py .         # 模块清单底稿
```
