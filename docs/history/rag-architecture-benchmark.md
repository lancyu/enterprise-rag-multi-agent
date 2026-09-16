# RAG 架构对标分析报告

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> 对象：`langgraph-enterprise-bot`（企业智能助手 · RAG 知识引擎）
> 方法：选取 5 个高质量开源 RAG 项目，通过 GitHub raw 拉取**核心源码原文精读**（非 README 摘要），逐模块对照本项目真实代码结构
> 日期：2026-09-03

---

## 一、调研选型与依据

选型标准：**检索链路完整性**（解析→切分→索引→召回→重排→生成→评测是否闭环）、**代码规范性**（抽象分层、降级设计、类型标注）、**社区活跃度**。

| 项目 | 定位 | 精读源码（本地路径 `/tmp/rag_research/`） | 借鉴它的理由 |
|---|---|---|---|
| **langgenius/dify** | 企业级 LLM 应用平台，RAG 完整度最高 | `retrieval_service.py`(45KB)、`data_post_processor.py`、`cached_embedding.py`、`jieba_kw.py`、`text_splitter.py`、`rerank_base/factory.py`、`dataset_retrieval.py`(95KB) | 与本项目同为 Python + 企业知识库场景；多租户、并发召回、重排、元数据过滤、缓存、审计全部落地 |
| **infiniflow/ragflow** | 深度文档理解标杆 | `rag/nlp/__init__.py`(67KB, 1900+ 行)、`deepdoc/parser/pdf_parser.py`(88KB)、`task_executor.py`(89KB) | 文档解析与切分策略的工程深度最强，正是本项目最薄弱一环 |
| **stanfordnlp/dspy** | Prompt 工程范式 | `predict/chain_of_thought.py`、`predict/predict.py`、`evaluate/evaluate.py`、`teleprompt/random_search.py` | 把「提示词」从字符串升级为可编程、可自动优化的模块——回答"提示词工程怎么做" |
| **explodinggradients/ragas** | RAG 评测事实标准 | `evaluation.py`、`metrics/__init__.py`、`dataset.py` | 评测体系的事实标准，给出指标选型依据 |
| **HKUDS/LightRAG** | 增量索引 + 图谱增强 | `lightrag.py`(324KB)、`operate.py`、`base.py` | 增量索引的内容哈希 ID 设计，直击本项目 ID 生成缺陷 |

> 说明：`langchain-ai/chat-langchain` 因仓库重构（main/master 分支 `retrieval_graph` 路径均 404）放弃，改选 LightRAG 补位，保证 5 个项目全为 Python 技术栈，可比性更强。

---

## 二、标杆项目的核心设计发现（源码级）

### 2.1 dify —— 检索工程的三个关键模式

**① 线程池并发多路召回**（`retrieval_service.py:781-916`）

```python
with ThreadPoolExecutor(max_workers=dify_config.RETRIEVAL_SERVICE_EXECUTORS) as executor:
    futures = []
    if retrieval_method == RetrievalMethod.KEYWORD_SEARCH and query:
        futures.append(executor.submit(propagate_context(self.keyword_search), ...))
    if RetrievalMethod.is_support_semantic_search(retrieval_method):
        futures.append(executor.submit(propagate_context(self.embedding_search), ...))
    if RetrievalMethod.is_support_fulltext_search(retrieval_method) and query:
        futures.append(executor.submit(propagate_context(self.full_text_index_search), ...))
    if futures:
        for future in concurrent.futures.as_completed(futures, timeout=300):
            if future.exception():
                for f in futures: f.cancel()   # 首个异常即取消其余，避免无谓等待
                break
```

- 四路召回模式枚举：`SEMANTIC_SEARCH` / `KEYWORD_SEARCH` / `HYBRID_SEARCH` / `FULL_TEXT_SEARCH`
- **混合模式下把 `score_threshold` 置 0**（`:329`），把过滤权交给后处理，避免阈值在融合前误杀
- **无 rerank 时才用 score_threshold 兜底**（`:910`）——rerank 优先、阈值兜底的分层决策

**② 后处理层 rerank 与 reorder 正交分离**（`data_post_processor.py`）

```python
def invoke(self, query, documents, score_threshold=None, top_n=None, query_type=...):
    if self.rerank_runner:
        documents = self.rerank_runner.run(query, documents, score_threshold, top_n, query_type)
    if self.reorder_runner:
        documents = self.reorder_runner.run(documents)   # 缓解 lost-in-the-middle
    return documents
```

- 两种 rerank 模式：`WEIGHTED_SCORE`（向量分×权重 + 关键词分×权重，无模型）与 `RERANKING_MODEL`（独立 rerank 模型）
- **`ReorderRunner` 独立成层**：重排解决"排得准不准"，reorder 解决"排好后 LLM 看不看得见"——后者针对 lost-in-the-middle 效应
- `@trace_span()` 装饰器、`tenant_id` 贯穿全链路

**③ Embedding 三层缓存**（`cached_embedding.py`）

| 缓存对象 | 机制 | 收益 |
|---|---|---|
| 文档向量 | 按 content hash 落 DB（`Embedding` 表） | 重建索引不重复付费 |
| 查询向量 | Redis `setex` TTL 600s，**命中则 `expire` 续期**（滑动过期） | 重复提问零成本 |
| Cache key | `f"{provider}_{model_name}_{hash}"` | **换模型自动失效**，杜绝跨模型向量混用 |

**④ 可选依赖的三级降级**（`jieba_kw.py`）：`jieba.analyse.default_tfidf` → `TFIDF 类` → 自建 `_SimpleTFIDF` 词频回退。每段降级都有明确注释说明触发条件。

### 2.2 ragflow —— 切分与解析的工程深度

`rag/nlp/__init__.py` 单文件 1900+ 行，能力远超"一个切分器"：

| 函数 | 作用 | 本项目对应 |
|---|---|---|
| `MergeStrategy` 枚举（`:1240`） | `UNDER_CAP`（绝不超过 token_size）/ `OVER_CAP`（允许一次边界溢出） | 无，仅固定参数 |
| `_merge_paragraph_groups` | 明确原则 **"No atom-split is ever performed"**（超长段落自成一 chunk） | `RecursiveCharacterTextSplitter` 末位分隔符 `""` 会切到**字符级** |
| `_compute_overlap_prefix` | overlap 按**百分比**从尾部切，且在分组时预留额度 | 固定 60 字符 |
| `hierarchical_merge` / `tree_merge` + `Node` 类 | 按**标题层级**建树合并 | 无 |
| `tokenize_table` / `tokenize_chunks_with_images` | 表格、图文混合 tokenize | 无 |
| `attach_media_context` | 表格/图片挂载上下文窗口 | 无 |
| `remove_contents_table` / `make_colon_as_title` / `bullets_category` | 去目录、冒号识别为标题、项目符号分类 | 无 |
| `extract_position` / `add_positions` | **页面坐标溯源**（chunk → 原文档位置） | 仅文件名 |
| `find_codec` / `is_english` / `is_chinese` | 编码检测、语言识别 | 无 |

### 2.3 DSPy —— 提示词不是字符串，是 Signature

```python
class ChainOfThought(Module):
    def __init__(self, signature, rationale_field=None, rationale_field_type=str, **config):
        signature = ensure_signature(signature)          # "question -> answer"
        rationale_field = rationale_field or dspy.OutputField(desc="${reasoning}")
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, ...)
        self.predict = dspy.Predict(extended_signature, **config)
```

核心范式差异：
- **Signature 是类型化的 I/O 声明**（`question -> answer`），prompt 由 `Predict` 模块编译生成，不是人写字符串
- **ChainOfThought 是"签名变换器"**：通过 `prepend` 动态插入 `reasoning` 输出字段，CoT 能力是一个可组合的模块而非一段咒语
- **提示词可优化**：`teleprompt/` 下的优化器（如 `random_search.py`）能针对评测指标**自动搜索**最优 few-shot 示例与指令

对照本项目：5 个 prompt 全是模块级字符串常量，无注册表、无版本、无优化回路。

### 2.4 ragas —— 评测指标矩阵

| 类别 | 指标 | 本项目是否覆盖 |
|---|---|---|
| 检索质量 | `context_precision`、`context_recall`、`context_entity_recall` | ❌ 仅有 HitRate@K / MRR |
| 生成质量 | `answer_correctness`、`answer_relevancy`、`answer_similarity` | ❌ |
| 忠实度 | `faithfulness`（LLM-as-judge 逐句核对） | ⚠️ 本项目用字符二元组覆盖率的零成本启发式 |
| 多轮 | `MultiTurnMetric` / `SingleTurnMetric` | ❌ |
| 工程 | `discrete_metric` / `numeric_metric` / `ranking_metric` 类型装饰器 | ❌ |

### 2.5 LightRAG —— 增量索引的 ID 设计

```python
# lightrag.py:1898-1899 注释原文
# Identity is deterministic and document-scoped: chunk ids hash
# ``(doc_id, chunk_content)`` and the operation id hashes the ordered ...
doc_key = compute_mdhash_id(full_text, prefix="doc-")     # :1927
# :1931  Deterministic, document-scoped identity: chunk ids hash
```

- chunk ID 哈希 **(doc_id, chunk_content)** → 内容不变则 ID 不变，真增量 diff
- `enable_llm_cache` / `enable_llm_cache_for_entity_extract` + `_build_role_llm_cache_identity`（按角色分层的 LLM 缓存）

**对照本项目缺陷**：`indexer.py:107` 用 `uuid5(source::chunk_index)`——**位置型 ID**。文档中间插入一段，后续所有 `chunk_index` 偏移，ID 全变。当前靠 `delete_by_source` + 全量重写掩盖，实为"伪增量"。

---

## 三、逐模块对标表

> 差距等级：**缺失**（完全没有）/ **薄弱**（有但关键能力缺）/ **可用**（够用，有优化空间）/ **对齐**（与标杆相当）/ **领先**（优于大多标杆）

### 1. 文档解析与清洗

| 项 | 内容 |
|---|---|
| **参考要点** | ragflow `deepdoc` 版面分析+OCR+表格（pdf_parser 88KB）；`remove_contents_table` 去目录、`make_colon_as_title`、`bullets_category`、`find_codec` 编码检测；dify `extractor_base` 抽取器抽象 |
| **本项目现状** | `doc_loader.py`(96行)：仅 `.pdf`(PyPDFLoader)/`.md`/`.txt`，无 DOCX/PPTX/XLSX/HTML，无 OCR/表格/版面；`prepare.py`(200行) 清洗扎实：NFKC 归一化→控制字符→页码噪声→空白压缩→sha1 指纹精确去重→Jaccard(0.95) 近重复→元数据注入，全流程有统计 |
| **差距** | **薄弱**（清洗层对齐，解析层缺失） |
| **改进建议** | ① 补 DOCX(`python-docx`)/HTML(`bs4`)——半天，企业文档主力格式；② PDF 换 `pdfplumber` 保留表格结构；③ OCR/版面分析最后做（`deepdoc` 重，且需视觉模型）；④ 近重复 O(n²) 在语料 >1 万时换 MinHash 分桶 |

### 2. 切分策略

| 项 | 内容 |
|---|---|
| **参考要点** | ragflow `MergeStrategy` 枚举 + **绝不原子切分** + overlap 百分比预留 + `hierarchical_merge`/`tree_merge` 标题树 + `tokenize_table` + `extract_position` 坐标；dify `TokenTextSplitter`(tiktoken 精确计长) + `add_start_index` 原文位置 |
| **本项目现状** | `indexer.py`：`RecursiveCharacterTextSplitter(chunk_size=300, overlap=60, separators=["\n\n","\n","。","！","？","；","，"," ",""])` + 自研 `_merge_short_chunks` 短片段向前合并（防纯标题碎片）；按 `len()` 计长 |
| **差距** | **薄弱** |
| **改进建议** | ① **末位分隔符去掉 `""`**——当前会切到字符级，破坏语义原子性；② 长度函数改 token 计数（`tiktoken`，与 dify 一致，chunk_size 语义才准确）；③ 注入 `start_index` 元数据（溯源到原文位置，成本极低）；④ 制度类文档可选标题层级合并 |

### 3. 向量化与 Embedding 管理

| 项 | 内容 |
|---|---|
| **参考要点** | dify `CacheEmbedding`：文档向量按 content hash 落库 + 查询向量 Redis TTL 600s 命中续期 + cache key 含 `{provider}_{model}_{hash}` 换模型自动失效；LightRAG `enable_llm_cache` + 角色分层缓存身份 |
| **本项目现状** | `embedding.py`(217行)：`LocalHashEmbeddings`（本地哈希+IDF，无 Key 时零依赖降级，设计优秀）+ `APIEmbeddings`（batch=32）+ `_l2_normalize` + 模式自适应阈值；**全文无任何缓存** |
| **差距** | **薄弱** |
| **改进建议** | ① 查询向量 Redis 缓存（key 必须含模型名，否则换 embedding 后向量空间错乱）；② 文档向量按 `fingerprint` 落盘（重建索引省全部 embedding 费用）；③ 降级链已完善，保持 |

### 4. 索引与向量库

| 项 | 内容 |
|---|---|
| **参考要点** | dify 向量库抽象 + 元数据过滤（三模式：disabled/LLM 自动推断/manual）；LightRAG 内容哈希 ID 实现真增量 |
| **本项目现状** | `vector_db.py`(250行)：自研 `MemoryVectorStore`（numpy 暴力全量扫描，npz 落盘）+ `ChromaVectorStore` 可选；`search(query_vector, k)` **无 where 过滤参数**；`_chunk_id = uuid5(source::chunk_index)` 位置型 ID |
| **差距** | **薄弱** |
| **改进建议** | ① `search()` 增加 `metadata_filter` 参数（为后续权限/租户铺路，改动小）；② chunk ID 改 **内容哈希** `md5(source::content)`——真增量，避免文档插入导致全量重写；③ 语料 >10 万条再引入 ANN（HNSW），当前暴力检索在千级语料下反而更快 |

### 5. 召回（关键词 / 向量 / 混合）

| 项 | 内容 |
|---|---|
| **参考要点** | dify 四路枚举 + `ThreadPoolExecutor` 并发 + `as_completed(300s)` 早错早停 + 混合模式阈值置 0 + rerank 优先阈值兜底 |
| **本项目现状** | `retriever.py`(320行)：dense 召回 `candidate_k = min(max(top_k*4, 30), count)` → 对候选集算词面分 → **RRF 融合(k=60, 权重 0.7/0.3)** → 阈值过滤 → 去重 → 软回退；`lexical_score` 自研（中文实义字符集覆盖度 + 连续子串 +0.3 奖励，过滤 `_CJK_STOP` 虚词，中英双路取 max） |
| **差距** | **可用**（融合策略领先，召回路径薄弱） |
| **改进建议** | ① **词面路改独立倒排召回**——当前只是对向量候选集重打分，池外纯词面命中被漏掉（代码注释已自承此债）；② 两路用线程池并发（对齐 dify）；③ **RRF 融合保留不动**——加权求和需归一化且随语料漂移，RRF 只看排名绕开量纲问题，这一点本项目优于 dify 的 `WEIGHTED_SCORE` |

### 6. 重排序

| 项 | 内容 |
|---|---|
| **参考要点** | dify `DataPostProcessor`：rerank（模型/加权两模式）与 `ReorderRunner` 正交分离，rerank 管"排得准"，reorder 管"LLM 看得见"（lost-in-the-middle） |
| **本项目现状** | **全库 grep `rerank\|reorder\|cross.?encoder\|cohere` → 0 命中**。召回后直接组装 prompt |
| **差距** | **缺失** |
| **改进建议** | ① **先加 `ReorderRunner`**——纯排序变换、零模型调用、零延迟，按 `[1,3,5,...,6,4,2]` 交错把高相关片段放首尾，半天可完成，对长上下文收益直接；② 再接 rerank 模型（`bge-reranker-base` 可本地部署，或 Cohere API）；③ rerank 分数并入 citations |

### 7. 查询改写与意图路由

| 项 | 内容 |
|---|---|
| **参考要点** | FastGPT `classifyQuestion` 官方模板首项即 Greeting；dify 元数据条件 LLM 自动推断 |
| **本项目现状** | `intent_router.py`：规则优先 + LLM 兜底四分类（`knowledge`/`tool`/`smalltalk`/`unknown`）；`smalltalk.py` 6 类寒暄整句锚定直出；`model_router.py` 三层动态路由（L0 闸门/L1 规则/L2 分类器/降级锁/abstain）+ `/routing/calibrate` 阈值标定；`retriever.rewrite_query` 已实现，默认关闭（限频账号友好） |
| **差距** | **领先** |
| **改进建议** | ① 查询改写保持默认关闭（每改写一次多一次模型调用，对 RPM=3 账号不划算）；② 若启用，改写结果应**多 query 融合召回**而非简单替换；③ 意图路由与动态路由是本项目的差异化优势，建议补 A/B 数据沉淀 |

### 8. Prompt 管理与上下文组装

| 项 | 内容 |
|---|---|
| **参考要点** | **DSPy**：prompt = 类型化 Signature，由 `Predict` 模块编译，CoT 等能力是签名变换器，teleprompt 可针对指标自动优化；dify `prompt_transform` 集中管理 |
| **本项目现状** | 5 个 prompt 全为模块级字符串常量：`generator.py:36` ANSWER_PROMPT、`intent_router.py:190` INTENT_PROMPT、`retriever.py:139` 查询改写、`dream.py:92`、`consolidator.py:78`；上下文组装 `build_context` → `[i] 来源：xxx\n内容`，tool_result 追加为 `[业务查询结果]`；拒答文案硬编码在 `prepare_generation` 内 |
| **差距** | **薄弱** |
| **改进建议** | ① **建 prompt 注册表**（`app/prompts/` 集中 + 版本号 + 变量声明与校验），先把 5 个常量收敛——半天，收益是 prompt 改动不再散落各模块、可追溯；② **上下文 token 预算裁剪**：当前 `docs[:top_k]` 无预算控制，长文档会撑爆上下文；③ 拒答文案外置为配置项（当前改话术要改代码）；④ 进阶：引入 DSPy 式 Signature，让意图分类/查询改写可针对评测指标自动优化 |

### 9. 生成与引用溯源

| 项 | 内容 |
|---|---|
| **参考要点** | dify 引用 + 片段命中次数统计；ragflow `extract_position`/`add_positions` 页面坐标级溯源 |
| **本项目现状** | `generator.py`(300行)：编号引用制度 + `extract_cited_indexes` 抽取实际引用 + citations 清单（source/file_name/score/lexical/fallback）+ 归一化置信度（RRF 分 / 理论上限，跨模型可比）+ `REFUSE_THRESHOLD=0.25` 优雅拒答 + 流式空流自愈；溯源粒度到**文件名** |
| **差距** | **可用** |
| **改进建议** | ① 补 `start_index`/`page` 元数据，溯源从"哪个文件"细化到"文件哪个位置"（配合模块 2 的元数据注入）；② 引用覆盖率可做**强校验**：答案声称引用 [1] 但 [1] 内容与结论不符时告警；③ 幻觉治理当前靠 prompt 约束 + 二元组粗筛，可加引用-结论一致性校验 |

### 10. 评测体系

| 项 | 内容 |
|---|---|
| **参考要点** | ragas：检索侧 `context_precision`/`context_recall`/`context_entity_recall`，生成侧 `answer_correctness`/`answer_relevancy`，`faithfulness` LLM-as-judge，`MultiTurnMetric` 多轮，指标类型装饰器 |
| **本项目现状** | `evaluator.py`(304行)：HitRate@K + MRR + `score_faithfulness`（字符二元组覆盖率，零成本启发式）+ `score_citation_coverage` + 用户反馈收集 + `build_report` + `suggest_improvements` 调优建议闭环；但**内置用例仅 4 条且硬编码**在 `DEFAULT_EVAL_CASES` |
| **差距** | **薄弱** |
| **改进建议** | ① **用例外置 YAML**（当前加一条用例要改代码，是评测落不了地的主因）——半天；② 补 `context_precision`/`context_recall`（ragas 口径，可直接参考实现）；③ LLM-as-judge 做成**可选开关**（限频账号默认关闭）；④ 评测接入 CI，防止调参改坏召回 |

### 11. 可观测性与链路追踪

| 项 | 内容 |
|---|---|
| **参考要点** | dify `@trace_span()` OTel 装饰器 + `DatasetQuery` 审计日志 + 片段命中次数 + OpenTelemetry 追踪任务；LangSmith/Langfuse 全链路 |
| **本项目现状** | `logger.py` 仅 44 行，纯 `logging` 封装；工作流 `trace` 字段记录节点名列表（前端可见），但**无 trace_id、无 span、无耗时打点、无 OTel**；grep `otel\|opentelemetry\|trace_id\|langsmith\|langfuse` 在 app/ 下 0 命中 |
| **差距** | **缺失** |
| **改进建议** | ① **加 `request_id`/`trace_id` 贯穿全链路**（含 SSE 事件），1 天——这是所有可观测性的地基，没有它日志无法串联；② 各节点耗时打入 trace（已有 `time.perf_counter` 用法可复用）；③ 对外接 Langfuse/LangSmith 或 OTel exporter；④ 检索侧记录"哪路召回贡献了最终结果"（多路召回归因必需） |

### 12. 缓存与并发

| 项 | 内容 |
|---|---|
| **参考要点** | dify Redis 缓存 + `ThreadPoolExecutor` 召回并发 + Redis 知识库**入站限流**；LightRAG 分层 LLM 缓存 |
| **本项目现状** | `redis_db.py`(131行) 用于会话/短期记忆，Redis 不可用时降级进程内内存（降级设计良好）；召回链路**串行**；embedding 无缓存；**无入站限流**（`llm_factory` 的 429 处理是针对上游重试，不是保护自身） |
| **差距** | **薄弱** |
| **改进建议** | ① 召回两路并发化（配合模块 5）；② embedding 缓存（配合模块 3）；③ **加入站限流**（`slowapi`，按 session/IP）——当前任何人可刷爆 API 配额；④ 热点 query 结果缓存（含检索结果与答案，TTL 分级） |

### 13. 权限与多租户

| 项 | 内容 |
|---|---|
| **参考要点** | dify `tenant_id` 贯穿全链路（`ModelManager.for_tenant`、`DataPostProcessor(tenant_id)`、`DatasetQuery` 审计）+ 按租户限流与用量统计 |
| **本项目现状** | **全库 grep `tenant\|租户` → 0 命中**。单租户设计，`user_id` 仅用于记忆隔离 |
| **差距** | **缺失** |
| **改进建议** | 若定位内部单部门助手，**可延后**（P2）；若需多部门共用：① chunk metadata 加 `tenant_id`/`acl` 字段；② `search()` 加过滤（依赖模块 4 的 `metadata_filter`）；③ 检索前做可见性裁剪——**必须在召回层过滤而非生成层**，否则会泄露。 |

### 14. 配置与部署

| 项 | 内容 |
|---|---|
| **参考要点** | dify 分层配置 + docker-compose 全栈编排 |
| **本项目现状** | `config.py`(254行)：集中配置 + 环境变量覆盖 + `_env()` 空值保护 + `mask_secret` 脱敏 + `dump_config` 快照 + **按实际 embedding 模式自适应阈值**（`effective_score_threshold`/`effective_fallback_min`，解决跨供应商分数尺度漂移）；Dockerfile + docker-compose + `/health` 自检 |
| **差距** | **对齐**（本项目强项） |
| **改进建议** | ① 引入 `pydantic-settings` 做配置 schema 校验，启动即发现类型错误；② 多环境 profile（dev/staging/prod）；③ 自适应阈值设计值得保留并写进文档——比固定阈值稳健 |

### 15. 编排与工作流（本项目额外强项）

| 项 | 内容 |
|---|---|
| **参考要点** | LangGraph 条件图；dify 工作流画布 |
| **本项目现状** | `workflow_graph.py` 8 节点 + 2 条件分支：`memory_load → intent_recognize → {knowledge_retrieve / tool_invoke / smalltalk_reply / model_route} → generate_answer → human_fallback`；`nodes.py`/`edges.py` 分层清晰；`human_fallback` 异常兜底；`/workflow` 接口暴露 mermaid 拓扑 |
| **差距** | **领先** |
| **改进建议** | 编排层已成熟，无需大改；可考虑把 rerank/reorder 作为独立节点接入图，保持"一个节点一职责"的现有风格 |

---

## 四、企业级优化要点

| 优化项 | 参考做法 | 本项目现状 | 落地优先级 |
|---|---|---|---|
| **召回准确率** | dify 多路并发 + rerank + reorder | RRF 融合已较优；词面非独立召回 | **P0**：词面独立召回 + ReorderRunner |
| **延迟优化** | 线程池并发、embedding 缓存 | 串行召回、无缓存 | **P0**：并发化 + 查询向量缓存 |
| **增量索引** | LightRAG 内容哈希 ID | 位置型 ID + `delete_by_source` 全量重写（伪增量） | **P1**：chunk ID 改内容哈希 |
| **流式输出** | SSE 逐 token | 已实现，且带空流自愈 + 温度自愈 | 已达标，保持 |
| **成本控制** | 查询/文档向量缓存、动态路由分档 | 动态路由（Flash/Pro）已落地；**无 embedding 缓存** | **P0**：embedding 缓存直接省钱 |
| **灰度与降级** | MOCK 降级、限流重试 | `MockChatModel` + 429 退避重试 + temperature 自愈（优秀） | 已达标；补**入站限流** |
| **安全合规** | 租户隔离、审计日志、密钥脱敏 | `mask_secret` 有；租户/审计**无** | **P2**（单租户可延后） |

---

## 五、落地清单（按投入产出比排序）

### 🔴 P0 必需项 —— 补齐可用性，约 1~2 周，全部低成本高收益

> **落地状态（2026-09-03）**：7 项 P0 已全部实现并离线回归通过（`tests/test_rag.py`、`tests/test_infra.py`）。
> 实现细节见 `docs/history/p0-implementation-design.md`。其中 #9「chunk ID 内容哈希」因 #6 词面索引的 key 依赖，
> 一并从 P1 提前纳入本次，需一次性重建知识库。

| # | 事项 | 对应模块 | 工作量 | 状态 |
|---|---|---|---|---|
| 1 | **加 Reorder**（召回后交错重排，1,3,5,…,6,4,2） | 6 重排序 | 半天 | ✅ 已实现 `app/rag/reorder.py` |
| 2 | **Prompt 注册表**（5 个常量收敛 + 变量校验） | 8 Prompt | 半天 | ✅ 已实现 `app/core/prompts.py` |
| 3 | **Embedding 缓存**（查询 LRU+TTL + 文档内容哈希持久化，key 含模型名） | 3/12 | 1 天 | ✅ 已实现 `app/utils/cache.py` |
| 4 | **评测用例外置 YAML** | 10 评测 | 1 天 | ✅ 已实现 `app/rag/eval_cases.yaml`（CI 未接，demo 无需） |
| 5 | **trace_id 贯穿全链路** + 节点耗时打点 | 11 可观测 | 1 天 | ✅ 已实现 `app/core/tracing.py` + logger Filter |
| 6 | **词面路独立召回 + 线程池并发** | 5 召回 | 2~3 天 | ✅ 已实现 `app/rag/lexical.py` + retriever 三路并发 |
| 7 | **入站限流**（滑动窗口，按 IP） | 12 并发 | 半天 | ✅ 已实现 `app/core/rate_limit.py`（20 次/分） |

### 🟠 P1 高 ROI —— 企业级增强，约 3~6 周

| # | 事项 | 对应模块 | 工作量 | 收益 |
|---|---|---|---|---|
| 8 | 接入 **rerank 模型**（bge-reranker 本地 / Cohere API） | 6 | 2~3 天 | 召回准确率的最大单项提升；本地部署无持续成本 |
| 9 | **chunk ID 改内容哈希** → 真增量索引 | 4 | 1 天 | ✅ 已实现（随 P0 #6 提前纳入，需重建知识库） |
| 10 | **文档格式扩展**（DOCX / HTML / PDF 表格） | 1 | 2~3 天 | 企业文档主力是 DOCX，当前完全不支持 |
| 11 | **切分升级**：token 计长 + 去掉字符级分隔符 + `start_index` 元数据 | 2 | 2 天 | 语义原子性 + 溯源精度 |
| 12 | **上下文 token 预算裁剪** | 8 | 1 天 | 防长文档撑爆上下文 |
| 13 | 补 `context_precision` / `context_recall` 指标 | 10 | 1~2 天 | 检索质量可量化归因 |

### 🔵 P2 企业级增强 —— 按需，非必需

| # | 事项 | 对应模块 | 前置依赖 |
|---|---|---|---|
| 14 | 元数据过滤 + ANN 索引（语料 >10 万） | 4 | #9 |
| 15 | OTel / Langfuse 全链路追踪 | 11 | #5 |
| 16 | 多租户与权限（metadata + 召回层过滤） | 13 | #14 |
| 17 | OCR / 版面分析（扫描件 PDF） | 1 | — |
| 18 | LLM-as-judge 忠实度评测（默认关闭） | 10 | #4 |
| 19 | DSPy 式 Signature 与自动提示词优化 | 8 | #2 + #4 |
| 20 | 标题层级合并切分 | 2 | #11 |

---

## 六、结论

**本项目的真实水位**：

- **领先项（2 个）**：意图路由与动态路由（三层 + 阈值标定 + 降级锁）、编排工作流（LangGraph 8 节点 + human_fallback）——这两项达到或超过多数开源标杆，是项目的差异化资产。
- **对齐项（1 个）**：配置与部署（集中配置 + 空值保护 + 密钥脱敏 + **按 embedding 模式自适应阈值**——后者比多数项目的固定阈值更稳健）。
- **可用项（2 个）**：召回（RRF 融合策略优于 dify 的加权融合，但词面非独立召回路）、生成与引用溯源（编号引用 + 归一化置信度 + 优雅拒答，溯源粒度仅到文件名）。
- **薄弱项（7 个）**：文档解析与清洗、切分策略、Embedding 管理、索引与向量库、Prompt 管理、评测体系、缓存与并发——**共同点是有实现但停留在"能跑"层面，缺少企业级所需的扩展性**。
- **缺失项（3 个）**：重排序（含 reorder）、可观测性（trace_id/span）、多租户。

**核心判断**：项目的**架构分层是对的**（五层 RAG 架构 + 图编排），问题不在设计而在**深度**。最该先做的是 7 项 P0——它们总计约 1~2 周，且都不需要架构改动，只是在现有分层里补关键零件。其中 **#1 ReorderRunner、#2 Prompt 注册表、#4 评测用例外置** 三项各只需半天到一天，却是"有没有"的性质差别。

**最不该先做的**：多租户（#16）、OCR（#17）、DSPy 自动优化（#19）——在没有 trace 数据和评测基线之前做这些，等于在没有仪表盘的情况下调发动机。
