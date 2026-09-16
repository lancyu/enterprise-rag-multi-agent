# P0 七项落地设计方案（demo 级简化版）

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> 定位：类比 nanobot 与 OpenClaw 的关系——**保留同类工程骨架，大幅简化实现**。
> 该有的模块一个不少，但去掉复杂边界处理、性能优化、过度抽象。
> 重心：**RAG 三块（词面路独立召回 / Embedding 缓存 / Reorder）详细设计，横切四块保留但轻量实现。**

## 编码约定（贯穿全部实现）

| 约定 | 规则 |
|---|---|
| 注释 | **只解释「为什么」，不解释「是什么」**。现状代码（如 `retriever.py:254`）已是这个风格，保持一致 |
| 配置 | 沿用 `config.py` 现有模式：模块级常量 `X: T = getattr(config, "X", default)`，**不引入 Pydantic Settings** |
| 日志 | 沿用 `app/utils/logger.py`，**不新建 logging 模块** |
| 异常 | 新增统一基类，业务层只捕基类 |
| 命名 | 与现有模块保持一致：文件名小写、类用驼峰、私有函数 `_` 前缀 |
| 依赖 | **不新增第三方依赖**（用标准库实现倒排索引与限流），避免 demo 膨胀 |

---

## 一、参考项目工程约定调研

### 1.1 配置管理

| 项目 | 实现 | 评价 |
|---|---|---|
| nanobot `config/schema.py`(708行) | Pydantic `BaseSettings` + 嵌套 Config 类，`AliasChoices` 多别名、`Field(ge/le)` 校验、`model_validator` | 功能强，但**对 demo 过重** |
| dify | `pydantic-settings` 分层配置 | 同上 |

**结论**：沿用本项目现有 `getattr` 常量模式。理由：现有 20+ 个模块已依赖该模式，改成 Pydantic 是纯重构成本，无功能收益。新配置项只在 `config.py` 追加常量。

### 1.2 错误处理

| 项目 | 实现 |
|---|---|
| dify `core/errors/error.py` | 扁平异常类 + `description` 类属性（供 API 错误响应直接用） |
| ragas `exceptions.py` | **单一基类 `RagasException` + 子类内置默认文案**（调用方不必写文案） |

**结论**：采用 **ragas 范式**——单基类 + 内置文案子类。比 dify 的 `description` 更省调用方代码，且 30 行内可完成。

### 1.3 日志

**现状已满足**：`app/utils/logger.py` 已有 RotatingFileHandler（5MB×3）+ 统一格式 `%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s`。

**结论**：不新建模块。P0-5 只需加一个 `logging.Filter` 把 trace_id 注入每条日志（约 15 行）。

### 1.4 测试

| 项目 | 实现 |
|---|---|
| nanobot | 极少单测，用 `core_agent_lines.sh` 自证规模 |
| ragas | pytest + 评测数据集驱动 |

**结论**：新增 **2 个测试文件**（RAG 详细 / 横切简略），沿用现有 `tests/*.py` 的 `main()` 可直跑风格（对齐 `test_smalltalk.py`）。

### 1.5 目录分层

nanobot 按职责扁平分组（agent/ bus/ config/ cron/ session/ utils/），单进程取代 OpenClaw 的分布式 gateway/node。

**本项目现状分层已清晰**（`app/{api,core,rag,graph,db,memory,utils,tools}` = 五层 RAG + 图编排），**保持不动**，只在对应层内加文件。

---

## 二、可借鉴点清单：保留 vs 裁剪

### 保留（核心机制，必须落地）

| # | 借鉴点 | 来源 | 落到本项目 |
|---|---|---|---|
| 1 | **RRF 融合优于加权融合** | 对比结论（dify 用加权需归一化且随语料漂移） | `rrf_fuse` **原样保留，一行不改** |
| 2 | **词面路必须独立召回** | dify `retrieval_service.py:781` ThreadPool 三路并发 + `as_completed` 早错早停 | 新增 `LexicalIndex`，`retrieve()` 改并发三路 |
| 3 | **Embedding 缓存 key 含模型名** | dify `cached_embedding.py` | 换 embedding 模型自动失效，避免脏缓存 |
| 4 | **倒排索引 + BM25 打分** | ragflow/dify 关键词索引 | 语料千级内，BM25 标准参数即够 |
| 5 | **单基类异常 + 内置文案** | ragas `exceptions.py` | 新增 `app/core/errors.py` |
| 6 | **trace 用 `contextvars`** | 通用范式 | 异步/线程池场景自动透传，无需逐层传参 |
| 7 | **rerank 与 reorder 正交分离** | dify `data_post_processor.py` | 只做 reorder（零成本），rerank 留接口不实现 |
| 8 | **chunk ID 用内容哈希** | LightRAG `lightrag.py:1898` | 顺手改掉伪增量（1 行改动） |

### 裁剪（过度设计，demo 不需要）

| 裁剪项 | 来源 | 理由 |
|---|---|---|
| jieba 三级降级（default_tfidf → TFIDF → 自建词频） | dify `jieba_kw.py` | 项目已有 `_cjk_runs` 分词，加 jieba 是**新增依赖 + 三级 fallback**，过重 |
| rerank 模型双模式（WEIGHTED_SCORE / RERANKING_MODEL） | dify | rerank 需引入 cross-encoder 模型（数百 MB），demo 不做 |
| ReorderRunner 的可配置策略类 | dify | 直接写成一个纯函数 |
| 图索引 / 实体关系抽取 | LightRAG | 与企业制度问答场景不匹配 |
| Signature 编译 + teleprompt 自动优化 | DSPy | 需要训练/验证集，demo 无 |
| asyncio 消息总线 | nanobot | 本项目是同步图 + FastAPI，非消息驱动 |
| Prometheus / OTLP 导出 | 通用 | 只做内存内 trace 统计 + `/routing/stats` 式接口 |
| Redis 分布式令牌桶 | 通用 | 单机内存滑动窗口即可（Redis 不可用时降级） |
| 多级缓存（L1/L2/L3） | 通用 | 单层 dict + TTL |

---

## 三、目录结构

```
app/
├── core/
│   ├── errors.py          ★新增  统一异常基类（ragas 范式）
│   ├── tracing.py         ★新增  trace_id + span 计时（contextvars）
│   ├── prompts.py         ★新增  Prompt 注册表（收敛 5 处散落常量）
│   ├── rate_limit.py      ★新增  入站限流（内存滑动窗口）
│   └── …（现有 8 个文件不动）
├── rag/
│   ├── lexical.py         ★新增  倒排索引 + BM25 —— RAG 核心
│   ├── reorder.py         ★新增  lost-in-the-middle 重排 —— RAG
│   ├── eval_cases.yaml    ★新增  评测用例数据外置
│   ├── retriever.py       ✎改造  三路并发召回 + 阈值过滤修复
│   ├── indexer.py         ✎改造  同步建词面索引 + chunk_id 改内容哈希
│   ├── evaluator.py       ✎改造  从 YAML 加载用例
│   └── generator.py       ✎改造  build_context 接入 reorder
├── utils/
│   ├── cache.py           ★新增  Embedding 缓存 —— RAG
│   ├── embedding.py       ✎改造  接入缓存
│   └── logger.py          ✎改造  加 trace_id Filter
├── config.py              ✎改造  追加约 14 个配置项
├── main.py                ✎改造  限流中间件 + trace 中间件
├── graph/nodes.py         ✎改造  节点打点
└── api/chat.py            ✎改造  trace_id 透传 + 限流装饰器

tests/
├── test_rag.py            ★新增  覆盖 RAG 三块（详细，约 145 行）
└── test_infra.py          ★新增  覆盖横切四块（简略，约 75 行）
```

**新增 8 个文件 / 改造 8 个文件**，现有分层一个不动。

---

## 四、RAG 三块详细设计

### 4.1 #6 词面路独立召回（最大项，重中之重）

#### 现状缺陷（读码确认，两个）

```
retriever.py:183-302  当前流程
  query → embed → store.search(k=候选50) → raw_hits
                                             ↓
                        dense_pool + lexical_pool（仅对 raw_hits 打分）
                                             ↓
                                        rrf_fuse
                                             ↓
                      阈值过滤（用 score = 向量分）  ← 缺陷②
                                             ↓
                              去重 → top_k → 软回退
```

- **缺陷① 词面非独立召回**：`lexical_pool` 遍历的是 `raw_hits`（向量候选集），**向量池外的纯词面命中根本进不来**。`retriever.py:206` 的注释自己承认了这笔债：「语料膨胀到万级后，词面路应改为独立的倒排索引召回」。
- **缺陷② 阈值过滤用错了分**（`retriever.py:250`）：排序依据 `score`（向量分），过滤也用 `score`。**一个词面高分、向量低分的优质片段会被直接过滤掉**。词面路独立后这个 bug 会更明显，必须一并修。

#### 改造后数据流

```
query ─┬─→ [线程1] embed_query → store.search(k=候选)      → dense_ranked
       ├─→ [线程2] LexicalIndex.search(query, k=候选)      → lexical_ranked  ← 新增独立路
       └─→ [线程3] rewrite_query → 各自再检索（默认关闭）    → rewrite_ranked
                        ↓  ThreadPoolExecutor + as_completed 早错早停
                   rrf_fuse([dense, lexical], weights=[0.7, 0.3])
                        ↓
              阈值过滤（改用 fused 分）                     ← 修复缺陷②
                        ↓
         去重 → top_k → 软回退 → [generator] reorder → 组装上下文
```

#### `app/rag/lexical.py` 设计（约 135 行）

```python
class LexicalIndex:
    """BM25 倒排索引。与向量库同生命周期，json 落盘便于调试。"""

    postings: Dict[str, Dict[str, int]]   # term -> {doc_key: 词频}
    doc_len:  Dict[str, int]              # doc_key -> 词数
    doc_text: Dict[str, str]              # doc_key -> 原文（组装结果用）
    avg_len:  float

    def build(self, chunks) -> None        # 从 Document 列表建索引
    def search(self, query, top_k) -> List[Tuple[str, float]]   # 返回 (doc_key, bm25分)
    def save(self, path) / load(self, path)                     # json 落盘
```

**关键决策与理由**

| 决策 | 选择 | 理由 |
|---|---|---|
| 打分函数 | **BM25**（k1=1.5, b=0.75） | 无外部依赖；比 TF-IDF 多了文档长度归一化，避免长片段占便宜。dify 用 jieba TF-IDF 是要处理大规模语料，demo 不需要 |
| 中文分词 | **复用 `retriever.py:49` 的 `_cjk_runs` + bigram** | 已有现成逻辑，不加 jieba 依赖 |
| 英文/数字 | 正则 `\w+` 小写化 | 与现有 `_latin_lexical_score` 一致 |
| 落盘格式 | **json** | demo 语料百~千条，json 可读可 diff，比 npz 好调试 |
| 并发方式 | **ThreadPoolExecutor** | `retrieve()` 是同步函数（被 graph 以 `to_thread` 调用），改动最小；`as_completed` 首异常即 cancel 其余 future |
| 索引重建时机 | `index_chunks` 后同步重建 | 与向量库保持一致；删除文档后同样重建（demo 级全量重建可接受） |
| 软回退 | 保留 | 现有逻辑正确，不动 |

**顺带修复（1 行）**：`indexer.py:105` 的 `_chunk_id` 从 `uuid5(source::chunk_index)` 改为 `uuid5(source::sha1(content))`。未变的 chunk ID 稳定，增量入库不再全量偏移。

### 4.2 #3 Embedding 缓存

#### 现状
`app/utils/embedding.py`（217 行）**全文无任何缓存**。每次建索引对全量文本重新 embed，每次问答对 query 重新 embed——重建一次索引就重复付费一次。

#### `app/utils/cache.py` 设计（约 85 行）

```python
class EmbeddingCache:
    """两层缓存：查询向量内存 LRU+TTL，文档向量按内容哈希持久化。"""

    def get_query_vec(self, model: str, text: str) -> Optional[List[float]]
    def put_query_vec(self, model: str, text: str, vec) -> None
    def get_doc_vec(self, model: str, text: str) -> Optional[List[float]]
    def put_doc_vec(self, model: str, text: str, vec) -> None
```

| 决策 | 选择 | 理由 |
|---|---|---|
| cache key | `f"{model}:{sha1(text)}"` | **含模型名**——换 embedding 模型自动失效（dify 同款），否则会拿到跨模型的脏向量 |
| 查询向量 TTL | 600s，命中续期 | 与 dify 一致 |
| 文档向量 | 持久化 json（无 TTL） | 文档内容不变则向量永不变，重建索引不重复付费 |
| 并发安全 | `threading.Lock` | #6 引入线程池后 embed 会并发调用 |
| Redis | **不用** | 纯内存 dict + json 落盘即可，Redis 不可用时还要处理降级，过重 |

### 4.3 #1 Reorder（缓解 lost-in-the-middle）

`app/rag/reorder.py`（约 38 行），纯函数，零模型调用零延迟：

```python
def reorder_docs(docs: List[Dict]) -> List[Dict]:
    """按 1,3,5,…,6,4,2 重排：高相关片段放首尾，低相关塞中间。

    LLM 对长上下文首尾的注意力显著高于中部（lost-in-the-middle），
    按相关性递降排列会让最弱片段占据注意力最好的位置之一。
    """
```

**插入位置**：`generator.py:86` 的 `build_context()` 内，**不是** `retrieve()` 内。
理由：reorder 改变的是「喂给 LLM 的顺序」，属于上下文组装阶段；检索结果的 score 顺序应保持原样，否则评测指标和前端展示看到的顺序会被污染。这与 dify 把 `ReorderRunner` 放在 `data_post_processor`（检索后、生成前）位置一致。

---

## 五、横切四块简要设计

| 项 | 文件 | 要点 |
|---|---|---|
| **#2 Prompt 注册表** | `app/core/prompts.py` | 把散落 5 处的常量（generator 22 行 / dream 20 / consolidator 19 / retriever 15 / intent 12 ≈ 88 行原文）搬进一个模块，配 `PROMPTS` 字典 + `get(name, **vars)` 渲染函数 + 变量缺失校验。**不做**版本管理、不做 token 预算、不做 DSPy 式签名编译 |
| **#4 评测 YAML 化** | `app/rag/eval_cases.yaml` + `evaluator.py` 改造 | 用例从 `evaluator.py:53` 的 `DEFAULT_EVAL_CASES`（现仅 4 条硬编码）外置为 YAML，`load_eval_cases()` 读取 + 字段校验。**不做** CI 流水线（demo 无需）、不做 ragas 式 context_precision（无标注数据） |
| **#5 trace_id 贯穿** | `app/core/tracing.py` + logger Filter + main 中间件 | `contextvars` 存 trace_id；`span()` 上下文管理器采集节点耗时；日志 Filter 注入；SSE 与响应头透出。**不做** span 嵌套树、不做 OTLP 导出 |
| **#7 入站限流** | `app/core/rate_limit.py` | 内存滑动窗口，按 IP 维度，默认 30 次/分钟；超限抛统一异常 → 429。**不做** Redis 令牌桶、不做按用户分级配额 |

**统一异常**（`app/core/errors.py`，约 32 行，ragas 范式）：

```python
class AppError(Exception):
    """项目异常基类，业务层只捕这一个。"""
    default_message = "服务内部错误"
    def __init__(self, message=None): ...

class KnowledgeBaseEmpty(AppError): default_message = "知识库为空，请先上传文档"
class EmbeddingFailed(AppError):    default_message = "向量化失败"
class RateLimitExceeded(AppError):  default_message = "请求过于频繁，请稍后再试"
class EvalCaseInvalid(AppError):    default_message = "评测用例格式不合法"
```

---

## 六、文件清单与行数预算

| 文件 | 类型 | 行数 | 说明 |
|---|---|---:|---|
| `app/rag/lexical.py` | 新增 | 135 | 倒排索引 + BM25 + json 落盘（**RAG 核心**） |
| `app/rag/retriever.py` | 改造 | 85 | 三路并发 + 阈值改用 fused |
| `app/rag/indexer.py` | 改造 | 30 | 同步建索引 + chunk_id 改哈希 |
| `app/rag/reorder.py` | 新增 | 38 | 纯函数重排 |
| `app/rag/generator.py` | 改造 | 5 | build_context 接入 reorder |
| `app/utils/cache.py` | 新增 | 85 | Embedding 两层缓存 |
| `app/utils/embedding.py` | 改造 | 18 | 接入缓存 |
| `app/utils/logger.py` | 改造 | 15 | trace_id Filter |
| **RAG 小计** | | **411** | |
| `app/core/errors.py` | 新增 | 32 | 统一异常 |
| `app/core/prompts.py` | 新增 | 105 | 含搬运的 88 行原文 |
| 5 处 prompt 调用点 | 改造 | 10 | 改为 `prompts.get(...)` |
| `app/core/tracing.py` | 新增 | 65 | contextvars + span |
| `app/main.py` | 改造 | 22 | 限流 + trace 中间件 |
| `app/graph/nodes.py` | 改造 | 18 | 节点打点 |
| `app/api/chat.py` | 改造 | 8 | trace 透传 + 限流装饰器 |
| `app/core/rate_limit.py` | 新增 | 52 | 内存滑动窗口 |
| `app/rag/eval_cases.yaml` | 新增 | 55 | 用例数据 |
| `app/rag/evaluator.py` | 改造 | 20 | YAML 加载 |
| `app/config.py` | 改造 | 28 | 约 14 个配置项 |
| **横切小计** | | **415** | |
| `tests/test_rag.py` | 新增 | 145 | RAG 三块（详细） |
| `tests/test_infra.py` | 新增 | 75 | 横切四块（简略） |
| `README.md` | 改造 | 40 | 更新能力表 |
| **测试与文档小计** | | **260** | |
| **总计** | | **1086** | 新增 8 文件 / 改造 8 文件 |

对比上一版估算 **1579 行 → 1086 行（-31%）**，砍掉的正是：jieba 三级降级、rerank 双模式、Redis 限流与缓存、Prometheus 导出、CI 流水线、多级缓存。

---

## 七、需要你确认的 5 个决策点

1. **阈值过滤改用 fused 分**——会改变现有召回结果（部分原先被向量分过滤掉的词面优质片段会回来）。是否接受行为变化？还是保守起见保留向量分过滤、只对纯词面路豁免？
2. **词面路索引全量重建 vs 增量更新**——demo 级建议全量重建（删除文档时重建整个索引，代码少 50 行）。若你预期文档频繁增删，再上增量。
3. **chunk_id 改内容哈希**——会让现有已建索引的全部 ID 变化，需要**重建一次知识库**（一次性成本）。这一项不在原 P0 清单里，是我读码发现的伪增量问题，是否纳入本次？
4. **限流阈值**——默认 30 次/分钟（按 IP）。你的账号上游限流是 RPM=3，前端调试时可能容易误触发，是否放宽到 60？
5. **是否需要我顺带更新 `docs/history/rag-architecture-benchmark.md`** 的落地清单状态（把已实现的项标记完成）？
