# 切分契约（Chunking Contract）

> 本文是切分改造的**对外契约与运维手册**，对应任务 T5-2。
> 任何改动只要违反本文「冻结项」，视为破坏性变更。

## 1. 冻结项（改动 = 破坏性变更，必须先改本文）

### 1.1 函数签名

```python
# app/rag/indexer.py
def chunk_documents(documents: List[Document]) -> List[Document]
def index_chunks(chunks: List[Document], refit_idf: bool = False) -> int
def build_index() -> Dict[str, Any]
def add_document(file_name: str, content: str) -> int
def delete_document(file_name: str) -> int
```

入参出参类型、顺序、默认值均不可变。新增能力一律走**可选关键字参数**，
不传时行为必须与现状完全一致。

### 1.2 chunk_id 算法（不可变）

```python
digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{digest}"))
```

**这是全链路隐含依赖**：向量库用它做幂等覆盖，词面索引用它做更新，
L5 评测用它对齐结果。改动它等于让索引里所有旧块变成孤儿。

⚠️ 由此推导出的运维铁律见 §4。

### 1.3 chunk 元数据字段

| 字段 | 来源 | 说明 | 稳定性 |
|---|---|---|---|
| `source` | 文档路径 | 参与 chunk_id 计算 | 冻结 |
| `chunk_index` | 组内序号 | 词面索引 key 的一部分 | 冻结 |
| `chunk_chars` | 正文长度 | 不计量上下文头 | 冻结 |
| `structure` | `hierarchical` / `flat` | 结构解析结果 | 稳定 |
| `heading_path` | `章 > 节` | 需 `CHUNK_ENRICH_METADATA=true` | 稳定 |
| `chapter` / `section` | 由 `heading_path` 派生 | 同上 | 稳定 |
| `doc_title` / `doc_version` | 文档头解析 | 同上，可能为空 | 稳定 |

> **关于 `METADATA_MIGRATION` 映射表**：清单 T3-4 曾设计一张 Pydantic 式
> 四分类声明式映射表（新增/重命名/废弃/回退）。实际落地时**刻意未引入**：
> 当前字段集仅 8 个且已稳定，映射表增加的间接层成本大于收益。
> 改由上表承担「字段演进可追溯」的职责 —— 字段变更时同步更新本表即可。
> 若未来字段数 >15 或出现跨版本兼容需求，再引入声明式映射。

---

## 2. 兼容层清单与弃用时间表

| # | 兼容层 | 现状 | 处置 | 移除条件 |
|---|---|---|---|---|
| C1 | `recursive` 切分策略 | 活跃 | **永不删除** | 无 —— 见下方约束 |
| C2 | `STRUCTURE_FLAT_FALLBACK` | 默认开 | 保留 | 无 —— 同类约束 |
| C3 | `CHUNK_STRATEGY_SCOPE` 灰度 | 未启用（留空） | 观察 30 天后评估 | 全量稳定 30 天且未再需要灰度 |
| C4 | `CHUNK_FALLBACK_OVERLAP_RATIO` | 已接线（默认 0.12） | 保留 | 随 C1 |

### C1 / C2 是设计约束，不是临时兼容

`recursive` 与 `STRUCTURE_FLAT_FALLBACK` 处理的是**无结构文档**
（纯段落、无标题锚点的 PDF 抽取文本等）。语料里这类文档会持续出现，
因此这两条路径是**功能的一部分**，不因改造完成而失效。

判断标准：删掉它，新格式的文档能不能被正确处理？
答案是不能 → 保留。（这也符合 LangChain 的做法：弃用有时间表，
但降级路径不在弃用范围内。）

### C4 的接线语义（已落地）

`CHUNK_FALLBACK_OVERLAP_RATIO` 此前只在配置里声明、**无任何读取点**，属于
「文档说保留、代码没接线」——改它不会有任何行为变化。现已接到唯一一条降级路径上：

```
_chunk_structure()  →  parse_document() 判定为 flat
                    →  _chunk_recursive(doc, degraded=True)
                    →  _splitter(_fallback_overlap())     # ← 本配置在此生效
                       _fallback_overlap() = round(CHUNK_SIZE * ratio)
```

三条必须记住的边界：

1. **只影响降级路径**。显式 `CHUNK_STRATEGY=recursive` 仍用 `CHUNK_OVERLAP`，
   两条路径互不影响——降级文档段落边界弱、内容密度低，用大重叠换来的语义
   连续性有限，却会明显放大「同段内容被多个块重复持有」。
2. **改它要重建索引**（理由同 §4：块内容变化 → `chunk_id` 变化）。
3. **回退靠改配置，不改代码**。想精确复现接线前行为（降级路径也用
   `CHUNK_OVERLAP`），把本值设为 `CHUNK_OVERLAP / CHUNK_SIZE`（当前默认
   配置下即 `0.2`）即可。

> 当前语料下（13 篇中仅 `系统权限与审批速查.txt` 为 flat）接线**未改变任何
> 切分产物**，`artifacts/baseline_chunks.json` 无需重建——两条重叠值在该文档上
> 命中的是同一组分隔符边界。护栏 `tests/test_chunking_baseline.py` 保持全绿，
> 即「接线是零行为变化」的机器证明。

### C3 的明确时间表

- **移除条件**：`CHUNK_STRATEGY=structure` 全量运行满 30 天，且期间
  未出现「需要用 scope 隔离某篇文档」的场景。
- **检查日期**：2026-10-08
- **移除动作**：删除 `config.CHUNK_STRATEGY_SCOPE` 与
  `indexer._effective_strategy()`，让 `chunk_documents` 直接用
  `config.CHUNK_STRATEGY` 分发。

---

## 3. 指标口径（容易混淆，务必按此理解）

| 指标 | 含义 | 方向 | 当前值 |
|---|---|---|---|
| `cross_chapter_rate` | 块内混入 **≥2 个不同一级章** | **越低越好**（缺陷，目标 0%） | 2.3%（改造前 16.9%） |
| `cross_section_rate` | 块内混入 **≥2 个小节** | **不是缺陷**（设计目标） | 85.2% |
| `head_attributed_rate` | 块首 40 字内有章节锚点 | 越高越好 | 79.0% |
| `pct_in_150_350` | 块长落在 150–350 字 | 越高越好 | 62.5% |

⚠️ **不要把 `cross_section_rate` 当缺陷去优化。** 结构感知切分**刻意**
把同章相邻小节贪心合并到目标块长，所以它天然偏高。把它当缺陷会得出
完全相反的优化方向（把块切碎，反而破坏同章语义完整性）。

两者由 `scripts/chunk_metrics.py` 计算，零 API 成本。

---

## 4. 运维铁律：策略变更必须全量重建索引

因为 `chunk_id` 是内容哈希（§1.2）：

```
切分策略变更
  → 块内容变化
  → chunk_id 变化
  → 增量入库时旧块不会被覆盖，而是作为新块并存
  → 检索时新旧块混合返回，答案自相矛盾
```

**正确操作顺序**：

```bash
# 1. 改配置
vim .env          # CHUNK_STRATEGY / CHUNK_CONTEXT_HEADER / CHUNK_ENRICH_METADATA

# 2. 全量重建（会 store.clear() 后重灌，杜绝残留）
PYTHONPATH=. .venv/bin/python -c "from app.rag.indexer import build_index; print(build_index())"

# 3. 重生成基线快照（否则护栏会误报漂移）
PYTHONPATH=.:scripts .venv/bin/python scripts/baseline_snapshot.py --update

# 4. 跑护栏
PYTHONPATH=. .venv/bin/python -m pytest tests/test_chunking_baseline.py -q
```

**回退**：同样走上述 4 步，把配置改回 `recursive` + 两个 `false`。
回退不能只改配置不重建 —— 理由同上。

---

## 5. 回归护栏（无 git 环境下的唯一自动化检测）

| 文件 | 作用 |
|---|---|
| `artifacts/baseline_chunks.json` | 冻结当前策略的切分产物（含内容哈希 + 产出配置） |
| `tests/test_chunking_baseline.py` | 逐条比对哈希；块数零容忍，块长 ±10，比率 ±2pp |
| `scripts/backup.sh <task-id> <files>` | 改代码前的文件级备份 → `artifacts/backup/<task-id>/`（**只放进行中的任务**；已完结任务的快照归档在仓库外，见 `artifacts/backup/README.md`，避免旧版源码污染扫描与检索） |

护栏失败时先做这两件事：
1. 看快照里的 `config.chunk_strategy` 是否与当前配置一致（策略变了就要 `--update`）
2. 不一致 → `cp -R artifacts/backup/<最近任务ID>/. .` 回退（若该任务已归档，路径换成
   `../_archive/langgraph-enterprise-bot-source-backup-20260911/<任务ID>/`）
