# 切分改造 · 变更登记

> ⚠️ **提示：文中提到的上游账号配额（`RPM=3` / 「限频账号」/「免费档」）是写作当时的事实，
> 现已不成立**，请勿当作现状依据——说明见 [`docs/README.md`](../README.md) 第二节的告示。

> 无 git 环境下的变更追溯替代方案（见 T0-1）。
> 每次改动业务代码**之前**执行：`bash scripts/backup.sh <任务号> <文件...>`
> 回退：`cp -R artifacts/backup/<任务号>/. .`
>
> ⚠️ 历史任务（T1-1/T4-1/T4-3/T5-1/T5-2/T6-1/T6-2/DIFY/FIX-P0）的快照已于
> 2026-09-11 移出仓库，归档在 `../_archive/langgraph-enterprise-bot-source-backup-20260911/`
> ——目的是不让旧版源码污染静态扫描与全文检索。回退命令相应改为
> `cp -R ../_archive/langgraph-enterprise-bot-source-backup-20260911/<任务号>/. .`。
> 详见 `artifacts/backup/README.md`。本文件下方的历史登记保持原样，不再改写。

登记格式：任务号 / 改动文件 / 摘要 / 验证命令 / 结果

---

## T0-1 · 无 git 安全网

- **改动文件**：新增 `scripts/backup.sh`、新增 `artifacts/backup/`、本文件
- **摘要**：建立文件级备份 + 变更登记 + 三层回退（配置开关 / 文件备份 / 基线护栏）
- **验证**：`bash scripts/backup.sh T1-1 app/config.py .env .env.example`
- **结果**：✅ 3 个文件备份成功
- **踩坑**：`echo "$BACKUP_DIR（...）"` 中未加花括号的 `$BACKUP_DIR` 紧跟全角 `（`，
  bash 在 UTF-8 locale 下会把全角字符的字节并入变量名 → `unbound variable`。
  **修复**：全角标点前的变量一律写 `${VAR}`。
  —— 本案根因：**未加花括号的变量 + 紧邻多字节字符**。

---

## T0-2 · 基线快照与回归护栏

- **改动文件**：新增 `scripts/chunk_metrics.py`、`scripts/baseline_snapshot.py`、
  `tests/test_chunking_baseline.py`、`artifacts/baseline_chunks.json`
- **摘要**：冻结改造前产物（124 块 + 逐块 sha1），建立回归护栏测试
- **基线实测**：124 块；块长 mean 237.9 / median 246.5 / min 42 / max 298；
  [150,350] 占 94.3%；<80 字 4 个；**跨节率 79.0%**；块首归属率 81.5%
- **验证**：`PYTHONPATH=.:scripts .venv/bin/python -m pytest tests/test_chunking_baseline.py -v`
- **结果**：✅ 4/4 通过
- **护栏有效性已证明**：注入 `CHUNK_SIZE=200` 后 3/4 用例失败（块数 124→211、
  内容哈希消失 112 / 新增 199、块长 237.9→142.4），确认能捕获静默漂移。

---

## T1-1 · 切分参数配置化（默认关闭）

- **备份**：`artifacts/backup/T1-1/`（config.py / .env / .env.example）
- **改动文件**：`app/config.py`、`.env`、`.env.example`
- **摘要**：
  - 新增 `_env_bool()` / `_env_list()` 两个辅助函数（统一此前散落的布尔判定写法）
  - 新增 11 项配置：`CHUNK_STRATEGY` / `CHUNK_TARGET_CHARS` / `CHUNK_HARD_MAX_CHARS` /
    `CHUNK_MIN_CHARS` / `CHUNK_CONTEXT_HEADER` / `CHUNK_ENRICH_METADATA` /
    `CHUNK_FALLBACK_OVERLAP_RATIO` / `INDEX_MIN_CHUNK_CHARS` / `STRUCTURE_*_PATTERNS`(4) /
    `STRUCTURE_FLAT_FALLBACK` / `CHUNK_STRATEGY_SCOPE` / `PARENT_CHUNK_ENABLED`
  - **所有默认值 = 改造前行为**，新逻辑默认关闭
- **验证**：
  - `PYTHONPATH=.:scripts .venv/bin/python -m pytest tests/test_chunking_baseline.py -q`
  - `PYTHONPATH=. .venv/bin/python tests/test_rag.py`
- **结果**：✅ 护栏 4/4 + test_rag 7/7 + test_infra 7/7 + test_smalltalk 7/7 + span 树 3/3

---

## T2-1 · 结构解析层 L1.5

- **改动文件**：新增 `app/rag/structure.py`、新增 `tests/test_structure.py`
- **摘要**：
  - 纯正则行首锚点（章 / 一级 / 小节 / Q&A / Markdown），全部来自配置
  - 输出 `SectionTree`（章节树 + heading_path）+ `split_by_section()` 供 T3-2 消费
  - 无结构文档降级 `flat`；**任何异常一律降级，永不抛错**
  - 顺带抽取文档头（标题 / 版本 / 生效日期 / 更新日期）
- **真实语料分类结果**：13 条记录 → 12 条 `hierarchical`、1 条 `flat`
  （`系统权限与审批速查.txt` 正确判为 flat，与预期一致）
- **验证**：`PYTHONPATH=. .venv/bin/python -m pytest tests/test_structure.py -q`
- **结果**：✅ 20/20 通过
- **踩坑**：Markdown 文档标题抽取时剥掉前导 `#` 后，`_is_anchor_line()` 就认不出
  md 锚点了（md 锚点本身依赖 `^#{1,6}\s`）。**修复**：锚点判定用原文、
  标题值用剥 `#` 后的串；H1（`#[^#]`）单独放行作为文档名。

---

## T2-2 · 重构 chunk_documents 接入结构感知切分（默认关闭）

- **备份**：`artifacts/backup/T2-2/`（indexer.py）
- **改动文件**：`app/rag/indexer.py`
- **摘要**：
  - `chunk_documents` 改为**策略分发**：`_effective_strategy(doc)` 读 `CHUNK_STRATEGY`（+`CHUNK_STRATEGY_SCOPE` 按文档灰度）→ recursive / structure
  - 新增 `_chunk_recursive`（= 改造前逻辑，零变化）、`_structure_merge_blocks`（结构感知合并）、`_chunk_structure`（结构切分入口）
  - `_structure_merge_blocks` 算法：仅合并**同章（L1）内**相邻小节、缓冲 ≤ `TARGET_CHARS`、**跨章边界硬切**；单块 >`HARD_MAX` 走递归拆；overlap 固定 0
  - `_chunk_structure` 内 `CHUNK_CONTEXT_HEADER` 控制「【章节】heading_path」前缀、`CHUNK_ENRICH_METADATA` 写 `heading_path/chapter/section/doc_title/version/date`
  - flat 文档自动回退 `_chunk_recursive`，不抛错
- **实测收益**（13 篇语料）：recursive 124 块 / 跨章率 16% → structure 176 块 / **跨章率 0%**；索引膨胀 **1.42×**（≤2.0× 上限）；168/176 块带 heading_path
- **验证**：`PYTHONPATH=.:scripts .venv/bin/python -m pytest tests/test_structure_chunking.py -v` + 护栏回归
- **结果**：✅ 结构切分 3/3 + 护栏 4/4 + 结构解析 20/20（共 27 项）
- **踩坑**：首版 `_structure_merge_blocks` flush 后误将 `buf=candidate`（含已 flush 旧内容）重新入缓冲，导致 37 块爆成 214 且缓冲无限增长。**修复**：flush 后另起 `buf=content`；并补「跨章保护」防止两章被并一块。

---

## T3-1 / T3-2 · 上下文头 + 元数据增强（开关验证）

- 逻辑随 T2-2 一并落地（不单独改码，符合「优化」要求）
- `test_structure_header_and_metadata_toggle` 已覆盖：关 → 无【章节】前缀、无 heading_path；开 → 两者均出现
- 验收：✅ 开关行为正确，关闭即回退 recursive 等价产物

---

## T4-1 · 评测消费 expect_section（章节命中）

- **备份**：`artifacts/backup/T4-1/`（retriever.py / indexer.py / evaluator.py / eval_cases.yaml）
- **改动文件**：
  - `app/rag/retriever.py`：`_dense_recall` 与词面路补全两处 `meta_by_key` 新增 `"metadata"` 透传完整 chunk 元数据
  - `app/rag/indexer.py`：`index_chunks` 词面索引写入同步 `heading_path/chapter/section`（使词面路命中也能查章节）
  - `app/rag/evaluator.py`：`EvalCase.expect_section` + YAML 解析 + `run_retrieval_eval` 计算 `section_hit_rate` + 新增 `_section_hit` / `filter_by_section`
  - `app/rag/eval_cases.yaml`：4 条用例补 `expect_section`；报销来源修正为 `财务报销管理制度.txt`
- **实测**（本地哈希 embedding 免 API）：4/4 用例解析 expect_section；`section_hit_rate = 1.0`（Top-K 均命中正确章节）
- **验证**：`PYTHONPATH=.:scripts .venv/bin/python -m pytest tests/test_eval_section.py -v`
- **结果**：✅ 4/4 通过
- **风险**：`retrieve()` 返回 dict 新增 `metadata` 字段（加法，不影响 content/source 既有读取）；评测消费为旁路，绝不阻断主链路。

---

## T4-2 · A/B 影子对比（零成本 O2 + 可选 O3）

- **新增**：`scripts/chunking_ab.py`、`docs/history/chunking-ab-report.md`
- **设计（优化）**：默认只跑 **O2 零成本结构指标**（跨章率/膨胀/块长分布/章节覆盖），不调 embedding；`--full` 才跑需真实 API 的 O3 检索评测（RPM=3 限流下分批）。避免 12 篇 × 4 组 × 全量重建的配额浪费。
- **O2 结论（A=recursive 基线 vs D=structure+头+元数据）**：
  - 跨章率 **16% → 0%**（核心收益，结构边界硬隔离）
  - 索引膨胀 **1.42×**（≤2.0× 健康）
  - 平均块长 237.9 → 178.8（更细但语义自洽）
  - 带 heading_path 元数据块 0 → 168
  - 块长分布：150-350 区间 A 117 / D 109（主体健康），D 在 80-150 段更多（同章内小节合并未满 220 即遇章节边界）
- **O3 状态**：脚本就绪，未跑（需 embedding API + 配额）。命令：
  `PYTHONPATH=. .venv/bin/python scripts/chunking_ab.py --full --save docs/history/chunking-ab-report.md`
- **验证**：`PYTHONPATH=. .venv/bin/python scripts/chunking_ab.py` 正常运行并产出报告

---

## T4-2 补 · O3 全量检索评测 + 评测集扩充（2026-09-08）

- **改动文件**：`app/rag/eval_cases.yaml`（4 条 → 25 条）、`docs/history/chunking-ab-report.md`
- **摘要**：跑通 O3 真实检索评测；发现并修复「评测集过小导致指标天花板」
- **结果（25 条用例，只变切分策略）**：

  | 组 | hit_rate | MRR | section_hit_rate |
  |---|---|---|---|
  | A（recursive） | 0.963 | 0.815 | 0.0 |
  | D（structure+头+元数据） | 0.963 | **0.870（+6.7%）** | **1.0** |

  验收标准（hit_rate 不低于 A / MRR 提升 ≥5% / section_hit 显著更高）**全部达标**。
- **踩坑（值得记）**：首轮用 4 条评测集跑，A 与 D 的 hit_rate、MRR **都是 1.000** ——
  天花板效应让两个核心指标完全失去区分度，A/B 会得出「两组一样好」的假结论。
  扩到 25 条（覆盖 11 篇文档，并加入跨文档同名概念如「分级」）后指标才恢复区分度。
  **教训：A/B 前先确认评测集规模足以区分，否则等于白跑。**

---

## T5-1 · 全量切换 + 重建索引（2026-09-08）

- **改动文件**：`.env`（`CHUNK_STRATEGY=structure` + `CHUNK_CONTEXT_HEADER=true`
  + `CHUNK_ENRICH_METADATA=true`）、`artifacts/baseline_chunks.json`、
  `scripts/chunk_metrics.py`、`scripts/baseline_snapshot.py`、`tests/test_chunking_baseline.py`
- **摘要**：L3 全量切换并重建索引；基线快照同步冻结新策略产物（176 块）
- **验证**：索引重建 124 → **176 块**（旧块已 `clear()`，无残留）；
  离线抽检 3 个问题全部命中正确章节（年假→3.2 年休假 / 报销→3.1 提交申请 /
  VPN→2.3 常见故障）；护栏 31/31 → 36/36 通过
- **踩坑**：护栏原先把 `cross_section_rate`（跨**小节**合并率）当缺陷断言 —— 这是错的。
  结构感知切分**刻意**把同章相邻小节合并到目标块长，该指标天然偏高（85%），
  把它当缺陷会得出完全相反的优化方向。
  **修复**：新增真正的缺陷指标 `cross_chapter_rate`（跨**章**率），
  并改写断言为「跨章率 ≤10%」+ 快照记录产出策略。

---

## T5-2 · 兼容层清理与弃用时间表（2026-09-08）

- **改动文件**：`app/config.py`、`app/rag/indexer.py`、`.env.example`、
  新增 `docs/chunking-contract.md`
- **摘要**：移除 `recursive` 的默认特殊地位；明确兼容层清单与移除条件
- **改动要点**：
  - `CHUNK_STRATEGY` 代码默认值 `recursive` → `structure`（此前漏配 `.env` 会静默回落旧策略）
  - 修复 `_effective_strategy()` **硬编码 `"structure"`** 的缺陷：
    「`CHUNK_STRATEGY=recursive` + 设了 scope」时自相矛盾，
    scope 内文档会被切成 structure。改为取 `config.CHUNK_STRATEGY`
  - `recursive` 实现**保留不删**：它是无结构（flat）文档的降级路径，属设计约束
- **文档**：`docs/chunking-contract.md` 冻结接口/元数据字段，写明运维铁律
  （策略变更必须全量重建索引）与 C3 灰度层的移除时间表（2026-10-08 评估）

---

## T6-1 · 父子双层索引（2026-09-08，已实现·默认关）

- **改动文件**：新增 `app/rag/parent_store.py`、`app/rag/indexer.py`、`app/rag/retriever.py`、
  `app/rag/generator.py`、`app/config.py`、新增 `tests/test_parent_chunk.py`
- **摘要**：small-to-big —— 检索用子块（细、准），生成用父块（上下文完整）
- **设计选择**：父块**不进向量库**，存旁路 JSON 按 `parent_id` 回捞。
  相比父子双索引：向量数量与 embedding 成本**零增长**，收益相同。
- **实测**：176 子块 → 95 父块（父子比 1.85x，父块均长 332 字）
- **默认关闭**：收益有限（与清单预判的 1.90x 一致），触发条件满足再开
  （语料 ≥100 篇 / 单篇 >5000 字 / 「命中正确但答案不完整」>20%）
- **验证**：`tests/test_parent_chunk.py` 5/5，含「关闭时必须零变化」的硬断言

---

## T6-2 · PDF 图片抽取（2026-09-08，已实现·默认关）

- **改动文件**：`app/utils/doc_loader.py`、`app/config.py`、新增 `tests/test_pdf_image.py`
- **摘要**：pdfplumber 路径新增内嵌图片抽取（渲染落盘 + 可选 OCR）
- **关键设计**：`PDF_IMAGE_MIN_AREA`（默认 10000 pt²）过滤装饰性小图 ——
  企业 PDF 里大量图标/分隔线/水印会被识别为「图片」，不过滤会污染切分与检索
- **降级**：`pytesseract`/系统 tesseract 缺失 → OCR 文本为空，不影响 PDF 文本抽取
- **实测**：白皮书 PDF 第 1 页图表被抽出（439×247pt，落盘 + 正文占位描述）
- **验证**：`tests/test_pdf_image.py` 6/6

---

## T-DIFY · Dify 外部知识库兼容接口（2026-09-08）

- **改动文件**：新增 `app/api/dify.py`、`app/main.py`、`app/config.py`、
  新增 `docs/dify-integration.md`、新增 `tests/test_dify_api.py`
- **摘要**：实现 Dify External Knowledge API（`POST /retrieval`），
  文档不用再往 Dify 传一份，切分/检索/拒答策略仍由本项目掌控
- **⚠️ 最大的坑（已处理）**：Dify 的 `score_threshold` 是 **0~1** 语义，
  而本项目 `fused` 是 RRF 分，量级仅约 **0.016**。直接透传会让
  「阈值 >0.02」的查询**过滤掉全部结果**，表现为「知识库明明有内容却检索不到」，
  且极难从 Dify 侧定位。适配层已除以 RRF 理论上限归一化
- **其余坑**：`metadata` 不能为 null（已保证返回 `{}`）；
  Dify 只拼 `/retrieval`（已同时注册 `/retrieval` 与 `/dify/retrieval`）
- **验证**：`tests/test_dify_api.py` 18/18；
  真实索引端到端：score 0.99/0.98/0.97，章节元数据正确

---

## 环境备注（与改造相关）

- `pytest` 原先不在项目 venv 中，已安装（9.1.1）
- **既有测试多为「脚本式」**（模块级 `sys.exit`），不能用 pytest 收集：
  `test_rag.py` / `test_infra.py` / `test_smalltalk.py` / `test_span_tree_smoke.py`
  需 `PYTHONPATH=. .venv/bin/python tests/xxx.py` 运行
- `test_service.py` 需本地 8001 端口起服务，离线跑不了
- pytest 可直接收集的：`test_chunking_baseline.py`、`test_structure.py`
