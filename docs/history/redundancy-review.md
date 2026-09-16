# 冗余实现与过度特化审查报告

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> 审查对象：`langgraph-enterprise-bot`（企业级 LangGraph RAG 智能助手，13,853 行 Python）
> 审查角度：**冗余实现 / 多余的特殊处理（special-casing）/ 过度定制化 / 硬编码 / 可被通用方案替代**
> 审查方法：4 路并行精读（rag / core+config / api+graph+main / 基础设施）+ 逐条二次验证
> 与 `docs/history/project-assessment.md` 的区别：那份报告看的是**性能/安全/错误处理**，本报告只看**通用性与可维护性**，不重复其结论。

## 判定标准

| 标记 | 含义 |
|---|---|
| ✅ 已验证 | 本报告作者用 Grep/Read 二次核实过 `file:line` 与论断 |
| 类型·冗余 | 同一功能有第二份实现，或写了无人使用的代码/配置 |
| 类型·特化 | 为单一场景/单一语料/单一数据写死，换场景即失效 |
| 类型·通用替代 | 自造了标准库或已依赖框架已提供的能力 |

**"合理特化"不计为问题**：兼容上游缺陷、明确的降级策略、性能必需的缓存等，已在文末单独列出。

---

# 一、S0：双份实现与逻辑分叉（通用性风险最高）

## S0-1 ✅ 已验证 · `/ask` 与 `/ask/stream` 是两套独立的工作流实现，且已实质分叉

- **位置**：非流式 `app/api/chat.py:82`（`enterprise_workflow.invoke`）vs 流式 `app/api/chat.py:190-208`（`_pre()` 手工串联节点）；分支判定被重写于 `app/api/chat.py:198-208`，其"真身"是 `app/graph/edges.py:9-25` 的 `intent_route_edge`
- **具体表现**：流式链路绕过编译图，自己写了一遍意图分支：
  ```python
  # chat.py:197-207（与 edges.py:18-25 同义的第二份拷贝）
  if intent == "tool":        state = tool_invoke_node(state)
  elif intent == "knowledge": state = knowledge_retrieve_node(state)
  elif intent == "smalltalk":
      state = smalltalk_reply_node(state); return state
  state = model_route_node(state)
  ```
  函数 docstring（`chat.py:163-164`）声称"与非流式链路共享同一套实现，避免逻辑分叉"，实际只共享了**节点函数**，没有共享**图拓扑**。
- **已分叉的 5 个具体点**：
  1. **异常兜底链缺失**：非流式经 `nodes.py:311-350`（`generate_answer_node`）→ `edges.py:28-50`（`error_route_edge`）→ `nodes.py:356-367`（`human_fallback_node`）；流式根本不调用这些，生成失败直接在 `chat.py:277-281` 发 `error` 事件返回，**永不产生转人工回答**。
  2. **`need_human` 语义丢失**：流式 state 从不写 `need_human`，前端 `meta.need_human`（`chat.py:316`）恒为 `False`。
  3. **错误信息策略相反**：非流式刻意脱敏（`chat.py:102-104`），流式直接回显 `f"对话服务异常：{exc}"`（`chat.py:280`）。
  4. **兜底文案不一致**：`chat.py:84` "抱歉，未能生成有效回答…" vs `chat.py:261` "抱歉，本次回答生成失败…"。
  5. **寒暄字段二次硬编码**：`smalltalk_reply_node` 已在 `nodes.py:236-239` 写好 `citations/confidence/refused`，流式却在 `chat.py:225` 再硬写一遍。
- **影响**：**换前端只要用到流式，就拿到与非流式不同的错误行为、字段与文案**。新增意图/节点/兜底策略必须同时改 `edges.py` 与 `chat.py`，漏改**无任何报错**，只静默走错分支。
- **通用方案**：用 LangGraph 原生流式替代手写串联——`enterprise_workflow.astream(state, stream_mode="updates")` 或 `astream_events(..., version="v2")`，节点内用 `StreamWriter` 推送 token。拓扑、条件边、`human_fallback` 全部由编译图保证，`/ask/stream` 只需把事件映射成 SSE 帧。
- **置信度**：高

## S0-2 ✅ 已验证 · 两条链路的响应体近 30 行重复，且字段已漂移

- **位置**：`app/api/chat.py:119-149`（`/ask` 返回体）vs `app/api/chat.py:290-321`（`/ask/stream` 的 `meta` 事件）
- **具体表现**：`"sources"` 列表的 11 行构造在 `chat.py:127-137` 与 `298-308` **逐字相同**；`intent/intent_source/model_tier/route_decision/citations/confidence/refused/tool_result/trace/span_tree/need_human/elapsed_ms/llm_mode/history_rounds` 全部重复。唯一差异是流式多了 `"cited": extract_cited_indexes(...)`（`chat.py:310`）——**字段已经分叉**。同处还复制了魔数 `min(len(history) // 2 + 1, 10)`（`chat.py:148` 与 `320`），无常量无注释。
- **影响**：改任一响应字段要改两处；`cited` 的存在证明漂移已真实发生；客户端无法用同一套解析逻辑处理两个端点。
- **通用方案**：抽纯函数 `build_chat_meta(state, history, answer, span_tree, elapsed) -> dict`，或用 Pydantic `response_model` 定义 `ChatResponse` 并序列化；`history_rounds` 上限提为 config。
- **置信度**：高

## S0-3 · RRF 归一化上限公式在两处各算一遍

- **位置**：`app/api/dify.py:85`（`upper = (DENSE_WEIGHT + LEXICAL_WEIGHT) / (RRF_K + 1)`）vs `app/rag/generator.py:144`（`fused_max = (DENSE_WEIGHT + LEXICAL_WEIGHT) / (RRF_K + 1)`）
- **具体表现**：`dify.py:73-88` 的注释明写"与项目内置信度计算口径一致"，但它是**再算一遍**而非调用。
- **影响**：一旦引入第三路召回或改 RRF 权重，两处必须同改，否则 Dify 的 `score_threshold` 与内部置信度静默错位——正是 `dify.py:76-80` 注释描述的"知识库有内容却检索不到"那个坑。
- **通用方案**：在 `generator.py` 暴露 `fused_upper_bound() -> float`，`dify.py` import 使用。
- **置信度**：高

## S0-4 · 健康检查有两个端点，鉴权豁免却不一致

- **位置**：`app/main.py:186-195`（`GET /health`）vs `app/api/test.py:27-33`（`GET /test/health`），二者都包装 `app/core/self_check.py:220` 的 `get_health()`
- **具体表现**：差异仅在于 `/health` 多包一层 `startup_check`、`/test/health` 多一个 `verbose` 分支。但 `/health` 在 `AUTH_EXEMPT_PATHS` 内，`/test/health` **不在**。
- **影响**：开启 `AUTH_ENABLED` 后，前端面板调用的 `/test/health`（`static/index.html:926,1066`）会 401，而容器探针 `/health` 正常；两个端点返回结构还不同，客户端需分别处理。
- **通用方案**：只保留 `/health`，用查询参数区分详略；或把 `/test/health` 设为别名并纳入豁免清单。
- **置信度**：高

## S0-5 · Bearer 密钥解析有两份，FastAPI 已内置 `HTTPBearer`

- **位置**：`app/main.py:101-109`（`_extract_api_key`，切 `X-API-Key` 与 `"bearer "` 前缀）vs `app/api/dify.py:147-156`（`_check_auth`，用 `split(None, 1)` 判 `bearer`）
- **具体表现**：两处各自实现"从 Authorization 头取 token"，规则不同（前缀匹配 vs 分词，大小写宽容度不同）。
- **影响**：鉴权头兼容性修复要改两处；对 `Bearer\txxx`、大小写混写的边界行为不一致。
- **通用方案**：`fastapi.security.HTTPBearer` + `APIKeyHeader` 组合成 `Depends`，统一提取。
- **置信度**：高

## S0-6 · 同一"词面相关性"有两套打分，最后用 `max()` 跨量纲合并

- **位置**：`app/rag/retriever.py:67-120`（0~1 字符覆盖度 + 0.3 子串奖励）与 `app/rag/lexical.py:186-229`（无上界 BM25）；合并在 `app/rag/retriever.py:370-379`
  ```python
  lexical_merged[key] = max(lexical_merged.get(key, 0.0), bm25)
  ```
- **具体表现**：两个数不同量纲，`max` 的结果由"哪个分支恰好命中"决定而非相关性；注释也自认"BM25 分无上界，与 0~1 的词面分不在同一量纲"。
- **影响**：词面路排序在两种分数交界处不可解释、不可调参；换语料/换切分粒度时混合点整体漂移，L5 的调优建议失去依据。
- **通用方案**：只保留 BM25 作为词面路（其自带长度归一化），`lexical_score` 降级为纯展示字段；若要两路，则作为**两个独立 RRF 输入**分别给权重，让排名融合处理量纲。
- **置信度**：高

## S0-7 ✅ 已验证 · `_RouteStats` 在两处重复实现，其中一份是死代码

- **位置**：`app/core/model_router.py:400`（`class _RouteStats`）与 `app/core/intent_router.py:98`（同名类）；`app/api/routing.py:18-26` 只导入 model_router 版本
- **具体表现**：两个类结构近似，都手写 `rule_hit_rate()`/`to_dict()`/`rule_detail_hits` 排序取前 8；`intent_router.py:122-129` 的 `get_route_stats`/`reset_route_stats` 全仓无调用者。并发语义还不一致（model_router 用 `_STATS_LOCK`，intent_router 直接自增）。
- **影响**：意图路由的规则命中率**无处可查**（尽管 docstring 称它是"该不该再加规则"的依据）；两套统计口径无法统一展示；intent 侧多线程计数有竞态。
- **通用方案**：抽通用 `RouteStats`（或 `collections.Counter` + 计数器数据类）供两处复用。
- **置信度**：高

---

# 二、S1：同一规则/常量在多处复制（改一处必漏）

## S1-1 ✅ 已验证 · 工单号正则 `T\d{3,}` 存在 4 份

- **位置**：`app/graph/nodes.py:40`（`_TICKET_PATTERN`）、`app/providers/llm.py:39`、`app/core/model_router.py:102`、`app/core/intent_router.py:40`
- **具体表现**：同一正则四种写法（后两者加了 `\b` 边界、规则名不同）。关联地，`nodes.py:41` 的 `_USER_HINT` 与 `intent_router.py:41` 的员工信息关键词是同一张表的两次手抄，**已不一致**（后者有"分机号/隶属"，前者是"分机"）。
- **影响**：改工单号规则要动 4 处；关键词不一致会导致"意图判为 tool、但 `tool_invoke_node` 抽不出标识"的组合，落入 `nodes.py:181` 的兜底文案。换业务域时 4 处全要重写。
- **通用方案**：抽 `app/core/patterns.py` 暴露 `TICKET_PATTERN`/`USER_HINT_WORDS` 单一常量；或让 `tool_invoke_node` 复用 `intent_router` 已命中的规则元数据。
- **置信度**：高

## S1-2 ✅ 已验证 · 中文停用词表与分词函数在 retriever / lexical 逐字复制

- **位置**：`app/rag/retriever.py:42-64`（`_CJK_CHAR`/`_cjk_runs`/`_CJK_STOP`）与 `app/rag/lexical.py:37-50`（同名三份）
- **具体表现**：`lexical.py:40` 的注释自己写"**与 retriever._CJK_STOP 一致**"、第 12 行写"分词复用 retriever._cjk_runs 的思路"——但实现是复制粘贴，不是复用。`lexical.py:48` 与 `retriever.py:52` 的函数体一字不差。
  > 说明：这两个**词表/分词本身**的存在是**合理特化**（中文无空格、虚词不剔除则词面路失效），问题只在于"两份"。
- **影响**：停用词表是中文检索判别力的来源，只在一处补词（如加"麻烦/请问下"）会让两路召回口径分叉，且无任何机制提醒。
- **通用方案**：抽 `app/rag/_text.py` 导出 `CJK_CHAR`/`CJK_STOP`/`cjk_runs`，两处 import；CJK 码位范围可用 `unicodedata` 或 `regex` 的 `\p{Han}`。
- **置信度**：高

## S1-3 ✅ 已验证 · 支持的文件后缀列表重复定义

- **位置**：`app/utils/doc_loader.py:20`（`SUPPORTED_SUFFIX = {".pdf",".md",".markdown",".txt"}`）与 `app/rag/prepare.py:95`（内联 `for ext in (".pdf", ".md", ".markdown", ".txt"):`）
- **影响**：新增格式（如 `.docx`）只改一处时，`file_type` 元数据静默变成 `"unknown"`，而 L4 引用溯源/L5 报告都消费该字段——不报错、只数据错。
- **通用方案**：`prepare._infer_file_type` 引用 `doc_loader.SUPPORTED_SUFFIX`，并优先用 `Path(source).suffix`。
- **置信度**：高

## S1-4 ✅ 已验证 · 文档 key 格式 `{source}::{chunk_index}` 三处拼装，其中一处是死代码

- **位置**：`app/rag/retriever.py:216-220`（`_hit_key`）、`app/rag/indexer.py:434`（写入）、`app/rag/lexical.py:74`（`_doc_key`，**全仓无调用者**）
- **影响**：这是 RRF 融合/去重/词面回捞的共同主键，格式一旦某处漂移（如默认值 `-1` vs 缺失），融合会静默错配成两条不同 key。
- **通用方案**：收敛为 `rag/keys.py: doc_key(source, chunk_index)` 单一实现，删除 `_doc_key`。
- **置信度**：高

## S1-5 · 取文件名动作在手写 6+ 处，跨平台处理不一致

- **位置**：`app/rag/prepare.py:112`、`app/rag/generator.py:117`（用 `split("/")[-1].split("\\")[-1]`）；`app/rag/evaluator.py:170`、`app/rag/retriever.py:478`、`app/api/chat.py:130,301`、`app/api/dify.py:196,211`（仅 `split("/")[-1]`）
- **影响**：Windows 来源（`C:\...\员工手册.txt`）在多数位置会显示成整条路径；修一处漏五处。而 `prepare.enrich_metadata` 早已写入 `file_name` 元数据，却未被复用。
- **通用方案**：统一 `os.path.basename` / `PurePath(source).name`；消费方直接读 `doc["metadata"]["file_name"]`。
- **置信度**：高

## S1-6 ✅ 已验证 · `_MIN_RULE_CONFIDENCE` 两份，且是永不触发的死分支

- **位置**：`app/core/model_router.py:98` 与 `:128`；`app/core/intent_router.py:69` 与 `:159`
- **具体表现**：两张规则表里最低置信度分别是 0.70（`model_router.py:119`）与 0.75（`intent_router.py:60`），全部 > 0.6，因此 `if confidence < _MIN_RULE_CONFIDENCE: continue` **永不执行**；常量名/值/注释在两文件重复。
- **影响**：读者会误以为存在"低置信规则被过滤"的保护；将来想加 0.5 的弱规则时，过滤位置与阈值分散在两模块。
- **通用方案**：删除该分支（置信度已写在表内即代表取舍），或收敛到一处共享常量并真正生效。
- **置信度**：高

## S1-7 · 通用问候语文案两处硬编码

- **位置**：`app/core/smalltalk.py:66`（greet 候选）与 `app/graph/nodes.py:212-215`（`_GENERIC_SMALLTALK_REPLY`）
- **影响**：改对外话术要同改两处，否则同一用户在不同输入上看到两种自我介绍。
- **通用方案**：把通用回复并入 smalltalk 规则表（如 `category="generic"`），`nodes.py` 从该模块取。
- **置信度**：高

## S1-8 · 寒暄词表与意图正则跨 4 个文件重复，部分规则结论等价

- **位置**：`app/core/smalltalk.py:57-143`；`app/core/model_router.py:102,113-119`；`app/core/intent_router.py:40`
- **具体表现**：同一组"你好/谢谢/再见"枚举在 smalltalk 与 model_router 各写一遍；`T\d{3,}` 在 2 处重复（见 S1-1）。更关键的是 model_router 的 3 条寒暄降档规则（0.95）与同表末条 `^[\s\S]{1,14}$`（0.70）产出的**都是 flash 档**——置信度在 `route_model` 里只用于拼 reason 字符串，不参与决策，故这 3 条属冗余项。
- **通用方案**：model_router 直接消费 `smalltalk.detect_smalltalk()` 的类别，删除等价降档规则。
- **置信度**：高

## S1-9 · 请求校验器逐字重复

- **位置**：`app/utils/validator.py:60-66`、`:119-125`、`:134-137`（`clean_query` 三份）；`:68-73` 与 `:75-80`（`validate_user_id` 与 `validate_session_id` 逻辑同构）
- **影响**：改 `SESSION_ID_PATTERN` 或过滤规则需同步 5 处；三份 `clean_query` 的异常文案已开始漂移。
- **通用方案**：用 `Annotated[str, AfterValidator(clean_query)]` / 共享 `@field_validator`，或定义 `SessionId = Annotated[str, StringConstraints(pattern=...)]` 复用。
- **置信度**：高

## S1-10 · 脚本间的指标计算与章节正则重复（第三份）

- **位置**：`scripts/chunking_ab.py:32`（`_CHAPTER_RE`）、`:35-38`（`_strip_header`）、`:52-81`（`_metrics`）vs `scripts/chunk_metrics.py:25-31,43-48,84-136`（`compute_metrics`）
- **具体表现**：`chunk_metrics.py` 已提供 `compute_metrics` 并声明"被 scripts/ 与 tests/ 消费"，但 `chunking_ab.py` 又独立定义章节正则与 `_metrics`，且**分箱阈值不同**（80/150/350 vs 150-350 区间）。
- **影响**：A/B 报告与 `baseline_snapshot.py:87` 冻结的"跨章率"来自两套定义，数值不可直接比较，可能得出相反结论。
- **通用方案**：`chunking_ab.py` 直接 `from chunk_metrics import compute_metrics`（`baseline_snapshot.py:48` 已示范），删除重复实现。
- **置信度**：高

## S1-11 · `chunking_ab.py` 验收判定重复 4 份 + 基线块数硬编码

- **位置**：阈值 `scripts/chunking_ab.py:107-109`；同一三元判定拷贝于 `:159-161`、`:167-170`、`:297-299`、`:305-308`；硬编码 `baseline_count: int = 124`（`:112`、`:205`、`:248`）
- **影响**：调阈值需改 4 处，漏改会出现"表格显示候选、正文推荐为空"的自相矛盾；换数据集后 `expansion` 失真、结论静默错误。
- **通用方案**：抽 `is_eligible(row)` / `recommend(rows)`；基线块数从 `artifacts/baseline_chunks.json`（`baseline_snapshot.py:32` 已产出）读取。
- **置信度**：高

---

# 三、S1：死代码 / 死配置（写了没人用的冗余）

## S2-1 ✅ 已验证 · 死配置：定义了、写了注释、但无人读取

| 配置项 | 位置 | 核查结果 |
|---|---|---|
| `CHUNK_MIN_CHARS` | `config.py:322` | 仅 `scripts/baseline_snapshot.py:80` 读来做快照，**业务逻辑零引用**；结构切分实际用 `INDEX_MIN_CHUNK_CHARS`(`indexer.py:37`) + `CHUNK_TARGET/HARD_MAX_CHARS` |
| `CHUNK_FALLBACK_OVERLAP_RATIO` | `config.py:332` | **全仓零引用** |
| `CASCADE_MAX_PER_SESSION` | `config.py:130` | **全仓零引用** |
| `CASCADE_MIN_CONFIDENCE` | `config.py:132` | **全仓零引用** |
| `CASCADE_ENABLED` | `config.py:129` | 仅作展示字段被读（`api/routing.py:65`、`config.py:541`），**不存在任何级联逻辑**；`model_router.py:12` 的 docstring 却宣称"L3 级联 Flash 不达标升 Pro" |

- **影响**：这是 config 自述在 `config.py:205-211` 已修复的"伪配置"问题在切片组/路由组的**残留复现**。运维照 `.env.example` 调 `CASCADE_*` 毫无效果；`CHUNK_MIN_CHARS` 与 `INDEX_MIN_CHUNK_CHARS` 两个近义配置并存，读者不知该改哪个。
- **通用方案**：要么在对应路径真正消费，要么删除配置项并同步删除 docstring 中的能力声明。
- **置信度**：高（全仓引用扫描）

## S2-2 ✅ 已验证 · 死代码：仅被测试调用、或完全无调用

| 符号 | 位置 | 调用情况 | 备注 |
|---|---|---|---|
| `lexical._doc_key` | `app/rag/lexical.py:74` | **零调用** | 与 `retriever._hit_key`/`indexer` 内联重复（见 S1-4） |
| `tracing.get_span_tree` | `app/core/tracing.py:142` | **零调用** | 与 `end_trace`（`:126-139`）逻辑重复，调用方都用 `end_trace()` |
| `api/evaluation.EvalCaseRequest` | `app/api/evaluation.py:18-23` | **零调用** | 同文件 `:35-43` 反而手工拼 `EvalCase`，Pydantic 能力没用上 |
| `prompts.render` | `app/core/prompts.py:103-112` | 仅 `tests/test_infra.py:78` | 生产一律走 `ChatPromptTemplate`；该函数还需手工转义 `{{ }}` |
| `evaluator.filter_by_section` | `app/rag/evaluator.py:214` | 仅 `tests/test_eval_section.py` | docstring 称"供生成端使用"，但生成端未使用 |
| `structure.detect_structure` | `app/rag/structure.py:285` | 仅 `tests/test_structure.py:84` | 为拿一个字符串却跑完整 `parse_document` 建树 |

- **影响**：测试反向塑形生产 API；维护者会误以为存在"章节过滤""模板渲染""结构探测"等生产特性。属"为让测试通过而保留的接口"。
- **通用方案**：删除无生产调用者的符号，测试改为直接验证底层函数（如 `parse_document(...).structure`）；若确要保留能力，应在生产路径真正接入。
- **置信度**：高

## S2-3 ✅ 已验证 · `soft_warnings` 是只写不读的状态字段

- **位置**：写入 `app/graph/nodes.py:137`；声明 `app/graph/state.py:47,78`；`edges.py:45` 仅注释提及
- **具体表现**：除写入点与注释/测试外，全仓无任何**读取**点；两个 API 响应体也不含该字段。
- **影响**：注释说它"只用于观测"，但实际观测不到——检索失败的降级信息既不进结构化日志、也不进响应或 trace。运维层面等于没有观测通道。
- **通用方案**：并入 `trace` 的 detail（`_trace` 已具备承载能力），或在响应/日志中真正暴露；否则删除以免制造"已有观测"的错觉。
- **置信度**：高

## S2-4 ✅ 已验证 · 消费方 `getattr(config, X, 默认值)` 兜底 13 处，与 config 自述直接冲突

- **位置**：`app/utils/doc_loader.py:90-93,133`、`app/core/model_router.py`（转口）、`app/core/rate_limit.py:20`、`app/core/source_acl.py:31`、`app/rag/retriever.py:36-39`、`app/rag/indexer.py:37`、`app/rag/prepare.py:31-32`、`app/rag/generator.py:36`、`app/rag/lexical.py:34-35`、`app/rag/evaluator.py:32`、`app/rag/parent_store.py:41`、`app/utils/cache.py:25-26`、`app/providers/embeddings.py:149`
- **具体表现**：`app/config.py:204-211` 的注释明确自述"现已补齐定义，并把消费方改为**运行时读取**"，但这些消费方仍是**模块级 import 期快照**，默认值在 config 与消费方各写一份（如 `RRF_K=60` 见 `config.py:212` 与 `retriever.py:36`）。`source_acl.py:31` 的兜底更是完全多余——`config.SOURCE_ACL` 在 `config.py:476` 已定义。
- **影响**：(a) 测试里 `monkeypatch.setattr(config, "RRF_K", ...)` 对已导入模块无效（`retriever.RRF_K` 仍是旧值）——这是"改了配置没生效"的经典陷阱；(b) 默认值两处维护，改一处另一处照旧兜底，问题被静默掩盖。
- **通用方案**：消费方改为函数内 `config.X` 直接属性访问；或统一用 `pydantic-settings` 提供带类型校验的单一来源。
- **置信度**：高

---

# 四、S1：过度特化 / mock 数据写进生产代码

## S2-5 ✅ 已验证 · 为 5 个 mock 用户写死的姓名正则

- **位置**：`app/graph/nodes.py:196`
  ```python
  m = re.search(r"([张李王赵孙][三四五六七])", query)
  ```
- **具体表现**：该字符类恰好只能匹配"张三/李四/王五/赵六/孙七"——即 `app/tools/user_tool.py:7-15` 的 `_MOCK_USERS` 全部 5 个中文姓名。这不是"中文姓名提取"，而是**把 mock 数据反写进了业务代码**。
- **影响**：`user_tool._resolve()` 明确支持"工号/用户名/中文姓名/邮箱前缀"，但 `nodes.py` 只放行英文账号（`[A-Za-z]{3,20}`）与这 5 个写死姓名。换任何真实通讯录（"欧阳靖""陈晓"）立刻失效，fail 到 `nodes.py:181`"未识别到有效业务参数"。典型"能过测试但上不了生产"。
- **通用方案**：删掉姓名正则，把候选标识交给工具层解析（`query_user_info` 已实现别名解析）；真正的 NER 应由 LLM 层完成。
- **置信度**：高

## S2-6 · 三处 mock 业务数据写进生产工具，且默认即数据源

- **位置**：`app/tools/ticket_tool.py:15-20,34,65,75-77`；`app/tools/user_tool.py:7-13,15,33,44-46`；`app/tools/rule_tool.py:7-17,30,38-40`
- **具体表现**：`user_tool` 根本没有远程实现，`_MOCK_USERS` 是唯一数据源；`ticket_tool` 因 `TICKET_USE_REMOTE` 默认 `false`，本地 mock 库是**默认生效路径**；`rule_tool` 用 `k in lowered` 子串匹配返回整块制度原文（与 RAG 形成两套并行知识来源）。
- **影响**：换业务/换企业时三个工具全部失效，且"演示数据"与"生产数据"没有边界；迁移到真实 LDAP/工单系统必须改动工具内部而非替换数据源。
- **通用方案**：定义 `TicketRepository`/`UserDirectory`/`RuleRepository` 抽象（`Protocol`），实现 `HttpXxxRepo` / `MockXxxRepo`；示例数据移到 `data/`、`tests/fixtures/`，由配置选择实现——与 `vector_db` 已有的"主实现 + 降级实现"模式保持一致。
- **置信度**：高

## S2-7 · Mock 大模型整段写进生产 provider，且用 prompt 魔法字符串做分支开关

- **位置**：`app/providers/llm.py:32-130`（规则库 35-55、`_mock_answer` 73-102、`MockChatModel` 105-130）
- **具体表现**：约 100 行规则引擎常驻生产模块；`llm.py:124` 的 `if "knowledge / tool / unknown" in prompt:` 通过**匹配 prompt 模板里的示例文本**来区分"这是意图分类还是答案生成"；`_CONTEXT_PATTERN`/`_QUERY_PATTERN`（`:57-58`）同样依赖其他模块 prompt 的确切措辞（"参考内容："、"用户提问："）。这是依赖外部字符串格式的隐式协议。
- **影响**：任何 prompt 改写（改措辞/标点/系统提示）都会让 mock 静默走错分支且不报错；Mock 实现与 `get_chat_model()` 降级分支（`:291,303`）强耦合，无法独立替换。
- **通用方案**：把 `MockChatModel` 及规则库移出（如 `app/providers/mock_llm.py` 或测试层），以依赖注入接入；分支判定改为显式入参或结构化消息类型。
- **置信度**：高

## S2-8 · 自检探针写死本企业业务问题

- **位置**：`app/core/self_check.py:79`（`retrieve_knowledge_docs("年假有多少天", top_k=3)`）、`:107`（`user_query="工单 T20240101 进度如何"`）
- **具体表现**：`_check_retrieval` 把"检索返回空即 fail"作为判据，探针问句硬编码为 HR 语料；`_check_chat` 用固定工单号跑端到端。
- **影响**：换一套知识库（无 HR 文档）或跑 `deep=True` 自检时，`/test/all` 稳定报 fail，把"环境变了"误报成"系统坏了"，掩盖真实故障。
- **通用方案**：探针问句取自 config 或 `data/` 中实际存在的文档片段（如随机取一条已索引 chunk 的标题词）；判定改为"索引非空且能返回结果"。
- **置信度**：高

## S2-9 · 评测基线硬编码示例语料文件名，且作为长期兜底

- **位置**：`app/rag/evaluator.py:60-65`（`DEFAULT_EVAL_CASES`），兜底路径 `:79-80,104`
- **具体表现**：`EvalCase("年假有多少天", ["年休假","年假"], "员工手册.txt")` 等，注释坦承"关键词取自 data/ 示例文档"；在 yaml 缺失/为空时被当兜底返回。
- **影响**：换语料后评测仍跑这几条，`hit_rate` 恒为 0，`suggest_improvements`（`evaluator.py:374-377`）会持续吐出"命中率偏低，请补充语料"的误导建议，掩盖真实问题。
- **通用方案**：无 yaml 时返回空列表并告警，不返回示例基线；示例用例移到 `tests/`。
- **置信度**：高

## S2-10 · `structure.py` 4 条抽取正则硬编码，与同文件"不写死任何值"声明矛盾

- **位置**：`app/rag/structure.py:134-137`（对照 `:10-11` 的 docstring 声明），另有 `:148` 魔数 `[:10]`
- **具体表现**：docstring 写"全部锚点来自 `config.STRUCTURE_*_PATTERNS`，新增文档格式改配置即可，不改代码（遵循『不写死任何值』原则）"，但紧接着 `_TITLE_RE`/`_VERSION_RE`/`_EFFECTIVE_RE`/`_UPDATED_RE` 四条正则写死在代码里；`_TITLE_RE` 的分隔符集合 `[·•|\-—–]` 与"公司名 · 文档名"形态来自样例文档。
- **影响**：换文档模板（无"·"分隔、日期写成"2024年1月1日"）就抽不到标题/版本，且这是代码级硬编码，与同文件其它"改配置即可"的锚点行为不一致——运维会以为也能改配置，实际不能。
- **通用方案**：把这 4 条正则与"头部扫描行数"并入 `config.STRUCTURE_*`；日期用一条兼容式 `\d{4}[-/年]\d{1,2}[-/月]\d{1,2}`。
- **置信度**：高

## S2-11 · 动态路由的原型句与超参全部硬编码

- **位置**：`app/core/model_router.py:142-166`（`_HARD_CANDIDATES`/`_EASY_CANDIDATES`）、`:170`（`_PROTOTYPE_TEMPERATURE = 0.08`）、`:173`（`_PROTOTYPE_TOP_K = 3`）
- **具体表现**：20 条"难/易"样例全是年假/报销/差旅/竞业/离职等本企业 HR 语料；温度、top-k 是模块级字面量。而同文件的阈值/带宽/粘性/超时全部走 `config.*`。注释还写"修改这里即可适配你自己的业务分布"——即换业务要改源码。
- **影响**：换语料（法务/医疗）后原型语义失配，分数分布漂移，`ROUTING_THRESHOLD` 标定随之失效；调试扫参必须改代码重启。`describe()` 还把这两个常量当只读指标暴露，暗示不该被改。
- **通用方案**：候选句改为 config（`ROUTING_HARD_CANDIDATES`/`ROUTING_EASY_CANDIDATES`，JSON 或 `_env_list`），温度/top-k 纳入 `ROUTING_*` 配置族。
- **置信度**：高

## S2-12 · 大量启发式阈值以魔数散落，未走 config

- **位置**：`app/rag/retriever.py:46`（`_LEXICAL_FLOOR=0.01`）、`:109`（子串奖励 `0.3`）、`:285`（`min(max(top_k*4,30),count)` 的 `4`/`30`）、`:146`（改写变体 `[:2]`）、`:462`（软回退 `min(3,...)`）；`app/rag/generator.py:142,144,149,150,151,153`（置信度系数 `0.6/0.1/0.2`**
- **具体表现**：置信度计算几乎全是裸数，而同一批模块里 `RRF_K/DENSE_WEIGHT/REFUSE_THRESHOLD` 却是 config 项，标准不统一。
- **影响**：换语料/供应商时这些阈值必须改源码（config 那组只需改 `.env`），调优无法闭环回 L5；`retriever.py:285` 的候选池扩张系数直接决定召回率与成本，却不可调。
- **通用方案**：收敛到 config（`LEXICAL_FLOOR`/`LEXICAL_SUBSTRING_BONUS`/`CANDIDATE_OVERSAMPLE`/`CANDIDATE_MIN`/`FALLBACK_TOPK`/`CONFIDENCE_*`），并在 `retrieval_stats()`/`generator_stats()` 暴露以便 L5 归因。
- **置信度**：高

---

# 五、S2：自造轮子（本可用标准库 / 已依赖框架）

## S3-1 ✅ 已验证 · 手写限流重试，且 `ChatOpenAI` 未传 `max_retries`（双重退避）

- **位置**：`app/providers/llm.py:167-243`（核心 191-209、211-243）；`ChatOpenAI(...)` 构造见 `:250-260`（**未传 `max_retries`**）
- **具体表现**：手写 `for attempt in range(self._max_retries + 1)` + `delay = min(base*2**attempt, max) + random.uniform(...)`，并重写 `_stream` 实现"流式中断回退一次性生成"。但：(a) 未传 `max_retries` 意味着 langchain-openai 的默认 `max_retries=2`（openai SDK 层自带指数退避 + 抖动）**仍在生效**，与手写重试叠加；(b) `langchain_core` 的 `Runnable.with_retry(stop_after_attempt=..., wait_exponential_jitter=True)` 就是这段逻辑的标准实现。
- **影响**：换掉 langchain-openai 时该包装器全部失效；且只覆写 `_generate`/`_stream`，未覆写 `_agenerate`/`_astream`，异步路径重试行为与同步不一致。
- **通用方案**：删除该包装类，在 `_build_raw_model()` 显式传 `max_retries=config.LLM_MAX_RETRIES`，或 `model.with_retry(...)`；流式回退用 `Runnable.with_fallbacks`。
- **置信度**：高

## S3-2 · `_is_rate_limit` 用正则猜 429，而异常类型已给出确定答案

- **位置**：`app/providers/llm.py:148-164`
- **具体表现**：四层判定 `status_code == 429` → `"RateLimit" in type(exc).__name__` → 正则 `rate.?limit|too many requests` → 正则 `(?:status|code|error)[^\d]{0,12}429`；注释明说要"避免裸子串误判 request id / content hash 里的 429"——这正是放弃类型判定的代价。
- **影响**：openai SDK 已定义 `openai.RateLimitError`（带 `status_code`）。手写字符串匹配对上游文案改动、本地化消息、其他供应商措辞都脆弱。
- **通用方案**：`except openai.RateLimitError` / `except httpx.HTTPStatusError as e: if e.response.status_code == 429`。
- **置信度**：高

## S3-3 · 自造 LRU + TTL 缓存，且手写 embedding 差集

- **位置**：`app/utils/cache.py:33-98`（LRU 续期 74-76、淘汰 83-85）；`app/providers/embeddings.py:196-213`（`missing_idx` 差集）
- **具体表现**：用 `dict` 插入序手工实现 LRU（命中 `pop` + 重插）、手工 TTL、手工容量淘汰。这些正是 `cachetools.TTLCache`（`maxsize`+`ttl`+LRU）的标准功能；向量缓存也可直接用 langchain 的 `CacheBackedEmbeddings` + `ByteStore`，由框架负责"哪些文本需重新 embed"的差集——`CachedAPIEmbeddings.embed_documents` 手写的 `missing_idx` 正是该能力的重实现。
- **影响**：缓存策略（LFU/分片/命中率统计）要自研；换用 langchain embeddings 生态时两层自定义缓存成为迁移障碍。
- **通用方案**：`cachetools.TTLCache` 替换 `EmbeddingCache`；向量缓存改用 `CacheBackedEmbeddings.from_bytes_store`。
- **置信度**：高（自造事实）／中（适配成本）

## S3-4 · `APIEmbeddings` 用 httpx 手写 OpenAI `/embeddings`

- **位置**：`app/providers/embeddings.py:136-175`（请求 151-163、分批 165-170）
- **具体表现**：手写 URL 拼接、Bearer 头、payload、`raise_for_status()`、按 `index` 重排、`batch=32` 分批、L2 归一化。项目**已依赖** `langchain_openai`（`llm.py:248` 在用 `ChatOpenAI`），`OpenAIEmbeddings` 提供同一能力且自带批处理/重试/校验。
- **影响**：同一供应商两套接入方式——LLM 走 langchain 封装（可观测、可重试、可换供应商），embedding 走裸 httpx（无重试、无 LangSmith trace、无统一超时）。换 Azure 等兼容端点需手改 URL 与鉴权。
- **通用方案**：改用 `langchain_openai.OpenAIEmbeddings`，仅保留 `query_prefix` 这一条已配置化的差异。
- **置信度**：中高

## S3-5 · `LocalHashEmbeddings` 自造完整向量化 + IDF 训练链路

- **位置**：`app/providers/embeddings.py:48-133`（分词 30-40、IDF 拟合/持久化 72-107、hashing trick 113-127）
- **具体表现**：约 86 行自研：中英混合分词 + 中文 bigram、`Counter` 统计 DF、手写 BM25 风格 IDF、MD5 双哈希 + 符号哈希、亚线性 TF×IDF、`idf.json` 自管持久化与版本。等价于 `sklearn` 的 `HashingVectorizer(alternate_sign=True)` + `TfidfTransformer`。
- **影响**：这是检索质量的下限路径，但实现细节（`_default_idf`、双哈希 salt、分词规则）全部私有，无法与成熟库做调参/评测对比；`idf.json` 与向量库共用持久化目录，换向量库时该路径失效。
- **通用方案**：若必须保留零依赖兜底，用 `HashingVectorizer`+`TfidfTransformer` 替代；否则把兜底限定为"不检索、仅提示"，自研算法降级为脚本/测试工具。
- **置信度**：中高

## S3-6 · 手写 JSON 解析，重复框架自带解析器

- **位置**：`app/memory/dream.py:34-56`（围栏剥离 43-44、`_JSON_BLOCK` 兜底 50-55、正则 31）
- **具体表现**：手写"去 ```json 围栏 → `json.loads` → 失败则正则抓第一个 `{...}`"三级降级。`langchain_core.utils.json.parse_json_markdown` 正是剥离围栏后解析，且本文件已依赖 `langchain_core.prompts`（`:68`）。
- **影响**：模型输出形态一变（尾随逗号/单引号/截断 JSON）需自行维护正则；这类"模型输出健壮性"由框架持续迭代，自研版会逐渐落后。
- **通用方案**：`parse_json_markdown`，或更彻底用 `PydanticOutputParser` 约束结构。
- **置信度**：中高

## S3-7 · 三处手写 FIFO/LRU 淘汰 + 手写百分位

- **位置**：淘汰 `app/core/model_router.py:191-193,248-252`（`PrototypeScorer._cache`，`pop(0)` 是 O(n)）、`:450-451,486-493`（`_LAST_TIER`）、`:482-483`（latencies 截尾）；百分位 `:419-424`（`_RouteStats._pct`）与 `:676-679`（`calibrate_threshold`）
- **具体表现**：三处各自实现容量+淘汰，语义还不一致；两处按"索引=长度×分位"取分位数，边界钳制写法不同（`min(len-1,...)` vs `max(0,min(size-1,...))`），标定阈值与报表 p50/p95 可能口径不一致。
- **通用方案**：单值映射用 `functools.lru_cache` 或 `collections.OrderedDict`；延迟采样用 `collections.deque(maxlen=...)`；百分位统一走 `statistics.quantiles`。
- **置信度**：高

## S3-8 · `self_check` 同时用"装饰器注册"与"手写清单"两套机制

- **位置**：`app/core/self_check.py:25-33`（`_item` 装饰器）、`:36-113`（打在 7 个函数上）、`:116-125`（`_collect_checks` 手写返回同样的 7 个）、`:144-145`（`getattr(check, "_check_name")` 无默认值）；`FAST_ITEMS`/`DEEP_ITEMS` 又是第三份名字集合（`:17,22`）
- **影响**：三处名字集合靠人工同步，漏加即静默不执行；`getattr` 无默认值在注册遗漏时抛 `AttributeError` 而非可读报错。
- **通用方案**：显式注册表（`@_item(...)` 直接 append 进列表），`_collect_checks` 由注册表派生，消除三份清单。
- **置信度**：高

## S3-9 · 工具绕过 `app.config` 自行解析环境变量

- **位置**：`app/tools/ticket_tool.py:9-12`
- **具体表现**：`TICKET_USE_REMOTE = os.getenv("TICKET_USE_REMOTE", "false").strip().lower() == "true"`，而 `config.py:25-33` 的 `_env_bool` 已统一布尔解析（接受 `1/yes/on`）；`TICKET_API_TIMEOUT` 也是裸 `os.getenv` + `float(...)`。
- **影响**：同一份 `.env`（`docker-compose.yml:24` 已声明该变量）在不同模块语义不同——`TICKET_USE_REMOTE=1` 会被解析为 `False`。
- **通用方案**：移入 config，用 `_env_bool`。
- **置信度**：高

## S3-10 · 8 个节点的 try/except + trace 样板重复，其中 2 个节点还漏了 `span`

- **位置**：样板见 `app/graph/nodes.py:44-48`（`_trace`）及各节点；**不走 span 的两个**：`nodes.py:150-188`（`tool_invoke_node`，用 `time.perf_counter()` 手工计时，trace+return 复制 4 次）、`nodes.py:218-258`（`smalltalk_reply_node`，`:224` 同样手工计时）
- **具体表现**：每个节点重复 `dict(state)` → `with span(...)` → `try/except` → 写 trace → return 这一套。`tool_invoke` 与 `smalltalk_reply` **不会出现在 span 树里**（`tracing.py:83-106` 只记录 `with span`），而其他 6 个会。
- **影响**：新增节点要抄样板；响应里的 `span_tree`（`chat.py:143/315`）缺节点，用 span 树做耗时归因会得出"工具耗时不计入"的错误结论。
- **通用方案**：写 `@node("name")` 装饰器统一处理 state 拷贝、span、异常降级与 trace 追加，节点只写业务体，降级策略参数化。
- **置信度**：高

## S3-11 · 记忆子系统的开关判定散落 6 处，无单一门面

- **位置**：`app/memory/long_term.py:49`、`app/memory/__init__.py:67,89`、`app/memory/consolidator.py:92,118`、`app/memory/dream.py:122`
- **具体表现**：`MEMORY_ENABLED` 被 6 处各自判断；`CONSOLIDATE_ENABLED` 被 2 处重复判断（`__init__.py:89` 与 `consolidator.py:118` 两层等价门检查）；`DREAM_ENABLED` 实际未参与 `run_dream` 判定（`__init__.py:101-108` 无检查）。
- **影响**：改开关语义要同改多处，漏改会导致"配置关了但某条路径仍执行/仍计费"。
- **通用方案**：门控集中在 `app/memory/__init__.py` 一处，或构造期一次性决定并注入 no-op 实现，下游不再读 config。
- **置信度**：高

## S3-12 · 双后端无共享接口，`list_sources` 聚合体逐字重复

- **位置**：`app/db/vector_db.py:254-259`（`MemoryVectorStore.list_sources`）vs `:316-322`（`ChromaVectorStore.list_sources`）；降级实现整体 `:76-259` 与 `:262-322`；同类模式见 `app/db/redis_db.py:14-81`（`MemoryRedis` 手写 7 个 redis 方法）
- **具体表现**：`list_sources` 的聚合体（`agg[src]=agg.get(src,0)+1` + `sorted(..., key=lambda x:-x[1])`）在两个类中逐字重复；两类无共同基类/`Protocol`，"对外一致接口"仅靠 docstring 声明。`MemoryRedis` 手工复刻 `end==-1`、`ltrim` 边界等 redis 语义。
- **影响**：新增后端（Milvus/pgvector/valkey）必须再抄一份；接口漂移无法被类型检查发现（如 `search` 的 `k<=0` 边界、`clear` 返回值语义不同）。
- **通用方案**：抽 `VectorStore(Protocol)`（`app/providers/base.py` 已有 `Embedder`/`Reranker` 抽象先例）+ `list_sources` 模板方法；`MemoryRedis` 与真实分支共用 `Protocol`，或直接用 `fakeredis`。
- **置信度**：中高

## S3-13 · `doc_loader` 目录扫描两份、同义格式两套加载实现

- **位置**：`app/utils/doc_loader.py:230-235`（`load_all_documents` 的 `rglob`+过滤）vs `:263-283`（`list_data_files` 同一段再写一遍）；`:209-211`（`_load_markdown` 自读文件）vs `:214-217`（`_load_text` 走 `TextLoader`）
- **具体表现**：两处 `sorted(data_dir.rglob("*"))` + `is_dir()`/`startswith(".")`/`suffix not in SUPPORTED_SUFFIX` 过滤完全重复；`.md`/`.markdown` 与 `.txt` 走两套加载路径但语义相同。
- **影响**：新增格式需改 4 处；两套加载路径的 metadata 不一致（`_load_markdown` 只给 `source`，`TextLoader` 带自有 metadata），下游 `file_name` 依赖补丁兜底。
- **通用方案**：抽 `_iter_supported_files(data_dir) -> Iterator[Path]` 复用；md/txt 统一走一个 `_load_plaintext`。
- **置信度**：中高

---

# 六、S2：配置解析与豁免清单的一致性

## S3-14 ✅ 已验证 · `config.py` 内布尔解析有两套实现（11 处内联 vs `_env_bool`）

- **位置**：`app/config.py:25-34`（`_env_bool`，把 `"none"` 视为 False）；内联写法出现在 `:67,79,89,117,125,129,238,250,272,280,286`（**共 11 处**）
- **具体表现**：`_env_bool` 在 40 行前刚定义且被 8 处使用，同一文件另 11 处仍写 `os.getenv(k,"false").strip().lower() not in ("0","false","no","off")`。差异在于 `_env_bool` 认 `"none"` 为 False，内联版不认——于是 `MEMORY_ENABLED=none` 与 `CHUNK_CONTEXT_HEADER=none` 行为相反。
- **影响**：新增开关靠抄袭邻近行，错误持续扩散；批量改判定集合要动 11 处以上，必然漏改。
- **通用方案**：全部改用 `_env_bool`，删除内联表达式。
- **置信度**：高

## S3-15 · 免鉴权路径与免限流路径是两套机制，限流那份硬编码

- **位置**：`app/main.py:122`（`path in config.AUTH_EXEMPT_PATHS`，可配置）vs `app/main.py:155`（`path in ("/health", "/", "/dify/info")`，**硬编码**）
- **具体表现**：`config.AUTH_EXEMPT_PATHS`（`config.py:442-445`）默认含 7 项；限流豁免元组只有 3 项。
- **影响**：两份清单已不同步——`/retrieval`、`/dify/retrieval`（Dify 高频检索）在鉴权层豁免，却**不在限流豁免里**，会消耗 IP 限流额度；运维想加豁免路径只能改代码。
- **通用方案**：统一为一份配置（或拆 `AUTH_EXEMPT_PATHS` / `RATE_LIMIT_EXEMPT_PATHS` 两个显式配置项），`main.py:155` 改读配置。
- **置信度**：高

## S3-16 · `_format_history` 的 6 与 `SHORT_TERM_WINDOW` 语义重复

- **位置**：`app/rag/generator.py:171`（`def _format_history(history, limit: int = 6)`），调用点 `:219` 未传参；`config.py:275` 已有 `SHORT_TERM_WINDOW = 6`（`memory/short_term.py:39` 在用）
- **影响**：两处都表达"注入 prompt 的最近对话轮数"却各自为政；调 `SHORT_TERM_WINDOW` 时 L4 prompt 历史不变。且"6 条消息"与 memory 的 `SHORT_TERM_WINDOW*2` 用法并存，单位语义（条 vs 轮）需澄清。
- **通用方案**：`_format_history` 默认值改为 `config.SHORT_TERM_WINDOW` 并统一单位。
- **置信度**：中高

## S3-17 · 文档长度上限两个来源，语义不一致（报错 vs 截断）

- **位置**：`app/utils/validator.py:17`（`MAX_DOC_CONTENT = 50_000` 硬编码）vs `app/config.py:418`（`MAX_DOC_CONTENT_CHARS`）；使用点 `app/api/knowledge.py:126-131`（`content[:max_chars]` 截断）与 `validator.py:108-109`（`if len(v) > MAX_DOC_CONTENT: raise` 报错）
- **影响**：调大 `MAX_DOC_CONTENT_CHARS` 后，`/upload-file` 放宽了，但 JSON 接口 `/knowledge/upload` 仍卡在 50000 且直接 400——同一份配置只对一半入口生效；"报错 vs 截断"两种语义本身也不一致。
- **通用方案**：`validator.py` 改读 `config.MAX_DOC_CONTENT_CHARS`；统一两条入口的越限语义。
- **置信度**：高

## S3-18 · Dify 的 OpenAPI schema 手写，与 FastAPI 自动 schema 平行维护

- **位置**：`app/api/dify.py:239-315`（约 75 行手写 dict）
- **具体表现**：`query/knowledge_id/top_k/score_threshold` 的字段定义被手写一遍，而同份契约已由 `RetrievalRequest`（`:53-57`）与 `RetrievalSetting`（`:48-50`）用 Pydantic 声明过；FastAPI 也已为 `/retrieval` 自动生成 `/openapi.json`。
- **影响**：改 `RetrievalSetting` 默认值/范围后，手写 schema 不同步，Dify 侧展示的契约与实际行为漂移。
- **通用方案**：从 `app.openapi()` 取 `/retrieval` 路径改写 `servers`；或至少用 `RetrievalRequest.model_json_schema()` 填充 `requestBody.schema`。
- **置信度**：中（Dify 需要 OpenAPI 格式属实，但"必须手写"不成立）

## S3-19 · `model_route_node` 重复计算 `estimate_confidence`

- **位置**：`app/graph/nodes.py:275-279`
- **具体表现**：注释自认"检索置信度此刻尚未写入 state，按需自算一次"，于是同一纯函数在路由阶段算一次、`prepare_generation`（`generator.py:200-201`）又算一次。
- **影响**：同一请求重复计算；两个消费点输入若未来不同会得到不一致置信度。
- **通用方案**：把 `confidence`/`citations` 作为独立状态字段由检索节点写入 state，路由与生成都读 state（`state.py:34-38` 已有该设计）。
- **置信度**：高

## S3-20 · `EvalCaseRequest` 是死代码，旁边却有手写的等价转换

- **位置**：`app/api/evaluation.py:18-23`（定义，**零引用**）与 `:35-43`（手工 `evaluator.EvalCase(...)`）；接口用 `cases: Optional[list]`（`:27`）收裸 list
- **影响**：两套并存的用例模型，改字段不知改哪个；Pydantic 的校验/文档能力完全没用上。
- **通用方案**：签名改为 `cases: Optional[List[EvalCaseRequest]]`，删除手工转换与 `if not eval_cases` 分支。
- **置信度**：高

---

# 七、明确判定为"合理特化，非问题"

以下虽属定制，但有可证实的正当理由，**不计为问题**（列出以免整改时误伤）：

| 项 | 位置 | 理由 |
|---|---|---|
| 自实现 BM25 + 中文 bigram 分词 | `app/rag/lexical.py:1-20,186-229` | 零依赖取舍明确；问题仅在"与 retriever 重复"（S1-2/S0-6），而非不该自造 |
| flat 文档回退递归切分 | `app/rag/indexer.py:210-211`、`structure.py:280-282` | 降级契约明确、永不抛错 |
| 内容哈希作 chunk id | `app/rag/indexer.py:367-376` | 增量入库 ID 稳定的必需设计 |
| `reorder.py` 首尾强中间弱重排 | `app/rag/reorder.py` | 纯函数、零成本、与 rerank 职责正交 |
| PDF 三级降级 | `app/utils/doc_loader.py:190-206` | 表格识别能力差异真实，逐级只降不抛 |
| 内存降级策略本身 | `app/db/redis_db.py:88-116`、`vector_db.py:329-348` | 显式声明的降级路径（问题在实现重复，见 S3-12） |
| rerank 懒加载 + 失败记忆 | `app/providers/rerank.py:46-69` | 避免每次请求重试加载，默认关闭 |
| 记忆游标自愈 / 硬上限 | `app/memory/store.py:37,146-172` | 针对"模型回显原文""外部写坏单调性"的兜底 |
| 无 LLM 时规则抽取 / 原文快照 | `app/memory/long_term.py:23-34,149-174`、`consolidator.py:73-78` | 有意的降级链路 |
| Dify 双路径注册 / 200+error_code / 未知运算符放行 | `app/api/dify.py:60-70,91-137,221-236` | 对上游 Dify 的具体行为迁就，理由可验证 |
| 上传流式分块 + Content-Length 预检 | `app/api/knowledge.py:31-50,88-96` | 防 OOM 的性能/安全必需 |
| `origins=*` 时关闭 credentials | `app/main.py:87-98` | 浏览器规范约束下的正确处理 |
| 内存滑动窗限流（非 Redis） | `app/core/rate_limit.py:1-11` | 单机部署的有意简化，文档已说明 |
| langsmith 缺失时 `traceable` 退化为恒等 | `app/core/observability.py:44-67` | 可选依赖的合理处理 |
| 自建轻量 span 树（非 OTel） | `app/core/tracing.py` | 成本取舍；仅其中重复+死代码（S2-2）应处理 |
| `llm_factory` re-export 兼容层 | `app/core/llm_factory.py:9-16` | 仍有 6 处消费，属过渡层 |

---

# 八、跨模块模式总结

本项目的"通用性债务"可归纳为 5 个反复出现的模式：

1. **双份实现且已分叉**：最典型是 `/ask` 与 `/ask/stream`（S0-1，5 处分叉）；其次是响应体（S0-2）、RRF 公式（S0-3）、健康检查（S0-4）、Bearer 解析（S0-5）、`_RouteStats`（S0-7）。这类问题**不会报错**，只在换场景时静默出错。
2. **同一规则/常量在多文件复制**：工单正则 4 份、中文词表 2 份、后缀表 2 份、文档 key 3 份、basename 6+ 份、校验器 5 份、脚本指标 3 份。改一处必漏。
3. **自造轮子覆盖框架已有能力**：重试（llm.py）、缓存（cache.py）、embedding 客户端（embeddings.py）、JSON 解析（dream.py）、节点样板（nodes.py）、百分位/淘汰（model_router.py）。
4. **mock/示例数据写进生产代码**：姓名正则（nodes.py:196）、三个工具的 mock 数据源、MockChatModel、自检探针问句、评测基线用例——换业务即失效，且"演示"与"生产"无边界。
5. **死代码/死配置残留**：`CHUNK_MIN_CHARS`、`CHUNK_FALLBACK_OVERLAP_RATIO`、`CASCADE_MAX_PER_SESSION`、`CASCADE_MIN_CONFIDENCE`、`_doc_key`、`get_span_tree`、`EvalCaseRequest`、`prompts.render`、`filter_by_section`、`detect_structure`、只写不读的 `soft_warnings`。与 config 自述"已修完伪配置"相矛盾，说明当时只修了被点名的那一批。

**附带发现（工程卫生）**：仓库存在 `artifacts/backup/{DIFY,T5-2,T6-1,T6-2,FIX-P0}/` 多份旧版源码副本，会污染全文检索（本次核查时多次命中）。虽已被 `.gitignore` 排除，建议归档或清理，避免误导。

---

# 九、建议的整改顺序

| 优先级 | 项 | 理由 |
|---|---|---|
| P0 | S0-1 流式链路复用编译图（`astream_events`） | 通用性风险最高，且能顺带消除 S0-2/S3-19 |
| P0 | S2-5 / S2-6 / S2-7 mock 数据出生产代码 | "能测不能上生产"，阻塞真实落地 |
| P1 | S1-1~S1-10 抽出单一常量/函数模块 | 成本低、收益直接（防漏改） |
| P1 | S2-1~S2-3 清理死配置/死代码 | 低风险，消除"改了没用"的陷阱 |
| P1 | S2-4 统一运行时读 config（去 getattr 快照） | 修复"改配置不生效"的根因 |
| P2 | S3-1/S3-2 重试与 429 判定改用框架能力 | 消除双重退避；但需回归流式链路 |
| P2 | S3-3~S3-8 自造轮子逐步替换 | 收益中等、有回归风险，建议随迭代进行 |
| P2 | S3-14/S3-15 配置解析与豁免清单统一 | 一致性修复 |
| P3 | S3-12/S3-13 抽象 `Protocol` 与复用扫描 | 结构性改进，优先级取决于是否要加后端 |

> 说明：以上为**静态审查结论**，未运行压测或端到端回归；标注 ✅ 的条目已由报告作者二次核实 `file:line`，其余来自分模块精读，建议整改前再以运行时行为确认一次（尤其 S0-1 流式链路的分叉表现）。
