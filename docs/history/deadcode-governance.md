# 未使用代码 / 依赖 检测与治理报告

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> 目标：为本项目建立一套**可复现、可进 CI** 的「未使用代码 / 依赖」检测能力，
> 清理存量冗余，并用机制防止回流。

---

## 0. 摘要

| 项目 | 结果 |
|---|---|
| 扫描范围 | 72 个 Python 文件，约 13,853 行 |
| 检测手段 | 自建 AST 扫描器 + ruff + vulture + deptry（四路互补） |
| 发现未使用问题 | **51 项**（自建扫描器 32 + ruff 未使用 import/变量 19） |
| 已清理 | **47 项**（删除未使用符号 24 个、死配置 3 项、未使用依赖 1 项、未使用 import/变量 19 处） |
| 登记豁免 | **4 项**（每条写明「为什么不能删」） |
| 新增治理机制 | `scripts/deadcode_scan.py` + `tests/test_deadcode.py`（11 个用例）+ `tests/deadcode_allowlist.py` + `pyproject.toml` 三层工具配置 + CI `deadcode` job + `requirements-dev.txt` |
| 验证 | `pytest` **89 passed**；ruff / vulture / deptry **全部零发现**；门禁经"注入死代码→失败、移除→通过"实测有效 |

---

## 1. 为什么不能只装一个 vulture

本项目有三处**框架约定导致的隐式引用**，纯名字统计的通用工具必然大面积误报：

| 约定 | 现象 | 通用工具的表现 |
|---|---|---|
| FastAPI 路由 | `@router.get("/x")` 注册，函数名无调用点 | 实测 vulture 60% 档把 `chat.py`、`knowledge.py`、`memory.py` 等**全部路由处理函数**报为"未使用" |
| LangGraph 节点 | 除 `add_node("name", fn)` 外，节点名还以**字符串**出现在拓扑描述里 | 字符串引用识别不到 |
| 配置常量 | 消费方用 `getattr(config, "X", default)` 读取 | 属性访问统计不到 |

实测：vulture 在 60% 档报出 50+ 项，其中约一半是假的；即使加上 `--ignore-decorators`，仍需人工逐条过滤。

**结论**：通用工具负责「广度」，项目专用扫描器负责「精度」。两者互补，缺一不可。

---

## 2. 检测方案

### 2.1 四路互补

| 工具 | 负责 | 为什么是它 |
|---|---|---|
| `scripts/deadcode_scan.py`（自建） | 未引用模块、未使用函数/类/常量、**死配置**、未使用依赖 | 通用工具看不到的三类约定，只有结合项目知识才能准确判定 |
| `ruff` | 未使用 import / 局部变量 / 未定义名（`F` 规则集） | 零误报，可直接作为硬门禁 |
| `vulture` | 未使用函数 / 类 / 变量（80% 档） | 对"定义域内确定未使用"最敏感，作为交叉验证 |
| `deptry` | 未使用 / 缺失 / 传递依赖 | 依赖维度的唯一权威工具 |

### 2.2 自建扫描器做了什么

在标准 AST 名字统计之上补齐：

1. **装饰器豁免** —— `@app.get` / `@router.post` / `@pytest.fixture` / `@field_validator` 等由框架隐式调用，不算未使用；
2. **字符串引用识别** —— 收集所有**非 docstring** 的字符串常量并切分为标识符 token，用于识别 `add_node("intent_recognize", …)`、`getattr(config, "RRF_K")` 这类动态引用；
3. **配置消费双通道** —— `config.X` 属性访问 **与** `getattr(config, "X")` 字符串访问；**刻意不把「出现在 .env」当作被消费**（见 §2.4）；
4. **`from app import config` 子模块识别** —— 否则整个 `app/config.py` 会被误判为"无人 import"；
5. **基类同名方法识别** —— 用 `dir(基类)` 判断方法是否为框架回调（如 `logging.Filter.filter`），比维护硬编码名单更通用；
6. **`__all__` / `__init__.py` re-export / 模型字段识别**；
7. **元数据文件排除** —— 豁免清单自身不能算作"引用来源"（见 §2.4）；
8. **行内豁免** `# deadcode: ignore`（与 ruff 的 `noqa`、deptry 的 `# deptry: ignore` 同风格）。

### 2.3 工具给出的假阳性，逐条核验后修正

按"工具/子代理结论必须二次核验"的纪律，用 `grep -w` 逐条复核，纠正了 4 类误报：

| 误报 | 原因 | 修正 |
|---|---|---|
| `app/config.py` 整个模块"无人引用" | 只认 `import app.config`，不认 `from app import config` | 补齐子模块导入识别 |
| `BASE_DIR`、`_ROUTING_THRESHOLD_RAW` 等 5 个"死配置" | 只统计**其他文件**的引用，漏掉 config.py 自身的使用 | 纳入自身引用（但排除字符串，否则 `X = _env("X",…)` 会把所有配置项洗白） |
| `TraceIdFilter.filter` 未使用 | 它是 `logging.Filter` 的回调，无显式调用点 | 新增基类同名方法识别 |
| `MemoryRedis.ttl` 等 shim 方法 | — | 见 §3.5，判定为接口契约，登记豁免 |

**一次关键的"避免误删"**：`smalltalk._HIT_COUNTS` 表面上"只写不读"（唯一读者 `get_smalltalk_stats` 已死），核验发现它**驱动 `_pick_reply` 的确定性回复轮换**（`smalltalk.py:155-161`）。若按表面结论连计数器一起删，会静默破坏寒暄回复的轮换行为。最终只删 `get_smalltalk_stats`，计数器与 `reset_smalltalk_stats`（测试在用）保留。

### 2.4 修正的 4 个扫描器自身缺陷（重要）

检测器本身也必须被检测。开发过程中通过自检用例发现并修复了 4 个缺陷——**其中两个会让整套测试变成"永远通过"的假门禁**：

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | `Lambda` / `IfExp` 节点的 `body` 是**单个表达式而非 list**，`body[0]` 抛 `TypeError` | 扫描真实代码时崩溃 | 加 `isinstance(body, list)` 判断 |
| 2 | 排除目录按**绝对路径**匹配 | 扫描根落在 `.pytest_tmp/` 时**整个扫描恒为空**，自检用例假通过 | 改为按"相对扫描根"的路径判定 |
| 3 | docstring 判定拿 `Constant` 去比对其父节点 `ast.Expr`，而 `Expr` 没有 `.body`，判断**恒为 False** | **docstring 排除从未生效**：注释/文档里顺口提一句函数名就能把死代码"洗白"。修复后立即多暴露出 4 项真问题 | 先按 `Module/FunctionDef/ClassDef` 的 `body[0]` 算出 docstring 节点 id，再统一过滤 |
| 4 | 豁免清单文件被当作"引用来源" | 清单里以字符串登记符号名，反而把它**自己豁免的死代码标记为"已使用"**，导致豁免刚登记就变成僵尸、僵尸校验立刻失败 | 引入 `REFERENCE_EXCLUDED_FILES`，元数据文件不参与引用统计 |
| 5 | 按**裸名**统计引用（同名即算命中） | 方法名与高频内置/三方 API 撞名时被"洗白"。实测：为会话序号加的 `MemoryRedis.get()` 全程零调用，扫描器却没报——因为仓里有大量 `dict.get(...)` / `os.getenv(...)` 贡献了 `get` 这个名字 | 本轮采用**人工复核 + 删除**：该方法是本任务新增代码，可直接确认零调用 |

缺陷 2、3 尤其值得记录：**它们不会报错，只会让检测静默失效**。若无"往临时包种死代码、断言必被检出"的自检用例，这两个 bug 会一直潜伏。

缺陷 5 是**方法论层面的已知局限，本轮未修**（属结构性权衡，不是 bug）：只统计裸名可以让扫描器零依赖、零配置、跨语言风格稳定，代价是 `get` / `save` / `run` / `load` 这类高频名一旦撞名就会漏报。修它需要改成"按 `模块.类.方法` 的限定名绑定关系求解"，复杂度约等于重写一遍静态作用域分析——性价比不合适。**处理方式**：

- 该局限已写进扫描器模块文档与 CLI 运行输出（有发现时附带提示），同步收进技能版本，提醒使用者不要把它当唯一裁判；
- 撞名高发区（数据访问层、工具类）优先用 `ruff` 的 `F401/F811` 与**人工复核调用点**兜底；
- 已知漏报不影响门禁可信度：门禁是"**新增即拦截**"（新写的死代码通常在自己模块内，不会恰好在同文件撞上高频名），而不是"存量全清"。

---

## 3. 检测结果与清理清单

### 3.1 未使用符号（24 项，全部删除）

| 位置 | 符号 | 说明 |
|---|---|---|
| `app/rag/lexical.py:74` | `_doc_key` | 私有辅助函数，零调用；`retriever._key` 已有等价实现 |
| `app/core/tracing.py:142` | `get_span_tree` | span 树已持久化到 `trace.jsonl`，无运行时读取方 |
| `app/api/evaluation.py:18` | `EvalCaseRequest` | 未被任何路由用作入参（路由收的是 `Optional[list]` 字典） |
| `app/core/errors.py:23` | `EmbeddingFailed` | 从未被 `raise` |
| `app/providers/llm.py:35` | `_TOOL_KEYWORDS` | 被 tool-calling 取代的旧关键词表 |
| `app/core/observability.py:44` | `traceable` | LangSmith 装饰器，零使用；项目自建的 `tracing.span()` 已在用（重复能力，取其一） |
| `app/core/rag_engine.py:99,104` | `run_evaluation`、`get_improvement_suggestions` | 与 `api/evaluation.py` 直连 `evaluator.*` 重复；且 `suggest_improvements()` 少传参，本就是坏的 |
| `app/core/chat_memory.py:50,62` | `list_sessions`、`get_ttl` | 运维观测辅助函数，无任何调用方与端点 |
| `app/db/vector_db.py:357` | `reset_vector_store` | 单例重置，无调用方；全量重建由 `indexer.build_index()` 的 `clear()` 承担 |
| `app/rag/lexical.py:258` | `reset_lexical_index` | 同上 |
| `app/utils/cache.py:113` | `reset_embedding_cache` | 同上 |
| `app/tools/rule_tool.py:38` | `list_rule_keys` | 注释称"供前端快捷提示"，但从未有端点暴露 |
| `app/tools/user_tool.py:44` | `list_user_keys` | 同上 |
| `app/tools/ticket_tool.py:75` | `list_ticket_ids` | 同上 |
| `app/core/smalltalk.py:196` | `get_smalltalk_stats` | 无观测出口（**计数器本身保留**，见 §2.3） |
| `app/memory/long_term.py:100` | `update_user_profile` | Dream 实际走 `add_user_note`（带去重），本方法被取代 |
| `app/memory/long_term.py:121,124` | `get_soul`、`set_soul` | 无调用方；`get_soul` 与 `store.read_soul` 重复 |
| `app/memory/store.py:122` | `write_soul` | 上一条删除后的**级联孤儿** |
| `app/rag/evaluator.py:249` | `score_citation_coverage` | 从未被评测流程调用 |
| `tests/test_eval_section.py:14` | `_sample_cases` | 测试辅助函数，零引用 |
| `tests/test_deadcode.py` | `KIND_LABELS` | 本报告作者新写代码中的多余常量——**被自己的扫描器当场抓出并删除** |

### 3.2 死配置（3 项，已删除）

| 配置 | 位置 | 依据 |
|---|---|---|
| `CASCADE_MAX_PER_SESSION` | `config.py:130` | 全仓零引用 |
| `CASCADE_MIN_CONFIDENCE` | `config.py:132` | 全仓零引用 |
| `APP_HOST` | `config.py:479` | 全仓零引用（`APP_PORT` 被 `main.py`/`self_check.py` 读取）；监听地址实际由启动命令决定，Dockerfile 里是 `uvicorn --host 0.0.0.0`。已同步从 `.env`、`.env.example` 移除并加注释说明 |

`CASCADE_ENABLED` 保留（被 `api/routing.py:65`、`config.py:541` 作为展示字段读取），但**注释已改为如实说明**："仅保留开关与展示字段，级联逻辑尚未实现，改这个值不会有任何行为变化"。

### 3.3 未使用依赖（1 项，已删除）

`requirements.txt:17` 的 `langchain` —— 全仓无任何 import，`pip show` 显示 `Required-by:` 为空，是纯孤立依赖。项目实际只用 `langchain-core / -community / -openai / -text-splitters` 四个子包。
→ 已从 `requirements.txt` 删除、从虚拟环境卸载、重新生成 `requirements.lock`。

### 3.4 未使用 import / 变量（19 处，已删除）

初始 13 处：`intent_router` 的 `config`、`model_router._ensure_ready` 的 `numpy`/`torch`（实际使用在 `_predict` 内各自 import，提前 import 还会让"可选依赖缺失"的探测范围失真）、`rag_engine` 的 `logger`、`vector_db` 的 `Tuple`、`evaluator` 的 `field`、`gen_favicon` 的 `math`/`io`、`chat.py`/`nodes.py` 的未使用 `exc` 绑定、`test_rag` 的 `json`、`test_span_tree_smoke` 的重复导入。

另 6 处是**删除符号后新产生的级联孤儿 import**（`evaluation.py` 的 `BaseModel`/`Field`、`chat_memory.py` 的 `Optional`、`observability.py` 的 `Any`/`Callable`/`Optional` 等），由 ruff 复扫捕获后一并修复。这也说明**改完后必须重跑门禁**。

### 3.5 登记豁免（4 项，保留并写明理由）

| 条目 | 保留理由 |
|---|---|
| `app/static/gen_favicon.py`（未引用模块） | 构建期资源生成脚本，手工执行；产出 favicon.png/ico 供静态引用，不是运行时模块 |
| `MemoryRedis.ttl` | `redis.asyncio` 降级替身的接口方法。删除后，未来任何 `await redis.ttl(k)` 只会在**无 Redis 的降级环境**失败——最难排查的环境相关 bug。保留模块 docstring 声明的"最小方法集"以保证可替换性 |
| `MemoryRedis.dbsize` | 同上 |
| `CHUNK_FALLBACK_OVERLAP_RATIO` | `docs/chunking-contract.md` 将其列为兼容层 C4「保留（随 C1）」，而 C1 标注"永不删除"，故未擅自删除。但**当前全仓无任何读取点**（recursive 路径实际用 `CHUNK_OVERLAP`，见 `indexer.py:50`）——属"文档说保留、代码没接线"的矛盾，**需产品/架构决策**：接线，或删除并同步修改契约文档 |

---

## 4. 成因分析

### 根因一：功能"做了一半"——写入了状态/配置，但没有出口

- `_HIT_COUNTS` 有写入、有 `get_smalltalk_stats` 读取函数，但**没有任何端点或调用方**读取它；
- `CASCADE_ENABLED` 有开关、有展示字段、有设计文档，但**没有一行级联逻辑**；
- `CHUNK_FALLBACK_OVERLAP_RATIO` 写进 `.env`、写进契约文档，但**没有代码读它**；
- `SOUL.md` 有读取，但**没有写入路径**（唯一写入者 `set_soul` 未被接线）。

**特征**：配置项、统计项、存储原语都齐了，唯独缺"最后一公里"的消费方。这类代码最危险——**看起来是完整的**，`CASCADE_ENABLED=false` 改起来毫无反应也不报错，属于静默失效。这也是本项目 `.env` 里那些"配了但没用"的键的来源。

### 根因二：能力被新实现取代，旧的没删

- `_TOOL_KEYWORDS` 被 tool-calling 取代；
- `observability.traceable` 与项目自建的 `tracing.span()` 能力重叠，后者已在用；
- `update_user_profile` 被 `add_user_note`（带去重）取代；
- `_doc_key` 被 `retriever._key` 取代；
- `list_*_keys` 的"前端快捷提示"设想未落地。

**特征**：新旧并存、语义重叠，读者无法判断该用哪个。这是演进过程的沉积物。

### 根因三：门面/兼容层过度铺开

`app/core/rag_engine.py` 定位是"五层架构的兼容门面"，于是把每层的能力都原样转发一遍。但 API 层实际**直接调用 `evaluator.*`**，门面成了无人经过的中间层。其中 `get_improvement_suggestions()` 还漏传了必填参数——转发函数写出后从没被运行过。

**特征**：为了让"对外签名保持不变"而机械复制的转发函数，但项目是**应用而非可发布库**（无打包元数据、无外部使用者），"对外兼容"这个前提并不存在。

### 根因四：调试/测试辅助函数缺少明确归属

`reset_*` 系列单例重置函数、`list_*_keys` 调试函数、`_sample_cases` 测试辅助——都是"当时有用、事后无人用"。它们既不属于生产路径，也没被测试纳入，处于生产代码与测试代码之间的灰区。

### 根因五（元凶）：缺少回归门禁

前四条的共同前提是——**没有任何机制在新增死代码时给出反馈**。删除一个函数后产生的孤儿 import（本次就产生了 6 处）、新写的无用辅助函数（本次连我自己写的 `KIND_LABELS` 都被抓出），都要等下一次人工全量审查才发现。

---

## 5. 同类开源项目调研

调研对象：`langchain`、`langgraph`、`llama_index`、`dify`、`ragflow`、`fastapi`、`full-stack-fastapi-template`，以及真实跑 deptry 的 `cookiecutter-uv`、`contextgem`。

### 5.1 关键发现

| 发现 | 证据 |
|---|---|
| **点名的几个大型项目都没有跑 vulture/deptry**，只靠 ruff 的 `F` 类规则兜底 | langgraph `pyproject.toml` 的 `lint.select` 含 `E/F/I/PLC0415/RUF100/TID251/UP`，无 vulture/deptry 配置 |
| fastapi 官方用 `per-file-ignores` 处理 re-export 误报 | `"__init__.py" = ["F401"]` |
| full-stack-fastapi-template 显式开 `ARG001`（未使用参数），并 `ignore = ["B008"]`（FastAPI `Depends()` 经典误报） | 其 `backend/pyproject.toml` |
| langgraph 显式启用 `RUF100`，防止"僵尸 noqa"堆积 | 同上 |
| llama_index 反过来 `ignore` 掉 `RUF100`（允许 `# noqa`） | 该取舍被官方注释明确写出——**是否允许裸 noqa 是需要明确表态的团队决策** |
| **真正跑 deptry 的是中小型高质量项目**，做法是"工具全开 + 显式豁免 + 每条写理由" | `contextgem` 的 `[tool.deptry.per_rule_ignores]` 逐条注释；`cookiecutter-uv` 把 `deptry src` 放进 `make check` |
| vulture 官方推荐用**白名单文件**而非 `# noqa` 抑制误报 | 官方 README；`--make-whitelist` 生成的是合法 Python 模块，可被解释器校验 |
| vulture / deptry **都没有官方 baseline 机制** | "基线 + 只禁新增"需自行用白名单/豁免清单实现 |

### 5.2 借鉴并落地的做法

| 借鉴来源 | 落地方式 |
|---|---|
| fastapi：`__init__.py` 豁免 F401 | `pyproject.toml` 的 `per-file-ignores` |
| langgraph：启用 `RUF100` 治理僵尸 noqa | 转化为**僵尸豁免校验**：`test_allowlist_entries_are_not_stale` 断言豁免清单里的条目必须仍被报出，否则测试失败 |
| contextgem：`per_rule_ignores` 每条写理由 | `tests/deadcode_allowlist.py` **每条豁免必须写"为什么不能删"，由 `test_every_allowlist_entry_has_reason` 强制校验** |
| vulture 官方：用 `--ignore-decorators` 处理框架注册 | `[tool.vulture] ignore_decorators` 覆盖 FastAPI 路由与 pytest fixture |
| 大型项目共识：**不追求一次性全绿，先上零误报的一层** | ruff 只启用 `F`；`B/C4/UP` 实测有 836 处历史问题，明确不纳入本次范围（避免门禁长期红着失去约束力） |

**未采纳**：`select = ["ALL"]`（langchain 做法）。本项目是应用不是库，且存量风格问题量大，一次全开会让门禁失去可执行性。

---

## 6. 落地机制

### 6.1 结构

```
scripts/deadcode_scan.py          ← 检测引擎（AST，懂项目约定）；支持 --strict
tests/deadcode_allowlist.py       ← 豁免清单（每条必须写理由）
tests/test_deadcode.py            ← 执行契约（14 个用例）
pyproject.toml                    ← ruff / vulture / deptry 配置
requirements-dev.txt              ← 三个工具的声明
```

### 6.2 测试集设计（`tests/test_deadcode.py`）

| 用例 | 作用 |
|---|---|
| `test_scanner_detects_planted_dead_code` | **地基**：往临时包种一处死代码，扫描器必须报出。正是这条用例暴露了 §2.4 的缺陷 2 与 3 |
| `test_scanner_respects_fastapi_and_fixture_decorators` | 防止路由处理函数被整片误报 |
| `test_no_unreferenced_modules` | 无"整个模块无人 import" |
| `test_no_unused_functions_or_methods` | 无零调用点的函数/方法 |
| `test_no_unused_classes_or_constants` | 无未引用类/模块常量/类属性 |
| `test_no_dead_config` | 无"定义了但无人消费"的伪配置 |
| `test_no_unused_declared_dependencies` | 无"声明了却无 import"的依赖 |
| `test_allowlist_entries_are_not_stale` | **僵尸豁免校验**（对齐 RUF100 思路） |
| `test_every_allowlist_entry_has_reason` | 禁止静默豁免 |
| `test_no_unused_imports_or_variables` | ruff 交叉验证（未装则 skip，不失败） |
| `test_deadcode_scan_cli_is_runnable` | CLI 可用性 |
| `test_strict_mode_reports_test_only_production_api` | `--strict` 视角有效：仅被测试引用的生产函数必须报出 |
| `test_strict_mode_does_not_report_test_helpers` | `--strict` 不报测试目录内的定义（否则噪音淹掉信号） |
| `test_strict_mode_keeps_script_only_dependencies` | `--strict` 不影响依赖判定（只给脚本用的库仍是真实依赖） |

失败信息直接给出 `file:line`、问题说明与处理方式（删除 / 登记豁免 / 行内 ignore）。

### 6.3 处理一条发现的三条路径

```python
# 1) 直接删除（默认，绝大多数情况）
# 2) 确需保留 → 在 tests/deadcode_allowlist.py 登记并写明理由
# 3) 就近标注 → 在定义处加 `# deadcode: ignore`
```

### 6.4 CI 门禁

`deadcode` job 与功能测试分离，任一环节非零退出即失败：

```yaml
- pytest tests/test_deadcode.py -q          # 自建扫描器 + 豁免清单
- ruff check app scripts tests              # 未使用 import / 变量 / 未定义名
- vulture                                   # 未使用函数 / 类 / 变量
- deptry .                                  # 未使用 / 缺失 / 传递依赖
```

命令均不带参数，**规则集单一来源于 `pyproject.toml`**，避免"测试里一套规则、CI 里另一套"的漂移。

---

## 7. 验证

| 验证项 | 结果 |
|---|---|
| 自建扫描器 | 32 项发现 → 清理 28 项 + 豁免 4 项 → **剩余 4 项全部已登记且理由齐备** |
| ruff `F` 规则集 | 13 项初始 + 6 项级联孤儿 → **All checks passed** |
| vulture（80% 档） | 1 项 → **0 项** |
| deptry | 19 项（含 artifacts 噪音）→ 配置收敛后 **No dependency issues found** |
| 全量 pytest | **89 passed**（原 78 + 新增 11） |
| **门禁有效性实测** | 注入一个死函数 → 门禁**立即失败**并打印 `app/_deadcode_probe.py:4`；移除后 → **11 passed** |
| 自检有效性实测 | 修复 §2.4 缺陷 3 后，扫描器**当场多报出 4 项真问题**（`traceable`、`MemoryRedis.ttl`、`MemoryRedis.dbsize`、`CHUNK_FALLBACK_OVERLAP_RATIO`），证明该缺陷此前确实在削弱检测力 |

---

## 8. 遗留项处置（2026-09-11 全部落地）

本节替代原先的「遗留与后续建议」——9 项已逐条处置完毕，**保留原判断与结论的对错记录**，
以便复查时知道哪些是当时的判断、哪些是后来的修正。

### 8.1 原「需产品/架构决策」4 项

| 项 | 决定 | 落地 |
|---|---|---|
| `CHUNK_FALLBACK_OVERLAP_RATIO` | **接线**（不删） | 接到唯一一条降级路径：`_chunk_structure()` → flat → `_chunk_recursive(degraded=True)` → `_splitter(_fallback_overlap())`。语义与边界见 `docs/chunking-contract.md` C4。接线后语料未变，基线快照无需重建（护栏全绿即"零行为变化"的机器证明）。豁免条目已从 allowlist 移除 |
| `CASCADE_ENABLED` | **补实现**（不摘开关） | 新增 `app/core/cascade.py`（触发判据为纯函数 + 三道闸门 + 会话预算 + 观测统计），接入 `nodes.py::generate_answer_node`。流式链路显式跳过并在 `route_decision` 记 `skipped=streaming`。回归测试 23 项见 `tests/test_cascade.py` |
| `SOUL.md` 写入路径 | **新增带鉴权的端点**（不恢复裸写入） | `PUT /memory/soul`：`AUTH_ENABLED=false` 时直接 403（fail-closed），开启后由全局中间件校验 API Key。同时把 `MemoryStore._write_file` 改为原子写（tmp + `fsync` + `os.replace`）——SOUL.md 是全局共享文件，半截内容会污染所有人的 `memory_context` |
| `artifacts/backup/**` | **归档到仓库外**（不删除） | 9 份历史快照移到 `../_archive/langgraph-enterprise-bot-source-backup-20260911/`。**刻意不删**：本项目无 git，这些快照是唯一的回退手段。`artifacts/backup/` 保留并加 README 说明搬运方式与理由 |

### 8.2 原「可继续推进」5 项

| # | 项 | 落地 |
|---|---|---|
| 1 | 收敛 DEP003 | `starlette` / `PyYAML` / `Pillow` 升格为 `requirements.txt` 直接依赖。判据不是"是不是传递依赖"，而是**失效代价**：starlette 缺失会在启动时炸；Pillow 缺失会被 `doc_loader` 的 `except` 吞掉，表现为"扫描版 PDF 静默变空文本"——最难排查的一类故障。**并顺带查出两条僵尸豁免**：`langsmith`（全仓已无直接 import）与 DEP004 的 `pytest`（从未真的报过 DEP004），二者已删除 |
| 2 | 启用 `RUF100` + `E4` + `BLE` | ruff `select` 扩为 `["F", "E4", "BLE", "RUF100"]`。实测 16 项：**12 项僵尸 noqa**（其中 3 项是"规则没开就显得多余、开了才成为必要"的陷阱，见下）、1 项真 E402（`retriever.py` 中部 `import fnmatch`，已上移）、3 处裸 except 已补 `# noqa: BLE001` + 理由 |
| 3 | 区分「生产使用」与「仅测试/工具使用」 | `scripts/deadcode_scan.py` 新增 `--strict`：只把 `app/` 内的引用算作已使用，只报 `app/` 内的定义。默认 3 项 → 严格 **9 项**，新暴露 6 项：`CHUNK_MIN_CHARS`、`filter_by_section`、`detect_structure`（正是原先预测的三项）、`reset_smalltalk_stats`、`KnowledgeBaseEmpty`、`SectionTree.path_at`。**刻意不进 CI 门禁**（结论需逐条人工判断），定位是定期巡检；依赖判定不受 strict 影响（只给脚本用的库仍是真实依赖）。`app/main.py` 属入口模块（`uvicorn app.main:app` 加载），已连同 `__init__.py`/`__main__.py` 一起列入 `ENTRY_MODULE_NAMES` 豁免，否则严格模式会稳定误报 |
| 4 | `soft_warnings` 只写不读 | 读出点 `app/api/chat.py::collect_soft_warnings`：落 WARNING 日志（含细节）+ 在响应体透出 `soft_warning_count`。**只透数量不透原文**——异常文本可能含内网地址/路径，与 500 处理器"不回显异常原文、只给 trace_id"的口径一致 |
| 5 | `__init__.py` re-export 显式化 | 已核对：需要 `__all__` 的 3 个包（`app/memory`、`app/providers`、`app/rag`）已声明；其余 7 个 `__init__.py` 只有一行 docstring、不 re-export 任何符号，加 `__all__` 属多余。**无需改动** |

### 8.3 处置后复验

```
全量 pytest        147 passed（治理前 89 → 本轮 +58）
ruff check         All checks passed（规则集已扩至 F+E4+BLE+RUF100）
vulture            零输出
deptry             Success! No dependency issues found
自建扫描器（默认）  3 项，全部在 allowlist 内且理由齐备
自建扫描器（--strict）10 项，待人工判断（不进门禁）
```

**一个必须记住的顺序陷阱（已写进 `pyproject.toml` 注释）**：开 `RUF100` 之前必须先开
`BLE`。RUF100 的两类报错含义完全不同，务必分清再动手：

- `non-enabled: X` —— 规则 X 没开，该 noqa 当前无用，**但规则一开就可能变成必需**。
  先清后开 = 删掉必需的抑制注释，规则一开立刻全红。
- `unused: X` —— 规则 X 开着、此处确实不违规，是**真僵尸**，可放心删。

已用最小样例实测验证（`select=F,RUF100` 报 2 处 `non-enabled`；`select=F,BLE,RUF100`
只剩 1 处 `unused`）。本轮 12 处清理全部属第二类。其中 3 处尤其值得记录：它们抑制的是
`BLE001`，而 ruff 的 BLE001 **不报"捕获后重新抛出/转换抛出"的处理器**，所以
`providers/llm.py`（重试与流式回退）、`rag/evaluator.py`（异常类型转换）那 3 处
从一开始就不需要 noqa。判据是**异常有没有被吞掉**，而不是"有没有写 `except Exception`"。

**门禁有效性再次被自证**：本轮为新测试 `tests/test_soft_warnings.py` 写代码时多写了一个
`import pytest`，`ruff check` 当场拦下（`F401 tests/test_soft_warnings.py:17`）。
即"新增即拦截"的定位成立，不是永远绿灯的假门禁。

**一处仍存的漏报（已知局限的实例，未修）**：`prompts.render`（仅 `tests/test_infra.py`
引用）在 `--strict` 下**仍未报出**——因为 `app/static/gen_favicon.py` 里有个同名函数
`render`，"裸名撞名"把死代码洗白了。这与 §7 记录的 `MemoryRedis.get` 是同一类问题，
根因是扫描器刻意不限定名解析（零依赖/零配置的代价）。要根治需要引入限定名解析——
成本明显高于收益，故保留为**已知局限**：定期巡检时请把 `render`/`get`/`run`/`save`
这类高频名单独人工过一遍。

---

## 9. 一句话结论

> 51 项未使用问题已处理 47 项、登记豁免 4 项。根因不是"写错代码"，而是**功能做了一半没有出口**、**能力被取代后旧实现未删**、**门面层过度铺开**，以及最关键的——**从来没有机制在死代码产生时给出反馈**。本次除了清理存量，更把"未使用代码"变成了 CI 里可回归的失败：三条工具 + 一套懂项目约定的扫描器 + 一份每条都写明理由的豁免清单。而在构建这套机制的过程中，**检测器自身的 4 个"静默失效"缺陷也被自检用例逐个逼了出来**——这恰恰印证了同一件事：没有反馈机制，缺陷就会一直潜伏。
