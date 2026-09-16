# 业务正确性故障排查与修复记录

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


**故障现象（用户报告）**
1. 正常问答被拒绝回答；
2. 工具调用一直加载（长时间无响应）。

**排查方式**：`logs/app.log` + `logs/trace.jsonl` 时间戳回溯 + 代码审读 + 进程内真实链路复现。
本项目非 git 仓库，无 diff 可比对，结论全部由日志与实测支撑。

---

## 一、症状一：正常问答被拒绝回答

两个独立根因，任一都能单独造成拒答。

### 根因 A · 软回退片段 `fused=0` → 置信度恒为 0 → 必然拒答

**现场证据**（`logs/app.log`，session `46b2cb98209c4e5b`）：

```
19:24:46 意图路由[规则]：knowledge（0.78，knowledge:流程咨询）
19:24:46 阈值过滤后无结果，触发软回退：top1_vec=0.0536
19:24:46 生成前置拒答：置信度 0.000 低于阈值 0.25
19:24:46 流式对话完成 耗时=7ms 命中=3
```

**关键矛盾**：命中 3 条却拒答。

**成因链**：

```python
# app/rag/retriever.py（修复前）
item["fused"] = 0.0     # ← 根因

# app/rag/generator.py
fused_max = (DENSE_WEIGHT + LEXICAL_WEIGHT) / (RRF_K + 1)
base = min(1.0, float(top.get("fused", 0.0)) / fused_max)
if top.get("fallback"):
    base *= 0.6          # 软回退降权：但 0 × 0.6 仍然是 0
```

设计本意是「软回退证据弱，故降权 0.6」，但 `fused` 被写死为 0，
归一化结果恒为 0，降权逻辑形同虚设，置信度必然跌破 `REFUSE_THRESHOLD=0.25`。

**修复**：写入该片段**真实的 RRF 融合分**。这些 key 全部来自 `dense_pool`，
已在上方 `rrf_fuse` 中算过，直接取用即可：

```python
item["fused"] = round(fused.get(key, 0.0), 6)
```

修复后 rank1 软回退片段的置信度为 `0.7 × 0.6 = 0.42 > 0.25`，降权语义真正生效。

### 根因 B · 意图规则冲突：「邮箱」裸词同时属 tool 与 knowledge

**现场证据**：

```python
# app/core/intent_router.py（修复前）
_TOOL_RULES = [
    ...
    (re.compile(r"(工号|谁的|联系方式|邮箱|哪个部门|分机号|隶属)"), "tool", 0.85, "员工信息查询"),
]
_KNOWLEDGE_RULES = [
    ...
    (re.compile(r"(VPN|vpn|密码|账号|邮箱|工牌|门禁|打印机|网络|系统|软件|安装)"),
     "knowledge", 0.75, "IT 运维主题"),
]
_RULE_GROUPS = [("tool", _TOOL_RULES), ("knowledge", _KNOWLEDGE_RULES)]   # tool 组在前
```

`match_rules` 命中即返回，tool 组在前 → 任何含「邮箱」的问句都被判成工具调用，
`_KNOWLEDGE_RULES` 永远拿不到：

| 问句 | 修复前 | 应为 |
|---|---|---|
| 怎么申请邮箱扩容 | tool 0.85 | knowledge |
| 邮箱密码忘了怎么办 | tool 0.85 | knowledge |
| 我的邮箱怎么配置 | tool 0.85 | knowledge |

判成 `tool` 只保证走 `tool_invoke_node`，而该节点还需从 query 里抽取参数
（工单号正则 / 员工标识）。抽不到就返回

> 未识别到有效业务参数，无法查询。请提供工单编号（如 T20240101）或员工账号。

用户看到的与拒答无异。

**修复一（提高精度）**：员工信息查询改为**结构匹配**——必须出现「索取属性取值」的句式：

| 结构 | 例 |
|---|---|
| A 查询动词 + 属性词 | 查一下张三的工号 |
| B1 谁/哪位 + 的 + 属性词 | 谁的邮箱 |
| B2 哪个/什么 + 属性词（到句尾） | 哪个部门 |
| C 属性词 + 是/为 + 疑问词 | 工号是什么 |
| D 人称代词 + 的 + 属性词（到句尾） | 我的邮箱 |

且整句不得含 how-to 操作词（`怎么 / 如何 / 申请 / 配置 / 重置 / 找回 / 扩容…`）——
「我的邮箱怎么配置」问的是操作步骤，属知识问答。
B2 要求锚定句尾，否则会吞掉「哪个部门负责报销」这类知识问句。

判定函数 `looks_like_employee_query` 同时供意图路由与工具节点调用，
消除 `_USER_HINT` 与 `_TOOL_RULES` 两张手抄表漂移的问题（见 `docs/history/redundancy-review.md`）。

**修复二（fail-open 兜底）**：即使仍被误判成 tool，抽不到参数时也**回退知识检索**
而不是回一句等同拒答的话术。参数抽取失败属**可降级**故障，真正需要转人工的只有
「工具抛异常」。回退检索本身失败也只记 `soft_warnings`，绝不写 `error_msg`
（那会把可降级抖动升级成人工转接，即 `docs/history/project-assessment.md` P0-1 的教训）。

---

## 二、症状二：工具调用一直加载

### 根因 C · 超时未被约束，30s 被放大到 100s+

两层重试叠加：

```python
# app/providers/llm.py（修复前）
return ChatOpenAI(...)   # 未传 max_retries → OpenAI SDK 默认 2 次
```

```python
def _stream(...):
    except Exception as exc:
        if yielded: raise
        # ← 不区分异常类型：超时也回退 _generate，等于把完整等待重新付一遍
    result = self._generate(...)
```

**现场证据**：`logs/trace.jsonl`，`trace_id=d2d3424188744932`，
`span=generate_answer 耗时=101399ms`（`.env` 中 `LLM_TIMEOUT=30`）。

**修复**：
1. `_build_raw_model` 显式 `max_retries=0`，禁用 SDK 内置重试，重试策略统一收口到
   `_RateLimitRetryModel`（**只对 429 退避重试**）；
2. 新增 `_is_timeout()`，`_stream` 遇到超时**不回退**，快速失败交由上层降级——
   超时意味着供应商侧拥堵或输出过长，重试几乎不会成功，只会让用户干等翻倍。

非超时错误（如不支持流式）仍保留整段回退，不误伤原有能力。

### 根因 D · 工单号查询被硬升 Pro 档

```python
_DIFFICULTY_RULES = [
    (re.compile(r"\bT\d{3,}\b", re.IGNORECASE), TIER_PRO, 0.95, "工单号查询"),   # ← 误放 Pro 段
]
```

工单号查询是**单事实直取**（查表返回一行），不需要归纳或推理。
放在 Pro 段会让一次查工单升到最贵档位，Pro 占比被这类请求灌高、成本归因失真。

**修复**：移至 Flash 段（单事实直问）。真正的难点在理解制度条款，不在读取确定字段。

### 根因 E · 流式链路缺少人工兜底分支

非流式图有 `generate_answer → error_route_edge → human_fallback`，
而 `app/api/chat.py::_pre` 手工复刻意图分支时**漏了这一步**：工具异常时图会把答案
换成「已转接人工」，流式链路却照样去调模型生成，`meta` 里还报 `need_human=true`——
两条链路对同一状态给出不一致的回答。

**修复**：`_pre` 末尾补 `error_route_edge → human_fallback_node`，并放在**生成之前**：
此处已确定「无法自动处理」，再花一次模型调用产出一个立刻被覆盖的答案纯属浪费。
SSE 侧新增 `need_human` 分支整段推送该文案。

---

## 三、验证

### 真实链路端到端（真实 embedding + 真实 LLM）

| 问句 | 意图 | 档位 | 置信度 | 拒答 | 命中 |
|---|---|---|---|---|---|
| 怎么申请邮箱扩容 | knowledge | flash | 1.0 | False | 5 |
| 邮箱密码忘了怎么办 | knowledge | flash | 1.0 | False | 5 |
| 我的邮箱怎么配置 | knowledge | flash | 1.0 | False | 5 |
| 年假有多少天 | knowledge | flash | 1.0 | False | 5 |
| 张三的邮箱是什么 | tool | flash | 0.6 | False | 员工信息已返回 |
| T20240101 什么状态 | tool | **flash**（原 pro） | 0.6 | False | 工单状态已返回 |

流式端点实测：「怎么申请邮箱扩容」→ `refused=False`，给出带引用 `[1]` 的真实答案。

### 回归测试

新增 `tests/test_biz_correctness.py`（37 项），覆盖 A–E 五个根因，
含「固化故障机理」的对照用例（`fused=0` 时置信度必为 0，防止回退）。

全量门禁：`pytest` 200 passed；`ruff check .` All checks passed；
`deadcode_scan --strict` 无新增项（9 项为存量）。

---

## 四、遗留与后续

1. `_extract_identifier` 原实现硬编码 `([张李王赵孙][三四五六七])`，只能命中五个特定姓名
   （换「张伟」「李明」静默失效）。已改为复用工具侧别名表 `_ALIASES`
   （新增 `user_tool.extract_identifier`），但**「我 / 我的」这类第一人称尚未映射到
   `user_id`** —— 目前会走 fail-open 回退检索，可用但不是最优。
2. `app/main.py` 的 `_run_startup_check` 在 local-hash 模式下会把「语义检索能力」判为
   `[FAIL]`（误报），需要按 embedding 模式分级判定。
3. 检索链路在 API embedding 模式下 `SCORE_THRESHOLD=None`（不过滤，取 Top-K），
   软回退分支实际只在 local-hash 模式或用户显式配阈值时触发；
   根因 A 的修复对两种模式都有效，但该分支的线上覆盖率偏低，值得补一条观测。
