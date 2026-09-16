# 工具调用能力启用方案：`bind_tools` 包装层修复

> ⚠️ **本文写作于「单 Agent 架构」时期，其中提到的部分模块已随多 Agent 重构删除**
> （`core/{intent_router,model_router,complexity_scorer,query_signals,intent_catalog,cascade,smalltalk}.py`、
> `tools/{rule,ticket,user}_tool.py`、`api/routing.py`，存档见 `_archive/`）。
> **当前架构以 [`docs/multi-agent-architecture.md`](multi-agent-architecture.md) 为准**；
> 本文的**问题分析、实测数据与判定方法仍然有效**，读时把模块名当作"当时的现场"。


> 状态：**§4.2 方案 B 已实施**（2026-09-15）。`bind_tools` 委托已落地，
> 既有 369 条测试零影响、ruff 与死代码门禁全绿。
> 实施记录与后续见 §八；消费方方案见 `docs/history/offline-slot-extraction-migration.md`。
> 前置发现见 `docs/history/tool-invocation-online-vs-offline.md` §9
> 原型与实测脚本：`artifacts/proto_bind_tools_retry.py`、`artifacts/diag_function_calling2.py`、
> `artifacts/diag_tool_schema.py`（新增：schema 与委托的离线验证）

---

## 一、结论

function calling 在本项目**不是「没选择用」，而是「物理上不可用」**：

```python
get_chat_model().bind_tools([query_user_info])
# → NotImplementedError
```

根因在自造的限流重试包装器：`app/providers/llm.py:188-271` 的
`_RateLimitRetryModel` 只实现了 `_llm_type` / `_generate` / `_stream`，
**没有把 `bind_tools` 委托给内层模型**，于是落到 `BaseChatModel` 的默认实现
（`raise NotImplementedError`）。底层 `ChatOpenAI`（`app/providers/llm.py:274-305`）
是支持的——能力被自己加的包装层挡住了。

**本方案改动约 30 行，不改任何现有行为**（全仓 0 处调用 `bind_tools`），
作用是**打开能力**，为后续「惰性升级 / 模型抽参」留出入口。

---

## 二、根因链路

```
get_chat_model()                         app/providers/llm.py:313-351
  └─ _build_real_model()                 app/providers/llm.py:308-310
       └─ _RateLimitRetryModel(           app/providers/llm.py:188-271
            delegate=_build_raw_model()  app/providers/llm.py:274-305
          )
          ├── _generate  ✓ 已实现         app/providers/llm.py:212-230
          ├── _stream    ✓ 已实现
          └── bind_tools ✗ 未委托  ← 断点
```

**实测证据**（`artifacts/diag_function_calling.py`）：

```
type(bind_tools(...)) = _ChatModelBinding      # 直接拿底层模型时，是 RunnableBinding 子类
                                                
get_chat_model().bind_tools([...]) → NotImplementedError   # 经包装层时，崩
```

两条命令的差别只在「是否经过 `_RateLimitRetryModel`」，定位唯一。

---

## 三、为什么不能用 LangChain 内置的 `with_retry`

`Runnable.with_retry(retry_if_exception_type=..., stop_after_attempt=...)` 看起来
正好能用，但**必须放弃**，理由是本项目已经犯过四次的同一个病：

`_is_rate_limit`（`app/providers/llm.py:148-164`）是**三层判据**：

1. `exc.status_code == 429`
2. 异常类型名含 `RateLimit`
3. 错误码语境匹配 `429` / `rate limit` / `too many requests`

它是**函数判断**，**无法降级成异常类型元组**。硬用 `with_retry` 只有两条路：
要么漏判（第 3 类异常不再重试），要么在调用点再写一遍判断逻辑——
**那就是第二套重试判据**，与本项目的 `_is_rate_limit` 各自漂移。

同理，退避参数必须继续来自 `LLM_MAX_RETRIES`（`app/config.py:91`）、
`LLM_RETRY_BASE_DELAY`（`app/config.py:92`）、`LLM_RETRY_MAX_DELAY`（`app/config.py:94`），
不能另立一套。

**结论：重试逻辑必须收敛成单一实现，`_generate` 与工具绑定路径共用。**

---

## 四、方案

### 4.0 前置：抽出一处重试实现

把 `_RateLimitRetryModel._generate`（`app/providers/llm.py:212-230`）现有的退避循环
原样抽成方法，逻辑**逐字不变**：

```python
def _call_with_rate_limit_retry(self, fn, *args, **kwargs):
    """限流退避重试的唯一实现，_generate 与工具绑定路径共用。"""
    last_exc = None
    for attempt in range(self._max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < self._max_retries:
                delay = min(self._base_delay * (2 ** attempt), config.LLM_RETRY_MAX_DELAY)
                delay += random.uniform(0.0, 0.3) * delay
                logger.warning("大模型触发限流(429)，%.1fs 后自动重试（%d/%d）",
                               delay, attempt + 1, self._max_retries)
                time.sleep(delay)
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc
```

`_generate` 改为 `return self._call_with_rate_limit_retry(self._delegate._generate, messages, ...)`。
**行为与现在完全一致**——这是纯提取，不是改写。

### 4.1 方案 A（最小改动，不新增类）

```python
def invoke_with_tools(self, messages, tools, **kwargs):
    """带工具绑定的一次调用（复用同一套限流重试）。"""
    bound = self._delegate.bind_tools(tools)
    return self._call_with_rate_limit_retry(bound.invoke, messages, **kwargs)
```

- **约 15 行**，不新增模块级符号，死代码门禁零风险。
- 代价：**不符合 LangChain 的 `bind_tools` 协议**，不能直接喂给
  `create_react_agent` 之类的预构建组件。

### 4.2 方案 B（协议兼容，推荐）

```python
from langchain_core.runnables import RunnableBinding

class _RetryBoundTools(RunnableBinding):
    """工具绑定模型 + 同源限流重试。只覆写 invoke：全链路无异步调用（已核实）。"""

    retry_host: Any = None

    def invoke(self, input, config=None, **kwargs):
        merged = {**self.kwargs, **kwargs}
        return self.retry_host._call_with_rate_limit_retry(
            lambda: self.bound.invoke(input, self._merge_configs(config), **merged)
        )
```

外加入口（`_RateLimitRetryModel` 上）：

```python
def bind_tools(self, tools, *, tool_choice=None, **kwargs):
    bound = self._delegate.bind_tools(tools, tool_choice=tool_choice, **kwargs)
    return _RetryBoundTools(bound=bound, retry_host=self)
```

- **约 35 行**，协议兼容：返回 `Runnable[LanguageModelInput, AIMessage]`，
  与 `BaseChatModel.bind_tools` 的签名一致。
- `_RetryBoundTools` 是模块级新类，但**被 `bind_tools` 直接引用**，不构成死代码。

> **实现约束（实测得出，不要"优化"掉）**
>
> `bind_tools` 的方法体**必须**保留 `self._delegate.bind_tools(...)` 这个**同名调用**。
>
> 项目自建的死代码扫描器**不认覆写关系**：`TRUSTED_STDLIB_ROOTS`
> （`scripts/deadcode_scan.py:158`）只含标准库，解析不出第三方 `BaseChatModel`
> 的成员，因此「覆写框架方法」这条豁免**对它无效**。实测两版实现：
>
> | 方法体写法 | 死代码门禁 |
> |---|---|
> | `self._delegate.bind_tools(tools, ...)` | ✅ 14 passed |
> | `return tools`（不出现同名） | ❌ `unused_method: _RateLimitRetryModel.bind_tools` |
>
> 也就是说：这个公有方法之所以过门禁，靠的是**方法名在方法体里被再次提到**
> （扫描器按裸名跨文件统计「被使用」），**不是**因为它是框架回调。
>
> 这是「能过」与「知道为什么能过」的差别——若将来有人把这一行重构成某种间接调用，
> 门禁会变红，而那**恰恰是提醒**：届时正确做法是在
> `tests/deadcode_allowlist.py` 显式登记并写明「覆写 LangChain 协议方法」，
> 而不是把同名调用硬塞回去。

### 4.3 选哪个

| | 方案 A | 方案 B |
|---|---|---|
| 适用 | 只在图节点里手动调一次模型抽参 | 要用 LangGraph 预构建 agent / 复用生态组件 |
| 改动 | ~15 行，无新符号 | ~35 行，1 个新类 |
| 协议 | 不兼容 | 兼容 |

本项目是**手写 LangGraph 节点**（无 `create_react_agent`），方案 A 其实够用；
但 B 的增量成本很小，且避免了「将来想用生态组件时再返工」。
**默认推荐 B。**

---

## 五、原型验证（零配额，4/4 通过）

`artifacts/proto_bind_tools_retry.py` 用**假 bound** 离线验证重试语义，
**不调用任何模型**：

| 场景 | 期望 | 实测 |
|---|---|---|
| 429 两次后成功 | 重试并成功，共 3 次调用 | ✅ 通过 |
| 429 超过上限 | 最终抛出，共 `max_retries+1` 次 | ✅ 通过 |
| 非 429（超时/参数错） | **不重试**，立即抛，共 1 次 | ✅ 通过 |
| `kwargs` 透传 | `tool_choice` 带到 bound | ✅ 通过 |

第三行是**刻意设计**，不是遗漏：超时快速失败是既有策略
（`_is_timeout` 的注释说明：重试只是把等待翻倍）。方案**保持**了它，
没有因为「顺手加个重试」而改变。

另经实测（`artifacts/diag_function_calling2.py`，2 次真实调用）：

| 问句 | function calling 输出 |
|---|---|
| 四月在哪个部门 | `query_user_info({"user_identifier": "四月"})` |
| 我的分机号是多少 | 无 `tool_calls`，反问用户补充信息 |

**边界要记住**：第二行证明「值不在句子里」时，**换用 function calling 一样抽不出**。
本方案只负责**打开能力**，不改变这条物理天花板。

---

## 六、验收清单

1. `get_chat_model().bind_tools([...])` 不再抛 `NotImplementedError`（原为红）。
2. **反向验证**：把 `_RetryBoundTools.invoke` 里的 `_call_with_rate_limit_retry`
   换成直接调用 `self.bound.invoke(...)`，429 场景必须**立即变红**。
   —— 这是证明「护栏真的在守」的唯一方式（本项目的同源护栏曾被 docstring 喂饱而空转）。
3. 反向验证 2：把 `_call_with_rate_limit_retry` 里的 `_is_rate_limit` 换成
   `True`（无条件重试），非 429 场景必须变红。
4. 既有 **369 条测试全绿**（预期零影响：0 处调用 `bind_tools`）。
5. `ruff` 全绿；**死代码门禁通过**——但要清楚它**不是**靠「覆写豁免」通过的：
   实测把方法体改成不出现同名调用，立即报 `unused_method`（见 §4.2 的实测表）。
6. **限流语义未分叉**：全仓 `_is_rate_limit` 仍只有一处定义、两处调用
   （`_generate` 与工具绑定路径）。

---

## 七、风险、边界与「不做会怎样」

**风险**：接近零。改动是「新增一个未被任何现有代码调用的入口」，
不触碰任何现有路径。唯一的失败模式是 `bind_tools` 委托写错——
由验收第 1、2 条覆盖。

**真正的成本在后面，不在本方案**：
要让 function calling 真正产生价值，还需要改
`app/graph/nodes.py`（工具节点改为模型驱动）、
`app/core/intent_catalog.py`（`IntentSpec` 增加参数声明）——
**那才是大工程**。本方案只是它的前置条件。

**不做会怎样**：维持现状。当前所有功能正常，只是
「模型抽参」这条路**走不通**——如果确定不用，这个方案可以不做。

**一处必须同时记住的约束**：即便能力打开，
**身份 / 会话类槽位永不走模型**（不能由模型决定查谁的 PII），
且 LLM 抽参必须放在开关之后（否则测试要么打满配额、要么退化为 mock）。

---

## 八、实施记录与后续

1. ~~现在做，还是等真要接入时再做？~~ → **已做**（2026-09-15）。
   抽参迁移就是本方案的直接消费方，不再是「零收益」的改动。
2. ~~选方案 A 还是 B？~~ → **选 B 并已实施**。落地三处：
   `_RetryBoundTools`（`app/providers/llm.py:189-210`）、
   `bind_tools` 委托（`app/providers/llm.py:268-279`）、
   重试实现抽成 `_call_with_rate_limit_retry`（`app/providers/llm.py:237-261`）。

**验证结果**（`artifacts/diag_tool_schema.py`，零配额）：

| 检查项 | 结果 |
|---|---|
| `bind_tools` 返回值 | `_RetryBoundTools`（原为 `NotImplementedError`） |
| 429×2 后成功 | 调用 3 次后返回，**重试走的是同源实现** |
| 369 条既有测试 | 全绿 |
| ruff / 死代码扫描 | 全绿（`bind_tools` 方法体保留同名调用） |

> 一个副产品发现：`RunnableBinding.bound` 有 pydantic 类型约束，
> 假对象必须是 `Runnable` 子类才能注入——**约束真的在生效**，
> 不是"随便包一层"。写测试替身时会遇到。

**剩余待拍板**：模型抽参的**降级出口**（Mock / 无配额时抽不出参数怎么办），
见 `docs/history/offline-slot-extraction-migration.md` §五 决策①。
