"""死代码豁免清单 —— 检测器报出、但经人工判定「应当保留」的条目。

为什么需要这个文件
------------------
静态检测必然有一类"技术上未被引用、语义上必须保留"的条目。业界做法不是把工具
关掉，而是**显式列出豁免 + 逐条写明理由**（见 deptry 官方文档的 `per_rule_ignores`
用法、contextgem 等真实项目的写法）。这样：

1. 豁免变成一次**有意识的决策**，而不是工具静默忽略；
2. 代码评审时任何人可以看到"豁免了什么、为什么"，避免豁免成为藏污纳垢之处；
3. `tests/test_deadcode.py` 会反向校验——**清单里的条目若不再被报出，就说明它是
   过期的僵尸豁免，测试会失败**（对齐 ruff `RUF100` 治理僵尸 noqa 的思路）。

新增条目的要求
--------------
键为 `Finding.key`（`kind:file:qualname`）。**必须**在 ALLOWLIST_REASONS 里写理由，
否则测试直接失败。写理由时请回答：为什么不能删？删了会怎样？

格式示例见下方。
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# 豁免条目：key -> 保留理由
# ---------------------------------------------------------------------------
ALLOWLIST_REASONS: Dict[str, str] = {
    # 构建期脚本，非运行时模块：由开发者手动执行 `python app/static/gen_favicon.py`
    # 生成 favicon 资源，产出物（png/ico）已被前端静态引用。它本就不该被 import。
    "unreferenced_module:app/static/gen_favicon.py:app.static.gen_favicon": (
        "构建期资源生成脚本，手工执行；产出 favicon.png/favicon.ico 供静态引用，"
        "不是运行时模块，无需被 import。"
    ),

    # ---- redis.asyncio 降级替身的接口完整性 ----
    # MemoryRedis 是真实 Redis 不可用时的进程内替身，价值在于「可替换性」。
    # 删掉这些方法不会立刻出错，但会制造最难查的环境相关 bug：真实 Redis 环境正常、
    # 只有降级环境才 AttributeError。保留模块 docstring 声明的「最小方法集」。
    "unused_method:app/db/redis_db.py:MemoryRedis.ttl": (
        "redis.asyncio 降级替身的接口方法。当前无调用方，但保留以保证替身可替换性："
        "删除后，未来任何 `await redis.ttl(k)` 只会在无 Redis 的降级环境失败。"
    ),
    "unused_method:app/db/redis_db.py:MemoryRedis.dbsize": (
        "同上：redis.asyncio 降级替身的接口方法，属模块 docstring 声明的"
        "「最小方法集」，保留以保证可替换性。"
    ),

    # ---- 请求作用域「工具 ↔ 调用方」通道的接口完整性 ----
    # RequestScope 是一份**模块 docstring 明文承诺的契约**：授权事实、工具产出的
    # 证据、工具吞掉的降级故障，三类数据都必须经由它流转，且**绝不出现在工具
    # 的 JSON Schema 里**（见 app/core/request_ctx.py 的「为什么必须走这里」）。
    #
    # 当前 3 个 SQLite 工具只用到 `add_tool_result`，于是通道的另外几个面
    # 暂时没有调用点。**但通道本身是活的**——读端就在工具实现里，而工具是按
    # 名字调用的，静态分析看不到这条边。删掉写端最隐蔽的代价是：下一个"检索型 /
    # 会吞异常的"工具加进来时，作者只会看到一套只会 `add_*`、永远读不到东西的
    # API，然后照着 `contextvars` 的直觉重新实现一遍——而那正是本模块
    # 踩过的坑（`Runnable.invoke` 里 `set()` 写进的是 context 副本，静默丢失）。
    # 保留一份自洽的通道，比事后重新推导这套约束便宜得多。
    "unused_function:app/core/request_ctx.py:get_allowed_sources": (
        "来源白名单的读端。当前检索已在五 Agent 架构里独立成 RAG Agent，"
        "白名单由 state 经函数参数显式传入（app/core/sub_agents.py），"
        "故读端暂无调用点；保留它是因为写端 set_allowed_sources 仍在 tool_node "
        "里逐请求调用——有写无读的通道是隐患，而删掉读端会让这个隐患更隐蔽。"
    ),
    "unused_function:app/core/request_ctx.py:add_retrieved_docs": (
        "「工具产出的检索片段」写端。3 个只读工具都不产片段（制度检索已交给 "
        "RAG Agent），故当前无调用点。它是 request_ctx 模块 docstring 承诺的"
        "三类数据之一，且带有按 (source, content) 去重、返回起始下标这两条"
        "被 L4 引用溯源依赖的语义；删除等于把这段约定从代码里抹掉。"
    ),
    # 注：`add_soft_warning` 曾在此豁免。混合路由落地后，层③/层④ 的
    # `app/core/routing/fusion.py::soft_warn` 真的会调用它（依赖抖动、仲裁失败
    # 都要留痕），豁免随之成为僵尸条目，由 allowlist 的反向校验报出后删除。
    # 这正是那套反向校验存在的意义：豁免不会随着被豁免者的复活而自动失效。

    # ---- Mock 模型：LangChain 受保护钩子，静态分析看不到那条边 ----
    # MockChatModel 继承 BaseChatModel，`_generate` 是它的**必需覆写点**：
    # 调用方写的是 `model.invoke(...)`，由基类分派进来，全仓没有一行 `._generate(`。
    # 它此前之所以"看起来被引用"，是因为 tests/test_infra.py 里那段已删除的
    # 退避重试用例凑巧写了 `model._generate([])` —— 一次偶然的文本命中，
    # 而不是真实的调用边。那次删除让这条边露了出来，属**检测器变准**而非新增问题。
    "unused_method:app/providers/llm.py:MockChatModel._generate": (
        "LangChain BaseChatModel 的受保护钩子，由 invoke() 分派调用，全仓无直接"
        "调用点（属框架约定，静态分析看不到）。删掉它 Mock 模型立刻不可用："
        "无 Key 时的离线链路、以及全部注入假模型的单测都会失效。"
    ),
}

#: 供测试直接使用的键集合
ALLOWLIST = set(ALLOWLIST_REASONS)
