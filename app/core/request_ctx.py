"""请求级上下文 —— 同一请求生命周期内共享的中间结果，避免重复计算与参数穿透。

三类数据放在这里，共同点是「当前请求的事实，不该出现在函数签名里」：

1. **query 向量**（`get_query_vector` / `set_query_vector`）
   检索 dense 路算出的 query 向量，同请求内再查同一句话时可复用。
   （原先还有「意图语义打分」「动态路由打分」两个复用方，已随自造路由删除。）

2. **来源白名单**（`get_allowed_sources` / `set_allowed_sources`）
   按用户部门/权限注入的知识隔离边界。

3. **本轮证据**（`add_retrieved_docs` / `get_retrieved_docs`、`add_tool_result`）
   检索片段与业务工具返回值，供 L4 生成与引用溯源消费。

为什么 2、3 必须走这里，而不是放进工具的参数
--------------------------------------------
工具是给**模型**调用的，其 JSON Schema 就是模型能看到的全部输入。而：

- ``allowed_sources`` 是**授权事实**，来自服务端鉴权 / ACL 层。把它做成工具参数，
  等于把「我能查谁的资料」交给模型决定——这是一条绝不能越过的边界
  （PII 越权不是"答错"，是合规事故）。
- 检索片段是**工具的产物**，模型只应看到它的文本形式（通过 ToolMessage），
  而不应负责把它回传。让模型回传，既浪费 token，又会引入"模型篡改证据"的可能。

⚠️ 为什么存的是**一个可变对象**，而不是几个独立的 ContextVar
----------------------------------------------------------------
这是本模块唯一一个反直觉、且踩过一次坑的地方，改动前务必读完。

**``ContextVar.set()`` 在工具内部是无效的。** LangChain 的 ``Runnable.invoke``
（``BaseTool.invoke`` 的父实现）会执行 ``contextvars.copy_context()``，再在
那个**副本**里调用工具函数。于是：

::

    v = ContextVar("v", default=None)
    v.set([])

    @tool
    def t(x: str) -> str:
        v.set("written-inside")     # 写进了副本，外面看不见
        return "ok"

    t.invoke({"x": "1"})
    print(v.get())                  # → []   ← 写入丢失

实测（langchain-core 当前版本）确认：**在工具内 set 的 ContextVar，
调用方读不到**。症状极其隐蔽：``search_knowledge`` 明明检索到了 5 条片段，
``agent_node`` 里 ``get_retrieved_docs()`` 却是空的 → 生成层拿不到任何证据 →
置信度 0 触发拒答，用户看到「知识库中没有找到」。**而工具日志显示检索成功。**

修法是把「可变对象」放进 ContextVar：``copy_context()`` 复制的是**映射**，
值仍是**同一个对象引用**，因此对对象内部状态的修改两边都可见。

    scope = _scope_var.get()   # 同一个 RequestScope 实例
    scope.docs.append(...)     # 就地修改 → 调用方可见 ✓

生命周期与线程
--------------
``reset_request_context()`` 每次请求调用一次（图的入口节点 ``memory_load_node``）。
**必须是「新建一个作用域对象」而不是「清空已有对象」**——线程池里的 worker 会被
复用，清空共享对象会让并发请求互相看到对方的证据；新建对象则天然隔离。

``asyncio.to_thread`` 与 ``ThreadPoolExecutor.submit`` 都会复制提交线程的 context，
复制的映射里携带的仍是同一个作用域对象引用，因此主线程写入的值子线程读得到、
子线程就地写入的值主线程也读得到。这是「请求级存储」能成立的前提。
"""
import contextvars
from typing import Any, Dict, List, Optional


class RequestScope:
    """一次请求的**可变**共享状态。字段直接暴露，由本模块的函数负责读写。"""

    __slots__ = (
        "query_vectors",
        "allowed_sources",
        "docs",
        "tool_results",
        "soft_warnings",
    )

    def __init__(self) -> None:
        #: query 文本 → 向量。存文本是为了校验：同一请求内同一句话才复用。
        self.query_vectors: Dict[str, List[float]] = {}
        #: 来源白名单。None = 不过滤（默认）；空列表 = 全部拒绝。
        self.allowed_sources: Optional[List[str]] = None
        #: 本轮检索到的片段（按首次出现顺序去重），供 L4 组装上下文与引用溯源。
        self.docs: List[Dict[str, Any]] = []
        #: 本轮业务工具返回值（employee / leave_balance），按调用顺序拼接。
        self.tool_results: List[str] = []
        #: 工具内部**吞掉**的可降级故障（如检索服务抖动被降级为「未命中」）。
        #:
        #: 为什么必须由工具自己上报，而不是让调用方看工具返回值判断：工具是刻意
        #: 「永不抛异常」的（见 ``tools/knowledge_tool.py``）——检索失败与真的没检索到
        #: 都返回 ``NO_HIT``，调用方**从返回值分不出来**。若不在这里留痕，
        #: 一次检索服务故障在日志里与「知识库确实没这条」完全同形，静默且不可查。
        self.soft_warnings: List[str] = []


_scope_var: contextvars.ContextVar[Optional[RequestScope]] = contextvars.ContextVar(
    "request_scope", default=None
)


def _scope() -> RequestScope:
    """取当前作用域；不存在则就地建一个。

    惰性创建服务于「不走图、直接调工具」的场景（自检脚本、Dify 检索端点、
    单元测试）。它建在这个 context 里，写不进任何父 context——所以**走图的
    链路必须先调用 `reset_request_context()`**，入口节点已经这么做了。
    """
    scope = _scope_var.get()
    if scope is None:
        scope = RequestScope()
        _scope_var.set(scope)
    return scope


def reset_request_context() -> None:
    """开启一个新的请求作用域。**每次请求开始必须调用一次**。

    为什么是「新建对象」而不是「清空旧对象」：线程池的 worker 会被复用，
    清空共享对象等于让并发请求在同一块内存上互相覆盖。
    """
    _scope_var.set(RequestScope())


# ---------------------------------------------------------------------------
# query 向量复用
# ---------------------------------------------------------------------------
def get_query_vector(query: str) -> Optional[List[float]]:
    """读取当前请求已算好的 query 向量（文本匹配才算命中）。"""
    return _scope().query_vectors.get(query)


def set_query_vector(query: str, vec: List[float]) -> None:
    """暂存当前请求的 query 向量，供后续复用。"""
    _scope().query_vectors[query] = vec


# ---------------------------------------------------------------------------
# 来源白名单（授权事实，模型不可见）
# ---------------------------------------------------------------------------
def set_allowed_sources(sources: Optional[List[str]]) -> None:
    """写入本轮允许访问的文档来源白名单。"""
    _scope().allowed_sources = sources


def get_allowed_sources() -> Optional[List[str]]:
    """读取本轮来源白名单。None 表示不做来源限制。"""
    return _scope().allowed_sources


# ---------------------------------------------------------------------------
# 本轮证据累计
# ---------------------------------------------------------------------------
def add_retrieved_docs(docs: List[Dict[str, Any]]) -> int:
    """把一批检索片段并入本轮证据集，返回**并入前**已有片段数。

    去重按 ``(source, content)``：模型可能用不同措辞多次调用检索，
    同一片段被召回两次是常态。不去重的话，引用清单里会出现重复条目，
    ``citation_coverage`` 这类指标会被稀释，Top-K 也会被同一段落占满。

    Returns:
        这批片段在全局证据集中的起始下标（0 基）。
    """
    scope = _scope()
    seen = {(str(d.get("source", "")), str(d.get("content", ""))) for d in scope.docs}
    start = len(scope.docs)
    for doc in docs or []:
        key = (str(doc.get("source", "")), str(doc.get("content", "")))
        if key in seen:
            continue
        seen.add(key)
        scope.docs.append(doc)
    return start


def get_retrieved_docs() -> List[Dict[str, Any]]:
    """读取本轮累计的检索片段（已去重、按发现顺序）。"""
    return list(_scope().docs)


def add_tool_result(text: str) -> None:
    """记下一次业务工具返回值（非检索类）。"""
    if not text:
        return
    _scope().tool_results.append(str(text))


def get_tool_results() -> List[str]:
    """读取本轮全部业务工具返回值。"""
    return list(_scope().tool_results)


# ---------------------------------------------------------------------------
# 可降级故障（工具内部吞掉的异常）
# ---------------------------------------------------------------------------
def add_soft_warning(text: str) -> None:
    """记录一条被工具吞掉的可降级故障。

    与 ``add_tool_result`` 的区别：那是**业务数据**（员工信息、假期余额），
    会被喂进 L4 生成上下文；本函数记录的是**观测信息**，只进日志与响应体计数，
    **绝不进生成上下文**——否则模型会对着一条「embedding 服务不可用」写解释。
    """
    if not text:
        return
    _scope().soft_warnings.append(str(text))


def get_soft_warnings() -> List[str]:
    """读取本轮工具内部记录的可降级故障。"""
    return list(_scope().soft_warnings)
