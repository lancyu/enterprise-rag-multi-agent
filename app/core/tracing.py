"""trace_id 贯穿 + 嵌套 span 树 + 结构化持久化 —— 让一次请求可观测、可回放。

为什么用 contextvars 而非全局变量或逐层传参：
    一次请求会横跨 FastAPI 事件循环、asyncio.to_thread 的线程池、以及检索里的
    ThreadPoolExecutor。contextvars 能跨线程自动透传，无需在每个函数签名里
    加 trace_id 参数（那是逐层传参的噪音，也容易漏传）。

span 树模型（对齐 OpenTelemetry / Langfuse / LangSmith 的 trace=span 树）：
    - span 是「一段代码的计时片段」，`with span("retrieve")` 自动计时；
    - 嵌套的 `with span` 通过 contextvars 维护的「当前 span 栈」自动形成父子树：
      进入时压栈（栈顶即父节点），退出时弹栈；
    - 一次请求的所有 span 汇总成一棵树，`end_trace()` 序列化后挂到响应字段、
      并追加写入 trace.jsonl 供离线回放。
"""
import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

# 每个请求上下文独立的 trace_id（线程/协程安全透传）
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)
# 当前 span 栈：进入 span 压栈、退出弹栈，栈顶即父节点
_span_stack_var: contextvars.ContextVar[List["SpanNode"]] = contextvars.ContextVar(
    "span_stack", default=None
)
# 顶层 span（根节点）集合：栈空时进入的 span 挂到这里
_root_spans_var: contextvars.ContextVar[List["SpanNode"]] = contextvars.ContextVar(
    "root_spans", default=None
)


@dataclass
class SpanNode:
    """一次 span 的结构化记录。start/end 为 perf_counter 单调时钟（秒）。"""

    name: str
    start: float
    end: float = 0.0
    attrs: Dict[str, Any] = field(default_factory=dict)
    children: List["SpanNode"] = field(default_factory=list)

    def elapsed_ms(self) -> int:
        return int((self.end - self.start) * 1000)

    def to_dict(self, base_start: float) -> Dict[str, Any]:
        """序列化为相对时间结构（供前端瀑布图与持久化消费）。"""
        return {
            "name": self.name,
            "start_ms": round((self.start - base_start) * 1000, 2),
            "duration_ms": round((self.end - self.start) * 1000, 2),
            "attrs": self.attrs,
            "children": [c.to_dict(base_start) for c in self.children],
        }


# ---------------------------------------------------------------------------
# trace_id 管理（既有 API 保持不变）
# ---------------------------------------------------------------------------
def new_trace_id() -> str:
    """生成新的 trace_id 并写入当前上下文。"""
    trace_id = uuid.uuid4().hex[:16]
    _trace_id_var.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(trace_id)


def get_trace_id() -> Optional[str]:
    return _trace_id_var.get()


# ---------------------------------------------------------------------------
# span 计时（升级为嵌套树）
# ---------------------------------------------------------------------------
@contextmanager
def span(name: str, **attrs: Any) -> Iterator["SpanNode"]:
    """计时一个执行段，自动挂到父 span 之下，结束打印耗时（含 trace_id）。

    用法：``with span("retrieve", top_k=5): ...``
    attrs 会被带入 span 的结构化记录（如检索的 chunk 数、生成的 token 数）。
    """
    node = SpanNode(name=name, start=time.perf_counter(), attrs=dict(attrs))
    stack = _span_stack_var.get() or []
    if stack:
        stack[-1].children.append(node)
    else:
        roots = _root_spans_var.get() or []
        roots.append(node)
        _root_spans_var.set(roots)
    _span_stack_var.set(stack + [node])
    try:
        yield node
    finally:
        node.end = time.perf_counter()
        _span_stack_var.set(stack)  # 弹栈：恢复到进入前的栈
        from app.utils.logger import logger

        # 注意：logger 的 TraceIdFilter 已自动注入 [trace=xxx] 前缀，这里不再重复拼接
        logger.info("span=%s 耗时=%dms", name, node.elapsed_ms())


# ---------------------------------------------------------------------------
# 请求级 span 树的开始 / 收集 / 持久化
# ---------------------------------------------------------------------------
def begin_trace(trace_id: Optional[str] = None) -> str:
    """开始一次请求的 span 收集：复用已有 trace_id（中间件已注入）或生成新的。

    必须在「执行工作流的那个线程」内调用，与 end_trace 成对出现——
    contextvars 按线程上下文隔离，跨线程（如 asyncio.to_thread 的调用方）
    读不到这里 set 的值。
    """
    trace_id = trace_id or get_trace_id() or uuid.uuid4().hex[:16]
    _trace_id_var.set(trace_id)
    _span_stack_var.set([])
    _root_spans_var.set([])
    return trace_id


def end_trace(persist: bool = True) -> List[Dict[str, Any]]:
    """收集整棵 span 树，返回其序列化结构，并（可选）持久化到 trace.jsonl。

    Returns:
        span 树列表（每个元素是根 span 的 dict，含嵌套 children）。
    """
    roots = _root_spans_var.get() or []
    if not roots:
        return []
    base_start = min(r.start for r in roots)
    tree = [r.to_dict(base_start) for r in roots]
    if persist:
        _persist(tree)
    return tree


def _persist(tree: List[Dict[str, Any]]) -> None:
    """把 span 树追加写入 trace.jsonl（每行一次请求，供离线回放与聚合）。"""
    try:
        from app import config

        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        total_ms = round(max(
            (sp["start_ms"] + sp["duration_ms"] for sp in tree), default=0.0
        ), 2)
        record = {
            "trace_id": get_trace_id(),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_ms": total_ms,
            "spans": tree,
        }
        with open(config.LOG_DIR / "trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        # 可观测性本身绝不能影响主流程：持久化失败只吞掉，不抛出
        pass
