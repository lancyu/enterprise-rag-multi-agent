"""零依赖的运行时标志位 —— 只放「初始化过程中产生、配置层又需要读」的值。

为什么会有这个模块
------------------
`app/config.py` 的几个**自适应默认值**（路由语义地板 / 检索阈值 / 软回退门槛）
需要知道「实际生效的 embedding 模式」：本地哈希向量与真实神经向量的余弦尺度
差一个数量级，套同一套阈值必然一边恒过、一边恒不过。而「实际模式」只有
`app/providers/embeddings.py` 把 embedder 真正建起来之后才知道 ——
于是 config 曾经直接去 import 它，形成

    config → utils.embedding → providers.embeddings → config

的**环**。环的代价不是"跑不起来"，而是：**任何想单独 import config 的地方，
都会被拖进整条 provider 初始化链**（包括一次真实的 embedding 健康检查请求）。
配置中心本该是一个可以放心 import 的叶子。

改法：把「当前实际模式」变成一个**由 provider 注入的普通值**。

- 本模块 **不 import 任何 `app.*` 的东西**（有契约守着：`pyproject.toml` 的
  `runtime_flags 必须零依赖`）——这是它存在的全部意义；
- provider 建好 embedder 后调 `set_embedding_mode()` **发布事实**；
- config 只读 `observed_embedding_mode()`，从此不认识 provider。

`None` 的含义
-------------
``None`` = **尚未初始化**，不是"没配"。读方自己决定兜底值 —— 配置层写成
``or "api"`` 是刻意的：真实接口是常规路径，本地哈希是降级路径；而且真实链路上
`get_embeddings()` 都先于阈值读取发生（`retriever.retrieve()` 先取 embedder
再读阈值，`router` 也是先算语义分再读地板），所以兜底只在**从未初始化过
embedding** 的场景生效。
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_embedding_mode: str | None = None


def set_embedding_mode(mode: str) -> None:
    """发布「实际生效的 embedding 模式」（由 provider 在建好 embedder 后调用）。

    只接受 ``"api"`` / ``"local-hash"`` —— 多一种取值就意味着多一个没人处理的
    分支，而阈值自适应是**按这两个值二分**的。写错时直接抛错，比悄悄降级好。
    """
    if mode not in ("api", "local-hash"):
        raise ValueError(f"未知的 embedding 模式：{mode!r}（只认 api / local-hash）")
    global _embedding_mode
    with _lock:
        _embedding_mode = mode


def clear_embedding_mode() -> None:
    """清掉已发布的值（``reset_embeddings()`` 会调它）。

    重置之后「实际模式」确实又是未知的了 —— 让这里如实反映状态，
    而不是留着一个可能已经过期的旧值。
    """
    global _embedding_mode
    with _lock:
        _embedding_mode = None


def observed_embedding_mode() -> str | None:
    """返回已观测到的 embedding 模式；从未初始化过时返回 ``None``。"""
    return _embedding_mode
