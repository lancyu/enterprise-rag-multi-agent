"""短期记忆 —— 当前会话的活跃对话窗口。

职责：决定「这一轮请求要把多少历史对话带进 prompt」。

三个约束必须同时满足，否则会出真实事故：
1. **轮数上限**——防止长对话把 prompt 撑爆，tokens 直接翻倍计费；
2. **字符预算**——轮数相同但每条很长时仍会超限，故必须有第二道防线；
3. **问答成对**——裁剪时不能把「问」留下而把「答」丢掉，否则模型会看到
   一个没有回答的提问，进而误以为自己还没回答过，产生重复答复。

为什么不用「保留最近 N 条消息」这种朴素写法：
    朴素写法会把一问一答切散。当窗口边界恰好落在问答中间时，
    模型收到的最后一条是「用户：年假有多少天」，而对应的回答被裁掉了，
    它会重新回答一遍已经说过的内容——这是多轮对话最常见的退化现象。
"""
from typing import Dict, List, Optional

from app import config


def build_window(
    history: List[Dict[str, str]],
    max_turns: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    """按「轮数 + 字符预算 + 问答成对」三重约束裁剪历史。

    Args:
        history: 完整历史，按时间正序，元素形如 {"role": "user"|"assistant", "content": "..."}
        max_turns: 保留的最近轮数（一问一答算一轮）
        max_chars: 字符预算上限

    Returns:
        裁剪后的历史（时间正序）。
    """
    if not history:
        return []

    max_turns = max_turns if max_turns is not None else config.SHORT_TERM_WINDOW
    max_chars = max_chars if max_chars is not None else config.SHORT_TERM_MAX_CHARS

    # 1) 轮数裁剪：从尾部取最近的 max_turns * 2 条
    window = list(history[-(max_turns * 2):]) if max_turns > 0 else list(history)

    # 2) 字符预算：从新到旧累加，找到能完整放下的最旧一条
    budget = max_chars if max_chars > 0 else float("inf")
    kept: List[Dict[str, str]] = []
    used = 0
    for msg in reversed(window):
        size = len(msg.get("content", "") or "")
        if used + size > budget and kept:
            break
        kept.append(msg)
        used += size
    kept.reverse()

    # 3) 问答成对：若窗口以 assistant 开头，说明配对的 user 被裁掉了，丢弃这条孤儿回复
    if kept and kept[0].get("role") == "assistant":
        kept = kept[1:]

    return kept


def format_history(history: List[Dict[str, str]]) -> str:
    """把**已裁剪**的历史格式化成 prompt 片段。

    **只格式化，不裁剪。** 裁剪点必须唯一（见 :func:`build_window`）：本函数若
    二次裁剪，上游按 ``SHORT_TERM_WINDOW`` 裁好的窗口会被再砍一刀，该配置随即
    静默失效——历史上真的发生过一次（原实现硬编码 ``history[-6:]``，把 6 轮
    12 条砍成 3 轮），现由
    ``tests/test_memory_pipeline.py::test_format_history_does_not_re_truncate`` 守住。

    生成层 ``app.rag.generator._format_history`` 委托本函数，保证格式化实现只有一份。
    """
    if not history:
        return "（无历史对话）"
    return "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
        for m in history
    )


def select_unarchived(
    history: List[Dict[str, str]],
    last_archived_seq: int = 0,
) -> List[Dict[str, str]]:
    """筛出尚未归档的消息（seq 大于归档水位）。

    为什么用序号而不是列表长度判断进度：
        history 会被 ltrim 封顶，会话饱和后**长度恒为常数**，
        任何基于长度的判据都会退化成「每一轮都触发」——既每轮多花一次
        模型调用，又把高度重叠的内容反复写进归档。seq 是唯一能跨越
        窗口滑动的稳定锚点。

    客户端直传的历史（请求体里的 chat_history）没有 seq，按 0 处理因而
    不参与归档——本就不该由服务端替客户端带来的历史做归档。
    """
    return [m for m in history if int(m.get("seq") or 0) > last_archived_seq]


def should_consolidate(to_archive: List[Dict[str, str]], threshold: Optional[int] = None) -> bool:
    """判断本次可归档的消息是否够一整批。

    语义是「够不够一次模型调用的本钱」，因此比的是**真正会被压缩的条数**，
    而不是待归档的积压总量——后者含尾部暂缓部分，会让阈值与
    实际压缩量对不上，导致触发比预期频繁。
    """
    limit = threshold if threshold is not None else config.CONSOLIDATE_THRESHOLD
    return limit > 0 and len(to_archive) >= limit


def split_for_consolidation(
    pending: List[Dict[str, str]],
    keep_tail: Optional[int] = None,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """把待归档消息切分为「归档」与「留在窗口里」两部分。

    尾部 keep_tail 条不归档：刚说完的话立刻被压成摘要，会让下一轮追问
    只能拿摘要当上下文，细节丢失反而伤体验。

    Returns:
        (to_archive, to_keep)
    """
    keep = keep_tail if keep_tail is not None else config.CONSOLIDATE_KEEP_TAIL
    if len(pending) <= keep:
        return [], list(pending)
    return list(pending[:-keep]), list(pending[-keep:])


class ShortTermMemory:
    """短期记忆视图 —— 与 ``LongTermMemory`` 对称的领域对象。

    **为什么 history 由构造参数注入，而不是自己加载**

    对话历史在 API 层就已经取出来了（``chat.py`` → ``await get_history(session_id)``
    → ``GraphState.chat_history``），一路传到生成节点才需要裁剪。若本对象再去读一次
    Redis，等于每轮多一次往返。注入式让对象语义上「持有消息」（对齐 nanobot
    ``Session`` 自带 messages 的设计），又不重复 I/O —— 这是 OO 载体与 LangGraph
    函数式数据流之间的折中点。

    一次构造、多处消费：同一轮里 ``window`` / ``format`` / ``stats`` 都基于同一份
    history，避免各算各的导致窗口口径不一致。
    """

    def __init__(
        self,
        session_id: str,
        history: List[Dict[str, str]],
        max_turns: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.history = list(history or [])
        self.max_turns = max_turns
        self.max_chars = max_chars

    @property
    def window(self) -> List[Dict[str, str]]:
        """裁剪后的窗口（三重约束：轮数 + 字符预算 + 问答成对）。"""
        return build_window(self.history, max_turns=self.max_turns, max_chars=self.max_chars)

    # 刻意不提供 format()：
    #   生产链路严格分两步——先在上游裁剪（本对象的 window），再由
    #   ``generator._format_history`` 只做格式化。本对象若提供"裁剪 + 格式化"
    #   一步到位的方法，调用方会以为拿到了窗口文本，实际却触发了第二次裁剪，
    #   让 SHORT_TERM_WINDOW 静默失效。这正是
    #   ``tests/test_memory_pipeline.py::test_format_history_does_not_re_truncate``
    #   守住的坑——**裁剪点必须唯一**。

    def stats(self) -> Dict[str, int]:
        """窗口统计：基于同一次裁剪，供观测与前端展示。"""
        window = self.window
        return {
            "session_id": self.session_id,
            "total_messages": len(self.history),
            "window_messages": len(window),
            "window_chars": sum(len(m.get("content", "")) for m in window),
            "dropped": len(self.history) - len(window),
        }
