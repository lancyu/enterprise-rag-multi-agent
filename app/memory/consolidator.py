"""记忆整理器（Consolidator）—— 把过期对话压缩成归档。

参考 HKUDS/nanobot 的两阶段记忆：Consolidator 负责「把旧对话变小」，
Dream 负责「把归档变成知识」。这里只做第一阶段。

触发时机由调用方决定（通常在会话消息数超过阈值时），本模块不主动轮询，
以避免在请求链路上引入不可控的延迟。

成本控制的三个考虑（每一条都对应一次真实的模型调用）：
1. 整段对话只做 **一次** 摘要调用，且只在超过阈值时触发；
2. 摘要输入先按字符预算裁剪，避免长对话把 token 打满；
3. 大模型不可用时退化为「规则化原文快照」而非直接丢弃——
   宁可存一份冗长的原文，也不要让这段对话凭空消失。

摘要要求保留什么，是这个模块真正的价值所在：
    寒暄可以丢，但「用户是谁、决定了什么、还有什么没办」必须留下。
    这三类信息是后续 Dream 提炼长期记忆的唯一原料。
"""
import time
from typing import Any, Dict, List, Optional

from app import config
from app.core.prompts import get as get_prompt
from app.memory.long_term import LongTermMemory
from app.utils.logger import logger

SUMMARIZE_PROMPT = get_prompt("summarize")

# 摘要输入的字符预算：留出足够空间给输出，同时避免超长对话打满 token
_SUMMARIZE_INPUT_MAX_CHARS = 4000
# 降级快照的字符上限
_RAW_SNAPSHOT_MAX_CHARS = 2000


class Consolidator:
    """对话整理器。"""

    def __init__(self, long_term: Optional[LongTermMemory] = None, user_id: str = "default"):
        self.long_term = long_term or LongTermMemory(user_id)

    # ------------------------------------------------------------------
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        used = 0
        # 从最新往回取，保证预算内装下的是「最近的对话」
        for msg in reversed(messages):
            role = "用户" if msg.get("role") == "user" else "助手"
            line = f"{role}：{msg.get('content', '')}"
            if used + len(line) > _SUMMARIZE_INPUT_MAX_CHARS and lines:
                break
            lines.append(line)
            used += len(line)
        lines.reverse()
        return "\n".join(lines)

    def _summarize_with_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """调用大模型生成摘要；失败返回 None 由调用方降级。"""
        try:
            from langchain_core.prompts import ChatPromptTemplate

            from app.core.llm_factory import get_chat_model

            prompt = ChatPromptTemplate.from_template(SUMMARIZE_PROMPT)
            raw = (prompt | get_chat_model()).invoke(
                {"conversation": self._format_conversation(messages)}
            ).content
            summary = (raw or "").strip()
            return summary or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("对话摘要生成失败，降级为原文快照：%s", exc)
            return None

    def _raw_snapshot(self, messages: List[Dict[str, str]]) -> str:
        """降级快照：按原文截断，并明确标注这是未压缩内容。"""
        body = self._format_conversation(messages)
        if len(body) > _RAW_SNAPSHOT_MAX_CHARS:
            body = body[:_RAW_SNAPSHOT_MAX_CHARS] + "...[截断]"
        return f"[RAW · 未压缩] {len(messages)} 条消息\n{body}"

    # ------------------------------------------------------------------
    def consolidate(
        self,
        messages: List[Dict[str, str]],
        session_key: Optional[str] = None,
        last_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """把一批待归档消息压缩归档。

        Args:
            last_seq: 本批覆盖到的最新消息序号，写入归档作为下次的起始水位。

        Returns:
            {"cursor": int, "degraded": bool, "chars": int, "last_seq": int}
            cursor 为 -1 表示写入失败。
        """
        if not messages:
            return {"cursor": -1, "degraded": False, "chars": 0, "last_seq": last_seq or 0}

        started = time.perf_counter()
        summary = self._summarize_with_llm(messages) if config.MEMORY_ENABLED else None
        degraded = summary is None
        if degraded:
            summary = self._raw_snapshot(messages)

        cursor = self.long_term.archive(summary, session_key=session_key, last_seq=last_seq)
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(
            "记忆整理完成：%d 条消息 → 归档 %d 字（%s，cursor=%s，水位=%s）",
            len(messages), len(summary), "降级快照" if degraded else "模型摘要", cursor, last_seq,
        )
        return {
            "cursor": cursor, "degraded": degraded, "chars": len(summary),
            "elapsed_ms": elapsed, "last_seq": last_seq or 0,
        }

    def maybe_consolidate(
        self,
        history: List[Dict[str, str]],
        session_key: Optional[str] = None,
        keep_tail: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """按需整理：仅当「**尚未归档的**新增消息」攒够阈值时执行。

        水位取自上一次归档记录（按 session_key 区分），因此历史窗口再怎么
        滑动，同一批内容也只会被归档一次。

        Returns:
            未触发整理时返回 None；否则返回 consolidate 的结果（附 pending 条数）。
        """
        from app.memory.short_term import (
            select_unarchived,
            should_consolidate,
            split_for_consolidation,
        )

        if not config.MEMORY_ENABLED or not config.CONSOLIDATE_ENABLED:
            return None

        last_seq = self.long_term.last_archived_seq(session_key)
        pending = select_unarchived(history, last_seq)
        to_archive, _to_keep = split_for_consolidation(pending, keep_tail=keep_tail)

        # 判据用「本次真正要压缩的条数」而不是积压总量：
        # 积压量里包含尾部暂缓归档的部分，用它做阈值会让实际触发频率
        # 高于直觉（压到阈值时其实只压了一部分，很快又够线）。
        # 用 to_archive 才能让 CONSOLIDATE_THRESHOLD 与
        # 「一次模型调用压缩多少条」严格对应。
        if not should_consolidate(to_archive):
            return None

        new_watermark = max(int(m.get("seq") or 0) for m in to_archive)
        result = self.consolidate(to_archive, session_key=session_key, last_seq=new_watermark)
        result["pending"] = len(pending)
        return result
