"""记忆蒸馏（Dream）—— 把对话归档沉淀为长期知识。

参考 HKUDS/nanobot 的 Dream 阶段，但做了一个关键的工程适配：

    nanobot 让模型「直接编辑整个记忆文件」（外科手术式最小改动），
    这在单用户本地 agent 场景很优雅，但要求模型输出完整文件内容——
    输出长、token 消耗大，且一旦模型理解偏差，可能一次性覆盖掉已有记忆。

    本项目改为「模型只输出新增条目，代码负责合并与去重」：
    - 输出短、token 量小（一次蒸馏的开销约为前者的 1/5）
    - 已有记忆不会被模型覆盖，只做增量追加，失败的最坏情况是"没记住新东西"，
      而不是"把旧记忆改坏了"
    - 去重由代码保证（LongTermMemory.add_facts 内部已处理）

另外提供完整的降级链路：
    大模型不可用 / 限频 → 退回规则化抽取（LongTermMemory.extract_facts_by_rules）→
    仍无结果 → 仅推进游标，不写任何记忆。任何一步失败都不会留下脏记忆。
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

from app import config
from app.core.prompts import get as get_prompt
from app.memory.long_term import LongTermMemory
from app.utils.logger import logger

DREAM_PROMPT = get_prompt("dream")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """容错解析模型输出的 JSON 对象。

    模型常会包裹 ```json 标记或前后加解释文字，这里逐级降级解析。
    """
    if not raw:
        return None
    text = raw.strip()
    # 去掉代码块围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    if match := _JSON_BLOCK.search(text):
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class Dream:
    """记忆蒸馏器。"""

    def __init__(self, long_term: Optional[LongTermMemory] = None, user_id: str = "default"):
        self.long_term = long_term or LongTermMemory(user_id)

    # ------------------------------------------------------------------
    def _extract_with_llm(self, entries: List[Dict[str, Any]]) -> Optional[Dict[str, List[str]]]:
        try:
            from langchain_core.prompts import ChatPromptTemplate

            from app.core.llm_factory import get_chat_model

            history_text = "\n".join(
                f"[{e.get('timestamp', '')}] {str(e.get('content', ''))[:800]}" for e in entries
            )
            prompt = ChatPromptTemplate.from_template(DREAM_PROMPT)
            raw = (prompt | get_chat_model()).invoke(
                {
                    "memory": self.long_term.get_facts() and "\n".join(f"- {f}" for f in self.long_term.get_facts()) or "（空）",
                    "user": self.long_term.get_user_profile() or "（空）",
                    "history": history_text,
                }
            ).content
            parsed = _parse_json_object(raw)
            if not parsed:
                logger.warning("蒸馏输出无法解析为 JSON，降级为规则抽取")
                return None
            return {
                "memory": [str(x).strip() for x in parsed.get("memory", []) if str(x).strip()][:20],
                "user": [str(x).strip() for x in parsed.get("user", []) if str(x).strip()][:20],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("蒸馏调用失败，降级为规则抽取：%s", exc)
            return None

    def _extract_by_rules(self, entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {"memory": [], "user": []}
        for entry in entries:
            notes = LongTermMemory.extract_facts_by_rules(str(entry.get("content", "")))
            result["memory"].extend(notes.get("fact", []))
            result["user"].extend(notes.get("name", []) + notes.get("preference", []))
        return result

    # ------------------------------------------------------------------
    def run(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """执行一次蒸馏。

        Returns:
            {"consumed", "cursor", "added_memory", "added_user", "degraded", "elapsed_ms"}
        """
        started = time.perf_counter()
        size = batch_size or config.DREAM_BATCH_SIZE
        entries = self.long_term.pending_entries()
        if not entries:
            return {
                "consumed": 0, "cursor": self.long_term.store.get_last_dream_cursor(),
                "added_memory": [], "added_user": [], "degraded": False,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "message": "没有待蒸馏的归档",
            }

        batch = entries[:size]
        extracted = self._extract_with_llm(batch) if config.MEMORY_ENABLED else None
        degraded = extracted is None
        if degraded:
            extracted = self._extract_by_rules(batch)

        added_memory = self.long_term.add_facts(extracted.get("memory", []))
        added_user = sum(self.long_term.add_user_note(n) for n in extracted.get("user", []))

        # 无论是否提炼出内容，都推进游标：
        # 否则同一批归档会被反复消费，蒸馏永远卡在同一处。
        last_cursor = max(int(e.get("cursor", 0)) for e in batch)
        self.long_term.mark_dreamed(last_cursor)
        self.long_term.store.compact_history()

        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(
            "记忆蒸馏完成：消费 %d 条归档 → 新增事实 %d 条、画像 %d 条（%s）",
            len(batch), added_memory, added_user, "规则降级" if degraded else "模型蒸馏",
        )
        return {
            "consumed": len(batch),
            "cursor": last_cursor,
            "added_memory": extracted.get("memory", []),
            "added_user": extracted.get("user", []),
            "degraded": degraded,
            "elapsed_ms": elapsed,
        }
