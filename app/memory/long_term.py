"""长期记忆 —— 跨会话沉淀的事实、画像与人格。

参考 HKUDS/nanobot 的分层理念，把长期记忆拆成三类文件，因为它们
「遗忘的成本」和「更新的频率」完全不同，混在一个文件里必然互相干扰：

- MEMORY.md：关于业务本身的长期事实与决策（更新最频繁）
- USER.md：关于用户是谁、偏好什么（一旦形成很稳定）
- SOUL.md：助手的人格与沟通风格（几乎不变，全局共享）

为什么用 Markdown 而不是 JSON/数据库：
    长期记忆的体量极小（KB 级），但会被人反复阅读和手工修正。
    Markdown 让人可以直接编辑，也让「模型改了什么」在 diff 里一目了然——
    这一点在调试"助手为什么记住了奇怪的东西"时价值极高。
"""
import re
from typing import Any, Dict, List, Optional

from app import config
from app.memory.store import MemoryStore
from app.utils.logger import logger

# 规则化事实抽取：Dream 不可用（限频 / 无 Key）时的降级路径
_NAME_PATTERNS = [
    re.compile(r"我(?:叫|是|的名字是)\s*([^\s，。,.!！?？]{1,20})"),
    re.compile(r"你可以?叫我\s*([^\s，。,.!！?？]{1,20})"),
]
_PREFERENCE_PATTERNS = [
    re.compile(r"我(?:喜欢|偏好|习惯|一般)\s*([^\n。！？]{2,40})"),
    re.compile(r"请(?:尽量|总是|以后)\s*([^\n。！？]{2,40})"),
]
_FACT_PATTERNS = [
    re.compile(r"我们(?:公司|部门|团队)\s*([^\n。！？]{2,60})"),
    re.compile(r"(?:记住|记一下|请注意)\s*[:：]?\s*([^\n。！？]{2,80})"),
]


class LongTermMemory:
    """长期记忆门面：封装文件读写与上下文注入。"""

    def __init__(self, user_id: str = "default", store: Optional[MemoryStore] = None):
        self.user_id = store.user_id if store else user_id
        self.store = store or MemoryStore(user_id)

    # ------------------------------------------------------------------
    # 上下文注入
    # ------------------------------------------------------------------
    def get_context(self, max_chars: Optional[int] = None) -> str:
        """返回注入 prompt 的长期记忆上下文（已按预算截断）。"""
        if not config.MEMORY_ENABLED:
            return ""
        return self.store.get_memory_context(max_chars=max_chars)

    # ------------------------------------------------------------------
    # 长期事实（MEMORY.md）
    # ------------------------------------------------------------------
    def add_facts(self, facts: List[str]) -> int:
        """追加长期事实，自动跳过已存在的条目。

        Returns:
            实际新增的条目数。
        """
        facts = [f.strip("- ").strip() for f in (facts or []) if f and f.strip("- ").strip()]
        if not facts:
            return 0
        # 用 get_facts() 而非原始文本做去重基准：标题行与格式差异不应影响判重
        existing = self.store.read_memory()
        existing_lines = set(self.get_facts())

        new_facts = [f for f in facts if f not in existing_lines]
        if not new_facts:
            return 0

        header = "" if existing.strip() else "# 长期记忆\n\n"
        block = "\n".join(f"- {f}" for f in new_facts)
        separator = "" if (not existing.strip() or existing.endswith("\n")) else "\n"
        self.store.write_memory(f"{existing}{separator}{header}{block}\n")
        logger.info("长期记忆新增 %d 条事实（user=%s）", len(new_facts), self.user_id)
        return len(new_facts)

    def get_facts(self) -> List[str]:
        """读取长期事实条目（只认 `- ` 开头的条目行，跳过标题与空行）。"""
        facts: List[str] = []
        for line in self.store.read_memory().splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                fact = stripped[2:].strip()
                if fact:
                    facts.append(fact)
        return facts

    def clear_facts(self) -> None:
        self.store.write_memory("# 长期记忆\n")

    # ------------------------------------------------------------------
    # 用户画像（USER.md）
    # ------------------------------------------------------------------
    def get_user_profile(self) -> str:
        return self.store.read_user()

    def add_user_note(self, note: str) -> int:
        """追加一条用户画像信息（去重）。"""
        note = (note or "").strip()
        if not note:
            return 0
        existing = self.store.read_user()
        if note in existing:
            return 0
        header = "" if existing.strip() else "# 用户画像\n\n"
        separator = "" if (not existing.strip() or existing.endswith("\n")) else "\n"
        self.store.write_user(f"{existing}{separator}{header}- {note}\n")
        return 1

    # ------------------------------------------------------------------
    # 归档（history.jsonl）
    # ------------------------------------------------------------------
    def archive(self, summary: str, session_key: Optional[str] = None, last_seq: Optional[int] = None) -> int:
        """写入一条对话归档，并记录本次覆盖到的会话消息序号。"""
        return self.store.append_history(summary, session_key=session_key, last_seq=last_seq)

    def last_archived_seq(self, session_key: Optional[str] = None) -> int:
        """本会话已归档到的消息序号（0 = 从未归档）。"""
        return self.store.get_last_archived_seq(session_key)

    def pending_entries(self) -> List[Dict[str, Any]]:
        """返回尚未被蒸馏消费的归档条目。"""
        return self.store.read_unprocessed_history(self.store.get_last_dream_cursor())

    def mark_dreamed(self, cursor: int) -> None:
        """推进蒸馏游标。"""
        self.store.set_last_dream_cursor(cursor)

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        entries = self.store.read_unprocessed_history(0)
        return entries[-limit:]

    # ------------------------------------------------------------------
    # 降级抽取（无大模型时）
    # ------------------------------------------------------------------
    @staticmethod
    def extract_facts_by_rules(text: str) -> Dict[str, List[str]]:
        """从对话文本中规则化抽取用户名、偏好与事实。

        这是 Dream 的降级路径：没有可用的大模型（或限频严重）时，
        仍能捕获「我叫张三」「我喜欢简洁回答」这类高价值显式陈述。
        规则抽取只处理用户主动说出的显式信息，不做任何推断。
        """
        notes: Dict[str, List[str]] = {"name": [], "preference": [], "fact": []}
        for chunk in re.split(r"[\n。！？]", text or ""):
            chunk = chunk.strip()
            if not chunk:
                continue
            for pattern in _NAME_PATTERNS:
                if (m := pattern.search(chunk)):
                    notes["name"].append(f"用户称呼：{m.group(1)}")
                    break
            for pattern in _PREFERENCE_PATTERNS:
                if (m := pattern.search(chunk)):
                    notes["preference"].append(f"用户偏好：{m.group(1)}")
                    break
            for pattern in _FACT_PATTERNS:
                if (m := pattern.search(chunk)):
                    notes["fact"].append(m.group(1))
                    break
        return notes

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return self.store.get_stats()

    def export(self) -> Dict[str, str]:
        """导出全部长期记忆（供前端查看与手工编辑）。"""
        return {
            "MEMORY.md": self.store.read_memory(),
            "USER.md": self.store.read_user(),
            "SOUL.md": self.store.read_soul(),
        }
