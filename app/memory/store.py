"""记忆存储层 —— 纯文件 I/O。

参考 HKUDS/nanobot（47k stars）的记忆设计，按企业服务场景做了三点适配：

1. **按 user_id 隔离**：nanobot 是单用户本地 agent，记忆文件直接放工作区根目录；
   本项目是多会话企业服务，长期记忆必须按用户隔离，否则 A 用户的偏好会污染
   B 用户的画像。人格文件 SOUL.md 是全局的，不随用户变化。

2. **history.jsonl 用游标而非全量重写**：追加写 + 单调递增 cursor，
   让"哪些归档还没被蒸馏"这件事可以被可靠地记住。用单个 Markdown 文件
   记录历史会随对话增长无限膨胀，且无法增量消费。

3. **零外部依赖**：不引入数据库。记忆文件天然小（KB 级）、schema 会演进，
   用 Markdown + JSONL 既能直接被人读懂和编辑，也便于 git 追踪变更。

文件布局：
    memory_store/
    ├── SOUL.md                    # 全局助手人格与沟通风格
    └── users/{user_id}/
        ├── USER.md                # 用户画像（身份、偏好）
        ├── MEMORY.md              # 长期事实与决策
        ├── history.jsonl          # 追加写入的对话归档（cursor 递增）
        ├── .cursor                # 归档写入游标
        └── .dream_cursor          # 蒸馏消费游标
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from app import config
from app.utils.logger import logger

# 单条归档的应急上限：正常情况下 Consolidator 会先截断，
# 这个 cap 只用于兜住「模型把输入原样回显成摘要」这类异常输出。
_HISTORY_ENTRY_HARD_CAP = 8000

_DEFAULT_SOUL = """\
# SOUL.md - 助手人格

你是企业内部智能助手，服务本公司员工。

## 沟通风格
- 专业、简洁、有礼貌，直接给结论再给依据
- 涉及制度条款时明确说明来源文档
- 不确定时如实说明，不编造
"""


def _safe_user_id(user_id: str) -> str:
    """把 user_id 规整为安全的目录名，防止路径穿越。"""
    cleaned = "".join(ch for ch in (user_id or "").strip() if ch.isalnum() or ch in "-_")
    return cleaned or "default"


class MemoryStore:
    """记忆文件的读写封装：只负责持久化，不含任何业务判断。"""

    def __init__(self, user_id: str = "default", base_dir: Optional[Path] = None):
        base = Path(base_dir) if base_dir else config.MEMORY_DIR
        self.user_id = _safe_user_id(user_id)
        self.base_dir = base
        self.soul_file = base / "SOUL.md"

        self.memory_dir = base / "users" / self.user_id
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.user_file = self.memory_dir / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"

        # cursor 分配与追加写必须原子，否则并发会话会读出同一个 cursor 产生重复
        self._append_lock = threading.Lock()
        self._ensure_files()

    # -- 初始化 ----------------------------------------------------------
    def _ensure_files(self) -> None:
        try:
            if not self.soul_file.exists():
                self.soul_file.parent.mkdir(parents=True, exist_ok=True)
                self.soul_file.write_text(_DEFAULT_SOUL, encoding="utf-8")
            for path in (self.memory_file, self.history_file, self.user_file):
                if not path.exists():
                    path.touch()
        except OSError:
            logger.exception("记忆文件初始化失败：%s", self.memory_dir)

    # -- 通用读写 --------------------------------------------------------
    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError:
            logger.exception("记忆文件读取失败：%s", path)
            return ""

    @staticmethod
    def _write_file(path: Path, content: str) -> None:
        """原子写入：先写同目录临时文件，fsync 后 os.replace 覆盖。

        记忆文件会被并发请求读写。直接 ``write_text`` 一旦中途失败（磁盘满、
        进程被杀）就会留下**半截内容**——对 SOUL.md 这种全局共享文件尤其致命：
        残缺的人格设定会污染所有人的 ``memory_context``。``os.replace`` 在同一
        文件系统上是原子的，读取方要么看到旧内容、要么看到新内容，没有中间态。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # -- 三类长期记忆文件 -------------------------------------------------
    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self._write_file(self.memory_file, content)

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self._write_file(self.user_file, content)

    def read_soul(self) -> str:
        """读取 SOUL.md（全局人格设定）。"""
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        """覆盖写入 SOUL.md（全局人格设定）。

        与 ``write_memory`` / ``write_user`` 对称，但**影响面完全不同**：
        后两者按 user_id 隔离，SOUL.md 是全局共享的——它会被拼进**每个用户**
        的 ``memory_context``，写坏一次全站受影响。

        因此本方法的唯一调用方是 ``PUT /memory/soul``（要求 ``AUTH_ENABLED=true``，
        见 ``app/api/memory.py``）。它解决的是"人格设定需要可运维"（改一次全站
        生效，手工进容器改文件不现实），而不是"允许对话过程自动改写人格"——
        不要让生成链路或记忆蒸馏流程调用它。
        """
        self._write_file(self.soul_file, content)

    # -- history.jsonl：追加写 + 游标 -------------------------------------
    def _read_entries(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        entries: List[Dict[str, Any]] = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 单条损坏不应让整个记忆系统不可用
                        logger.warning("归档条目解析失败，已跳过：%s", line[:80])
        except OSError:
            logger.exception("归档文件读取失败：%s", self.history_file)
        return entries

    @staticmethod
    def _valid_nonneg_int(value: Any) -> Optional[int]:
        """只接受非负 int；bool 是 int 的子类，必须显式排除。"""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def _iter_valid_entries(self) -> Iterator[Tuple[Dict[str, Any], int]]:
        for entry in self._read_entries():
            cursor = self._valid_nonneg_int(entry.get("cursor"))
            if cursor is None:
                continue
            if not isinstance(entry.get("timestamp"), str) or not isinstance(entry.get("content"), str):
                continue
            yield entry, cursor

    def _next_cursor(self) -> int:
        last_cursor = 0
        for _entry, cursor in self._iter_valid_entries():
            last_cursor = max(last_cursor, cursor)
        # 游标文件与文件尾部取最大值：即使外部程序写坏了单调性也能自愈
        try:
            if self._cursor_file.exists():
                persisted = int(self._cursor_file.read_text(encoding="utf-8").strip())
                last_cursor = max(last_cursor, persisted)
        except (ValueError, OSError):
            pass
        return last_cursor + 1

    def append_history(
        self,
        entry: str,
        session_key: Optional[str] = None,
        max_chars: Optional[int] = None,
        last_seq: Optional[int] = None,
    ) -> int:
        """追加一条归档，返回其游标。

        Args:
            last_seq: 本条归档覆盖到的会话消息序号。下次整理据此只处理更新的消息，
                避免历史窗口滑动时反复归档同一批内容。
        """
        limit = max_chars or _HISTORY_ENTRY_HARD_CAP
        content = (entry or "").rstrip()
        if len(content) > limit:
            logger.warning("归档条目超过 %d 字符（%d），已截断", limit, len(content))
            content = content[:limit]

        with self._append_lock:
            cursor = self._next_cursor()
            record: Dict[str, Any] = {
                "cursor": cursor,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "content": content,
            }
            if session_key:
                record["session_key"] = session_key
            if last_seq is not None:
                record["last_seq"] = int(last_seq)
            try:
                with open(self.history_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._write_file(self._cursor_file, str(cursor))
            except OSError:
                logger.exception("归档写入失败")
                return -1
        return cursor

    def get_last_archived_seq(self, session_key: Optional[str] = None) -> int:
        """读取某会话最近一次归档覆盖到的消息序号（0 表示从未归档）。

        按 session_key 过滤：同一个用户可能有多个会话共用一份 history.jsonl，
        不区分会话会让水位互相污染（A 会话归档后 B 会话误以为自己也归档过）。
        """
        watermark = 0
        for entry in self._read_entries():
            if session_key is not None and entry.get("session_key") != session_key:
                continue
            seq = self._valid_nonneg_int(entry.get("last_seq"))
            if seq is not None:
                watermark = max(watermark, seq)
        return watermark

    def read_unprocessed_history(self, since_cursor: int) -> List[Dict[str, Any]]:
        """读取游标之后的全部归档条目（供 Dream 消费）。"""
        return [entry for entry, cursor in self._iter_valid_entries() if cursor > since_cursor]

    # -- 蒸馏游标 --------------------------------------------------------
    def get_last_dream_cursor(self) -> int:
        try:
            if self._dream_cursor_file.exists():
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip() or 0)
        except (ValueError, OSError):
            pass
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._write_file(self._dream_cursor_file, str(int(cursor)))

    # -- 维护 ------------------------------------------------------------
    def compact_history(self) -> int:
        """丢弃最老的「已被蒸馏」条目，未消费的条目必须保留。"""
        max_entries = config.MAX_HISTORY_ENTRIES
        entries = self._read_entries()
        if len(entries) <= max_entries:
            return 0
        dream_cursor = self.get_last_dream_cursor()
        keep: List[Dict[str, Any]] = []
        dropped = 0
        for entry in entries:
            cursor = self._valid_nonneg_int(entry.get("cursor"))
            if len(entries) - dropped > max_entries and cursor is not None and cursor <= dream_cursor:
                dropped += 1
                continue
            keep.append(entry)
        if dropped:
            try:
                with open(self.history_file, "w", encoding="utf-8") as f:
                    for entry in keep:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                logger.info("归档压缩完成：丢弃 %d 条已蒸馏条目", dropped)
            except OSError:
                logger.exception("归档压缩失败")
                return 0
        return dropped

    # -- 上下文注入 -------------------------------------------------------
    def get_memory_context(self, max_chars: Optional[int] = None) -> str:
        """组装注入 prompt 的长期记忆上下文。

        只注入非空部分，避免把空标题也塞进 prompt 浪费 token。
        """
        limit = max_chars or config.LONG_TERM_MAX_CHARS
        sections: List[str] = []

        soul = self.read_soul().strip()
        user = self.read_user().strip()
        memory = self.read_memory().strip()

        if memory:
            sections.append(f"## 长期记忆\n{memory}")
        if user:
            sections.append(f"## 用户画像\n{user}")
        if soul:
            sections.append(f"## 沟通风格\n{soul}")

        context = "\n\n".join(sections)
        if len(context) > limit:
            context = context[:limit] + "\n...[记忆已截断]"
        return context

    # -- 观测 -------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        entries = self._read_entries()
        return {
            "user_id": self.user_id,
            "history_entries": len(entries),
            "dream_cursor": self.get_last_dream_cursor(),
            "pending_entries": len(self.read_unprocessed_history(self.get_last_dream_cursor())),
            "memory_chars": len(self.read_memory()),
            "user_chars": len(self.read_user()),
        }
