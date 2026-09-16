"""SOUL.md（全局人格设定）写入路径测试。

`SOUL.md` 与 `MEMORY.md` / `USER.md` 的关键区别是**影响面**：后两者按 user_id
隔离，SOUL.md 会被拼进每个人的 `memory_context`。所以这条写入路径必须钉住
三件事，缺一件都会变成隐患：

1. **未开鉴权时不可写** —— 一个整体不设防的服务不该存在"一次改掉全站人格"的入口；
2. **写的是全局文件，不是某个用户的目录** —— 写错位置会静默失效（改了但没生效）；
3. **写入是原子的** —— 半截的人格设定会污染所有人的上下文。
"""
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import config
from app.api.memory import SOUL_MAX_CHARS, SoulRequest, memory_write_soul
from app.memory.store import MemoryStore


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """把全局记忆目录指向临时目录，避免污染仓库内的 memory_store/。"""
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. 鉴权闸门
# ---------------------------------------------------------------------------
def test_write_soul_requires_auth_enabled(memory_dir, monkeypatch) -> None:
    """默认（AUTH_ENABLED=false）一律 403，且不落盘。"""
    monkeypatch.setattr(config, "AUTH_ENABLED", False)

    with pytest.raises(HTTPException) as exc:
        memory_write_soul(SoulRequest(content="新人格"))

    assert exc.value.status_code == 403
    assert "AUTH_ENABLED" in exc.value.detail, "错误信息要告诉调用方怎么才能用"
    assert not (memory_dir / "SOUL.md").exists()


def test_write_soul_succeeds_when_auth_enabled(memory_dir, monkeypatch) -> None:
    """开启鉴权后可用；此时请求已由 main 的中间件校验过 API Key。"""
    monkeypatch.setattr(config, "AUTH_ENABLED", True)

    resp = memory_write_soul(SoulRequest(content="# 人格\n简洁、直接。"))

    assert resp["code"] == 0 and resp["empty"] is False
    assert (memory_dir / "SOUL.md").read_text(encoding="utf-8") == "# 人格\n简洁、直接。"


def test_empty_content_clears_persona(memory_dir, monkeypatch) -> None:
    """"清除人格"是合法状态（memory_context 会省略该段），回复要说清楚。"""
    monkeypatch.setattr(config, "AUTH_ENABLED", True)

    resp = memory_write_soul(SoulRequest(content=""))

    assert resp["empty"] is True and "清空" in resp["message"]
    assert (memory_dir / "SOUL.md").read_text(encoding="utf-8") == ""


def test_content_length_is_capped() -> None:
    """超长内容在模型校验阶段就被挡住（防误操作写坏全站人格）。"""
    with pytest.raises(ValidationError):
        SoulRequest(content="长" * (SOUL_MAX_CHARS + 1))


# ---------------------------------------------------------------------------
# 2. 写的是全局文件（而非某个用户目录）
# ---------------------------------------------------------------------------
def test_soul_is_global_and_shared_across_users(memory_dir, monkeypatch) -> None:
    """SOUL.md 必须落在记忆根目录：所有用户读到的都是同一份。"""
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    memory_write_soul(SoulRequest(content="# 统一人格"))

    assert (memory_dir / "SOUL.md").exists(), "SOUL.md 应在记忆根目录，而不是 users/ 下"
    assert not list((memory_dir / "users").glob("*/SOUL.md")), "SOUL.md 不应按用户隔离"

    # 两个不同用户读到同一份人格
    assert MemoryStore("alice").read_soul() == "# 统一人格"
    assert MemoryStore("bob").read_soul() == "# 统一人格"


def test_soul_write_does_not_touch_user_files(memory_dir, monkeypatch) -> None:
    """改全站人格不得顺带改写任何用户的长期记忆（影响面要精确）。"""
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    store = MemoryStore("alice")
    store.write_memory("# 记忆\nalice 的私有事实")
    store.write_user("# 用户画像\nalice")

    memory_write_soul(SoulRequest(content="# 新人格"))

    assert MemoryStore("alice").read_memory() == "# 记忆\nalice 的私有事实"
    assert MemoryStore("alice").read_user() == "# 用户画像\nalice"


# ---------------------------------------------------------------------------
# 3. 原子写入
# ---------------------------------------------------------------------------
def test_write_is_atomic_and_leaves_no_temp_file(memory_dir) -> None:
    """临时文件必须被 os.replace 吃掉，不能残留在记忆目录里。"""
    store = MemoryStore("alice")
    store.write_soul("# 人格")

    leftovers = [p.name for p in memory_dir.rglob("*.tmp")]
    assert leftovers == [], f"原子写入残留了临时文件：{leftovers}"


def test_all_memory_writes_are_atomic(memory_dir) -> None:
    """`write_memory` / `write_user` 同样走原子路径（统一在一个原语上收敛）。"""
    store = MemoryStore("alice")
    store.write_memory("# 记忆")
    store.write_user("# 画像")

    assert [p.name for p in memory_dir.rglob("*.tmp")] == []


def test_rewrite_replaces_whole_content(memory_dir) -> None:
    """覆盖写：新的短内容不得残留旧内容的长尾（这正是非原子写最典型的症状）。"""
    store = MemoryStore("alice")
    store.write_soul("旧人格" * 100)
    store.write_soul("新人格")

    assert store.read_soul() == "新人格"
