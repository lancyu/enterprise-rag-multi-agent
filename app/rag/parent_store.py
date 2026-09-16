"""父子双层索引 · 父块旁路存储（T6-1）。

为什么父块不进向量库
--------------------
父子双层（small-to-big）的两种实现：

- **双索引**：父块、子块各自向量化建索引。检索时先查子块再查父块，
  向量数量翻倍，embedding 成本也翻倍，且父块粒度粗、语义模糊，
  作为检索单元反而会拉低 Top-K 精度。
- **旁路存储**（本实现）：只让**子块**进向量库（粒度细、匹配准），
  父块存在这个轻量 store 里，命中子块后按 `parent_id` 回捞完整上下文。

后者向量数量零增长、embedding 成本零增长，收益（生成端上下文更完整）一样拿到，
因此选它。代价是父块内容不参与检索匹配 —— 这本来也不是它的职责。

存储为什么用 JSON
-----------------
父块只在「生成端组装上下文」时被按 id 回捞一次，没有检索/相似度需求，
也不要求事务。JSON + 内存字典足够，无需引入新的存储依赖。

oom/失败策略：本模块**永不抛错打断主链路** —— 回捞失败就当没有父块，
退回子块内容（与未启用该特性时完全一致）。
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from app.utils.logger import logger

_store: Dict[str, Dict[str, Any]] = {}
_lock = Lock()
_loaded = False


def _path() -> Path:
    from app import config

    return Path(getattr(config, "PARENT_STORE_PATH", "vector_store/parents.json"))


def load() -> int:
    """从磁盘加载父块（幂等，只加载一次）。"""
    global _loaded
    with _lock:
        if _loaded:
            return len(_store)
        path = _path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _store.update(data)
                    logger.info("父块存储已加载：%d 条（%s）", len(_store), path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("父块存储加载失败，按空库继续：%s", exc)
        _loaded = True
        return len(_store)


def save() -> None:
    """持久化到磁盘。"""
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            path.write_text(
                json.dumps(_store, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("父块存储写入失败（不影响主链路）：%s", exc)


def add_many(items: List[Dict[str, Any]]) -> int:
    """批量写入父块。每项至少含 parent_id / content。"""
    if not items:
        return 0
    with _lock:
        for it in items:
            pid = it.get("parent_id")
            if pid:
                _store[pid] = it
    return len(items)


def get(parent_id: str) -> Optional[Dict[str, Any]]:
    return _store.get(parent_id)


def get_content(parent_id: str) -> str:
    """按 id 取父块正文；缺失返回空串（调用方据此退回子块）。"""
    load()
    item = _store.get(parent_id)
    return str(item.get("content", "")) if item else ""


def clear() -> None:
    """清空（全量重建索引时调用，与向量库 store.clear() 对齐）。"""
    with _lock:
        _store.clear()
    path = _path()
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("父块存储清理失败：%s", exc)


def remove_by_source(source: str) -> int:
    """删除指定来源文档的全部父块（增量入库/删除文档时与向量库对齐）。"""
    name = Path(source).name
    with _lock:
        dead = [pid for pid, it in _store.items()
                if Path(str(it.get("source", ""))).name == name]
        for pid in dead:
            _store.pop(pid, None)
    if dead:
        save()
    return len(dead)


def count() -> int:
    load()
    return len(_store)


def stats() -> Dict[str, Any]:
    load()
    if not _store:
        return {"parents": 0, "mean_chars": 0.0, "mean_children": 0.0}
    lens = [len(str(it.get("content", ""))) for it in _store.values()]
    kids = [int(it.get("child_count", 0)) for it in _store.values()]
    return {
        "parents": len(_store),
        "mean_chars": round(sum(lens) / len(lens), 1),
        "mean_children": round(sum(kids) / len(kids), 2),
    }
