"""切分基线回归护栏（无 git 环境下唯一的自动化回归检测手段）。

设计意图
--------
把**当前线上切分策略的产物**钉死在 `artifacts/baseline_chunks.json`，
逐条比对内容哈希。任何改动只要让切分产物漂移，测试立刻失败：

- **通过了** → 切分产物稳定
- **失败了** → 要么是有意为之（策略切换后要用 `--update` 重生成基线），
               要么是改动引入了静默漂移（去查 `artifacts/backup/<task-id>/` 回退）

基线演进记录
------------
- 改造前：recursive，124 块，跨章率 16.9%（此快照已随切换作废）
- 2026-09-08（T5-1 全量切换）：**structure + 上下文头 + 元数据**，176 块，跨章率 2.3%

为什么「产物漂移」值得单独钉死：切分是 RAG 的地基，块内容变了 →
chunk_id（内容哈希）变了 → 索引里新旧块并存 → 检索结果自相矛盾，
而这类问题在常规单测里完全看不出来。

容忍度说明
----------
- 内容哈希集合：**必须完全一致**（零容忍，这是硬证明）
- 块数 / 块长：允许 ±10（浮点与排序噪声）
- 跨节率等比率：允许 ±2pp（避免单块边界抖动造成误报）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SNAPSHOT_PATH = REPO_ROOT / "artifacts" / "baseline_chunks.json"

for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# 容忍度
# ---------------------------------------------------------------------------
COUNT_TOL = 0          # 块数：零容忍
LEN_MEAN_TOL = 10      # 平均块长：±10 字
RATE_TOL = 0.02        # 比率：±2pp


def _load_snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"基线快照缺失：{SNAPSHOT_PATH}\n"
            f"请先生成：PYTHONPATH=.:scripts .venv/bin/python scripts/baseline_snapshot.py"
        )
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _current_chunks():
    from app.rag.indexer import chunk_documents
    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    return chunk_documents(prepare_documents(load_all_documents()))


# ---------------------------------------------------------------------------
# 护栏
# ---------------------------------------------------------------------------
def test_chunk_count_matches_baseline():
    """块数必须与基线一致。"""
    snap = _load_snapshot()
    chunks = _current_chunks()
    expected = snap["counts"]["chunks"]
    assert abs(len(chunks) - expected) <= COUNT_TOL, (
        f"块数漂移：基线 {expected} → 当前 {len(chunks)}\n"
        f"若为有意的切分策略变更，请重生成基线："
        f"PYTHONPATH=.:scripts .venv/bin/python scripts/baseline_snapshot.py --update"
    )


def test_chunk_content_hashes_identical():
    """每一块的内容哈希集合必须与基线完全一致 —— 这是「零变化」的硬证明。"""
    import hashlib

    snap = _load_snapshot()
    chunks = _current_chunks()

    def sha1(t: str) -> str:
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    baseline = {c["content_sha1"] for c in snap["chunks"]}
    current = {sha1(c.page_content) for c in chunks}

    missing = baseline - current      # 基线的块消失了
    added = current - baseline        # 多出了新块

    assert not missing and not added, (
        f"切分产物内容漂移：消失 {len(missing)} 块 / 新增 {len(added)} 块\n"
        f"  消失样例: {list(missing)[:2]}\n"
        f"  新增样例: {list(added)[:2]}\n"
        f"这是重构引入的行为变化，请回退到 artifacts/backup/ 对应任务目录。"
    )


def test_structural_metrics_within_tolerance():
    """结构指标（跨节率 / 块长 / 归属率）必须在容忍带内。"""
    from chunk_metrics import compute_metrics

    snap = _load_snapshot()
    chunks = _current_chunks()
    base_m = snap["metrics"]
    cur_m = compute_metrics(chunks)

    assert abs(cur_m["len_mean"] - base_m["len_mean"]) <= LEN_MEAN_TOL, (
        f"平均块长漂移：基线 {base_m['len_mean']} → 当前 {cur_m['len_mean']}"
    )

    for key in ("cross_chapter_rate", "cross_section_rate",
                "head_attributed_rate", "pct_in_150_350"):
        assert abs(cur_m[key] - base_m[key]) <= RATE_TOL, (
            f"{key} 漂移超容忍带（±{RATE_TOL:.0%}）："
            f"基线 {base_m[key]:.1%} → 当前 {cur_m[key]:.1%}"
        )


def test_baseline_metrics_record_defect():
    """固化「跨章率」的修复成果 —— 基线已从 recursive 切到 structure（T5-1）。

    两个极易混淆的指标，方向**相反**，不要改错：

    - **跨章率 `cross_chapter_rate`：缺陷指标，目标 0%**。
      一块里揉进两个不同的一级章节 → 语义污染，检索到它必然串味。
      改造前（recursive）实测 16.9%，切换后 2.3%。本测试钉住这个成果。
    - **跨小节合并率 `cross_section_rate`：设计目标，不是缺陷**。
      结构感知切分**刻意**把同章相邻小节贪心合并到目标块长，
      所以它天然偏高（当前 85%）—— 把它当缺陷会得出完全相反的优化方向。

    快照的 `config` 段记录了产出它的策略，护栏失败时先看策略是否被改动。
    """
    snap = _load_snapshot()
    m = snap["metrics"]
    cfg = snap.get("config", {})

    assert cfg.get("chunk_strategy") == "structure", (
        f"基线快照对应策略为 {cfg.get('chunk_strategy')!r}，期望 'structure'。"
        f"若刚做了策略切换/回退，请重生成基线："
        f"PYTHONPATH=.:scripts .venv/bin/python scripts/baseline_snapshot.py --update"
    )

    # T3-2 的修复目标：跨章率 ≤10%（本轮实测 2.3%）
    assert m["cross_chapter_rate"] <= 0.10, (
        f"跨章率超标：{m['cross_chapter_rate']:.1%} > 10%（改造前 16.9%）。"
        f"这通常是结构解析的章级锚点没覆盖新文档格式，去查 "
        f"STRUCTURE_CHAPTER_PATTERNS / STRUCTURE_MD_PATTERNS。"
    )
