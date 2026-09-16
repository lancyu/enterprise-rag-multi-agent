"""生成切分基线快照（改造前的冻结基准）。

用途
----
1. **回归护栏的数据源**：`tests/test_chunking_baseline.py` 读取本快照，
   任何切分重构只要改变了产物，测试立刻失败 —— 这是「行为零变化」的硬证明。
2. **A/B 的对照组**：借鉴 LlamaIndex 冻结 `llama-index-legacy` 的做法，
   把改造前产物钉死，任何时刻都能切回去对照。
3. **无 git 环境下的唯一自动化回归检测手段**（见 T0-1）。

用法
----
    # 生成/更新快照
    PYTHONPATH=. .venv/bin/python scripts/baseline_snapshot.py
    PYTHONPATH=. .venv/bin/python scripts/baseline_snapshot.py --update   # 确认后覆盖

默认**拒绝覆盖**已存在的快照，避免误把「改坏之后」的产物当成基线。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SNAPSHOT_PATH = REPO_ROOT / "artifacts" / "baseline_chunks.json"

# 快照格式版本：结构变更时递增，测试据此校验兼容性
SNAPSHOT_VERSION = 1


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_snapshot() -> Dict[str, Any]:
    """跑完整 load → prepare → chunk，产出快照字典。"""
    from app.rag.indexer import chunk_documents, _chunk_id
    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    from chunk_metrics import compute_metrics  # 同目录脚本模块

    raw = load_all_documents()
    prepared = prepare_documents(raw)
    chunks = chunk_documents(prepared)

    items: List[Dict[str, Any]] = []
    for c in chunks:
        content = c.page_content
        source = c.metadata.get("source", "unknown")
        items.append({
            "source": Path(source).name if source else "unknown",
            "chunk_index": c.metadata.get("chunk_index"),
            "chunk_chars": len(content),
            "content_sha1": _sha1(content),
            "chunk_id": _chunk_id(source, content),
        })

    # 把「产出这份快照的切分配置」一并冻结：
    # 否则快照只有块数，事后无法判断它对应哪种策略，护栏失败时排查成本极高。
    from app import config as app_config

    return {
        "version": SNAPSHOT_VERSION,
        "config": {
            "chunk_size": int(os.getenv("CHUNK_SIZE", "300")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "60")),
            "chunk_strategy": app_config.CHUNK_STRATEGY,
            "context_header": bool(app_config.CHUNK_CONTEXT_HEADER),
            "enrich_metadata": bool(app_config.CHUNK_ENRICH_METADATA),
            "target_chars": int(app_config.CHUNK_TARGET_CHARS),
            "hard_max_chars": int(app_config.CHUNK_HARD_MAX_CHARS),
            "min_chars": int(app_config.CHUNK_MIN_CHARS),
        },
        "counts": {
            "documents": len(raw),
            "prepared": len(prepared),
            "chunks": len(chunks),
        },
        "metrics": compute_metrics(chunks),
        "chunks": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成切分基线快照")
    parser.add_argument(
        "--update", action="store_true",
        help="允许覆盖已存在的快照（缺陷修复后确认更新时使用）",
    )
    parser.add_argument(
        "--out", default=str(SNAPSHOT_PATH), help="快照输出路径",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.update:
        print(
            f"[snapshot] 快照已存在，拒绝覆盖：{out_path}\n"
            f"[snapshot] 确认要更新基线请加 --update（务必确认当前产物是正确的）",
            file=sys.stderr,
        )
        return 1

    snap = build_snapshot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    m = snap["metrics"]
    cfg = snap["config"]
    print(f"[snapshot] 已写入：{out_path}")
    print(f"[snapshot] 策略：{cfg['chunk_strategy']} "
          f"(header={cfg['context_header']}, meta={cfg['enrich_metadata']})")
    print(f"[snapshot] 文档 {snap['counts']['documents']} 篇 "
          f"→ 片段 {snap['counts']['chunks']} 条")
    print(f"[snapshot] 跨章率 {m['cross_chapter_rate']:.1%} | "
          f"跨小节合并率 {m['cross_section_rate']:.1%} | "
          f"块首归属率 {m['head_attributed_rate']:.1%} | "
          f"块长 mean {m['len_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
