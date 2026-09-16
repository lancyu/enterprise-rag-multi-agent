#!/usr/bin/env python3
"""切分改造 T4-2：A/B 影子对比（离线，不碰线上）。

设计原则（按「优化」要求）：
- **O2 零成本结构指标**（默认即跑）：块数 / 平均块长 / 跨章率 / 索引膨胀 / 章节覆盖率 /
  块长分布。全部来自本地切分结果，不调用任何 embedding API，秒级完成。
- **O3 检索评测**（--full 才跑）：真实调用 embedding 跑 run_retrieval_eval，对比
  hit_rate / MRR / section_hit_rate。受账号 RPM=3 限流影响会较慢，分批执行。

实验组：
  A   recursive            （基线，改造前行为）
  D   structure+header+meta（推荐方案）

用法：
  python scripts/chunking_ab.py            # 仅 O2 结构指标
  python scripts/chunking_ab.py --full     # 再跑 O3 检索评测
  python scripts/chunking_ab.py --full --save docs/history/chunking-ab-report.md
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from dataclasses import dataclass, field
from typing import Dict, List

from app import config
from app.rag import indexer
from app.rag.prepare import prepare_documents
from app.utils.doc_loader import load_all_documents

_CHAPTER_RE = re.compile(r"(第[一二三四五六七八九十百]+章|^\s*[一二三四五六七八九十]+、)", re.M)


def _strip_header(text: str) -> str:
    if text.startswith("【章节】"):
        return text.split("\n\n", 1)[1] if "\n\n" in text else text
    return text


@dataclass
class ChunkStat:
    strategy: str
    chunks: int = 0
    mean_len: float = 0.0
    cross_chapter_rate: float = 0.0
    with_heading_path: int = 0
    len_hist: Dict[str, int] = field(default_factory=dict)
    details: dict = field(default_factory=dict)


def _metrics(chunks) -> ChunkStat:
    lens = [len(c.page_content) for c in chunks]
    cross = 0
    with_path = 0
    for c in chunks:
        body = _strip_header(c.page_content)
        if _CHAPTER_RE.search(body):
            if len(set(m.group(0) for m in _CHAPTER_RE.finditer(body))) >= 2:
                cross += 1
        if c.metadata.get("heading_path"):
            with_path += 1
    n = len(chunks)
    hist = {"<80": 0, "80-150": 0, "150-350": 0, ">350": 0}
    for l in lens:
        if l < 80:
            hist["<80"] += 1
        elif l <= 150:
            hist["80-150"] += 1
        elif l <= 350:
            hist["150-350"] += 1
        else:
            hist[">350"] += 1
    return ChunkStat(
        strategy="",
        chunks=n,
        mean_len=round(st.mean(lens), 1) if n else 0.0,
        cross_chapter_rate=round(cross / n, 3) if n else 0.0,
        with_heading_path=with_path,
        len_hist=hist,
    )


def build_with(strategy: str, header: bool, meta: bool) -> ChunkStat:
    """用指定配置建索引并计算 O2 结构指标（不写报告，纯度量）。"""
    indexer.config.CHUNK_STRATEGY = strategy
    indexer.config.CHUNK_CONTEXT_HEADER = header
    indexer.config.CHUNK_ENRICH_METADATA = meta
    prep = prepare_documents(load_all_documents())
    chunks = indexer.chunk_documents(prep)
    stat = _metrics(chunks)
    stat.strategy = f"strategy={strategy} header={header} meta={meta}"
    stat.details = {
        "chunks": stat.chunks,
        "mean_len": stat.mean_len,
        "cross_chapter_rate": stat.cross_chapter_rate,
        "with_heading_path": stat.with_heading_path,
        "len_hist": stat.len_hist,
    }
    return stat


# T4-3 参数敏感性网格（零 API 成本：只算结构指标，不建索引、不调 embedding）
GRID_TARGETS = [180, 220, 280]      # 目标块长
GRID_HARD_MAX = [300, 340, 400]     # 硬上限
# 硬约束（来自清单 T4-3 验收标准）：不满足直接淘汰
MAX_FRAGMENT_PCT = 0.05             # <80 字碎片占比
MAX_EXPANSION = 2.0                 # 相对基线的索引膨胀
MAX_CROSS_CHAPTER = 0.05            # 跨章率


def scan_grid(baseline_count: int = 124, verbose: bool = True) -> List[dict]:
    """目标块长 × 硬上限 网格扫描（**零 API 成本**）。

    为什么扫描与评测必须解耦：全量检索评测要重建索引 + 调 embedding，跑 9 组
    成本很高；而结构指标（块长分布 / 碎片率 / 跨章率 / 膨胀）是纯字符串统计，
    秒级可跑完，足以淘汰明显不合适的参数。只有进入决赛的候选才值得付费用
    检索指标去区分 —— 这是本轮最重要的成本优化（9 次全量 → 1~2 次）。
    """
    indexer.config.CHUNK_STRATEGY = "structure"
    indexer.config.CHUNK_CONTEXT_HEADER = True
    indexer.config.CHUNK_ENRICH_METADATA = True
    saved = (indexer.config.CHUNK_TARGET_CHARS, indexer.config.CHUNK_HARD_MAX_CHARS)
    prep = prepare_documents(load_all_documents())

    rows: List[dict] = []
    try:
        for target in GRID_TARGETS:
            for hard_max in GRID_HARD_MAX:
                if hard_max <= target:      # 硬上限必须大于目标块长，否则恒触发二次切
                    continue
                indexer.config.CHUNK_TARGET_CHARS = target
                indexer.config.CHUNK_HARD_MAX_CHARS = hard_max
                chunks = indexer.chunk_documents(prep)
                lens = [len(c.page_content) for c in chunks]
                n = len(lens) or 1
                in_100_300 = sum(1 for l in lens if 100 <= l <= 300) / n
                lt80 = sum(1 for l in lens if l < 80) / n
                stat = _metrics(chunks)
                rows.append({
                    "target": target,
                    "hard_max": hard_max,
                    "chunks": len(chunks),
                    "mean_len": stat.mean_len,
                    "pct_100_300": round(in_100_300, 3),
                    "pct_lt_80": round(lt80, 3),
                    "cross_chapter": stat.cross_chapter_rate,
                    "expansion": round(len(chunks) / baseline_count, 2) if baseline_count else None,
                })
    finally:
        # 必须还原：网格扫描只是临时改内存配置，不能污染进程内后续步骤
        indexer.config.CHUNK_TARGET_CHARS, indexer.config.CHUNK_HARD_MAX_CHARS = saved

    if verbose:
        print(f"\n{'目标':>6}{'硬上限':>8}{'块数':>7}{'均长':>8}"
              f"{'100-300':>10}{'<80':>8}{'跨章':>8}{'膨胀':>7}  判定")
        print("-" * 74)
        for r in rows:
            ok = (r["pct_lt_80"] <= MAX_FRAGMENT_PCT
                  and (r["expansion"] or 0) <= MAX_EXPANSION
                  and r["cross_chapter"] <= MAX_CROSS_CHAPTER)
            print(f"{r['target']:>6}{r['hard_max']:>8}{r['chunks']:>7}{r['mean_len']:>8}"
                  f"{r['pct_100_300']:>10.1%}{r['pct_lt_80']:>8.1%}"
                  f"{r['cross_chapter']:>8.1%}{str(r['expansion']) + 'x':>7}  "
                  f"{'✅ 候选' if ok else '—'}")

        eligible = [r for r in rows
                    if r["pct_lt_80"] <= MAX_FRAGMENT_PCT
                    and (r["expansion"] or 0) <= MAX_EXPANSION
                    and r["cross_chapter"] <= MAX_CROSS_CHAPTER]
        if eligible:
            best = max(eligible, key=lambda r: r["pct_100_300"])
            print(f"\n推荐（满足硬约束中 100-300 占比最高）："
                  f"目标块长={best['target']} 硬上限={best['hard_max']} "
                  f"→ {best['pct_100_300']:.1%} 落在 100-300 字")
        else:
            print("\n⚠️ 无组合同时满足全部硬约束，需放宽阈值或调整切分算法")
    return rows


def _o3_retrieval_eval(strategy: str, header: bool, meta: bool) -> dict:
    """O3：真实 embedding 检索评测（受 RPM 限流影响，可能较慢）。"""
    from app.rag import evaluator as E
    indexer.config.CHUNK_STRATEGY = strategy
    indexer.config.CHUNK_CONTEXT_HEADER = header
    indexer.config.CHUNK_ENRICH_METADATA = meta
    # 用对应策略重建索引
    indexer.build_index()
    cases = E.load_eval_cases()
    return E.run_retrieval_eval(cases, top_k=config.SIMILARITY_TOP_K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="额外跑 O3 检索评测（需 embedding API）")
    ap.add_argument("--scan-only", action="store_true",
                    help="只跑 T4-3 参数网格扫描（零 API 成本，用于参数定稿）")
    ap.add_argument("--save", default="", help="把对比报告写到指定 md 路径")
    args = ap.parse_args()

    if args.scan_only:
        print("=" * 64)
        print("T4-3 参数敏感性网格扫描（零 API 成本）")
        print("=" * 64)
        scan_grid(baseline_count=124)
        return

    print("=" * 64)
    print("切分改造 A/B 影子对比（O2 零成本结构指标）")
    print("=" * 64)

    # 基线 A
    a = build_with("recursive", False, False)
    # 推荐 D
    d = build_with("structure", True, True)

    print(f"\n{'指标':<22}{'A 基线(recursive)':>20}{'D 结构+头+元数据':>20}")
    print("-" * 64)
    print(f"{'块数':<22}{a.chunks:>20}{d.chunks:>20}")
    print(f"{'平均块长':<22}{a.mean_len:>20}{d.mean_len:>20}")
    print(f"{'跨章率':<22}{a.cross_chapter_rate:>20.0%}{d.cross_chapter_rate:>20.0%}")
    print(f"{'带 heading_path 块':<22}{a.with_heading_path:>20}{d.with_heading_path:>20}")
    infl = round(d.chunks / a.chunks, 2) if a.chunks else 0
    print(f"{'索引膨胀(相对A)':<22}{'1.00x':>20}{str(infl) + 'x':>20}")
    print(f"{'块长分布 <80':<22}{a.len_hist['<80']:>20}{d.len_hist['<80']:>20}")
    print(f"{'块长分布 80-150':<22}{a.len_hist['80-150']:>20}{d.len_hist['80-150']:>20}")
    print(f"{'块长分布 150-350':<22}{a.len_hist['150-350']:>20}{d.len_hist['150-350']:>20}")
    print(f"{'块长分布 >350':<22}{a.len_hist['>350']:>20}{d.len_hist['>350']:>20}")

    o3 = None
    if args.full:
        print("\n" + "=" * 64)
        print("O3 检索评测（真实 embedding，分批跑）")
        print("=" * 64)
        print("[A] 重建索引 + 评测...")
        o3_a = _o3_retrieval_eval("recursive", False, False)
        print(f"    hit_rate={o3_a['hit_rate']} mrr={o3_a['mrr']} section_hit={o3_a.get('section_hit_rate')}")
        print("[D] 重建索引 + 评测...")
        o3_d = _o3_retrieval_eval("structure", True, True)
        print(f"    hit_rate={o3_d['hit_rate']} mrr={o3_d['mrr']} section_hit={o3_d.get('section_hit_rate')}")
        o3 = {"A": o3_a, "D": o3_d}

    scan_rows = None
    if args.save or args.full:
        print("\n" + "=" * 64)
        print("T4-3 参数网格扫描（零 API 成本）")
        print("=" * 64)
        scan_rows = scan_grid(baseline_count=a.chunks or 124)

    if args.save:
        _save_report(args.save, a, d, infl, o3, scan_rows)
        print(f"\n报告已写入：{args.save}")


def _save_report(path: str, a: ChunkStat, d: ChunkStat, infl: float, o3, scan_rows=None):
    lines = [
        "# 切分改造 A/B 影子对比报告",
        "",
        "> 离线影子对比，不碰线上。A=recursive 基线，D=structure+头+元数据。",
        "",
        "## O2 零成本结构指标",
        "",
        f"- 块数：A={a.chunks} → D={d.chunks}（膨胀 {infl}x，≤2.0x 为健康）",
        f"- 平均块长：A={a.mean_len} → D={d.mean_len}",
        f"- **跨章率（核心收益）：A={a.cross_chapter_rate:.0%} → D={d.cross_chapter_rate:.0%}**",
        f"- 带 heading_path 元数据块数：A={a.with_heading_path} → D={d.with_heading_path}",
        f"- 块长分布 D：<80={d.len_hist['<80']} / 80-150={d.len_hist['80-150']} / 150-350={d.len_hist['150-350']} / >350={d.len_hist['>350']}",
        "",
        "## O3 检索评测（需 embedding API）",
        "",
    ]
    if o3:
        lines += [
            f"- A：hit_rate={o3['A']['hit_rate']} | MRR={o3['A']['mrr']} | section_hit={o3['A'].get('section_hit_rate')}",
            f"- D：hit_rate={o3['D']['hit_rate']} | MRR={o3['D']['mrr']} | section_hit={o3['D'].get('section_hit_rate')}",
            "",
            "判定：D 的 section_hit_rate 应 ≥ A（章节命中更准）；hit_rate/MRR 不应低于 A。",
        ]
    else:
        lines += [
            "_未运行（O3 需真实 embedding，受 RPM 限流影响较慢）。_",
            "运行：`python scripts/chunking_ab.py --full --save docs/history/chunking-ab-report.md`",
        ]

    if scan_rows:
        lines += [
            "",
            "## T4-3 参数敏感性网格扫描（零 API 成本）",
            "",
            "扫描**只算结构指标**（块长分布 / 碎片率 / 跨章率 / 膨胀），不建索引、不调 embedding；"
            "只有决赛候选才送去跑付费的全量检索评测。",
            "",
            "| 目标块长 | 硬上限 | 块数 | 平均块长 | 100-300 占比 | <80 占比 | 跨章率 | 膨胀 | 判定 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in scan_rows:
            ok = (r["pct_lt_80"] <= MAX_FRAGMENT_PCT
                  and (r["expansion"] or 0) <= MAX_EXPANSION
                  and r["cross_chapter"] <= MAX_CROSS_CHAPTER)
            lines.append(
                f"| {r['target']} | {r['hard_max']} | {r['chunks']} | {r['mean_len']} "
                f"| {r['pct_100_300']:.1%} | {r['pct_lt_80']:.1%} | {r['cross_chapter']:.1%} "
                f"| {r['expansion']}x | {'✅ 候选' if ok else '—'} |"
            )
        eligible = [r for r in scan_rows
                    if r["pct_lt_80"] <= MAX_FRAGMENT_PCT
                    and (r["expansion"] or 0) <= MAX_EXPANSION
                    and r["cross_chapter"] <= MAX_CROSS_CHAPTER]
        if eligible:
            best = max(eligible, key=lambda r: r["pct_100_300"])
            lines += [
                "",
                f"**推荐**：目标块长 `{best['target']}` + 硬上限 `{best['hard_max']}`"
                f"（满足硬约束的组合中，100-300 字占比最高 {best['pct_100_300']:.1%}）。"
                f"硬约束：<80 字碎片 ≤{MAX_FRAGMENT_PCT:.0%}、"
                f"膨胀 ≤{MAX_EXPANSION}x、跨章率 ≤{MAX_CROSS_CHAPTER:.0%}。",
            ]
        else:
            lines += ["", "⚠️ 无组合同时满足全部硬约束，需放宽阈值或调整切分算法。"]

    from pathlib import Path
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
