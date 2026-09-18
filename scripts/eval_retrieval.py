"""检索侧离线评测 —— 补上「只有命中率」时看不到的那半边。

为什么还需要一个脚本
--------------------
``app/rag/evaluator.run_retrieval_eval`` 已经能给出 ``hit_rate`` 与 ``mrr``。
这两个指标回答的是同一件事：**Top-K 里有没有正确答案、它排第几**。
它们回答不了另外两件事：

    Recall@K  正确答案一共有 5 条，你召回了 1 条 —— 命中率 1.0，但你漏了 4 条
    NDCG@K    正确答案排在第 5 位 —— 命中率同样是 1.0，但用户（和 rerank）看到的是第 5 位

命中率是**天花板效应**最严重的指标：25 条用例里有 22 条命中，改动之后还是 22 条，
A/B 就看不出差别（这个坑在扩充用例时踩过一次：4 条用例时两组都是 1.000）。
Recall / NDCG 是**排序敏感**的，rerank 的收益只能靠它们看出来 ——
这也是本条排在 P1-3（rerank）之前做的原因。

指标怎么算（以及为什么这么算）
----------------------------
相关性判据只有一个：**关键词命中 + 来源匹配**（与 ``run_retrieval_eval`` 现有的
判定口径一致）。它不是人工标注的黄金集，是**粗粒度代理** —— 本项目没有标注数据，
与其假装有，不如把代理口径写成唯一一处、并且让分子分母共用它。

- ``n_gold``：用**同一个判据**扫全量语料得到的「相关片段总数」。
  这是 Recall 的分母。分子分母不共用判据，Recall 就没有意义
  （这正是本脚本最容易写错、也最不容易被发现的地方）。
- ``recall@K = |Top-K 中相关片段| / n_gold``
- ``NDCG@K``：二元增益，``DCG = Σ rel_i / log2(i+2)``；
  ``IDCG`` 取「前 min(n_gold, K) 位全是相关」的理想排列。
  二元相关性下 NDCG 退化为「相关片段是不是都排在前面」，正是 rerank 要优化的东西。
- ``n_gold == 0`` 的用例是**用例本身的缺陷**（关键词在语料里根本不存在），
  单独列出来并**排除出均值** —— 让它静默变成 recall=0 会把均值拖下去却看不出为什么。

用法
----
    PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py              # 跑一遍并和基线 diff
    PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py --worst 8    # 列出最差的 8 条
    PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py --update     # 确认后刷新基线

会真实调用 embedding 接口（每条用例一次）。语料扫描（算 n_gold）是**纯离线**的，
不发任何请求。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BASELINE_PATH = REPO_ROOT / "artifacts" / "retrieval_baseline.json"
BASELINE_VERSION = 1


# ---------------------------------------------------------------------------
# 相关性判据 —— 唯一一处
# ---------------------------------------------------------------------------
def is_relevant(content: str, source: str, keywords: Sequence[str],
                expect_source: Optional[str]) -> bool:
    """判定一个片段对某条用例是否相关。

    ⚠️ 这是**唯一**的相关性判据：Top-K 结果与全量 gold 集都必须走它。
    任何一处自己再写一遍「关键词 in content」，Recall 的分子分母就不再是同一个东西，
    而指标看上去仍然正常 —— 这类错误没有任何报错会提醒你。
    """
    if expect_source and expect_source not in source:
        return False
    lowered = (content or "").lower()
    return any(str(kw).lower() in lowered for kw in keywords if kw)


# ---------------------------------------------------------------------------
# 纯指标（不碰检索，可直接单测）
# ---------------------------------------------------------------------------
def reciprocal_rank(flags: Sequence[bool]) -> float:
    """MRR 的单条取值：第一个相关结果排名的倒数；没命中为 0。"""
    for idx, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / idx
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    """折损累计增益（排名越靠后折损越大）。"""
    return sum(g / math.log2(idx + 2) for idx, g in enumerate(gains))


def ndcg_at_k(flags: Sequence[bool], n_gold: int) -> float:
    """二元相关性下的 NDCG@K。``n_gold`` 为 0 时返回 0（调用方应已排除该用例）。

    ⚠️ 增益**封顶在 n_gold**：不可能存在比 gold 集更多的相关片段。
    不封顶的话，一旦 Top-K 里的相关片段多于 gold 集（说明语料扫描与索引用的
    切片配置不一致），NDCG 会算出 >1，而没有任何人会觉得这个数不对 ——
    指标越界本身是**最该报警**的信号，却被当成"效果变好了"。
    """
    if n_gold <= 0:
        return 0.0
    gains: List[float] = []
    for flag in flags:
        gains.append(1.0 if (flag and sum(gains) < n_gold) else 0.0)
    ideal_dcg = dcg([1.0] * min(n_gold, len(gains)))
    if ideal_dcg <= 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def recall_at_k(flags: Sequence[bool], n_gold: int) -> float:
    """Top-K 召回率：召回到的相关片段数 / 语料里的相关片段总数。"""
    if n_gold <= 0:
        return 0.0
    return sum(1 for f in flags if f) / n_gold


# ---------------------------------------------------------------------------
# 全量语料的 gold 集（离线）
# ---------------------------------------------------------------------------
def load_corpus_chunks() -> List[Dict[str, str]]:
    """取全量片段（纯离线，不调 embedding）。"""
    from app.rag.indexer import chunk_documents
    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    chunks = chunk_documents(prepare_documents(load_all_documents()))
    return [
        {"content": c.page_content, "source": str(c.metadata.get("source", ""))}
        for c in chunks
    ]


def gold_counts(cases: Sequence[Any], corpus: Sequence[Dict[str, str]]) -> List[int]:
    """对每条用例数出语料里的相关片段总数（Recall 的分母）。"""
    counts: List[int] = []
    for case in cases:
        counts.append(sum(
            1 for c in corpus
            if is_relevant(c["content"], c["source"], case.expect_keywords, case.expect_source)
        ))
    return counts


# ---------------------------------------------------------------------------
# 跑一轮
# ---------------------------------------------------------------------------
def run(cases: Sequence[Any], gold: Sequence[int], top_k: int) -> Dict[str, Any]:
    """对每条用例跑一次真实检索，返回逐条明细 + 汇总。"""
    from app.rag.retriever import retrieve

    rows: List[Dict[str, Any]] = []
    for case, n_gold in zip(cases, gold):
        started = time.perf_counter()
        docs = retrieve(case.question, top_k=top_k)
        elapsed = int((time.perf_counter() - started) * 1000)

        flags = [
            is_relevant(str(d.get("content", "")), str(d.get("source", "")),
                        case.expect_keywords, case.expect_source)
            for d in docs
        ]
        rows.append({
            "question": case.question,
            "n_gold": n_gold,
            "defective": n_gold == 0,
            # 召回的相关片段比 gold 集还多 = 语料扫描与索引用的切片配置不一致，
            # 这时 Recall 的分母是错的。指标照出，但要**说出来**，不能吸收掉。
            "gold_conflict": sum(flags) > n_gold,
            "hit": any(flags),
            "rr": round(reciprocal_rank(flags), 3),
            "recall": round(recall_at_k(flags, n_gold), 3),
            "ndcg": round(ndcg_at_k(flags, n_gold), 3),
            "elapsed_ms": elapsed,
        })

    effective = [r for r in rows if not r["defective"]]
    n = len(effective) or 1
    return {
        "version": BASELINE_VERSION,
        "top_k": top_k,
        "cases": len(rows),
        "effective_cases": len(effective),
        "hit_rate": round(sum(r["hit"] for r in effective) / n, 3),
        "mrr": round(sum(r["rr"] for r in effective) / n, 3),
        "recall": round(sum(r["recall"] for r in effective) / n, 3),
        "ndcg": round(sum(r["ndcg"] for r in effective) / n, 3),
        "avg_ms": int(sum(r["elapsed_ms"] for r in rows) / (len(rows) or 1)),
        "defective": [r["question"] for r in rows if r["defective"]],
        "gold_conflicts": [r["question"] for r in rows if r["gold_conflict"]],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _print(report: Dict[str, Any], worst_n: int) -> None:
    for i, r in enumerate(report["rows"], 1):
        mark = "  ⚠️ 用例缺陷（语料里无相关片段）" if r["defective"] else ""
        print(f"[{i:>2}/{report['cases']}] {r['question'][:22]:<24} "
              f"hit={int(r['hit'])} rr={r['rr']:.3f} recall={r['recall']:.3f} "
              f"ndcg={r['ndcg']:.3f} gold={r['n_gold']:>3} {r['elapsed_ms']:>5}ms{mark}")

    print("\n" + "=" * 72)
    print(f"用例              {report['cases']} 条（有效 {report['effective_cases']} 条，Top-K={report['top_k']}）")
    print(f"命中率 hit_rate   {report['hit_rate']:.3f}")
    print(f"MRR               {report['mrr']:.3f}")
    print(f"Recall@K          {report['recall']:.3f}")
    print(f"NDCG@K            {report['ndcg']:.3f}")
    print(f"平均耗时          {report['avg_ms']}ms")
    if report["defective"]:
        print(f"\n⚠️ 用例缺陷 {len(report['defective'])} 条（已排除出均值，请修用例）：")
        for q in report["defective"]:
            print("   -", q)
    if report["gold_conflicts"]:
        print(f"\n⚠️ gold 集与索引不一致 {len(report['gold_conflicts'])} 条"
              "（召回的相关片段多于语料扫描结果 —— 多半是重建索引时的切片配置与当前不同，"
              "此时 Recall 的分母偏小）：")
        for q in report["gold_conflicts"]:
            print("   -", q)

    if worst_n:
        ranked = sorted((r for r in report["rows"] if not r["defective"]),
                        key=lambda r: (r["ndcg"], r["rr"]))
        print(f"\n最差 {min(worst_n, len(ranked))} 条（按 NDCG 升序）：")
        for r in ranked[:worst_n]:
            print(f"   ndcg={r['ndcg']:.3f} recall={r['recall']:.3f} gold={r['n_gold']:>3}  {r['question']}")


def _diff(report: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    print("\n与基线对比（正=变好）：")
    for key in ("hit_rate", "mrr", "recall", "ndcg"):
        old = baseline.get(key, 0.0)
        new = report[key]
        delta = new - old
        arrow = "↑" if delta > 0.0005 else ("↓" if delta < -0.0005 else "＝")
        print(f"   {key:<10} {old:.3f} → {new:.3f}  {arrow} {delta:+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="检索侧离线评测（HitRate / MRR / Recall / NDCG）")
    parser.add_argument("--top-k", type=int, default=0, help="Top-K（0=用 config.SIMILARITY_TOP_K）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    parser.add_argument("--worst", type=int, default=5, help="列出最差的 N 条（0=不列）")
    parser.add_argument("--update", action="store_true", help="允许覆盖已存在的基线")
    parser.add_argument("--out", default=str(BASELINE_PATH), help="基线路径")
    args = parser.parse_args()

    from app import config
    from app.rag.evaluator import load_eval_cases

    cases = load_eval_cases()
    if args.limit:
        cases = cases[: args.limit]

    print("扫描全量语料计算 gold 集（离线，不发请求）…")
    corpus = load_corpus_chunks()
    gold = gold_counts(cases, corpus)
    print(f"语料片段 {len(corpus)} 条 | 用例 {len(cases)} 条\n")

    report = run(cases, gold, args.top_k or config.SIMILARITY_TOP_K)
    _print(report, args.worst)

    out = Path(args.out)
    if args.update or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n基线已写入：{out}")
        return 0

    baseline = json.loads(out.read_text(encoding="utf-8"))
    _diff(report, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
