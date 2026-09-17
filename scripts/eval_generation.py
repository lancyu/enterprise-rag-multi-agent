"""生成侧离线评测 —— 补齐 ``run_retrieval_eval`` 只评检索的缺口。

``app/rag/evaluator.run_retrieval_eval`` 看的是「有没有召回到正确片段」，
但用户拿到的是**生成后的答案**。片段召回正确、答案却截断 / 拒答 / 编造，
在检索指标上完全看不出来——「回答不完整」那类故障就是这么漏掉的。

本脚本对 ``app/rag/eval_cases.yaml`` 的每条用例跑一遍**真实生成**，统计：

    拒答率        答案是否被判为「知识库没有」（应接近 0）
    截断率        是否撞上 max_tokens（finish_reason=length，应恒为 0）
    关键词覆盖率  答案里是否出现了期望关键词——正确性的粗粒度代理
    忠实度        答案与召回片段的字面重合度（evaluator.score_faithfulness）
    首 token / 端到端耗时

用法::

    PYTHONPATH=. .venv/bin/python scripts/eval_generation.py

注意：会真实调用大模型（每条用例一次），有真实耗时与成本。默认**不插入静默间隔**
—— 「每条之间等 20 秒」是为当年那个限频账号定的，前提早已不存在，留着它只是白等。
若你的账号确有频率上限、日志里出现 429，再用 ``--sleep`` 拉开节奏。
"""
import argparse
import time

from app.core.rag_engine import retrieve_knowledge_docs
from app.rag.evaluator import load_eval_cases, score_faithfulness
from app.rag.generator import generate_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="生成侧离线评测")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="每条用例之间的间隔秒数（默认 0；账号有频率上限、出现 429 时再调大）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    args = parser.parse_args()

    cases = load_eval_cases()
    if args.limit:
        cases = cases[: args.limit]
    print(f"用例数 = {len(cases)} | 间隔 = {args.sleep}s\n")

    rows = []
    for i, case in enumerate(cases, 1):
        docs = retrieve_knowledge_docs(case.question)
        t0 = time.perf_counter()
        res = generate_answer(case.question, docs)
        ms = int((time.perf_counter() - t0) * 1000)

        answer = res.answer or ""
        stats = res.stats
        row = {
            "q": case.question,
            "refused": bool(res.refused),
            "truncated": bool(getattr(stats, "truncated", False)),
            "hit": any(kw.lower() in answer.lower() for kw in case.expect_keywords),
            "faith": score_faithfulness(answer, [d.get("content", "") for d in docs]),
            "chars": len(answer),
            "ms": ms,
            "ttft": getattr(stats, "ttft_ms", 0),
            "hits": len(docs),
            "error": res.error,
        }
        rows.append(row)
        print(f"[{i:>2}/{len(cases)}] {case.question[:22]:<24} 拒答={int(row['refused'])} "
              f"截断={int(row['truncated'])} 含关键词={int(row['hit'])} "
              f"忠实度={row['faith']:.2f} {row['chars']:>4}字 {ms:>6}ms 命中={row['hits']}")
        if res.error:
            print("      生成错误:", res.error)
        if i < len(cases):
            time.sleep(args.sleep)

    n = len(rows)
    print("\n" + "=" * 68)
    print(f"用例数            {n}")
    print(f"拒答率            {sum(r['refused'] for r in rows)}/{n}")
    print(f"截断率            {sum(r['truncated'] for r in rows)}/{n}")
    print(f"关键词覆盖率      {sum(r['hit'] for r in rows)}/{n}")
    print(f"平均忠实度        {sum(r['faith'] for r in rows) / n:.3f}")
    print(f"平均首 token      {sum(r['ttft'] for r in rows) / n:.0f}ms")
    print(f"平均端到端        {sum(r['ms'] for r in rows) / n:.0f}ms")
    print(f"平均字数          {sum(r['chars'] for r in rows) / n:.0f}")
    print(f"生成报错          {sum(1 for r in rows if r['error'])}/{n}")

    suspect = [r["q"] for r in rows if r["refused"] or r["truncated"] or not r["hit"] or r["error"]]
    if suspect:
        print("\n需人工复核：", suspect)


if __name__ == "__main__":
    main()
