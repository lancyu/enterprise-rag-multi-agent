"""RAG 五层架构 · L5 评估迭代层。

职责边界：量化系统表现，并把结论回流给 L1/L2/L3 形成闭环。
    离线评测（命中率 / MRR） → 线上反馈收集 → 忠实度核算 → 报告与调优建议

为什么必须有这一层：
    没有评估的 RAG 只能靠"我问了两句感觉还行"来判断好坏。一旦语料增长、
    换了 embedding 供应商、调了切片参数，系统是在变好还是变坏，
    完全没有客观依据——这是 RAG 项目最常见的烂尾原因。

指标设计：
    - HitRate@K：Top-K 里有没有命中期望内容（召回能力的下限指标）
    - MRR：首个命中结果的排名倒数（衡量"好结果排得够不够前"）
    - Faithfulness：答案字符二元组被上下文覆盖的比例（无 LLM 的幻觉粗筛）
    - 引用覆盖率：答案里带引用标记的比例（溯源能力）

关于忠实度：
    严格做法是用 LLM-as-judge 逐句核对，但那需要额外模型调用，
    每次评测都要真实付出一轮往返，成本与耗时都不低。这里用
    「字符二元组覆盖率」做零成本启发式：
    答案中大量出现上下文里根本不存在的二元组合时，基本可以判定为编造。
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from app import config
from app.core.errors import EvalCaseInvalid
from app.utils.doc_loader import file_name
from app.utils.logger import logger

FEEDBACK_FILE: Path = Path(getattr(config, "FEEDBACK_FILE", config.LOG_DIR / "feedback.jsonl"))
EVAL_CASES_FILE: Path = Path(__file__).resolve().parent / "eval_cases.yaml"


# ---------------------------------------------------------------------------
# 评测用例
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    """一条评测用例。

    expect_keywords 用于判定「是否命中」：只要 Top-K 中任一片段包含
    其中任一关键词，即视为命中。比严格匹配标准答案更鲁棒，
    也更适合企业知识库「同一事实多种表述」的现实。

    expect_section（可选）：期望答案所在的章节锚点（如「第三章 假期管理」、
    「1.2 密码策略」）。用于「章节命中率」——衡量检索结果是否落在正确章节，
    而不只是碰巧命中关键词。结构感知切分（T2-2）上线后此字段才有意义。
    """

    question: str
    expect_keywords: List[str]
    expect_source: Optional[str] = None
    expect_section: Optional[str] = None


# 内置基线用例：覆盖企业知识库最典型的问答场景。
# 关键词取自 data/ 目录下的示例文档，实际部署时应按自己的语料替换。
DEFAULT_EVAL_CASES: List[EvalCase] = [
    EvalCase("年假有多少天", ["年休假", "年假"], "员工手册.txt"),
    EvalCase("报销流程怎么走", ["报销"], "员工手册.txt"),
    EvalCase("VPN 连接不上怎么办", ["VPN"], "IT支持指南.txt"),
    EvalCase("密码忘记了怎么重置", ["密码"], "IT支持指南.txt"),
]


def load_eval_cases(path: Optional[Path] = None) -> List[EvalCase]:
    """从 YAML 外置文件加载评测用例，字段缺失/非法时抛 EvalCaseInvalid。

    YAML 外置的好处：增删用例无需改代码（原实现硬编码 4 条在源码里），
    业务方/运维可直接编辑 yaml 扩充覆盖，不影响代码主线。
    """
    import yaml

    path = Path(path) if path else EVAL_CASES_FILE
    if not path.exists():
        # 文件缺失时退回内置用例，保证评测链路永远可用
        logger.warning("评测用例文件不存在：%s，使用内置用例", path)
        return list(DEFAULT_EVAL_CASES)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cases_raw = raw.get("cases", [])
    except Exception as exc:
        raise EvalCaseInvalid(f"评测用例文件解析失败：{exc}") from exc

    cases: List[EvalCase] = []
    for i, item in enumerate(cases_raw):
        question = str(item.get("question", "")).strip()
        keywords = item.get("expect_keywords") or []
        if not question or not keywords:
            raise EvalCaseInvalid(f"第 {i + 1} 条用例缺少 question 或 expect_keywords")
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            raise EvalCaseInvalid(f"第 {i + 1} 条用例 expect_keywords 必须是字符串列表")
        cases.append(
            EvalCase(
                question=question,
                expect_keywords=[str(k) for k in keywords],
                expect_source=item.get("expect_source"),
                expect_section=(str(item["expect_section"]).strip()
                                if item.get("expect_section") else None),
            )
        )
    return cases or list(DEFAULT_EVAL_CASES)


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(str(kw).lower() in lowered for kw in keywords if kw)


# ---------------------------------------------------------------------------
# 离线检索评测
# ---------------------------------------------------------------------------
def run_retrieval_eval(
    cases: Optional[List[EvalCase]] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """跑检索评测，返回逐条明细与汇总指标。

    会真实调用 embedding 接口，用例数量请控制（每一轮都是一次真实远端往返）。
    """
    from app.rag.retriever import retrieve

    if cases is None:
        cases = load_eval_cases()
    top_k = top_k or config.SIMILARITY_TOP_K
    if not cases:
        return {"count": 0, "hit_rate": 0.0, "mrr": 0.0, "details": []}

    details: List[Dict[str, Any]] = []
    hits = 0
    rr_sum = 0.0
    section_hits = 0
    section_cases = 0

    for case in cases:
        started = time.perf_counter()
        docs = retrieve(case.question, top_k=top_k)
        elapsed = int((time.perf_counter() - started) * 1000)

        hit_rank = 0
        for idx, doc in enumerate(docs, start=1):
            content = doc.get("content", "")
            source_ok = (not case.expect_source) or (case.expect_source in str(doc.get("source", "")))
            if source_ok and _contains_any(content, case.expect_keywords):
                hit_rank = idx
                break

        hit = hit_rank > 0
        rr = 1.0 / hit_rank if hit else 0.0
        hits += 1 if hit else 0
        rr_sum += rr

        # 章节命中：Top-K 任一片段的章节元数据命中 expect_section（仅在用例标注时统计）
        section_hit = False
        if case.expect_section:
            section_cases += 1
            if any(_section_hit(doc, case.expect_section) for doc in docs):
                section_hit = True
                section_hits += 1

        details.append(
            {
                "question": case.question,
                "hit": hit,
                "hit_rank": hit_rank,
                "reciprocal_rank": round(rr, 3),
                "section_hit": section_hit if case.expect_section else None,
                "top_sources": [file_name(str(d.get("source", ""))) for d in docs[:3]],
                "top_score": docs[0].get("score") if docs else None,
                "top_lexical": docs[0].get("lexical") if docs else None,
                "elapsed_ms": elapsed,
            }
        )

    count = len(cases)
    report = {
        "count": count,
        "top_k": top_k,
        "hit_rate": round(hits / count, 3),
        "mrr": round(rr_sum / count, 3),
        "section_hit_rate": round(section_hits / section_cases, 3) if section_cases else None,
        "section_cases": section_cases,
        "details": details,
    }
    logger.info("L5 检索评测完成：命中率 %.3f | MRR %.3f | 章节命中率 %s | 用例 %d 条",
                report["hit_rate"], report["mrr"],
                f"{report['section_hit_rate']}" if report["section_hit_rate"] is not None else "N/A", count)
    return report


def _section_hit(doc: Dict[str, Any], expect_section: str) -> bool:
    """判断单篇检索结果是否落在期望章节内。

    匹配逻辑：期望章节锚点（如「1.2 密码策略」「第三章 假期管理」）是否出现在
    该片段的 `heading_path` / `chapter` / `section` 任一字段中。用子串匹配而非精确相等，
    因为 expect_section 通常只写「章/节」关键词，而元数据是全路径。
    """
    meta = doc.get("metadata") or {}
    if not meta:
        return False
    target = expect_section.strip()
    if not target:
        return False
    candidates = [
        str(meta.get("heading_path", "")),
        str(meta.get("chapter", "")),
        str(meta.get("section", "")),
    ]
    return any(target in c for c in candidates if c)


def filter_by_section(docs: Sequence[Dict[str, Any]], section: str) -> List[Dict[str, Any]]:
    """按章节过滤检索结果（供生成端 / 章节级问答使用）。

    section 为空或片段无章节元数据时原样返回，保证「过滤失败不阻断」的硬约束。
    """
    if not section or not str(section).strip():
        return list(docs)
    out = [d for d in docs if _section_hit(d, section)]
    return out if out else list(docs)


# ---------------------------------------------------------------------------
# 忠实度与引用
# ---------------------------------------------------------------------------
def _bigrams(text: str) -> Set[str]:
    chars = [ch for ch in (text or "") if not ch.isspace()]
    return {"".join(chars[i : i + 2]) for i in range(len(chars) - 1)}


def score_faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """答案忠实度（0~1）：答案字符二元组被上下文覆盖的比例。

    零成本启发式，用于粗筛"答案里出现了上下文完全没有的内容"这类明显编造。
    """
    answer_bigrams = _bigrams(answer)
    if not answer_bigrams:
        return 0.0
    context_bigrams: Set[str] = set()
    for ctx in contexts:
        context_bigrams |= _bigrams(ctx)
    if not context_bigrams:
        return 0.0
    return round(len(answer_bigrams & context_bigrams) / len(answer_bigrams), 3)


# ---------------------------------------------------------------------------
# 线上反馈
# ---------------------------------------------------------------------------
def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    rating: str,
    comment: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """记录一次用户反馈（good / bad），追加写入 jsonl。

    采用 append-only 的 jsonl 而非数据库：反馈量小、schema 会演进，
    jsonl 便于后续直接 grep / pandas 分析，也避免为一个辅助功能引入依赖。
    """
    if rating not in ("good", "bad"):
        logger.warning("忽略非法反馈评分：%s", rating)
        return False
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "question": (question or "")[:500],
            "answer": (answer or "")[:2000],
            "rating": rating,
            "comment": (comment or "")[:500],
        }
        if extra:
            record.update(extra)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:  # noqa: BLE001
        logger.exception("反馈写入失败")
        return False


def load_feedbacks(limit: int = 200) -> List[Dict[str, Any]]:
    """读取最近的反馈记录。"""
    if not FEEDBACK_FILE.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        logger.exception("反馈文件读取失败")
        return []
    return rows[-limit:]


def feedback_stats(limit: int = 200) -> Dict[str, Any]:
    """反馈统计：好评率与最近的差评样本（差评是调优的第一手线索）。"""
    rows = load_feedbacks(limit)
    if not rows:
        return {"total": 0, "good": 0, "bad": 0, "good_rate": 0.0, "recent_bad": []}
    good = sum(1 for r in rows if r.get("rating") == "good")
    bad_rows = [r for r in rows if r.get("rating") == "bad"]
    return {
        "total": len(rows),
        "good": good,
        "bad": len(bad_rows),
        "good_rate": round(good / len(rows), 3),
        "recent_bad": [
            {"question": r.get("question", "")[:100], "comment": r.get("comment", "")}
            for r in bad_rows[-5:]
        ],
    }


# ---------------------------------------------------------------------------
# 报告与调优建议
# ---------------------------------------------------------------------------
def build_report(include_eval: bool = False) -> Dict[str, Any]:
    """汇总各层状态，产出一份可诊断的系统报告。"""
    report: Dict[str, Any] = {"feedback": feedback_stats()}
    try:
        from app.rag.indexer import get_index_stats
        from app.rag.prepare import get_last_stats
        from app.rag.retriever import retrieval_stats

        report["prepare"] = get_last_stats()
        report["index"] = get_index_stats()
        report["retrieval"] = retrieval_stats()
    except Exception:  # noqa: BLE001
        logger.warning("报告汇总时部分层状态不可用")
    if include_eval:
        report["retrieval_eval"] = run_retrieval_eval()
    return report


def suggest_improvements(report: Optional[Dict[str, Any]] = None) -> List[str]:
    """基于报告给出可执行的调优建议（规则化，便于直接落地）。"""
    report = report if report is not None else build_report()
    suggestions: List[str] = []

    eval_block = report.get("retrieval_eval") or {}
    if eval_block:
        hit_rate = eval_block.get("hit_rate", 1.0)
        mrr = eval_block.get("mrr", 1.0)
        if hit_rate < 0.7:
            suggestions.append(
                f"检索命中率偏低（{hit_rate}）：优先补充语料覆盖，其次检查切片是否把关键段落切碎（L1/L2）"
            )
        if mrr < 0.6:
            suggestions.append(
                f"好结果排名靠后（MRR={mrr}）：提高词面路权重 LEXICAL_WEIGHT，或调小 CHUNK_SIZE（L3/L2）"
            )

    fb = report.get("feedback") or {}
    if fb.get("total", 0) >= 5 and fb.get("good_rate", 1.0) < 0.6:
        suggestions.append(
            f"用户好评率偏低（{fb.get('good_rate')}）：检查 recent_bad 中的样本，定位是召回错还是生成错（L3/L4）"
        )

    retrieval = report.get("retrieval") or {}
    if retrieval.get("score_threshold") is not None:
        suggestions.append(
            "当前使用绝对分数阈值过滤，跨 embedding 供应商迁移时容易失效，"
            "建议保持 SCORE_THRESHOLD 为空以启用自适应模式（L3）"
        )

    if not suggestions:
        suggestions.append("各项指标正常，暂无调优建议")
    return suggestions
