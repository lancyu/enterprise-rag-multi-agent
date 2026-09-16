"""切分改造 T4-1：评测消费 expect_section（章节命中）。

验证点：
1. load_eval_cases 正确解析 expect_section（必填字段缺失仍抛错）；
2. _section_hit 对 heading_path / chapter / section 子串匹配正确；
3. filter_by_section 按章节过滤，且无匹配时回退原样返回（不阻断）。
"""
from pathlib import Path

from app.rag import evaluator as E


def test_load_eval_cases_parses_expect_section():
    """读真实 yaml：规模足够，且**标注了 expect_section 的用例**能被正确解析。

    为什么用「带 expect_section 的条数」而不是「全部条数」断言：
    无结构（flat）文档（如速查表、PDF 抽取文本）产出的块没有 `heading_path`，
    这类用例**不该**标注 expect_section —— 硬标注只会制造永远不命中的负样本，
    反而污染 section_hit_rate。`run_retrieval_eval` 也只统计标注过的用例
    （`section_cases`），未标注的不参与，两者口径一致。
    """
    cases = E.load_eval_cases()
    assert len(cases) >= 20, f"评测集应 ≥20 条以保证指标有区分度，当前 {len(cases)} 条"
    with_section = [c for c in cases if c.expect_section]
    assert len(with_section) >= 20, (
        f"带 expect_section 的用例应 ≥20 条，当前只有 {len(with_section)} 条"
    )
    # 报销那条来源应已修正为详细的财务制度文档
    reimburse = next(c for c in cases if "报销" in c.question)
    assert reimburse.expect_source == "财务报销管理制度.txt"


def test_load_eval_cases_rejects_missing_required():
    # 临时写一个缺 keywords 的坏用例，确认抛 EvalCaseInvalid
    import tempfile
    bad = "cases:\n  - question: 测试\n    expect_keywords: []\n"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(bad)
        path = Path(f.name)
    try:
        import pytest
        with pytest.raises(E.EvalCaseInvalid):
            E.load_eval_cases(path)
    finally:
        path.unlink()


def test_section_hit_substring_match():
    doc = {
        "metadata": {
            "heading_path": "第三章 假期管理 > 3.2 年休假",
            "chapter": "第三章 假期管理",
            "section": "3.2 年休假",
        }
    }
    assert E._section_hit(doc, "第三章 假期管理")
    assert E._section_hit(doc, "3.2 年休假")
    assert not E._section_hit(doc, "第八章 培训")
    # 无 metadata 的片段一律不命中
    assert not E._section_hit({"content": "..."}, "第三章 假期管理")


def test_filter_by_section_fallback_when_no_match():
    docs = [
        {"metadata": {"chapter": "第三章 假期管理", "section": "3.2 年休假"}},
        {"metadata": {"chapter": "二、VPN 远程接入", "section": "2.2 VPN 连接步骤"}},
    ]
    # 命中：只留假期管理
    kept = E.filter_by_section(docs, "第三章 假期管理")
    assert len(kept) == 1
    assert kept[0]["metadata"]["chapter"] == "第三章 假期管理"
    # 空 section 或无匹配 -> 回退原样返回（不阻断）
    assert E.filter_by_section(docs, "") == docs
    assert E.filter_by_section(docs, "不存在的章") == docs
