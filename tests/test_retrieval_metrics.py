#!/usr/bin/env python3
"""检索指标（P1-4）的单元测试 —— 只测**纯函数**，不碰检索、不发请求。

为什么单独测
------------
`scripts/eval_retrieval.py` 的产出是「rerank 有没有用」的唯一判据（P1-3 依赖它）。
指标算错了不会报错，只会让 A/B 得出一个**看起来很合理**的错误结论 ——
这比不测更糟。所以这里既验数值，也验两条**结构性不变量**：

1. **分子分母共用同一个相关性判据**（`is_relevant`）。Recall 的分子来自 Top-K、
   分母来自全量语料；两处各写一遍判据，指标照样输出、没有任何报错，
   但它已经不是在算 Recall 了。这种错误只能靠结构判据抓
   —— 数值测试抓不到，因为两处都"自洽"。
2. **`n_gold == 0` 必须被当成用例缺陷暴露**，而不是静默变成 recall=0 混进均值。
"""
from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest

from eval_retrieval import (
    dcg,
    gold_counts,
    is_relevant,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    run,
)


def _case(keywords, source=None) -> SimpleNamespace:
    return SimpleNamespace(question="q", expect_keywords=keywords,
                           expect_source=source, expect_section=None)


# ---------------------------------------------------------------------------
# 1. 数值
# ---------------------------------------------------------------------------
def test_reciprocal_rank_uses_the_first_relevant_position():
    assert reciprocal_rank([True, True]) == 1.0
    assert reciprocal_rank([False, True]) == 0.5
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0


def test_dcg_discounts_lower_positions():
    """同一个相关片段排第 1 位和第 2 位，增益必须不同 —— 否则 NDCG 退化成命中率。"""
    assert dcg([1.0, 0.0]) > dcg([0.0, 1.0])
    assert dcg([1.0, 0.0]) == 1.0                       # 1 / log2(2)
    assert dcg([0.0, 1.0]) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_one_when_the_relevant_hit_is_on_top():
    assert ndcg_at_k([True, False, False], n_gold=1) == 1.0


def test_ndcg_penalises_a_hit_buried_at_the_bottom():
    """命中率看不出差别（都是命中），NDCG 必须能看出来。"""
    top = ndcg_at_k([True, False, False, False], n_gold=1)
    buried = ndcg_at_k([False, False, False, True], n_gold=1)
    assert buried < top


def test_ndcg_never_exceeds_one_even_if_more_relevant_than_gold_are_returned():
    """理想排列只取前 ``min(n_gold, K)`` 位 —— 否则 NDCG 会 >1 而没人觉得不对。"""
    assert ndcg_at_k([True, True, True], n_gold=1) == 1.0


@pytest.mark.parametrize("flags,n_gold", [
    ([True], 1), ([True, True], 2), ([False], 5), ([True] * 5, 100),
])
def test_recall_stays_within_zero_and_one(flags, n_gold):
    assert 0.0 <= recall_at_k(flags, n_gold) <= 1.0
    assert 0.0 <= ndcg_at_k(flags, n_gold) <= 1.0


def test_recall_counts_how_much_of_the_gold_set_was_found():
    assert recall_at_k([True, False], n_gold=4) == pytest.approx(0.25)
    assert recall_at_k([True, True], n_gold=2) == 1.0


def test_zero_gold_yields_zero_instead_of_dividing_by_zero():
    assert recall_at_k([True], 0) == 0.0
    assert ndcg_at_k([True], 0) == 0.0


# ---------------------------------------------------------------------------
# 2. 相关性判据
# ---------------------------------------------------------------------------
def test_source_filter_is_part_of_the_judgement():
    assert is_relevant("年假规定", "员工手册.txt", ["年假"], "员工手册.txt")
    assert not is_relevant("年假规定", "财务报销.txt", ["年假"], "员工手册.txt")
    assert is_relevant("年假规定", "财务报销.txt", ["年假"], None)


def test_gold_count_uses_the_same_source_filter():
    corpus = [
        {"content": "年假有五天", "source": "员工手册.txt"},
        {"content": "年假有五天", "source": "财务报销.txt"},   # 关键词同、来源不同
        {"content": "无关内容", "source": "员工手册.txt"},
    ]
    assert gold_counts([_case(["年假"], "员工手册.txt")], corpus) == [1]
    assert gold_counts([_case(["年假"])], corpus) == [2]


# ---------------------------------------------------------------------------
# 3. 结构性不变量（数值测试抓不到的那部分）
# ---------------------------------------------------------------------------
def test_gold_and_retrieval_share_one_relevance_judgement():
    """**反向验证**：分子（Top-K）与分母（全量 gold）必须调用同一个 `is_relevant`。

    为什么用源码判据而不是数值断言：两处各自写一遍判据时，各自都是自洽的，
    任何数值断言都会通过 —— 因为「两处一致」这件事本身没有被执行到。
    ⚠️ 把 `run()` 里的 `is_relevant(...)` 换成内联的关键词判断，本用例必须变红。
    """
    for func in (gold_counts, run):
        src = inspect.getsource(func)
        assert "is_relevant(" in src, (
            f"{func.__name__} 不再调用 is_relevant —— Recall 的分子/分母判据分叉了"
        )


def test_relevance_judgement_is_defined_exactly_once():
    """判据只能有一处定义：多一处就多一个分叉点。"""
    import eval_retrieval

    module_src = inspect.getsource(eval_retrieval)
    assert module_src.count("def is_relevant(") == 1


def test_zero_gold_is_surfaced_as_a_defective_case_not_silently_averaged():
    """`n_gold == 0` 的用例要**列出来**并从均值里排除。

    让它静默变成 recall=0 是最坏的处理方式：均值被拖下去，而看报表的人
    只会以为"检索变差了"，想不到是**用例**坏了。
    """
    rows = [
        {"question": "正常", "n_gold": 4, "defective": False,
         "hit": True, "rr": 1.0, "recall": 0.25, "ndcg": 0.5, "elapsed_ms": 1},
        {"question": "坏的", "n_gold": 0, "defective": True,
         "hit": False, "rr": 0.0, "recall": 0.0, "ndcg": 0.0, "elapsed_ms": 1},
    ]
    effective = [r for r in rows if not r["defective"]]
    assert len(effective) == 1
    assert [r["question"] for r in rows if r["defective"]] == ["坏的"]
    # 均值只算有效用例 —— 若把缺陷用例算进来，这里会是 0.125 而不是 0.25
    assert sum(r["recall"] for r in effective) / len(effective) == pytest.approx(0.25)
