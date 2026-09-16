"""切分产物结构指标（**零 API 成本**，纯字符串与正则统计）。

设计意图
--------
本模块是「两阶段评测」里**阶段 1** 的核心：
参数网格扫描时只跑这里的结构指标（跨节率 / 块长分布 / 碎片率 / 归属率 / 膨胀比），
**不建索引、不调 embedding、不调 LLM**，因此 9 组网格可在秒级跑完；
只有进入决赛的 1~2 个候选才送去跑阶段 2 的付费检索评测。

同时它也被 `tests/test_chunking_baseline.py` 复用，作为「行为零变化」的回归护栏。

依赖方向：`scripts.chunk_metrics` ← 被 scripts/ 与 tests/ 消费，不依赖 app 业务模块
（只接受 `List[Document]`，不关心 chunk 是怎么产生的）。
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Dict, Iterable, List, Sequence

# ---------------------------------------------------------------------------
# 章节锚点：与 app/rag/structure.py 的层级定义保持一致
# （此处刻意独立实现，避免脚本依赖业务模块；两处修改需同步）
# ---------------------------------------------------------------------------
ANCHOR_PATTERN = re.compile(
    r"第[一二三四五六七八九十百]+章"      # 第三章
    r"|[一二三四五六七八九十]+、"          # 一、
    r"|\d+\.\d+\s"                        # 3.2 
    r"|Q\d+[：:]"                         # Q8：
    r"|#{1,6}\s"                          # ## 6 知识库管理
)

# 判定「块首有归属」时向前看多少字符
HEAD_LOOKAHEAD = 40

# L1 章级锚点（跨章 = 真正的语义污染，必须趋近 0）
#
# 语义区分（重要，容易混淆）：
#   - 跨**章**：块里揉进了两个不同的第一章级章节 → 缺陷，目标 0%
#   - 跨**小节**：块里揉进同章的相邻小节 → **设计目标**，不是缺陷。
#     结构感知切分本来就是把同章相邻小节贪心合并到目标块长，
#     所以 cross_section_rate 天然偏高，把它当缺陷会得出相反的优化方向。
CHAPTER_ANCHOR = re.compile(
    r"第[一二三四五六七八九十百]+章"      # 第三章
    r"|^[一二三四五六七八九十]+、"        # 一、（行首）
    r"|^#{1,2}\s",                       # Markdown H1/H2
    re.M,                                # ^ 必须按行首匹配，否则正文引用会被误判
)


def _texts(chunks: Iterable[Any]) -> List[str]:
    """兼容 Document 对象与裸字符串两种输入。"""
    out: List[str] = []
    for c in chunks:
        out.append(c.page_content if hasattr(c, "page_content") else str(c))
    return out


def count_anchors(text: str) -> int:
    """文本内出现的章节锚点数量。"""
    return len(ANCHOR_PATTERN.findall(text))


def count_chapter_anchors(text: str) -> int:
    """文本内出现的 **不同** 章级（L1）锚点数量。

    用于判定「跨章」：>= 2 说明这块把两章揉在一起，是必须消灭的语义污染。
    只在行首匹配（`re.M`）避免把正文里引用的「第三章」误判为章节边界。
    """
    # re.M 已编译进 pattern，这里不能再传 flags（finditer 第二参是 pos 不是 flags）
    hits = {
        m.group(0).strip()
        for m in CHAPTER_ANCHOR.finditer(text)
        if m.group(0).strip()
    }
    return len(hits)


def has_head_anchor(text: str, lookahead: int = HEAD_LOOKAHEAD) -> bool:
    """块首 `lookahead` 字内是否存在章节锚点（即「有标题归属」）。"""
    return bool(ANCHOR_PATTERN.search(text[:lookahead]))


def compute_metrics(
    chunks: Sequence[Any],
    *,
    baseline_count: int | None = None,
) -> Dict[str, Any]:
    """计算切分产物的结构指标。

    Args:
        chunks: 切分产物（`Document` 列表或字符串列表）
        baseline_count: 基线块数，给定时额外计算索引膨胀比

    Returns:
        结构指标字典（全部为纯统计量，不含任何模型/索引依赖）
    """
    texts = _texts(chunks)
    n = len(texts)
    if n == 0:
        return {
            "count": 0, "len_mean": 0.0, "len_median": 0.0,
            "len_min": 0, "len_max": 0,
            "pct_in_150_350": 0.0, "pct_lt_80": 0.0, "n_lt_80": 0,
            "cross_section_rate": 0.0, "cross_chapter_rate": 0.0,
            "head_attributed_rate": 0.0,
            "expansion_ratio": None,
        }

    lens = [len(t) for t in texts]
    in_range = sum(1 for l in lens if 150 <= l <= 350)
    lt80 = sum(1 for l in lens if l < 80)

    # 跨小节合并：块内含 >=2 个章节锚点（同章内合并，属于设计目标，越高不代表越差）
    cross = sum(1 for t in texts if count_anchors(t) >= 2)
    # 跨章：块内含 >=2 个不同 L1 章级锚点（真正的语义污染，目标 0%）
    cross_ch = sum(1 for t in texts if count_chapter_anchors(t) >= 2)
    # 有归属：块首能直接看到锚点
    headed = sum(1 for t in texts if has_head_anchor(t))

    return {
        "count": n,
        "len_mean": round(statistics.mean(lens), 1),
        "len_median": round(statistics.median(lens), 1),
        "len_min": min(lens),
        "len_max": max(lens),
        "pct_in_150_350": round(in_range / n, 4),
        "pct_lt_80": round(lt80 / n, 4),
        "n_lt_80": lt80,
        "cross_section_rate": round(cross / n, 4),
        "cross_chapter_rate": round(cross_ch / n, 4),
        "head_attributed_rate": round(headed / n, 4),
        "expansion_ratio": (
            round(n / baseline_count, 3) if baseline_count else None
        ),
    }


def format_report(m: Dict[str, Any], title: str = "结构指标") -> str:
    """把指标渲染成人类可读的多行报告。"""
    exp = f"{m['expansion_ratio']}x" if m.get("expansion_ratio") else "-"
    return (
        f"── {title} ──\n"
        f"  块数            : {m['count']}"
        + (f"（相对基线膨胀 {exp}）" if m.get("expansion_ratio") else "") + "\n"
        f"  块长 mean/median: {m['len_mean']} / {m['len_median']} "
        f"(min {m['len_min']}, max {m['len_max']})\n"
        f"  落在[150,350]字 : {m['pct_in_150_350']:.1%}\n"
        f"  <80 字碎片      : {m['n_lt_80']} 个（{m['pct_lt_80']:.1%}）\n"
        f"  跨章率          : {m['cross_chapter_rate']:.1%}   ← 越低越好（缺陷指标，目标 0%）\n"
        f"  跨小节合并率    : {m['cross_section_rate']:.1%}   ← 设计目标，非缺陷\n"
        f"  块首有归属率    : {m['head_attributed_rate']:.1%}   ← 越高越好\n"
    )


if __name__ == "__main__":
    # 直接运行：对当前默认配置（recursive）产出基线指标
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from app import config
    from app.rag.indexer import chunk_documents
    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    prepared = prepare_documents(load_all_documents())
    chunks = chunk_documents(prepared)
    title = f"当前配置（{config.CHUNK_STRATEGY}）"
    print(format_report(compute_metrics(chunks), title=title))
