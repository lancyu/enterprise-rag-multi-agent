"""检索片段重排 —— 缓解「迷失在中间」（lost-in-the-middle）。

为什么需要重排：
    大模型对长上下文的首尾注意力显著高于中部。按相关性递降排列时，
    最弱的片段恰好落在注意力最好的位置之一，最强的片段反而挤在中间被忽略。
    把高相关片段放首尾、低相关塞中间，能让最关键的证据落在模型最"看得清"的地方。

为什么是纯函数、零模型调用：
    重排只改变「喂给 LLM 的顺序」，属于上下文组装阶段，不改变检索分数的
    客观排序。零成本、零延迟，是所有 RAG 优化里性价比最高的一项。
"""

from typing import Any, Dict, List


def reorder_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 1, 3, 5, …, 6, 4, 2 重排片段：高相关放首尾，低相关塞中间。

    输入已按相关性降序（retriever 的 fused 排序结果），奇数位升序排前面、
    偶数位降序排后面，即可得到「强-中-弱-中-强」的分布。
    少于 3 条时顺序不变（重排无意义）。
    """
    if len(docs) <= 2:
        return list(docs)
    front = docs[0::2]   # 第 1, 3, 5, … 位（较相关）
    back = docs[1::2]    # 第 2, 4, 6, … 位（较弱）
    back.reverse()
    return front + back
