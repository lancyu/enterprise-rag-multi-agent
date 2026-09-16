"""字面相似度 —— 词面打分的**唯一**定义处。

它替换掉了什么，以及为什么必须替换
----------------------------------
第一版词面分是 ``Σ len(命中关键词)``，每条能力自带一张手写关键词表。它的失效
方式是**结构性的**，不是"词表抄得不够全"：

1. **同义改写必然漏**。词表里写了"分机号"，用户说"座机"就漏；写了"部门"，
   用户说"团队"就漏。补词只能追着漏判跑，永远差一条——**用户说的话是开集，
   手写的词表是闭集**。
2. **穷举宾语的能力被系统性抬高**。分数是各命中词长度之和，于是"年假"+"调休"
   两个宾语（4.0）压过真正表达提问意图的"区别"（2.0）——
   *提到两个宾语比在问两者的关系得分更高*。这个偏向没有可调参数
   （权重就是词长），调不了。
3. **它制造了第二份事实来源**。``keywords`` 与 ``utterances`` 描述同一件事，
   两者不一致时没有任何一处会报错。

改成"与例句的字符 n-gram 相似度"之后，这三条一起消失：**例句是唯一的事实来源**。
新增一个领域只需要写例句与描述，词表由 :mod:`app.core.routing.derive` 反推。

为什么是字符 bigram 的 Dice 系数
--------------------------------
============  ==========================================================
判据           取舍
============  ==========================================================
字符（非词）     中文没有空格。jieba 一类分词器引入词典 = 又一份手写词表，
              且对"座机/分机号"这类未登录词恰恰分不对——而它们正是要救的。
bigram（非 1/3）  单字噪声太大（"的""是"到处都是）；trigram 对短句太脆，
              改一个字就归零。
Dice（非 Jaccard） ``2|A∩B| / (|A|+|B|)`` 对**长度不等**的句子更宽容。
              路由里几乎永远是"短提问 vs 长例句"（"我还有几天年假" vs
              "请问张三年假还剩多少天"），Jaccard 会把这句压到 0.2 以下。
============  ==========================================================

离线实测（22 条问句，同一份目录）：关键词法 11/22，本模块 17/22。
可解释性同时变好：命中的是**某一条具体例句**，而不是"哪几个词撞上了"。

⚠️ 本模块是**引擎**：不许出现任何领域词，也不许 import ``catalog``。
它只认字符串，不认"年假"是什么。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, FrozenSet, Sequence, Tuple

#: n-gram 的 n。改它等于改整个词面层的灵敏度，所以做成常量而不是参数——
#: 让它只有一个定义处，标定阈值时才不会出现"两处 n 不一样"这种查不出的不一致。
NGRAM_N = 2


def normalize(text: str) -> str:
    """归一化：转小写、只保留字母数字。

    去掉的是标点与空白，**保留汉字**——``isalnum()`` 对汉字为真。
    "张三的工号是多少？" 与 "张三的工号是多少" 归一化后必须相等，
    否则同一个问题加个问号就换个判定，是最难解释的一类抖动。
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


@lru_cache(maxsize=4096)
def ngrams(text: str, n: int = NGRAM_N) -> FrozenSet[str]:
    """取 ``text`` 的全部长度为 ``n`` 的连续片段（已归一化）。

    缓存是必要的：每个能力要对全部例句算一遍，同一份例句每轮路由都要用。
    键是 ``(text, n)``，所以改一条例句只会让它自己失效。

    短于 ``n`` 的文本返回 ``{整个文本}`` 而不是空集——空集会让 Dice 分母为 0，
    于是"嗨"这种单字提问对任何例句的相似度都是 0，白白掉进灰区。
    """
    norm = normalize(text)
    if len(norm) < n:
        return frozenset({norm}) if norm else frozenset()
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def dice(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Dice 系数 ``2|A∩B| / (|A|+|B|)`` ∈ [0, 1]。任一方为空则 0。"""
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def similarity(a: str, b: str, n: int = NGRAM_N) -> float:
    """两条文本的字面相似度。"""
    return dice(ngrams(a, n), ngrams(b, n))


@dataclass(frozen=True)
class BestMatch:
    """一条文本与一组例句的最佳匹配。**分数与"像哪一条"一起返回。**

    不返回例子的版本（只给分数）在排障时是没法用的：看到 0.42 分既不知道
    它像谁，也不知道该改哪条例句。第一版的 ``hits`` 记的是"命中了哪几个词"，
    那对应的是词的集合；换成例句之后，**例子本身**才是那个可操作的对象。
    """

    score: float
    example: str = ""


def best_match(query: str, examples: Sequence[str], n: int = NGRAM_N) -> BestMatch:
    """``query`` 与 ``examples`` 里最像的那一条。空例句集返回 0 分。

    取 ``max`` 而不是取平均：能力的例句是**同一个意图的不同说法**，
    用户说中其中任意一种就够判定，说中多种并不比说中一种更"是这个意图"。
    取平均反而会惩罚例句写得多的能力——那是"多写例句"的反向激励。
    """
    query_grams = ngrams(query, n)
    best = BestMatch(0.0, "")
    for example in examples:
        score = dice(query_grams, ngrams(example, n))
        if score > best.score:
            best = BestMatch(score, example)
    return best


def gram_union(texts: Iterable[str], n: int = NGRAM_N) -> FrozenSet[str]:
    """一组文本出现的全部 n-gram 之并。

    :mod:`app.core.routing.derive` 用它从例句反推词表：例句里出现过的
    连续片段就是"这个领域说过的词"。
    """
    union: FrozenSet[str] = frozenset()
    for text in texts:
        union |= ngrams(text, n)
    return union


def contains_any(haystack: str, needles: FrozenSet[str], n: int = NGRAM_N) -> Tuple[str, ...]:
    """``needles`` 里有哪些片段出现在 ``haystack`` 中（按字典序返回，便于断言）。

    只用于**观测与测试**：真正的判据不该靠"某个片段出现没出现"——
    那正是被本模块替换掉的做法。派生词表在启动期自检里用它做双向核对。
    """
    grams = ngrams(haystack, n)
    return tuple(sorted(needles & grams))
