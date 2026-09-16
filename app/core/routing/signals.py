"""句式判定 —— 只回答"这句话长什么样"，**绝不**回答"这句话是什么意图"。

为什么必须与意图解耦（本模块最重要的约定）
------------------------------------------
依赖方向是**单向**的::

    catalog ──引用──> signals          （signals 绝不 import catalog）

判据一旦反过来依赖具体意图名，就会出现"为了修 A 意图的句式、弄坏 B 意图的句式"
——这正是八模块规则路由时代的失效方式。这里每个函数只返回一个**关于句子的
客观事实**；它是否被用来否定某个意图，由 ``catalog.IntentSpec.guards`` 决定。

本模块**不含任何业务词**
------------------------
"部门""年假""工号"属于**某个领域**，不属于路由。它们由
:class:`app.core.routing.vocabulary.Vocabulary` 随目录一起提供，本模块只接收。

这里保留的是**汉语语法层**，与领域无关：系词（是/在/属于）、疑问词（哪个/什么/啥）、
结构助词（的）、问句标记（呢/吗/？），以及三张句式词表——操作问句、对比问句、
礼貌前缀。它们换到任何中文领域都一样，所以留在引擎里。

两个"词表"字段的去向（2026-09-16 改）
------------------------------------
- ``policy_nouns`` / ``has_policy_context`` **已删除**。它的职责是"含规范名词就不算
  在问某个人的值"，而这是词面打分的替代品：``Σ len(关键词)`` 分不出
  「哪个部门负责报销」与「张三在哪个部门」，只能靠一张额外的手写名词表去否掉。
  改成与例句做 n-gram 相似度之后（:mod:`app.core.routing.similarity`），
  这两句在``catalog`` 里各自有例句，相似度直接把它们分开（1.00 vs 0.43），
  guard 变成纯粹的冗余——而冗余的手写词表正是本项目要清掉的那类东西。
- ``business_nouns`` **改为由例句反推**（:mod:`app.core.routing.derive`），
  内容也从"词"变成"片段集合"，故 :func:`has_business_noun` 改用
  :func:`app.core.routing.similarity.contains_any` 而不是拼一条 200 多分支的正则。

判据函数一律**显式接收** ``vocab``，**不设默认值**：没有默认值，就没有
"悄悄用了错误词表"这种可能。上层（anchors / fusion / router）负责传下来，
默认取 ``catalog.vocabulary()``。

三态返回值（不要退化成"布尔 + 空串"两个函数）
--------------------------------------------
``extract_person_attr_slot`` 返回 ``Optional[str]``::

    None   → 不是这类问句
    ""     → 是这类问句，但没给对象（→ 该澄清）
    "张三" → 是这类问句且抽到了对象

``looks_like_person_attr_query`` 写成它的**派生**（``is not None``），
由构造保证两者不可能漂移。用两个函数表达同一件事，迟早会出现一个返回 True
而另一个返回 "" 的组合，而没有任何一处能看出它不该这样。

历史教训（``docs/history/intent-routing-redesign.md`` 记的）
----------------------------------------------------
「张三**在哪**个部门」曾被当成 how-to 问题，因为把"在哪"当成了 how-to 标志词。
真正的判据是**"属性词处于被索取位置"**：``(?:哪个|什么)`` 与属性词必须**相邻**
（中间只允许空白），而不是同句共现。
仅靠关键词表区分不了「提到某个词」与「要取这个值」——正因如此，下面这条正则的
两半不能互相牵制，所以它单独住在本模块，不散落到业务代码里。
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

from app.core.routing import similarity
from app.core.routing.vocabulary import Vocabulary

# ---------------------------------------------------------------------------
# 编译缓存：词表可哈希（元组），所以同一份词表只编译一次。
#
# 这些函数**必须**在词表为空时返回 ``None``，而不是 ``re.compile("")``：
# 空模式会匹配任意字符串，于是"没给词表"会变成"什么都命中"——
# 一个安静地把所有请求都判成越界的引擎。
# ---------------------------------------------------------------------------
def _alternation(words: Tuple[str, ...]) -> str:
    """把词表拼成正则交替式。**按长度降序**：长词必须先匹配。

    "早上好"若排在"早"之后，引擎会先吃掉"早"、剩下"上好"，
    于是一条最普通的写法反而漏判。``re.escape`` 让词表保持"词"的身份——
    想写正则请用 :attr:`Vocabulary.identifier_patterns`，那里才是正则。
    """
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


@lru_cache(maxsize=32)
def _person_attr_re(attr_words: Tuple[str, ...]) -> Optional[re.Pattern]:
    if not attr_words:
        return None
    attr = _alternation(attr_words)
    return re.compile(
        # 形态一：X 是/在/属于 哪个 <属性>          —— 「张三在哪个部门」
        #
        # ⚠️ 对象槽是**懒惰**的 ``{0,8}?``。用贪婪写法，「张三在哪个部门」会把
        # "张三在"整个吞进对象槽（系词是可选的，引擎乐得多吃一个字符），
        # 于是抽出来的对象是"张三在"。判通道不受影响（只用到"非 None"），
        # 但槽位会作为一个**看起来像人名**的脏值流出去——这种错最难发现，
        # 因为它不报错、只是安静地传下去。懒惰匹配从最少的字符开始试，先撞上
        # 系词的位置就停下，正好切出"张三"。
        r"(?P<obj>[\u4e00-\u9fa5A-Za-z0-9]{0,8}?)\s*(?:是|在|属于)?\s*(?:哪个|哪個|什么|啥)\s*"
        + r"(?P<attr>" + attr + r")"
        + _PERSON_ATTR_TAIL
        # 形态二：X 的 <属性>（是多少）             —— 「王五的分机号是多少」「他的邮箱是多少」
        + r"|(?P<obj2>[\u4e00-\u9fa5A-Za-z0-9]{1,8})\s*的\s*"
        + r"(?P<attr2>" + attr + r")"
        + _PERSON_ATTR_TAIL,
        re.I,
    )


@lru_cache(maxsize=32)
def _identifier_re(patterns: Tuple[str, ...]) -> Optional[re.Pattern]:
    if not patterns:
        return None
    return re.compile("|".join(patterns))


#: 索取结构的"尾巴"：**系词 + 疑问词（+ 量词）**。这是汉语语法，不是业务词。
#:
#: 这是本模块最关键的一处约束。放开成"任意短尾巴"之后，
#: 「公司的邮箱怎么申请」会命中（对象=公司、属性=邮箱、尾巴=怎么申请），
#: 于是一条制度类问题被误送进工具通道——而那正是本设计要修的 bug 之一。
#:
#: 写成「系词 × 疑问词 × 量词」三段而不是把「是多少 / 是什么 / 是谁 / 是哪天…」
#: 一条条列举：**列举是开集，语法是闭集**。上一版漏了「是谁」「是哪天」「是几号」，
#: 于是「李四的直属领导是谁」这条最普通的问法压根锚不到——症状是"它掉到后面楼层
#: 去了"，没有任何报错。改成三段之后，任何"X 的 Y 是谁/是多少/是哪天"都自然覆盖。
#:
#: 三段**整体可选**：`赵六的入职时间` 这类不带疑问词的写法同样是取值句。
_COPULA = r"是|在|属于|有|为|叫|算"
_MEASURE = r"号|天|个|次|位|名|时|年|月|日|分|点|块|元|岁|条|张"
_INTERROGATIVE = r"多少|多长|多久|几|什么|啥|谁|哪|哪个|哪個|哪里|哪儿|哪天|何时"
_PERSON_ATTR_TAIL = (
    r"(?:\s*(?:" + _COPULA + r")?\s*(?:" + _INTERROGATIVE + r")(?:" + _MEASURE + r")?)?"
    r"\s*(?:呢|吗|啊)?\s*[?？。!！\s]*$"
)

#: 抽取出的对象槽里可能混进的**礼貌前缀**。
#:
#: 它们是"对助手的礼貌"，不是"被查询的人"：「请问在哪个部门」懒匹配到的对象是
#: "请问"。剥掉它不改变任何判定（层① 只用"非 None"这个事实），但能让这个槽位
#: 真的可用——一个看起来像人名、其实是客套词的脏值传出去，比空值危险得多，
#: 因为它不报错、只是安静地误导下游。
#:
#: 注意正则本身要**动态拼**：前缀词表是汉语通用的，写死在这里没问题，
#: 但它必须与 ``_POLITE_PREFIX_RE`` 用同一个来源，避免两处漂移。
_POLITE_PREFIXES: Tuple[str, ...] = (
    "请问", "麻烦", "帮我", "帮忙", "想问问", "想问", "想了解", "问一下", "查一下", "查查",
    "看一下", "看下",
)
_POLITE_PREFIX_RE = re.compile(r"^(?:" + _alternation(_POLITE_PREFIXES) + r")+")


def extract_person_attr_slot(query: str, vocab: Vocabulary) -> Optional[str]:
    """抽取「谁 + 哪个属性」问句里的**对象**部分。三态返回，见模块 docstring。"""
    pattern = _person_attr_re(tuple(vocab.attr_words))
    if pattern is None:
        return None
    match = pattern.search(query or "")
    if not match:
        return None
    slot = (match.group("obj") or match.group("obj2") or "").strip()
    return _POLITE_PREFIX_RE.sub("", slot).strip()


def looks_like_person_attr_query(query: str, vocab: Vocabulary) -> bool:
    """是否"在向某个人索取某个属性值"。

    **写成抽取的派生**，由构造保证不可能与 :func:`extract_person_attr_slot` 漂移。
    """
    return extract_person_attr_slot(query, vocab) is not None


#: 属性槽的最大长度。**只约束"反推词表"这条路径**，不约束上面那条正式判据。
#:
#: 反推出的属性词要进 :attr:`Vocabulary.attr_words`，而那个词表是**判据本身**
#: （它一宽，误命中立刻回来）。所以这里取"宁窄勿宽"：超过 4 字的候选一律不要，
#: 因为 5 字以上的片段几乎必然是"属性 + 尾巴"粘在了一起
#: （离线实测：无约束的 ``公司的邮箱怎么申请`` 会抽出 ``邮箱怎么申请`` 这类 6 字垃圾）。
#: 真需要更长的属性词，直接把它写进 ``catalog`` 的 ``attr_words`` 即可。
FREE_ATTR_MAX = 4


@lru_cache(maxsize=1)
def _free_attr_re() -> re.Pattern:
    """同一套句式，但**属性槽不受词表约束**——属性槽的边界由尾巴定界。

    这条正则的用途只有一个：**从例句反推属性词表**（:mod:`app.core.routing.derive`）。
    它天生过宽（``公司的邮箱怎么申请`` 会被抽出属性槽 ``邮箱``），
    所以只允许喂给它**人工挑选过的正例**，绝不能用它做判据。
    自测见 ``tests/test_routing_funnel.py`` 里那条"反推词表不得误命中负样本"的用例。
    """
    return re.compile(
        r"(?:[\u4e00-\u9fa5A-Za-z0-9]{0,8}?\s*(?:是|在|属于)?\s*(?:哪个|哪個|什么|啥)\s*)"
        r"(?P<attr>[\u4e00-\u9fa5A-Za-z0-9]{1," + str(FREE_ATTR_MAX) + r"}?)"
        + _PERSON_ATTR_TAIL
        + r"|(?:[\u4e00-\u9fa5A-Za-z0-9]{1,8}\s*的\s*)"
        r"(?P<attr2>[\u4e00-\u9fa5A-Za-z0-9]{1," + str(FREE_ATTR_MAX) + r"}?)"
        + _PERSON_ATTR_TAIL,
        re.I,
    )


def person_attr_slot_of(query: str) -> Optional[str]:
    """抽出「X 的 <属性>」/「X 在哪个 <属性>」里的**属性槽**（不查词表）。

    返回 ``None`` 表示这句话没有这个句式。**这是反推工具，不是判据**：
    任何"这句话属于某意图"的判断都必须走 :func:`looks_like_person_attr_query`
    （带词表）或词面相似度层，不能拿它代替。
    """
    match = _free_attr_re().search(query or "")
    if not match:
        return None
    return (match.group("attr") or match.group("attr2") or "").strip()


def has_explicit_identifier(query: str, vocab: Vocabulary) -> bool:
    """句子里是否出现了可枚举的标识符（工号 / 邮箱字面量…）。

    格式封闭、可穷举，规则**优于**模型：比模型便宜、稳、可单测。
    """
    pattern = _identifier_re(tuple(vocab.identifier_patterns))
    return bool(pattern and pattern.search(query or ""))


# ---------------------------------------------------------------------------
# guard 判据
#
# guard 是「命中即把**声明它的那个能力**清零」。判据只描述**句式**，
# 至于它否定哪个能力，由 catalog 那边声明 —— 本模块绝不出现意图名。
# ---------------------------------------------------------------------------
#: how-to 问句的标志词：**纯粹的"询问方式"副词**，与领域无关。
#:
#: 刻意只收这一类。"流程""步骤""怎么申请"看着更"像"操作问句，但它们是
#: **领域词**——企业说"流程"，医院说"须知"，学校说"手续"。写在这里就等于
#: 让引擎认识某个业务；它们属于领域词表（``Vocabulary.policy_nouns``）。
#: 一条护栏测试盯着这条界线（``test_engine_modules_contain_no_domain_words``）。
#:
#: 也刻意**不含"在哪"**：它曾把「张三在**哪**个部门」判成 how-to 问句——
#: 取值句与操作句的区分靠的是"属性词是否处于被索取位置"，不是某个疑问词。
_HOWTO_RE = re.compile(r"怎么|如何|怎样|咋办|咋弄|咋整|怎么办")

#: 对比句式标志词：这句话在**比较两个对象**，而不是在索取某一个值。**汉语通用。**
#:
#: 为什么需要它（一个真实的漏判）：「年假和调休有什么区别」曾被判进
#: "查假期余额"那个能力。原因是词面分是 ``Σ len(关键词)``，两个宾语词各自累加，
#: 压过了真正表达提问意图的"区别"——**"提到两个宾语"比"在问两者的关系"得分更高**。
#: 这是加权求和打分的固有特性，靠调权重治不好（权重就是词长，没有可调参数），
#: 只能靠句式事实去否定错的那个意图。
#:
#: 刻意**不含**"分别"：「年假和调休我分别还剩几天」是正当的取值问句，
#: 含进去会把它误清空——一条 guard 反向制造一次误判，比它想修的那个 bug 更难发现。
_COMPARISON_RE = re.compile(
    r"对比|区别|差别|差异|有什么不同|有啥不同|有什么不一样|哪个更|哪个好|相比|冲突|一样吗"
)


def is_howto_query(query: str, vocab: Vocabulary) -> bool:
    """是否是"怎么做 / 流程是什么"这类问句。

    A.4 的判据还要求"主语不是具体人名"。离线层没有 NER，用
    :func:`looks_like_person_attr_query`（属性词处于被索取位置）作为代理——
    它比"出现人名"更严，且方向安全：判不出来最多多走一次语义，不会误清正确项。
    """
    text = query or ""
    if not _HOWTO_RE.search(text):
        return False
    return not looks_like_person_attr_query(text, vocab)


def has_policy_context(query: str, vocab: Vocabulary) -> bool:
    """**已废弃的 guard 名，保留只为让旧配置报错而不是静默失效。**

    第一版用它拦「含规范名词的问题不该被 ``employee_attr`` 抢走」。改成与例句做
    n-gram 相似度之后这件事由**例句本身**完成（负样本就写在它该去的那条能力里），
    于是它成了一张多余的手写名词表。留着它会让"引擎里还藏着企业域的词"这件事
    无法被测试证伪。

    ⚠️ 它现在**恒为 False**：不是"暂时关掉"，而是职责已被取代。
    任何 ``guards`` 里还写着 ``policy_context`` 的目录都能通过启动校验，
    但那个 guard 不再起作用——这条注释就是它唯一的痕迹。
    """
    return False


def has_business_noun(query: str, vocab: Vocabulary) -> bool:
    """句中是否含**任何**业务片段。用于越狱判据的**反向**保护。

    词表来源变了：不再是手写的"业务词"，而是
    :func:`app.core.routing.derive.derive_business_nouns` 从例句反推出的**片段集合**
    （本仓库是 257 个）。所以这里不能拼正则，改用集合求交——
    200+ 分支的正则编译开销与可读性都不可接受，而且每加一句例句都要重编译。

    ⚠️ 这个判据的方向是**宽进**：多认一个片段只会让一次越狱落到层④ 继续判（安全），
    少认一个才会误拦正常业务问题（不可逆）。所以反推规则刻意偏向"宁可多认"。
    """
    if not vocab.business_nouns:
        return False
    return bool(similarity.contains_any(query or "", vocab.business_nouns))


def is_comparison_query(query: str, vocab: Vocabulary) -> bool:
    """是否是"比较两个对象"的问句。

    与 :func:`is_howto_query` 同构：命中标志词还不够，还要求**不是**在向某个人
    索取属性——「张三和李四的工号有什么区别」问的是两个人，不是两条规范。
    """
    text = query or ""
    if not _COMPARISON_RE.search(text):
        return False
    return not looks_like_person_attr_query(text, vocab)


#: guard 名 → 判据函数。函数签名统一为 ``(query, vocab) -> bool``。
#: ``catalog`` 里声明的 guard 名必须在这里能解析到，否则启动期校验直接 raise。
#:
#: ``policy_context`` 从前也在这里，现已是恒假的占位（见 :func:`has_policy_context`）。
#: 保留名字**而不是删掉**：删掉之后，一份还写着它的老目录会在启动期报
#: "guard 未定义"——那条报错说的是"拼错了"，而真实原因是"它被取代了"，
#: 两者该给的提示完全不同。留着它，老目录照常启动，只是那个 guard 不再起作用。
GUARD_FUNCS: Dict[str, Callable[[str, Vocabulary], bool]] = {
    "howto": is_howto_query,
    "policy_context": has_policy_context,
    "comparison": is_comparison_query,
    "business_noun": has_business_noun,
}

#: guard 名闭集。catalog 只认这几个，拼错一个字母就会被启动期校验拦下。
GUARDS: Tuple[str, ...] = tuple(GUARD_FUNCS)
