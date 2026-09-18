"""统一异常基类 + 对外故障描述的唯一构造处。

设计来源：ragas `exceptions.py`。调用方只需 ``raise KnowledgeBaseEmpty()``，
不必每次重复写错误文案；业务层统一捕获 ``AppError`` 即可。
相比 dify 的 ``description`` 类属性，内置默认文案进一步省掉调用方的文案代码。

对外口径（改本模块之前先读完这一段）
------------------------------------
本项目对「出错时怎么跟调用方说」只有一条规则：

    响应体里**不出现任何异常信息**。故障描述只有一种形态 ——
    ``public_detail(前缀)``，即「固定前缀 + 本次请求的 trace_id」。

理由是异常字符串常含文件路径、内网地址、供应商返回体、依赖版本等内部细节，
直接下发等于把内部拓扑交给匿名调用方；而用户真正需要的是一个**能被客服检索的
凭证**，trace_id 正是那个凭证（完整堆栈已由 ``logger.exception`` 落盘）。

为什么必须有唯一实现处
----------------------
同一次故障在两条链路上曾经给出两种口径：``/chat/ask`` 只回 trace_id，
``/chat/ask/stream`` 回显 ``str(exc)``。谁都不会报错，但同一个问题在浏览器里
能看到文件路径、用 curl 看不到 —— 这类分歧只在事故复盘时才会被发现。
把「怎么对外说」收成一个函数，分歧就没有落脚点。

边界层的配套约束（护栏 ``tests/test_error_boundary.py`` 盯着它）
---------------------------------------------------------------
``app/api/`` 与 ``app/main.py`` 是唯一的外发出口，故这两处**不接触异常对象**：
``except`` 不把异常绑到名字上（``logger.exception`` 不需要变量），
需要 AppError 的自带文案时用 ``exc.message`` 而不是 ``str(exc)``。
理由同上 —— 只要异常对象到了边界层，就一定会有人顺手把它拼进响应里。
"""
from app.core.tracing import get_trace_id


def trace_ref() -> str:
    """本次请求对外的 trace 凭证。

    无 trace 时返回 ``"-"`` 而**不是空串**：空串会让客户端渲染出
    「trace_id: 」这种像是漏了字段的样子，诱导用户去报一个不存在的凭证。
    """
    return get_trace_id() or "-"


def public_detail(prefix: str) -> str:
    """把内部故障翻译成可对外的故障描述：固定前缀 + trace 凭证。

    唯一合法用法是 ``public_detail("评测失败")``。**不要**把异常拼进
    ``prefix`` —— 本函数存在的意义就是让那件事没有地方写。
    """
    return f"{prefix}（trace_id: {trace_ref()}）"


class AppError(Exception):
    """项目异常基类。业务层只捕这一个，即可覆盖所有可预期错误。"""

    default_message = "服务内部错误"

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class KnowledgeBaseEmpty(AppError):
    default_message = "知识库为空，请先上传文档"


class RateLimitExceeded(AppError):
    default_message = "请求过于频繁，请稍后再试"


class EvalCaseInvalid(AppError):
    default_message = "评测用例格式不合法"
