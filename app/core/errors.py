"""统一异常基类 —— 单一基类 + 子类内置默认文案。

设计来源：ragas `exceptions.py`。调用方只需 ``raise KnowledgeBaseEmpty()``，
不必每次重复写错误文案；业务层统一捕获 ``AppError`` 即可。
相比 dify 的 ``description`` 类属性，内置默认文案进一步省掉调用方的文案代码。
"""


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
