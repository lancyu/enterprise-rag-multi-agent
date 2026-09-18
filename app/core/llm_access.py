"""调用聊天模型的公共管道：取默认模型 + 读取模型输出的文本。

为什么单独成模块
----------------
这两个函数此前在 **4 个模块里各抄了一遍**（``tool_agent`` / ``sub_agents`` /
``router_agent`` / ``routing.arbitration``），共 8 份，函数体 sha1 完全相同。
它们总是成对出现——先 ``_default_model()`` 拿到模型，再用 ``_content_of()`` 读回
它吐的文本——所以本质是同一件事的两半，放在一起而不是各建一个模块。

抄 4 遍的代价不是多几行，而是**改一处忘一处**：真要支持新的 content 形态
（如 LangChain 将来新增的块类型），只改了走得通的那条链路，另外三条会静默地
把回复读成空字符串——而空字符串在上层表现为"模型没说话"，不报错。

为什么放在 ``app/core/``
------------------------
消费方全部在 ``core`` 内。``default_model`` 需要一个**惰性** import
（``app/providers/llm.py`` 反向依赖 ``app.core.observability``，在模块顶部直接
import 会把这层反向依赖提前到导入期），故函数体内 import，与收敛前的行为一致。
"""
from typing import Any


def content_of(message: Any) -> str:
    """取出消息的纯文本内容。

    ``content`` 可能是分块列表（LangChain 新式多模态消息），需把其中的文本块
    拼接起来；非文本块（图片等）忽略。
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return ""


def default_model():
    """取进程默认的聊天模型。

    惰性 import：``app/providers/llm.py`` 会 import ``app.core.observability``，
    在模块顶部直接 import 会把这条反向依赖提前到导入期。另外，取模型的时机交给
    调用方也更安全——``get_chat_model()`` 自身带实例缓存，重复调用不建新连接。
    """
    from app.providers.llm import get_chat_model

    return get_chat_model()
