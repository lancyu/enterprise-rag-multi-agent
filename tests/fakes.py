"""测试替身 —— 不联网、不消耗配额，但**真实走** function calling 的结构。

为什么需要一个"假模型"而不是用 MockChatModel
--------------------------------------------
``MockChatModel``（``app/providers/llm.py``）刻意**不实现 bind_tools**：它的定位是
"完全离线时跑通链路"，不是"模拟智能"。若让它用规则假装抽参，测试会以为
function calling 验证过了，实际验证的只是 Mock 里的规则——真正要测的接入被掩盖。

所以工具链路的回归测试需要另一种替身：**结构上真实**（返回标准
``AIMessage.tool_calls``、经 ``bind_tools`` 绑定）但**行为上可编排**（调什么由
测试指定）。本模块提供这个替身。

与 LangChain 类型约束的关系
---------------------------
``bind_tools`` 在真实模型上返回 ``RunnableBinding``，其 ``bound`` 字段有 pydantic
类型约束（必须是 ``Runnable`` 子类）。这里返回的 ``_RecordingBinding`` 刻意**不继承**
``Runnable``: 它只被本项目的代码直接 ``.invoke()``，不参与 LangChain 的组合与校验。
一旦哪天需要把它塞进 chain，就会立刻在构造处报错——**这是好事**，
说明约束在生效，而不是被一个宽松的替身糊过去。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage


class RecordingModel:
    """按脚本返回工具调用（或纯文本）的假模型。

    脚本每一项的形态决定这一轮返回什么，三种写法都常用::

        RecordingModel([{"name": "search_knowledge", "args": {"query": "年假"}}])
            → 单次工具调用（**最常用**）

        RecordingModel([[{...}, {...}]])
            → 同一轮并行发起多个工具调用

        RecordingModel(["这是直答内容"])
            → 纯文本，tool_calls 为空 → 走直答出口

        RecordingModel([[]])
            → 明确表达"本轮不调工具且不产出内容"

    每调用一次 ``invoke`` 消费脚本中的一项；脚本耗尽后重复最后一项
    （便于"多轮都返回同一个结果"的写法）。
    ``calls`` 记录每次实际收到的消息列表，供断言"对话结构是否成对"。
    """

    def __init__(self, script: List[Any], *, bind_error: Optional[Exception] = None) -> None:
        self.script = list(script) if script else [[]]
        self.bind_error = bind_error
        self.calls: List[List[Any]] = []
        self.bound_tools: Optional[List[Any]] = None

    # -- 供被测代码调用 ------------------------------------------------------
    def bind_tools(self, tools):
        if self.bind_error is not None:
            raise self.bind_error
        self.bound_tools = list(tools)
        return _RecordingBinding(self)


def _normalize_calls(item: Any) -> List[Dict[str, Any]]:
    """把脚本项规整成 tool_call 列表。

    ``dict`` 与 ``list[dict]`` 都要接受：前者是「一次调用」的直觉写法，
    若不特判会被 ``list(dict)`` 拆成键名列表，报出 ``string indices must be
    integers`` —— 一个与真实错误毫无关系的报错，排查时会浪费很多时间。
    """
    if isinstance(item, dict):
        return [item]
    return [c for c in (item or []) if isinstance(c, dict)]


class _RecordingBinding:
    def __init__(self, model: RecordingModel) -> None:
        self._model = model

    def invoke(self, messages, **_kwargs):
        model = self._model
        model.calls.append(list(messages))
        index = min(len(model.calls) - 1, len(model.script) - 1)
        item = model.script[index]
        if isinstance(item, str):
            # 纯文本：工具调用为空 → 走直答出口
            return AIMessage(content=item)
        calls = _normalize_calls(item)
        if not calls:
            return AIMessage(content="")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": c["name"],
                    "args": c.get("args") or {},
                    "id": c.get("id") or f"call_{index}_{i}",
                    "type": "tool_call",
                }
                for i, c in enumerate(calls)
            ],
        )
