"""自检项（``app/core/self_check.py``）的回归测试。

**回归背景**：``_check_sqlite_db`` 里那条「3 个工具能构建参数 schema」的判据
曾写成 ``tool.args.get("properties")``。而 LangChain 的 ``StructuredTool.args``
返回的是**属性映射本身**（形如 ``{"name": {...}}``），并不含 ``properties`` 键——
于是断言恒为假，该项在工具完全正常时也**永远判失败**。

这类故障的危险之处在于它**不是崩溃，而是一条长期假警报**：自检一直显示一条红色，
排查者会去翻工具代码（那里是对的），真正的故障反而被这条噪声淹没。
因此本文件同时守住两个方向——正常时必须 pass，真的没有参数时必须 fail。
"""
from __future__ import annotations

import pytest


def test_tool_args_is_a_properties_mapping_not_a_json_schema():
    """钉住 LangChain 的接口约定，避免判据再次被写成 ``.get("properties")``。

    ``StructuredTool.args`` 等价于 JSON Schema 的 ``properties`` 段本身，
    所以「工具声明了参数」等价于「这个 dict 非空」，而不是「它含 properties 键」。
    """
    from app.core.tool_agent import AGENT_TOOLS

    assert AGENT_TOOLS, "工具清单为空时本用例失去意义"
    for tool in AGENT_TOOLS:
        args = tool.args
        assert isinstance(args, dict) and args, f"{tool.name} 应声明至少一个参数"
        # 值形如 {"type": ..., "description": ...}，这是「属性映射」的形态特征
        assert all(isinstance(spec, dict) and "type" in spec for spec in args.values()), (
            f"{tool.name}.args 的形态不是属性映射：{args}"
        )


def test_sqlite_db_check_passes_when_tools_are_healthy(business_db):
    """把判据退回 ``args.get("properties")``，本用例即变红。"""
    from app.core.self_check import _check_sqlite_db

    info = _check_sqlite_db()

    assert info["detail"], "自检项应给出可读的 detail"
    assert info["extra"]["read_only"] is True, "只读约束必须真的生效"


def test_sqlite_db_check_still_fails_on_a_tool_without_parameters(business_db, monkeypatch):
    """非空转证明：面对「真的没有参数」的工具，本项仍必须报错。

    只有与前一条配对，才能说明这条护栏**两个方向**都有效——
    否则把它改成恒真也会让上一条通过。
    """
    from types import SimpleNamespace

    import app.core.tool_agent as tool_agent
    from app.core.self_check import _check_sqlite_db

    blank = SimpleNamespace(name="no_param_tool", args={})
    monkeypatch.setattr(tool_agent, "AGENT_TOOLS", [*tool_agent.AGENT_TOOLS, blank])

    with pytest.raises(RuntimeError, match="未能构建参数 schema"):
        _check_sqlite_db()
