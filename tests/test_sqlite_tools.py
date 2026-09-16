"""3 个只读业务工具的回归测试 —— 契约、边界情形与只读性。

本文件守三件事，都是"不测就会静默出错"的：

1. **对外契约**：JSON 信封的字段名与取值枚举（``ok`` / ``data`` / ``error`` /
   ``message``）。模型是按信封分流的，字段名改了它不会报错，只会**判错分支**。
2. **边界情形**：重名、无记录、查无此人。三种"不成功"的输入必须给出
   **三种不同**的回答——把它们混成一句"查不到"是本项目最反对的静默错答。
3. **只读性**：不是"读代码确认没写 INSERT"，而是**真的写一次**看是否被拒绝。

数据来自 ``tests/conftest.py::business_db``（独立临时库，不读 ``data/enterprise.db``）。
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from app.tools.sqlite_tools import (
    TOOL_AGENT_TOOLS,
    TOOLS_BY_NAME,
    find_employee_by_name,
    query_employee_info,
    query_leave_balance,
)


def _payload(raw: str) -> dict:
    """工具返回的必须是可解析的 JSON —— 这是"返回结构化数据"的底线。"""
    data = json.loads(raw)
    assert isinstance(data, dict)
    return data


# ===========================================================================
# 1. 只读性（三层强制的第三层）
# ===========================================================================
def test_database_rejects_writes(business_db):
    """数据库层**真的**拒绝写操作 —— 不是靠代码自觉。

    ``app/db/enterprise_db.py`` 用 URI ``file:...?mode=ro`` 打开连接，
    写权限只在部署期的 seed 脚本里存在。这条测试守的是"有人把 mode=ro 去掉"：
    那种改动不会让任何功能失败，只会让越权写入成为可能，**且没人会发现**。
    """
    from app.db import enterprise_db as db

    with pytest.raises(sqlite3.OperationalError):
        # 只读性验证必须走**真实连接**，否则验证的是另一条连接串。
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO employee VALUES ('X9999','黑客','x','x',NULL,'2024-01-01')"
            )

    assert db.is_read_only() is True


def test_tools_module_contains_no_write_statements():
    """代码层：工具模块里不得出现任何数据修改语句。

    这是对上一层的补充而非替代：``mode=ro`` 会在运行时挡住写入，
    但一句写语句混在只读代码里仍然说明设计被破坏了，值得直接失败。
    """
    from pathlib import Path

    import app.tools.sqlite_tools as mod

    source = Path(mod.__file__).read_text(encoding="utf-8").upper()
    for keyword in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE TABLE"):
        assert keyword not in source, f"只读工具模块里出现了写语句：{keyword}"


# ===========================================================================
# 2. 工具签名与 JSON Schema（规格的直接对应物）
# ===========================================================================
def test_three_tools_with_exact_names():
    """规格规定了 3 个工具且只有 3 个 —— 数量或名字变了必须立刻可见。"""
    assert [t.name for t in TOOL_AGENT_TOOLS] == [
        "find_employee_by_name",
        "query_employee_info",
        "query_leave_balance",
    ]


@pytest.mark.parametrize(
    "tool_name,expected_props,expected_required",
    [
        ("find_employee_by_name", ["name"], ["name"]),
        ("query_employee_info", ["employee_id"], ["employee_id"]),
        ("query_leave_balance", ["employee_id"], ["employee_id"]),
    ],
)
def test_schema_matches_spec(tool_name, expected_props, expected_required):
    """参数名与必填项必须与规格逐字一致 —— 模型看到的就是这份 schema。"""
    schema = TOOLS_BY_NAME[tool_name].args_schema.model_json_schema()

    assert sorted(schema["properties"]) == sorted(expected_props)
    assert sorted(schema.get("required", [])) == sorted(expected_required)
    for name in expected_props:
        assert schema["properties"][name].get("description"), f"{tool_name}.{name} 缺少描述"


def test_authorization_facts_are_not_tool_parameters():
    """``allowed_sources`` 这类授权事实**绝不能**出现在 schema 里。

    它是服务端鉴权层写入的（``request_ctx.set_allowed_sources``）。做成工具参数
    等于让模型决定"我能查谁的资料"——那是合规事故，不是功能问题。

    注：``employee_id`` 是**唯一**出现在 schema 里的标识类参数，且它只表示
    "用户想查谁的记录"（业务意图），不表示"我是谁"（授权事实）。没有登录态之后，
    这个区分必须靠"工号只能来自用户原话或工具返回值"来维持，见
    ``app/core/tool_agent.py`` 的 ``GROUNDED_ARGS``。
    """
    for tool_obj in TOOL_AGENT_TOOLS:
        props = tool_obj.args_schema.model_json_schema()["properties"]
        for forbidden in ("allowed_sources", "current_employee_id", "user_id"):
            assert forbidden not in props, f"{tool_obj.name} 把授权参数暴露给了模型"


# ===========================================================================
# 3. 工具 1：find_employee_by_name
# ===========================================================================
def test_find_employee_hit(business_db):
    data = _payload(find_employee_by_name.invoke({"name": "张三"}))

    assert data["ok"] is True
    assert data["data"]["ambiguous"] is False
    assert data["data"]["candidates"][0]["employee_id"] == "E1001"
    assert data["data"]["candidates"][0]["department"] == "技术部"
    # 只返回约定的三列：多返回的字段会扩大 PII 暴露面
    assert sorted(data["data"]["candidates"][0]) == ["department", "employee_id", "name"]


def test_find_employee_ambiguous_returns_all_candidates(business_db):
    """重名返回全部候选人，并要求模型向用户确认 —— **不得替用户挑一个**。

    库里有两个「王五」（E1003 运营部 / E1007 法务部）。挑错的后果不是"答得不好"，
    而是把**另一个人的**信息交给了用户，且看不出来。
    """
    data = _payload(find_employee_by_name.invoke({"name": "王五"}))

    assert data["ok"] is True
    assert data["data"]["ambiguous"] is True
    assert {c["employee_id"] for c in data["data"]["candidates"]} == {"E1003", "E1007"}
    assert "确认" in data["data"]["note"]


def test_find_employee_not_found(business_db):
    """查无此人 → ``not_found``，既不是语法错误、也不是空成功。"""
    data = _payload(find_employee_by_name.invoke({"name": "查无此人"}))

    assert data["ok"] is False
    assert data["error"] == "not_found"
    assert "查无此人" in data["message"]


def test_find_employee_partial_name_does_not_match(business_db):
    """姓名用**精确**匹配，不做前缀/模糊匹配。

    模糊匹配会让「王」命中「王五」，把另一个人的工号交到模型手里——
    而工号是访问其余数据的钥匙。宁可查不到。
    """
    data = _payload(find_employee_by_name.invoke({"name": "王"}))

    assert data["ok"] is False
    assert data["error"] == "not_found"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_required_argument_is_rejected_by_tool(business_db, blank):
    """schema 的 ``required`` 挡不住空串 —— 工具必须自己判一次。

    规格第 3 条要求"必填参数缺失时模型主动向用户追问"。若工具把空串当成合法
    输入，它会一路查到数据库、回一句"查无此人"——用户会以为自己问的那个人
    不存在，而真相是模型没填参数。
    """
    data = _payload(find_employee_by_name.invoke({"name": blank}))

    assert data["ok"] is False
    assert data["error"] == "invalid_argument"


# ===========================================================================
# 4. 工具 2：query_employee_info
# ===========================================================================
def test_query_employee_info_hit(business_db):
    data = _payload(query_employee_info.invoke({"employee_id": "E1001"}))

    assert data["ok"] is True
    assert data["data"]["department"] == "技术部"
    assert data["data"]["position"] == "高级工程师"
    assert data["data"]["leader_id"] == "E1009"
    assert data["data"]["entry_date"] == "2019-03-11"


def test_query_employee_info_not_found(business_db):
    data = _payload(query_employee_info.invoke({"employee_id": "E9999"}))

    assert data["ok"] is False
    assert data["error"] == "not_found"


@pytest.mark.parametrize("bad", ["1001", "E10", "张三", "  ", "E1001的" ])
def test_employee_id_pattern_rejects_non_ids(business_db, bad):
    """工号必须匹配 ``^[Ee]\\d{3,}$`` —— 非法值抛 ``ValidationError``。

    这是**给模型的提示**而不是给用户的错误：执行层会把它归为 ``rejected``
    （模型侧问题），让模型在下一轮自我纠正，而不是把用户推给人工。
    """
    with pytest.raises(ValidationError):
        query_employee_info.invoke({"employee_id": bad})


# ===========================================================================
# 5. 工具 3：query_leave_balance
# ===========================================================================
def test_query_leave_balance_hit(business_db):
    data = _payload(query_leave_balance.invoke({"employee_id": "E1001"}))

    assert data["ok"] is True
    assert data["data"]["annual_leave"] == 5.0
    assert data["data"]["compensatory_leave"] == 2.0


def test_missing_leave_record_is_not_zero(business_db):
    """**「无记录」≠「余额为 0」** —— 本工具最容易被做错的一处。

    E1006 有工号但 ``leave_balance`` 里没有记录。若这里返回
    ``{"annual_leave": 0, "compensatory_leave": 0}``，用户会以为自己的假被清零了；
    而正确的话术必须**明说**这不代表 0，并给出下一步（找 HR 核实）。
    """
    data = _payload(query_leave_balance.invoke({"employee_id": "E1006"}))

    assert data["ok"] is False
    assert data["error"] == "not_found"
    assert "不代表" in data["message"] and "0" in data["message"]
    assert "HR" in data["message"]


# ===========================================================================
# 6. 基础设施故障必须**上抛**，不得伪装成业务结论
# ===========================================================================
def test_database_unavailable_raises_instead_of_returning_not_found(tmp_path, monkeypatch):
    """库不存在时必须抛 ``EnterpriseDBError``，**不能**返回 ``not_found``。

    这是本项目判据「能不能从其他来源得到答案」的直接体现：把
    "数据库没初始化"伪装成"没有这位员工"，用户会以为自己或这位同事不存在。
    上抛之后由工具执行层统一转人工，用户看到的是可行动的信息。
    """
    from app.db.enterprise_db import EnterpriseDBError

    monkeypatch.setattr("app.config.SQLITE_DB_PATH", str(tmp_path / "missing.db"))

    with pytest.raises(EnterpriseDBError):
        query_employee_info.invoke({"employee_id": "E1001"})
    with pytest.raises(EnterpriseDBError):
        find_employee_by_name.invoke({"name": "张三"})
