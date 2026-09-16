"""业务结构化查询工具（SQLite，**全部只读**）—— 工具 Agent 的全部能力。

三个工具
--------
====================================  ====================================
工具                                   作用
====================================  ====================================
``find_employee_by_name(name)``        姓名 → 工号 / 部门
``query_employee_info(employee_id)``   工号 → 部门 / 岗位 / 直属领导 / 入职时间
``query_leave_balance(employee_id)``   工号 → 剩余年假 / 调休
====================================  ====================================

三条贯穿全模块的约定
--------------------

**1. 返回结构化 JSON，不返回长自然语言。**

工具的产物是要喂给**模型**的，不是直接给用户看的。自然语言会带来两个问题：
① 模型必须"读懂"一句话才能取出字段，多一层歧义；② 措辞会被模型复述进最终
答案，等于把措辞的控制权交给了工具实现。返回 JSON 后，"字段是什么"由 schema
固定，"话怎么说"由 L4 生成层决定——职责不重叠。

统一信封（模型据此可靠分支，不需要理解中文措辞）::

    {"ok": true,  "data": {...}}
    {"ok": false, "error": "not_found|invalid_argument",
     "message": "给用户看的简短中文说明"}

**2. 全部只读。** 两张表的写权限只存在于部署期的
``scripts/seed_enterprise_db.py``；运行期连接是 URI ``mode=ro``，
即使本模块被人加了写语句，执行时也会立即失败（见 ``app/db/enterprise_db.py``）。

**3. 失败分两类，处理方式刻意不同。**

这条最容易做错，所以单独说明。判据沿用全项目的同一条：
**「能不能从其他来源得到答案」**。

========================================  ====================================
失败类型                                   处理
========================================  ====================================
业务性结果（查不到 / 参数缺失）            返回 ``ok:false`` 的**结构化信封**
基础设施故障（数据库不可用 / 结构不符）    **向上抛** ``EnterpriseDBError``
========================================  ====================================

为什么基础设施故障要抛而不是也包成信封：吞掉它会让用户看到
「未找到工号 E1001 的员工」——一个**由降级造成的静默错答**。用户会以为
自己或这位同事不存在，而真相是数据库没初始化。这类故障下**没有任何来源
能回答**，所以正确的处置是让工具执行层统一转成「已转接人工」，而不是
伪装成一个业务结论。

（对比：**检索**失败是可以降级的，因为那时 L4 的「知识库中没有找到」与真实语义
一致——同一条判据，不同的结论。检索路径见 ``app/core/sub_agents.py``，
它是独立 Agent 而不是工具。）

为什么每个文件都显式写 ``args_schema``
--------------------------------------
不写类 docstring、参数说明全部写在 ``Field(description=...)``：pydantic v2 会把
类 docstring 渲染成 JSON Schema 的顶层 description 发给模型，而"为什么这样设计"
对模型是纯噪音（完整对照见 ``docs/tool-json-schema.md``）。

参数里的 ``pattern`` 与「参数不合 schema」的处置
------------------------------------------------
``pattern`` 是给模型的**提示**，显著降低"把整句问话当编号"的概率；代价是
非法值会在 ``tool.invoke`` 时被 pydantic 挡下抛 ``ValidationError``。
工具执行层（``tool_agent.execute_tool_calls``）把它归为 ``rejected`` 而**不是**
``error``——参数写错是模型侧问题，不该把用户推给人工。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.db import enterprise_db as db


def _ok(data: Any) -> str:
    """成功信封。``ensure_ascii=False`` 让中文以原样进入模型上下文（省 token）。"""
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


def _fail(error: str, message: str) -> str:
    """失败信封。``error`` 是给模型判断用的稳定枚举，``message`` 是给用户看的。"""
    return json.dumps({"ok": False, "error": error, "message": message}, ensure_ascii=False)


_NOT_FOUND = "not_found"
_INVALID_ARGUMENT = "invalid_argument"


def _require(value: Optional[str], field: str, hint: str) -> Optional[str]:
    """必填参数缺失时返回失败信封，否则返回 None（表示校验通过）。

    规格第 3 条要求「必填参数缺失时模型主动向用户追问，禁止编造参数」。
    schema 的 ``required`` 能挡住"没传这个键"，但挡不住"传了个空串"——
    模型完全可能产出 ``{"name": ""}``。所以工具侧仍要判一次空格。
    """
    if value is None or not str(value).strip():
        return _fail(_INVALID_ARGUMENT, f"缺少必填参数 {field}。{hint}")
    return None


# ===========================================================================
# 工具 1：按姓名查员工
# ===========================================================================
class FindEmployeeByNameArgs(BaseModel):
    name: str = Field(
        ...,
        description=(
            "员工姓名，取值必须来自用户原话中出现的那个名字。"
            "若用户只给了工号，请改用 query_employee_info；"
            "若用户没有指明是谁（如「我的部门」「谁的邮箱」），不要猜测、不要编造，"
            "直接不调用本工具，先向用户询问姓名或工号。"
        ),
    )


@tool(
    args_schema=FindEmployeeByNameArgs,
    description=(
        "按姓名查询员工，返回工号、姓名与部门。"
        "仅当用户说出了具体姓名时使用；若用户只给了工号，请改用 query_employee_info。"
        "重名时返回全部候选人，需向用户确认是哪一位。"
    ),
)
def find_employee_by_name(name: str) -> str:
    """按姓名查询员工，返回工号、姓名与部门。

    ``description`` 显式传入、不用 docstring：docstring 里写的是**给维护者看的
    理由**（为什么只返回三列），而它会被 LangChain 原样发给模型——对模型是纯噪音，
    且白白占掉每次请求的 token。同一个理由（pydantic 的类 docstring 也会泄漏成
    schema 的顶层 description）已经写在 ``args_schema`` 上，这里保持一致。
    完整对照见 ``docs/tool-json-schema.md``。

    只返回工号 / 姓名 / 部门三列：工号是后续查询的钥匙，但它不是"查得到更多"的
    理由——多返回的字段会进入模型上下文，既浪费 token，也扩大 PII 暴露面。
    重名时返回**多条**（如两位「王五」），由模型向用户追问是哪一位，
    而不是替用户在两条记录里挑一个。
    """
    invalid = _require(name, "name", "请提供要查询的员工姓名。")
    if invalid:
        return invalid
    # 不加 try/except：库不可用属于**基础设施故障**，让它抛到工具执行层统一处置
    # （转人工）。吞掉它就会退化成「未找到这位员工」——一个静默错答。
    # 判据见模块 docstring 第 3 条。
    rows = db.find_employee_by_name(name)

    if not rows:
        return _fail(_NOT_FOUND, f"未找到姓名为「{name.strip()}」的员工，请核对姓名后重试。")
    if len(rows) > 1:
        return _ok({
            "ambiguous": True,
            "candidates": rows,
            "note": "存在多名同名员工，请向用户确认是哪一位（可提供部门辅助区分）。",
        })
    return _ok({"ambiguous": False, "candidates": rows})


# ===========================================================================
# 工具 2：按工号查员工基础信息
# ===========================================================================
class EmployeeInfoArgs(BaseModel):
    employee_id: str = Field(
        ...,
        description=(
            "员工工号，形如 E1001（字母 E + 至少 3 位数字）。"
            "取值来源有二：用户原话中直接给出的工号，或 find_employee_by_name "
            "返回的 employee_id。**不要自行编造工号。**"
        ),
        pattern=r"^[Ee]\d{3,}$",
    )


@tool(
    args_schema=EmployeeInfoArgs,
    description="按工号查询员工基础信息：部门、岗位、直属领导、入职时间。",
)
def query_employee_info(employee_id: str) -> str:
    """按工号查询员工基础信息：部门、岗位、直属领导、入职时间。

    ``description`` 显式传入的理由见 ``find_employee_by_name`` 的说明。
    """
    invalid = _require(employee_id, "employee_id", "请提供员工工号（形如 E1001）。")
    if invalid:
        return invalid
    # 库不可用 → 抛出（见模块 docstring 第 3 条），由工具执行层转人工
    row = db.query_employee_info(employee_id)

    if not row:
        return _fail(_NOT_FOUND, f"未找到工号为「{employee_id.strip()}」的员工。")
    return _ok(row)


# ===========================================================================
# 工具 3：按工号查剩余假期
# ===========================================================================
class LeaveBalanceArgs(BaseModel):
    employee_id: str = Field(
        ...,
        description=(
            "员工工号，形如 E1001（字母 E + 至少 3 位数字）。"
            "取用户原话中给出的工号，或 find_employee_by_name 返回的 employee_id。"
            "**不要自行编造工号。**"
        ),
        pattern=r"^[Ee]\d{3,}$",
    )


@tool(
    args_schema=LeaveBalanceArgs,
    description=(
        "按工号查询剩余年假与调休天数。只回答这两项，"
        "不要推算总天数或已用天数。"
    ),
)
def query_leave_balance(employee_id: str) -> str:
    """按工号查询剩余年假与调休天数。

    ``description`` 显式传入的理由见 ``find_employee_by_name`` 的说明。

    刻意**不**返回"总天数"或"已用天数"：库里没有这两列，任何推算都可能是错的
    （比如年中入职的按比例折算规则）。宁可只答被问到的部分，也不给一个看起来
    合理、实际错误的数字。
    """
    invalid = _require(employee_id, "employee_id", "请提供员工工号（形如 E1001）。")
    if invalid:
        return invalid
    # 库不可用 → 抛出（见模块 docstring 第 3 条）
    row = db.query_leave_balance(employee_id)

    if not row:
        # 「无记录」与「余额为 0」必须区分：E1006 是前者。若这里返回 0，
        # 用户会以为自己的假被清零了——一个由"贴心默认值"造成的严重误导。
        return _fail(
            _NOT_FOUND,
            f"未查询到工号「{employee_id.strip()}」的假期记录。"
            "这不代表余额为 0，可能是记录尚未建立，建议联系 HR 核实。",
        )
    return _ok(row)


#: 工具 Agent 的全部能力（顺序即 schema 中的呈现顺序）。
#:
#: **只有这 3 个，且全部只读。** 制度类问题不在这里——它们由简单/复杂 RAG
#: Agent 负责（见 ``app/core/sub_agents.py``）。把两类能力混在一个工具列表里，
#: 会让模型在"查制度"与"查业务数据"之间做一次没有必要的选择。
TOOL_AGENT_TOOLS: tuple = (
    find_employee_by_name,
    query_employee_info,
    query_leave_balance,
)

TOOLS_BY_NAME: Dict[str, Any] = {t.name: t for t in TOOL_AGENT_TOOLS}
