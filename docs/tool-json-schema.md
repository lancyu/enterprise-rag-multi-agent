# 工具 Agent：3 个只读工具的实现与 Function Calling JSON Schema

> 本文是**规格交付物**：3 个工具的 Python 实现（SQLite 交互）与它们对应的完整
> Function Calling JSON Schema。除此之外还写清了 schema 里每一处的**取舍理由**
> ——schema 是模型能看到的**全部输入**，改一个字段描述就是改行为，因此它需要一份
> 可评审的说明。
>
> 对应代码：`app/tools/sqlite_tools.py`（工具层）、`app/db/enterprise_db.py`（数据层）、
> `scripts/seed_enterprise_db.py`（建表与演示数据）。
> 回归用例：`tests/test_sqlite_tools.py`（23 条）、`tests/test_tool_agent.py`（40 条）。
>
> **服务端没有登录态。** 工号只能来自用户原话或工具返回值，没有任何"当前登录用户"
> 可供对账——因此本文不含任何权限校验章节。历史上曾有一个带内置权限校验的
> `query_work_order` 工具与配套的会话身份链，已整条移除，存档见
> `_archive/removed-workorder-and-identity-20260915/`。

---

## 一、交付清单与三条硬约束

| # | 工具 | 作用 | 必填参数 |
|---|---|---|---|
| 1 | `find_employee_by_name` | 姓名 → 工号 / 姓名 / 部门 | `name` |
| 2 | `query_employee_info` | 工号 → 部门 / 岗位 / 直属领导 / 入职时间 | `employee_id` |
| 3 | `query_leave_balance` | 工号 → 剩余年假 / 调休天数 | `employee_id` |

规格给出的规则，在本项目里各自落到一个**结构性**位置（而不是靠自觉）：

| 规格要求 | 落地方式 | 被哪条测试钉住 |
|---|---|---|
| ① 职责单一、全部只读 | 连接层 URI `mode=ro`（SQLite 自己拒绝写）+ 代码层只有 `SELECT` + 测试层真的尝试写入并断言失败 | `test_database_rejects_writes`、`test_tools_module_contains_no_write_statements` |
| ② 返回结构化 json，不返回长自然语言 | 统一信封 `{"ok":true,"data":{...}}` / `{"ok":false,"error":…,"message":…}` | `test_*_hit` 系列 |
| ③ 参数缺失时追问、禁止编造 | schema 的 `required` + 工具侧空格校验（挡 `{"name":""}`）+ `GROUNDED_ARGS` 落地校验（参数必须能在用户原句里定位） | `test_blank_required_argument_is_rejected_by_tool`、`test_grounding_guard` |
| ④ 不许模型自己决定"能查谁" | 3 个工具的参数里**没有**任何授权类字段；`employee_id` 只是"用户想查谁的记录"这一业务意图，不代表"我是谁" | `test_authorization_facts_are_not_tool_parameters` |

**「全部只读」不是靠约定，而是靠三层强制。** 约定挡不住后来者：写一句 `INSERT`
与写一句 `SELECT` 在代码审查里长得一样。`mode=ro` 让越界在**运行时立即失败**
（`attempt to write a readonly database`），测试层则防止有人为了"图方便"
把第 1 层换成普通连接。

---

## 二、数据库

### 2.1 表结构（字段与规格严格一致，不增不减）

```sql
CREATE TABLE IF NOT EXISTS employee (
    employee_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL,
    position    TEXT NOT NULL,
    leader_id   TEXT,
    entry_date  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_balance (
    employee_id        TEXT PRIMARY KEY,
    annual_leave       REAL NOT NULL,
    compensatory_leave REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id)
);
```

初始化（**部署期**动作，幂等）：

```bash
PYTHONPATH=. .venv/bin/python scripts/seed_enterprise_db.py         # 建表 + 写入演示数据
PYTHONPATH=. .venv/bin/python scripts/seed_enterprise_db.py --check # 只校验结构
```

路径由 `SQLITE_DB_PATH` 决定，默认 `./data/enterprise.db`。

> 为什么建表在**脚本**里而不是应用启动时自动建：自动建表会让"数据库是空的"
> 变得不可见——表建好了、查也查得动，只是返回空。于是「查不到这位员工」与
> 「数据没导入」在调用方看来完全同形，而前者是业务结论、后者是部署事故。

### 2.2 演示数据里刻意造的两组边界

数据不是随便填的，每一组都对应一条必须被区分开的语义：

| 边界情形 | 数据 | 它防的是什么 |
|---|---|---|
| **重名** | `E1003 王五`（运营部）与 `E1007 王五`（法务部） | 工具必须返回**两个**候选人、由模型向用户确认；替用户挑一个 = 返回另一个人的工号 |
| **有工号但无假期记录** | `E1006 周八` 在 `leave_balance` 里**没有行** | 「记录缺失」≠「余额为 0」。回 0 会让用户以为自己的假被清零了 |

`employee` 共 9 行、`leave_balance` 共 8 行——差的正是 E1006 这一条，
不是漏导数据，而是刻意留的边界样本。

---

## 三、数据层：SQLite 交互

工具层不直接写 SQL，只调用数据层函数。**分开的理由**是让"取数"与"怎么组织返回"
各自单一：数据层只做"按主键取一行"这类纯操作，任何业务判断（查不到该怎么说、
重名该怎么办）都留在工具层。

### 3.1 连接与查询原语

```python
"""app/db/enterprise_db.py（节选）"""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import config


class EnterpriseDBError(RuntimeError):
    """业务库不可用（文件缺失 / 结构不符 / 查询失败）。"""


def db_path() -> Path:
    """数据库文件路径（配置项 ``SQLITE_DB_PATH``）。"""
    return Path(config.SQLITE_DB_PATH)


def _connect() -> sqlite3.Connection:
    """打开**只读**连接。文件不存在时抛 EnterpriseDBError。"""
    path = db_path()
    if not path.exists():
        raise EnterpriseDBError(
            f"业务数据库不存在：{path}。请先执行 `python scripts/seed_enterprise_db.py` 初始化。"
        )
    # mode=ro 由 SQLite 自身执行——写操作会在执行期失败，而不是靠调用方自觉。
    uri = f"file:{path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as exc:  # pragma: no cover - 环境问题
        raise EnterpriseDBError(f"无法打开业务数据库：{exc}") from exc
    conn.row_factory = sqlite3.Row
    return conn


def _query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """执行只读查询，返回 dict 列表。异常一律转成 EnterpriseDBError。"""
    try:
        with _connect() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise EnterpriseDBError(f"业务库查询失败：{exc}") from exc
    return [dict(row) for row in rows]
```

两处刻意的设计：

- **每次调用都新建连接**，不缓存。SQLite 的连接对象**不可跨线程共享**，而
  FastAPI 的同步端点跑在线程池里——缓存连接会得到"偶发 `ProgrammingError`"
  这类最难查的 bug。打开本地文件是微秒级，SQLite 有页缓存，代价可忽略。
- **所有取值走 `?` 占位符**。工具的参数来自模型，而模型可能被诱导产出任意
  字符串——拼接 SQL 就是典型的注入入口。

### 3.2 三个查询函数

```python
def find_employee_by_name(name: str) -> List[Dict[str, Any]]:
    """按姓名查员工（可能重名，故返回列表）。

    姓名用 ``=`` 精确匹配而不是 ``LIKE``：模糊匹配会让「王」命中「王小明」，
    从而把**另一个人的工号**交到模型手里；而工号是本项目里访问其余数据的钥匙，
    错发的代价远大于"查不到"。
    """
    return _query(
        "SELECT employee_id, name, department FROM employee WHERE name = ? LIMIT 20",
        (name.strip(),),
    )


def query_employee_info(employee_id: str) -> Optional[Dict[str, Any]]:
    """按工号查员工基础信息（部门 / 岗位 / 直属领导 / 入职时间）。"""
    rows = _query(
        "SELECT employee_id, name, department, position, leader_id, entry_date "
        "FROM employee WHERE employee_id = ? LIMIT 1",
        (employee_id.strip(),),
    )
    return rows[0] if rows else None


def query_leave_balance(employee_id: str) -> Optional[Dict[str, Any]]:
    """按工号查剩余年假与调休天数。"""
    rows = _query(
        "SELECT employee_id, annual_leave, compensatory_leave "
        "FROM leave_balance WHERE employee_id = ? LIMIT 1",
        (employee_id.strip(),),
    )
    return rows[0] if rows else None
```

---

## 四、工具层：三个工具函数

### 4.1 公共约定

```python
"""app/tools/sqlite_tools.py（节选）"""
import json
from typing import Any, Optional

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

    schema 的 ``required`` 能挡住"没传这个键"，但挡不住"传了个空串"——
    模型完全可能产出 ``{"name": ""}``。所以工具侧仍要判一次空格。
    """
    if value is None or not str(value).strip():
        return _fail(_INVALID_ARGUMENT, f"缺少必填参数 {field}。{hint}")
    return None
```

统一信封（模型据此可靠分支，不需要理解中文措辞）：

```json
{"ok": true,  "data": {...}}
{"ok": false, "error": "not_found|invalid_argument",
 "message": "给用户看的简短中文说明"}
```

**失败分两类，处理方式刻意不同**（判据沿用全项目那条：**能不能从其他来源得到答案**）：

| 失败类型 | 处理 | 为什么 |
|---|---|---|
| 业务性结果：查不到 / 参数缺失 | 返回 `ok:false` 的**结构化信封** | 换个名字、补个参数就能答——这是可纠正的输入问题 |
| 基础设施故障：库不可用 / 结构不符 | **向上抛** `EnterpriseDBError` | 吞掉它会得到「未找到工号 E1001 的员工」——**由降级造成的静默错答**，用户会以为自己或这位同事不存在 |

（对比：**检索**失败是可以降级的，因为那时 L4 的「知识库中没有找到」与真实语义
一致。同一条判据，不同的结论。）

### 4.2 工具 1 · `find_employee_by_name`

```python
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
    """按姓名查询员工，返回工号、姓名与部门。"""
    invalid = _require(name, "name", "请提供要查询的员工姓名。")
    if invalid:
        return invalid
    # 不加 try/except：库不可用属于**基础设施故障**，让它抛到工具执行层统一处置
    # （转人工）。吞掉它就会退化成「未找到这位员工」——一个静默错答。
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
```

返回示例（重名）：

```json
{"ok": true, "data": {"ambiguous": true, "candidates": [
  {"employee_id": "E1003", "name": "王五", "department": "运营部"},
  {"employee_id": "E1007", "name": "王五", "department": "法务部"}
], "note": "存在多名同名员工，请向用户确认是哪一位（可提供部门辅助区分）。"}}
```

只返回三列的取舍：工号是后续查询的钥匙，但"能查得更多"不是多返回的理由——
多出的字段会进入模型上下文，既浪费 token，也扩大 PII 暴露面。

### 4.3 工具 2 · `query_employee_info`

```python
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
    """按工号查询员工基础信息：部门、岗位、直属领导、入职时间。"""
    invalid = _require(employee_id, "employee_id", "请提供员工工号（形如 E1001）。")
    if invalid:
        return invalid
    row = db.query_employee_info(employee_id)

    if not row:
        return _fail(_NOT_FOUND, f"未找到工号为「{employee_id.strip()}」的员工。")
    return _ok(row)
```

返回示例：

```json
{"ok": true, "data": {"employee_id": "E1001", "name": "张三", "department": "技术部",
                      "position": "高级工程师", "leader_id": "E1009", "entry_date": "2019-03-11"}}
```

### 4.4 工具 3 · `query_leave_balance`

```python
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
    """按工号查询剩余年假与调休天数。"""
    invalid = _require(employee_id, "employee_id", "请提供员工工号（形如 E1001）。")
    if invalid:
        return invalid
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
```

返回示例（正常 / 无记录）：

```json
{"ok": true, "data": {"employee_id": "E1001", "annual_leave": 5.0, "compensatory_leave": 2.0}}
{"ok": false, "error": "not_found", "message": "未查询到工号「E1006」的假期记录。这不代表余额为 0，可能是记录尚未建立，建议联系 HR 核实。"}
```

刻意**不**返回"总天数"/"已用天数"：库里没有这两列，任何推算都可能是错的
（比如年中入职的按比例折算规则）。宁可只答被问到的部分，也不给一个看起来
合理、实际错误的数字。

---

## 五、完整 Function Calling JSON Schema

以下是 `convert_to_openai_tool()` 对上面三个工具产出的**逐字结果**——也就是实际
发给模型的那份 payload（顺序即 `TOOL_AGENT_TOOLS` 的顺序）。
自检命令：

```bash
PYTHONPATH=. .venv/bin/python -c "
import json
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.tools.sqlite_tools import TOOL_AGENT_TOOLS
print(json.dumps([convert_to_openai_tool(t) for t in TOOL_AGENT_TOOLS], ensure_ascii=False, indent=2))
"
```

```json
[
  {
    "type": "function",
    "function": {
      "name": "find_employee_by_name",
      "description": "按姓名查询员工，返回工号、姓名与部门。仅当用户说出了具体姓名时使用；若用户只给了工号，请改用 query_employee_info。重名时返回全部候选人，需向用户确认是哪一位。",
      "parameters": {
        "properties": {
          "name": {
            "description": "员工姓名，取值必须来自用户原话中出现的那个名字。若用户只给了工号，请改用 query_employee_info；若用户没有指明是谁（如「我的部门」「谁的邮箱」），不要猜测、不要编造，直接不调用本工具，先向用户询问姓名或工号。",
            "type": "string"
          }
        },
        "required": ["name"],
        "type": "object"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "query_employee_info",
      "description": "按工号查询员工基础信息：部门、岗位、直属领导、入职时间。",
      "parameters": {
        "properties": {
          "employee_id": {
            "description": "员工工号，形如 E1001（字母 E + 至少 3 位数字）。取值来源有二：用户原话中直接给出的工号，或 find_employee_by_name 返回的 employee_id。**不要自行编造工号。**",
            "pattern": "^[Ee]\\d{3,}$",
            "type": "string"
          }
        },
        "required": ["employee_id"],
        "type": "object"
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "query_leave_balance",
      "description": "按工号查询剩余年假与调休天数。只回答这两项，不要推算总天数或已用天数。",
      "parameters": {
        "properties": {
          "employee_id": {
            "description": "员工工号，形如 E1001（字母 E + 至少 3 位数字）。取用户原话中给出的工号，或 find_employee_by_name 返回的 employee_id。**不要自行编造工号。**",
            "pattern": "^[Ee]\\d{3,}$",
            "type": "string"
          }
        },
        "required": ["employee_id"],
        "type": "object"
      }
    }
  }
]
```

---

## 六、schema 里每一处的取舍

### 6.1 参数描述是**契约**，不是注释

`description` 是模型能看到的全部信息。这几句话就是防线本身，删掉它们等于删功能：

| 句子 | 防的失败模式 |
|---|---|
| 「取值必须来自用户原话中出现的那个名字」 | 模型对「怎么申请邮箱扩容」调 `find_employee_by_name`，抽出一个人名，于是返回**别人的**部门 |
| 「若用户没有指明是谁……不要猜测、不要编造，直接不调用本工具」 | 上一条的显式版本：把"不该调"也写清楚，而不只是"调了要填对" |
| 「若用户只给了工号，请改用 query_employee_info」 | 拿姓名工具去接工号，参数对不上 → 永远 `not_found`，用户以为查无此人 |
| 「重名时返回全部候选人，需向用户确认」 | 模型替用户挑一个 → 用户拿到另一个人的工号 |
| 「**不要自行编造工号。**」 | 模型对「我的年假还剩几天」直接猜一个工号去查——没有登录态时这是**唯一**能造成"答错人"的路径 |

`tests/test_biz_correctness.py::test_object_lookup_tools_forbid_guessing` 直接断言
这些句子还在——**改动它们等于改动业务行为**，必须被门禁拦住。

### 6.2 `pattern` 是给模型的**提示**，不是强校验

`pattern` 显著降低"把整句问话当成工号"的概率（「张三的年假还剩几天」这类句子
模型容易把整句塞进参数）。代价是非法值会在 `tool.invoke` 时被 pydantic
挡下抛 `ValidationError`。

执行层把它归为 **`rejected`** 而**不是** `error`：

- `rejected` = 模型侧问题 → 回一条说明让它在下一轮自我纠正，**不转人工**；
- `error` = 基础设施问题 → 可能确实答不了，**转人工**。

若归成 `error`，用户只是把工号打错一位，却被告知"已转接人工"——这是误伤。

### 6.3 `required` 挡不住空串，所以工具侧还要判一次

`{"name": ""}` 能通过 schema 校验（键存在、类型是 string、也匹配不了任何
`pattern` 之外的约束）。模型完全可能产出它。所以每个工具入口都过一次
`_require(..., value.strip())`，返回 `invalid_argument` 而不是抛错。

### 6.4 授权参数**不能**进 schema

`allowed_sources`（来源白名单）是**授权事实**，来自服务端鉴权 / ACL 层，
只存在于请求作用域（`app/core/request_ctx.py`），绝不出现在工具的 JSON Schema 里。

把它做成工具参数，等于把「我能查谁的资料」交给模型决定——这不是"答错"，
是合规事故。`tests/test_sqlite_tools.py::test_authorization_facts_are_not_tool_parameters`
对三个禁用名（`allowed_sources` / `current_employee_id` / `user_id`）逐个断言。

> 注：`employee_id` **在** schema 里，但它不是授权事实——它表示"用户想查谁的
> 记录"（业务意图）。没有登录态之后，"我是谁"这个事实不再存在，也就没有可对账的
> 对象；把关点因此前移到**姓名**（见 §7）。

### 6.5 工具描述与函数 docstring 是**两份文本**

`@tool(description=...)` 显式传入，docstring 只写"为什么这样设计"。

原因：LangChain 会把函数 docstring 原样当作工具描述发给模型，而 pydantic v2 会把
Args 类的 docstring 渲染成 JSON Schema 的顶层 `description`。所以：

- **给模型的**（`description=` / `Field(description=...)`）：做什么、什么时候**不要**用、参数从哪来。短句、无 markdown、无实现细节。
- **给维护者的**（函数 docstring / 模块 docstring）：为什么只返回三列、为什么无记录不等于 0、为什么基础设施故障要上抛。

一份文本同时服务两类读者时，必然有一方被牺牲：写成给模型看的，维护者失去上下文；
写成给维护者看的，模型每次请求都要读一段"为什么这样设计"——纯噪音，还占 token。
这就是"Args 类刻意不写类 docstring"的同一理由，此处保持一致。

---

## 七、工具 Agent 怎么用这些工具

工具 Agent（`app/core/tool_agent.py`）只负责 function calling，不负责回答：

```
用户问题 ──► 抽取参数 ──► 调工具 ──► 拿到 JSON ──► 再决策
              ▲                                        │
              └──── 必填参数缺失 → 直接向用户追问 ◄─────┘
```

| 规格要求 | 实现位置 | 说明 |
|---|---|---|
| 参数抽取 | 模型（`bind_tools` + `AGENT_TOOLS`） | 不写离线抽取规则——规则时代"修 A 坏 B"，见 `docs/history/tool-invocation-online-vs-offline.md` |
| 缺失信息追问 | `_decide` 的直答出口 | 本轮**一次都没成功取到证据**时，模型自己的话就是答案（典型即追问） |
| 链式调用 | `TOOL_AGENT_MAX_STEPS`（默认 3） | 先 `find_employee_by_name` 拿工号，再 `query_leave_balance`；同一轮也可并行发多个调用 |
| SQLite 查询 | 本模块的 3 个工具 | 制度类问题**不在这里**——由简单/复杂 RAG Agent 负责 |

### 7.1 「我的年假还剩几天」为什么会被追问

服务端**没有登录态**，工具 Agent 拿不到"当前用户是哪名员工"这个事实。因此遇到
「**我的**年假」这类问法，模型的正确行为是**向用户索要姓名或工号**（提示词第 1 条：
缺参数就追问，禁止编造），而不是猜一个工号去查库。

这是刻意的取舍：**没有身份就不假装有身份**。若要恢复这条路，正确做法是接真实
SSO 并把员工工号重新注入为服务端事实，而不是让模型去猜。

「张三的年假还剩几天」照常可用——「张三」在原句里，`find_employee_by_name`
换出 E1001，再用工号查余额，这是规格要求的链式调用。

### 7.2 两条护栏

两条都在 `execute_tool_calls` 里，`tests/test_tool_agent.py` 覆盖：

1. **候选集校验**：模型只能调 `AGENT_TOOLS` 里的工具，编一个名字会被拒绝。
2. **落地校验（`GROUNDED_ARGS`）**：只对"抽错会泄露他人信息 / 返回错误记录"的
   槽位要求参数能在用户原句里定位——

   ```python
   GROUNDED_ARGS = {
       # 姓名抽错 → 查到另一位同事；而姓名是自由文本，模型最容易"顺手编"。
       "find_employee_by_name": ("name",),
       # 刻意**不**登记 employee_id：它是「谁」的标识，用户往往不会把工号念一遍
       # （「张三的年假」里只有姓名）。模型应当先用 find_employee_by_name 换取工号，
       # 而不是编一个——这正是「必填参数缺失时向用户追问，禁止编造」那条规格。
   }
   ```

   刻意不做无差别的全参数校验：`employee_id` 在正常路径下本来就**不在**句子里
   （它要等第一轮 `find_employee_by_name` 返回），登记它等于让整条链式调用
   在第二轮全部失败。

   > 残余风险（模型直接猜一个工号去调 `query_leave_balance`）由提示词的
   > 「缺参数就追问、禁止编造」兜底。这是**有意的取舍**，不是漏检。

---

## 八、验收

```bash
# 工具契约 / 只读性 / 边界（23 条）
./.venv/bin/python -m pytest tests/test_sqlite_tools.py -q

# 工具 Agent：候选集、护栏、追问、链式调用、回捞（40 条）
./.venv/bin/python -m pytest tests/test_tool_agent.py -q

# 描述即契约：禁止猜测、不得揽下制度类问题
./.venv/bin/python -m pytest tests/test_biz_correctness.py -q
```

建库与连通性自检：

```bash
PYTHONPATH=. .venv/bin/python scripts/seed_enterprise_db.py --check
PYTHONPATH=. .venv/bin/python -m app.core.self_check    # 含 sqlite_db 一项
```

人工冒烟（服务已启动）：

| 提问 | 期望 |
|---|---|
| 张三在哪个部门 | `find_employee_by_name` → `E1001` → `query_employee_info`（链式） |
| 张三的年假还剩几天 | 链式调用后 `annual_leave: 5.0` |
| E1001 的年假是多少 | 工号直接来自原句 → `query_leave_balance` → `annual_leave: 5.0` |
| 周八的假期余额是多少 | `query_leave_balance(E1006)` → `not_found`，且答案**不是** 0 |
| 王五在哪个部门 | `find_employee_by_name` → `ambiguous: true`，向用户确认是哪一位 |
| 我的年假还剩几天 | **直接向用户索要姓名或工号**，不编造、不猜工号 |
