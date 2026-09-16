"""企业业务数据访问层（SQLite，**只读**）。

两张表（DDL 见 ``scripts/seed_enterprise_db.py``）
-------------------------------------------------
======================  ====================================================
表                        字段
======================  ====================================================
``employee``            employee_id / name / department / position /
                        leader_id / entry_date
``leave_balance``       employee_id / annual_leave / compensatory_leave
======================  ====================================================

「只读」是怎么被**强制**的（三层，缺一不可）
--------------------------------------------
1. **连接层**：用 URI ``file:...?mode=ro`` 打开，SQLite 自身拒绝任何写操作。
   这一层是硬约束——即便本模块被人加了一句 ``INSERT``，执行时也会立刻报
   ``attempt to write a readonly database``，而不是悄悄写进去。
2. **代码层**：本模块只出现 ``SELECT``，没有任何数据修改语句。
3. **测试层**：``tests/test_sqlite_tools.py`` 直接尝试写入并断言失败，
   防止有人把第 1 层改掉（比如换成普通连接"图方便"）。

为什么连接层要用 ``mode=ro`` 而不是"只读地写 SQL"：**约定挡不住后来者**。
写一句 ``INSERT`` 与写一句 ``SELECT`` 在代码审查里长得一样，而 `mode=ro`
让越界在**运行时**立即失败。这与本项目其余部分的取向一致——把约束放在
结构里，而不是放在注释里。

为什么每次调用都新建连接
------------------------
- SQLite 的连接对象**不可跨线程共享**，而 FastAPI 的同步端点跑在线程池里；
  缓存连接会得到最难查的一类 bug：偶发的 ``ProgrammingError``。
- 打开本地文件的成本在微秒级，且 SQLite 有页缓存，代价可忽略。
- 换来的是零共享状态：不需要锁、不受 worker 复用影响。

参数化查询
----------
所有取值一律走 ``?`` 占位符。用户输入**永不**参与 SQL 字符串拼接——
工具的参数来自模型，而模型可能被诱导产出任意字符串，这是典型的注入入口。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import config
from app.utils.logger import logger


class EnterpriseDBError(RuntimeError):
    """业务库不可用（文件缺失 / 结构不符 / 查询失败）。

    刻意**不吞**异常：数据访问层是全项目最需要"失败要响"的地方——静默地把
    查询失败变成"查不到"，会让用户看到「没有这位员工」而不是「系统故障」，
    是典型的静默错答。降级决策留给上层（工具层返回结构化错误）。
    """


def db_path() -> Path:
    """数据库文件路径（配置项 ``SQLITE_DB_PATH``）。"""
    return Path(config.SQLITE_DB_PATH)


def _connect() -> sqlite3.Connection:
    """打开**只读**连接。文件不存在时抛 :class:`EnterpriseDBError`。"""
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
    """执行只读查询，返回 dict 列表。

    异常一律转成 :class:`EnterpriseDBError`：调用方（工具层）只需要区分
    「业务库不可用」与「查不到」两种情形，不该关心 sqlite3 的异常层次。
    """
    try:
        with _connect() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        logger.warning("业务库查询失败：%s（sql=%s）", exc, sql[:80])
        raise EnterpriseDBError(f"业务库查询失败：{exc}") from exc
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# 查询：员工
# ---------------------------------------------------------------------------
def find_employee_by_name(name: str) -> List[Dict[str, Any]]:
    """按姓名查员工（可能重名，故返回列表）。

    只返回工号 / 姓名 / 部门三列——这正是 ``find_employee_by_name`` 工具承诺的
    输出。多查的列会经由工具进入模型上下文，既浪费 token，也扩大了 PII 暴露面。

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


def table_names() -> List[str]:
    """列出库中的表名（启动自检用，验证结构是否为预期的那两张）。"""
    return [r["name"] for r in _query(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )]


def is_read_only() -> bool:
    """**实测**只读约束是否生效：真的尝试建一张表，看 SQLite 是否拒绝。

    为什么不是"看代码里有没有 INSERT"：那是审阅，不是验证。把约束放在
    ``mode=ro`` 里的意义正是"即便有人加了一句写语句，运行时也会失败"——
    那么验证方式就必须是"真的写一次"。

    返回 True 表示写被拒绝（符合预期）。任何异常都视为"被拒绝"，
    唯一返回 False 的情形是写**成功**了，那说明只读层已被改掉。
    """
    try:
        with _connect() as conn:
            conn.execute("CREATE TABLE _selfcheck_write_probe(x)")
    except sqlite3.Error:
        return True
    logger.error("业务库可写！只读约束未生效：%s", db_path())
    return False


def table_counts(tables: List[str]) -> Dict[str, int]:
    """统计若干表的行数（启动自检用）。

    ``tables`` 必须是**调用方写死的白名单常量**，不来自用户输入或模型输出；
    这是本模块唯一一处拼接表名的地方，因此把这条约束写在签名旁边而不是注释里。
    """
    counts: Dict[str, int] = {}
    for name in tables:
        rows = _query(f"SELECT COUNT(*) AS n FROM {name}")
        counts[name] = int(rows[0]["n"]) if rows else 0
    return counts
