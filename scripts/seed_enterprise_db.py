#!/usr/bin/env python3
"""初始化业务结构化数据库（SQLite）。

用法::

    cd langgraph-enterprise-bot
    ./.venv/bin/python scripts/seed_enterprise_db.py            # 建表 + 写入演示数据
    ./.venv/bin/python scripts/seed_enterprise_db.py --check     # 只校验结构

产出：``data/enterprise.db``（路径由 ``SQLITE_DB_PATH`` 配置决定）。

为什么建表与数据在**脚本**里，而不是应用启动时自动建
----------------------------------------------------
自动建表看起来更省事，但会让"数据库是空的"这种状态变得不可见：表建好了、
查也查得动，只是返回空——于是「查不到这位员工」与「数据没导入」在调用方看来
完全同形。业务库属于**部署产物**，应当由部署流程显式产生，并在失败时立刻可见。

脚本本身是**幂等**的：重复执行只会把数据重置回演示状态。

⚠️ 本脚本会**写入**数据库；应用运行时的 ``app/db/enterprise_db.py`` 是只读连接
（URI ``mode=ro``），两者是刻意的分工——写权限只存在于部署期。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import config  # noqa: E402

# ---------------------------------------------------------------------------
# 表结构（与规格严格一致，字段名不增不减）
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS employee (
    employee_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL,
    position    TEXT NOT NULL,
    leader_id   TEXT,
    entry_date  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leave_balance (
    employee_id       TEXT PRIMARY KEY,
    annual_leave      REAL NOT NULL,
    compensatory_leave REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employee(employee_id)
);
"""

# ---------------------------------------------------------------------------
# 演示数据
# ---------------------------------------------------------------------------
# 刻意造出两组**边界情形**，让"查不到"与"查得到"能被区分开：
#   1. 重名（两个「王五」）—— find_employee_by_name 必须返回两个工号，
#      由模型去追问"是哪一位"，而不是替用户猜一个；
#   2. 有工号但无假期记录（E1006）—— 这是「记录缺失」，与「余额为 0」不同，
#      工具必须如实说"没有记录"而不是回 0。
EMPLOYEES = [
    ("E1001", "张三", "技术部", "高级工程师", "E1009", "2019-03-11"),
    ("E1002", "李四", "产品部", "产品经理", "E1009", "2021-07-01"),
    ("E1003", "王五", "运营部", "运营总监", "E1010", "2017-05-20"),
    ("E1004", "赵六", "人力资源部", "HRBP", "E1010", "2022-09-15"),
    ("E1005", "孙七", "财务部", "财务主管", "E1009", "2018-11-02"),
    ("E1006", "周八", "市场部", "市场专员", "E1010", "2024-02-19"),
    ("E1007", "王五", "法务部", "法务专员", "E1009", "2023-06-08"),   # 重名
    ("E1009", "陈九", "技术部", "技术总监", None, "2015-01-05"),
    ("E1010", "郑十", "人力资源部", "人力资源总监", None, "2016-04-18"),
]

LEAVE_BALANCES = [
    ("E1001", 5.0, 2.0),
    ("E1002", 10.0, 0.0),
    ("E1003", 1.5, 6.5),
    ("E1004", 8.0, 1.0),
    ("E1005", 0.0, 3.0),
    # E1006 刻意**不写**假期记录：验证「记录缺失」不被当成 0
    ("E1007", 12.0, 4.0),
    ("E1009", 15.0, 0.0),
    ("E1010", 9.0, 2.5),
]


def _check(path: Path) -> int:
    """只校验结构，不写入。"""
    if not path.exists():
        print(f"✗ 数据库不存在：{path}")
        return 1
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = sorted(
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        )
        print(f"数据库：{path}")
        print(f"表：{tables}")
        for table in ("employee", "leave_balance"):
            if table not in tables:
                print(f"✗ 缺少表：{table}")
                return 1
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<16} {count} 行")
    print("✓ 结构校验通过")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化企业业务 SQLite 库")
    parser.add_argument("--check", action="store_true", help="只校验结构，不写入")
    args = parser.parse_args()

    path = Path(config.SQLITE_DB_PATH)
    if args.check:
        return _check(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        # 幂等：先清空再写入，重复执行结果一致（避免"跑了三遍多出三行"的困惑）
        for table in ("leave_balance", "employee"):
            conn.execute(f"DELETE FROM {table}")
        conn.executemany(
            "INSERT INTO employee VALUES (?, ?, ?, ?, ?, ?)", EMPLOYEES
        )
        conn.executemany(
            "INSERT INTO leave_balance VALUES (?, ?, ?)", LEAVE_BALANCES
        )
        conn.commit()

    print(f"✓ 已初始化：{path}")
    for table, rows in (("employee", EMPLOYEES), ("leave_balance", LEAVE_BALANCES)):
        print(f"  {table:<16} {len(rows)} 行")
    print("\n提示：应用侧是**只读**连接（URI mode=ro），写权限只在部署期存在。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
