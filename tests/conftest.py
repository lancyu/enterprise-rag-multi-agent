"""pytest 全局配置。

为什么需要 `collect_ignore`
---------------------------
`tests/` 下混着两种风格的测试：

- **pytest 风格**（函数式、无模块级副作用）—— 可被正常收集；
- **脚本式**（模块级 `sys.exit(...)`，部分还需要真实服务在 8001 端口）——
  pytest 收集时会在 **import 阶段**就执行 `sys.exit`，直接抛出
  `INTERNALERROR: SystemExit`，**整个测试会话挂掉**，一条用例都跑不了。

被忽略的脚本式用例请用下面的方式单独运行：

```bash
PYTHONPATH=. .venv/bin/python tests/test_rag.py
PYTHONPATH=. .venv/bin/python tests/test_span_tree_smoke.py
PYTHONPATH=. .venv/bin/python tests/test_infra.py        # 需基础设施
PYTHONPATH=. .venv/bin/python tests/test_service.py      # 需服务已在 8001 端口运行
```

这里的忽略不是「不跑这些测试」，而是「不让它们污染 pytest 收集」——
否则每次跑单测都得手动拼 `--ignore=...` 四个参数。

`test_smalltalk.py` 已随自造路由机制一并归档
（`_archive/removed-selfbuilt-routing-20260915-1314/tests/`）。
寒暄在五 Agent 架构下由**闲聊 Agent** 处理（模板直出，不调模型），
回归用例见 `tests/test_multi_agent.py::test_smalltalk_never_touches_retrieval_or_tools`。
"""
import sqlite3

import pytest

collect_ignore = [
    "test_infra.py",
    "test_rag.py",
    "test_service.py",
    "test_span_tree_smoke.py",
]


@pytest.fixture
def business_db(tmp_path, monkeypatch):
    """一份**独立于 `data/enterprise.db`** 的业务库，供工具与多 Agent 测试使用。

    为什么不在测试里直接读 ``data/enterprise.db``：

    1. **它可能不存在**（CI / 新克隆的仓库要跑一次 seed 脚本才有），
       测试会以"库不存在"这种与被测逻辑无关的方式失败；
    2. **它会被改**（有人改了演示数据，测试跟着红/绿），而测试应当只依赖
       它自己写死的那几条边界数据；
    3. 边界用例需要**刻意造**的数据（重名、缺假期记录），
       直接用演示数据等于把"测试的前置条件"藏在另一个文件里。

    数据来自 ``scripts/seed_enterprise_db.py`` 的同一份常量：那两组边界情形
    （两个「王五」/ E1006 无假期记录）正是这里要断言的，
    复制一份必然漂移，故直接复用。
    """
    from seed_enterprise_db import EMPLOYEES, LEAVE_BALANCES, SCHEMA

    path = tmp_path / "enterprise_test.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO employee VALUES (?, ?, ?, ?, ?, ?)", EMPLOYEES)
        conn.executemany("INSERT INTO leave_balance VALUES (?, ?, ?)", LEAVE_BALANCES)
        conn.commit()
    monkeypatch.setattr("app.config.SQLITE_DB_PATH", str(path))
    return path


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch):
    """每个用例前重置入站限流器。

    限流器是**进程内全局单例**（``app/core/rate_limit.py``），命中记录跨用例
    累积：套件里打 HTTP 的用例一多，就会把默认 20 次/分钟的窗口提前打满，
    后面的用例直接拿到 429 —— 表现为「单独跑通过、整套跑失败」的顺序依赖，
    排查时极易误判成被测代码的问题。

    换成全新实例而不是清空内部字典：走公开构造函数，不依赖私有实现。
    """
    from app.core import rate_limit

    monkeypatch.setattr(rate_limit, "_limiter", rate_limit.RateLimiter())
    yield

