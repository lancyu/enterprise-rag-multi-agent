#!/usr/bin/env python3
"""分层依赖契约（P1-2）—— 把「api → core → … → utils」从文档里的说法变成门禁。

守什么
------
这条分层线在文档里写了很久，但**没有任何机制拦得住**：往 `app/rag/` 里加一行
`from app.api.chat import router`，代码照跑、测试照绿、类型检查照过，
只有下一个读代码的人才会发现分层已经名存实亡。
「文档里写了」不等于「发生时会有人知道」。

契约由 `pyproject.toml` 的 `[tool.importlinter]` 定义，本文件负责**驱动**它
（与 `tests/test_deadcode.py` 驱动 ruff / vulture / deptry 是同一个套路：
外部工具按需安装，pytest 负责口径与自检）。

为什么要有「自检」
------------------
一条永不失败的门禁比没有门禁更糟 —— 它会让人以为"这里已经被检查过了"。
所以本文件除了「现在绿不绿」，还额外证明两件事：

1. **契约真的会红**：临时往 `app/rag/` 里塞一个向上 import，跑一次契约检查，
   必须失败（这正是 P1-2 的验收条件）；
2. **豁免清单不会烂掉**：`unmatched_ignore_imports_alerting = error` 必须还在配置里，
   且清单条数有**上限**（只减不增的棘轮）。
"""
from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import tomllib
from typing import List

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

#: 契约必须覆盖的顶层包 —— 新增一个顶层包时若忘了归类，本条会红。
_EXPECTED_TOP_LEVEL = {
    "api", "core", "db", "graph", "memory", "providers", "rag", "tools", "utils",
}

#: 豁免条数的**上限**（棘轮）：只许减，不许增。
#: 调高这个数字应当是一次有意识的决定，并且在 review 里说得清为什么。
#: 当前 14 条的分组与理由见 `pyproject.toml` 的 `ignore_imports` 注释。
#: （原为 15 条：`app.utils.embedding -> app.providers.embeddings` 随 P1-6
#: 删掉兼容壳后消失 —— 这是**往下降**的唯一合法理由：真的还清了一条债。）
_IGNORE_BUDGET = 14


def _config() -> dict:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["importlinter"]


def _contract() -> dict:
    contracts = _config()["contracts"]
    assert len(contracts) >= 1, "import-linter 契约被删空了"
    return contracts[0]


def _named_contract(name: str) -> dict:
    for contract in _config()["contracts"]:
        if contract.get("name") == name:
            return contract
    raise AssertionError(f"契约 {name!r} 不见了")


def _run_lint_imports() -> subprocess.CompletedProcess:
    """跑一次契约检查。

    用 `console script` 而不是 `python -m importlinter.cli`：那个模块**没有**
    ``__main__`` 入口，`-m` 会静默地什么都不做并返回 0 —— 一条永远"通过"的
    门禁（实测踩到：stdout/stderr 全空、退出码 0）。
    """
    executable = pathlib.Path(sys.executable).parent / "lint-imports"
    if not executable.exists():   # 退化路径：直接调 console_scripts 的入口函数
        cmd = [
            sys.executable, "-c",
            "from importlinter.cli import lint_imports_command; lint_imports_command()",
        ]
    else:
        cmd = [str(executable)]

    return subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )


# ---------------------------------------------------------------------------
# 1. 契约现在成立
# ---------------------------------------------------------------------------
def test_layering_contract_is_kept():
    """跑一次真实的契约检查，必须通过。

    ⚠️ 反向验证：往 `app/rag/` 里加一行 `from app.api.chat import router`，
    本条必须变红（下一条用例自动做这件事）。
    """
    proc = _run_lint_imports()
    # 装不上时给出可执行的指引，而不是一句"命令找不到"——
    # 这个依赖不在 requirements.txt 里，新克隆的仓库要单独装。
    assert "No module named" not in proc.stderr, (
        "import-linter 未安装。执行：\n"
        "  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \\\n"
        "    pip install --index-url http://pypi.tuna.tsinghua.edu.cn/simple \\\n"
        "    --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements-dev.txt\n"
        f"{proc.stderr[-500:]}"
    )
    assert proc.returncode == 0, f"分层契约被破坏：\n{proc.stdout[-3000:]}"
    assert "KEPT" in proc.stdout, f"契约没有真的跑起来：\n{proc.stdout[-1000:]}"


def test_contract_actually_rejects_an_upward_import():
    """**自检**：往能力层塞一个向上 import，契约必须变红。

    为什么必须做这件事：如果契约的 layers 写错（例如层名拼错、全部落进同一层），
    它会一直"通过"，而所有人都会以为分层被守住了。这条用例把"契约还有牙齿"
    这件事本身变成可验证的。

    探针文件用 ``_`` 开头且用完即删：它不该被 pytest 收集，也不该留在仓库里。
    """
    probe = _REPO_ROOT / "app" / "rag" / "_layering_probe.py"
    assert not probe.exists(), f"上次运行的探针文件没被清理：{probe}"
    probe.write_text(
        '"""临时探针：验证分层契约仍然有效。本文件由测试创建并在 finally 中删除。"""\n'
        "from app.api.chat import router  # noqa: F401\n",
        encoding="utf-8",
    )
    try:
        proc = _run_lint_imports()
    finally:
        probe.unlink(missing_ok=True)

    assert proc.returncode != 0, (
        "往 app/rag/ 里加了 `from app.api.chat import router`，分层契约却没报错 —— "
        "契约已经失效（层的定义多半写错了）"
    )
    assert "app.rag" in proc.stdout and "app.api" in proc.stdout, proc.stdout[-1500:]
    assert not probe.exists(), "探针文件没有被清理"


# ---------------------------------------------------------------------------
# 2. 豁免清单不许烂掉
# ---------------------------------------------------------------------------
def test_stale_exemptions_are_an_error():
    """僵尸豁免必须报错，否则清单只会单向变长。

    豁免条目对应的那行 import 一旦被删或挪走，条目就变成"僵尸"：它不再豁免任何
    东西，却仍在列表里占位子，后来的读者会以为"这条已经确认过了"。
    `unmatched_ignore_imports_alerting = error` 就是治这个的
    —— 与 ruff `RUF100`、`tests/deadcode_allowlist.py` 的僵尸校验同一套语义。
    """
    assert _contract().get("unmatched_ignore_imports_alerting") == "error", (
        "僵尸豁免不再报错 —— 豁免清单会只增不减"
    )


def test_exemption_budget_does_not_grow():
    """豁免条数只减不增（棘轮）。

    契约的价值不在于"现在全绿"，而在于"新增一条向上依赖时必须有人做判断"。
    清单一旦可以随手加长，这个判断就退化成复制粘贴一行字符串。
    """
    count = len(_contract().get("ignore_imports", []))
    assert count <= _IGNORE_BUDGET, (
        f"豁免从 {_IGNORE_BUDGET} 条涨到了 {count} 条。"
        "新增豁免前请先判断：是真的需要这条向上依赖，还是分层放错了位置？"
        "确认必须新增时，同步调高 _IGNORE_BUDGET 并说明理由。"
    )


def test_every_top_level_package_is_classified():
    """`app/` 下的每个顶层包都必须出现在契约里。

    否则新增一个顶层包时，它默认处于"不受任何约束"的状态 ——
    而这恰恰是分层最容易悄悄破的地方。
    """
    declared: List[str] = []
    for layer in _contract()["layers"]:
        # 层写法形如 "app.rag : app.memory"，两种分隔符都表示"同层多个模块"
        for piece in layer.replace("|", ":").split(":"):
            piece = piece.strip()
            if piece.startswith("app."):
                declared.append(piece[len("app."):].split(".")[0])

    missing = sorted(_EXPECTED_TOP_LEVEL - set(declared))
    assert missing == [], f"这些顶层包没有被写进分层契约：{missing}"

    # 自检：反向也要成立 —— 契约里不该残留已经删掉的包
    actual = {p.name for p in (_REPO_ROOT / "app").iterdir() if p.is_dir()}
    ghosts = sorted(set(declared) - actual)
    assert ghosts == [], f"契约里还写着已经不存在的包：{ghosts}"


# ---------------------------------------------------------------------------
# 3. runtime_flags 的零依赖（P1-6 解环后的新增不变量）
# ---------------------------------------------------------------------------
_RUNTIME_FLAGS = _REPO_ROOT / "app" / "runtime_flags.py"


def _app_imports(source: str) -> List[str]:
    """返回源码里所有指向 `app` 的 import（AST 判据，不看字符串）。"""
    found: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names
                      if a.name == "app" or a.name.startswith("app.")]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "app" or module.startswith("app."):
                found.append(module)
    return found


def test_runtime_flags_has_no_app_imports():
    """`app/runtime_flags.py` 不许 import 任何 `app.*` 的东西。

    这个模块存在的**全部理由**就是「谁都能安全地 import 它」：它由 provider 单向
    发布运行时事实、供配置层读取（见它的模块 docstring 与 P1-6）。它一旦开始
    import `app.*`，就重新变成环的一部分 —— 而建它正是为了拆环。
    """
    assert _RUNTIME_FLAGS.exists(), "app/runtime_flags.py 被删了（P1-6 的载体）"
    found = _app_imports(_RUNTIME_FLAGS.read_text(encoding="utf-8"))
    assert found == [], f"runtime_flags 重新依赖了 app：{found}"


def test_zero_dependency_contract_actually_fires():
    """**自检**：往 `runtime_flags.py` 里塞一行 `from app import config`，契约必须变红。

    为什么单独验这一条：`forbidden` 契约的 `forbidden_modules` 写错（例如只写
    `"app"` 而不写 `"app.*"`）时，它会**一直通过**——而所有人都会以为
    「零依赖」被守住了。这是本项目反复踩的「护栏恒真」坑，所以这里用探针
    把「契约还有牙齿」变成可验证的事实。

    ⚠️ 反向验证：把 `pyproject.toml` 里那条 forbidden 契约删掉，本用例必须变红。
    """
    assert _named_contract("runtime_flags 必须零依赖")["type"] == "forbidden"

    original = _RUNTIME_FLAGS.read_text(encoding="utf-8")
    _RUNTIME_FLAGS.write_text(
        original + "\n\nfrom app import config  # noqa: F401  探针：本行由测试注入并还原\n",
        encoding="utf-8",
    )
    try:
        proc = _run_lint_imports()
    finally:
        _RUNTIME_FLAGS.write_text(original, encoding="utf-8")

    assert proc.returncode != 0, (
        "runtime_flags 里加了一行 `from app import config`，零依赖契约却没报错 —— "
        "契约已失效（多半是 forbidden_modules 的写法没匹配上 `app.config`）"
    )
    assert "runtime_flags" in proc.stdout, proc.stdout[-1500:]
