"""未使用代码 / 未使用依赖 检测测试集。

这套测试回答一个具体问题：**仓库里哪些代码写了却没人用？**

为什么不是「装个 vulture 跑一下」就完事
----------------------------------------
本项目的三处框架约定会让纯名字统计的通用工具大面积误报（实测 vulture 在本仓库
60% 档报出 50+ 项，其中约一半是假的）：

=============== ====================================================
约定              为什么误报
=============== ====================================================
FastAPI 路由     `@router.get("/x")` 注册，函数名无调用点
LangGraph 节点   除 `add_node("name", fn)` 外，节点名还以**字符串**形式
                 出现在拓扑描述里
配置常量         消费方用 `getattr(config, "X", default)` 读取，或只在
                 `.env` 里出现
=============== ====================================================

所以本项目自建 `scripts/deadcode_scan.py`：在 AST 名字统计之上补齐装饰器豁免、
字符串引用识别、`getattr(config, …)` 消费识别、`__all__` / re-export 识别。
本文件则是它的**执行契约**——在 CI 里把"死代码"变成可回归的失败。

测试分层
--------
1. 自检（meta）：先证明扫描器真的能抓出死代码，避免它静默失效变成"永远通过"；
2. 分类断言：未引用模块 / 未使用函数类常量 / 死配置 / 未使用依赖，逐类断言为空
   （扣除 `deadcode_allowlist.py` 中已写明理由的豁免）；
3. 僵尸豁免校验：清单里的条目必须仍被报出，否则说明该条目已过期，应删掉；
4. 外部工具交叉验证：ruff（未使用 import / 变量 / 未定义名）可用时一并执行；
5. 严格模式（`--strict`）：只把 `app/` 内的引用算作「已使用」，用于暴露
   "仅被测试或工具脚本引用"的生产 API。它**不进 CI 门禁**（结论需人工判断），
   但把它钉在测试里，可以保证这个视角本身不会失效。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from deadcode_scan import Finding, scan_findings
from tests.deadcode_allowlist import ALLOWLIST, ALLOWLIST_REASONS

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 夹具：整个模块共享一次扫描结果（扫描约 70 个文件，避免每个用例重复解析）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def findings() -> List[Finding]:
    return scan_findings(ROOT)


@pytest.fixture(scope="module")
def unexpected(findings: List[Finding]) -> List[Finding]:
    """扣掉已豁免的条目后，真正需要处理的发现。"""
    return [f for f in findings if f.key not in ALLOWLIST]


def _format(items: List[Finding]) -> str:
    lines = [
        "",
        "─" * 78,
        f"检测到 {len(items)} 项未使用问题（未在 deadcode_allowlist.py 中豁免）：",
        "─" * 78,
    ]
    for f in items:
        lines.append(f"  {f.file}:{f.lineno}  [{f.kind}] {f.qualname or f.name}")
        lines.append(f"      {f.detail}")
    lines.append("─" * 78)
    lines.append("处理方式：删除它；若确需保留，请在 tests/deadcode_allowlist.py")
    lines.append("中登记并写明「为什么不能删」，或在定义处加 `# deadcode: ignore`。")
    return "\n".join(lines)


def _by_kind(items: List[Finding], kinds: set) -> List[Finding]:
    return [f for f in items if f.kind in kinds]


# ---------------------------------------------------------------------------
# 1. 自检：扫描器本身必须是有效的
# ---------------------------------------------------------------------------
def test_scanner_detects_planted_dead_code(tmp_path: Path) -> None:
    """往临时包里种一处死代码，扫描器必须报出来。

    这条用例是整套测试的"地基"：如果哪天扫描器改成返回空列表（或异常被吞掉），
    下面所有"断言为空"的用例都会变成永远通过——那比没有测试更危险。
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text(
        "def used():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def never_called():\n"
        "    return 2\n"
        "\n"
        "\n"
        "ORPHAN_CONST = 3\n",
        encoding="utf-8",
    )
    (pkg / "consumer.py").write_text(
        "from pkg.mod import used\n"
        "\n"
        "VALUE = used()\n",
        encoding="utf-8",
    )

    found = scan_findings(tmp_path, roots=["pkg"])
    pairs = {(f.kind, f.name) for f in found}

    assert ("unused_function", "never_called") in pairs, "扫描器漏报了明显的死函数"
    assert ("unused_constant", "ORPHAN_CONST") in pairs, "扫描器漏报了死常量"
    assert not any(n == "used" for _, n in pairs), "扫描器把被调用的函数误判为死代码"


def test_scanner_respects_fastapi_and_fixture_decorators(tmp_path: Path) -> None:
    """带框架装饰器的函数必须被豁免（否则 FastAPI 路由会被整片误报）。"""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "routes.py").write_text(
        "class R:\n"
        "    def get(self, path):\n"
        "        def deco(fn):\n"
        "            return fn\n"
        "        return deco\n"
        "\n"
        "\n"
        "router = R()\n"
        "\n"
        "\n"
        "@router.get('/ping')\n"
        "def ping():\n"
        "    return 'pong'\n",
        encoding="utf-8",
    )

    found = scan_findings(tmp_path, roots=["pkg"])
    assert not any(f.name == "ping" for f in found), "路由处理函数被误报为死代码"


# ---------------------------------------------------------------------------
# 2. 分类断言
# ---------------------------------------------------------------------------
def test_no_unreferenced_modules(unexpected: List[Finding]) -> None:
    """没有"整个模块无人 import"的文件。"""
    items = _by_kind(unexpected, {"unreferenced_module"})
    assert not items, _format(items)


def test_no_unused_functions_or_methods(unexpected: List[Finding]) -> None:
    """没有被定义却从无调用点的函数 / 方法。"""
    items = _by_kind(unexpected, {"unused_function", "unused_method"})
    assert not items, _format(items)


def test_no_unused_classes_or_constants(unexpected: List[Finding]) -> None:
    """没有被定义却从无引用的类 / 模块常量 / 类属性。"""
    items = _by_kind(
        unexpected, {"unused_class", "unused_constant", "unused_class_attr"}
    )
    assert not items, _format(items)


def test_no_dead_config(unexpected: List[Finding]) -> None:
    """config.py 里没有"定义了但全仓无人消费"的伪配置。

    只写进 config.py 却没有任何读取点的配置项最危险：运维照它调 .env 完全没有效果，
    且没有任何报错——属于"静默失效"。
    """
    items = _by_kind(unexpected, {"dead_config"})
    assert not items, _format(items)


def test_no_unused_declared_dependencies(unexpected: List[Finding]) -> None:
    """requirements.txt 中没有"声明了却全仓无 import"的依赖。

    注：仅通过命令行/框架隐式使用的运行时依赖（uvicorn、python-multipart 等）
    已在 scripts/deadcode_scan.py 的 RUNTIME_ONLY_DEPS 中列明并注明理由。
    """
    items = _by_kind(unexpected, {"unused_dependency"})
    assert not items, _format(items)


# ---------------------------------------------------------------------------
# 3. 僵尸豁免校验（对齐 ruff RUF100 治理过期 noqa 的思路）
# ---------------------------------------------------------------------------
def test_allowlist_entries_are_not_stale(findings: List[Finding]) -> None:
    """豁免清单里的每条都必须仍然被报出；否则说明它已过期，应当删除。"""
    live = {f.key for f in findings}
    stale = sorted(ALLOWLIST - live)
    assert not stale, (
        "\n以下豁免条目已不再被检测到（代码已删除或规则已变更），"
        "请从 tests/deadcode_allowlist.py 中移除，避免清单腐化：\n"
        + "\n".join(f"  - {k}" for k in stale)
    )


def test_every_allowlist_entry_has_reason() -> None:
    """每个豁免必须写明理由——不允许"静默豁免"。"""
    missing = sorted(k for k in ALLOWLIST if not ALLOWLIST_REASONS.get(k, "").strip())
    assert not missing, (
        "\n以下豁免条目缺少理由说明，请在 ALLOWLIST_REASONS 中补充"
        "「为什么不能删、删了会怎样」：\n" + "\n".join(f"  - {k}" for k in missing)
    )


# ---------------------------------------------------------------------------
# 4. 外部工具交叉验证（ruff：未使用 import / 未使用变量 / 未定义名 / 过期 noqa）
# ---------------------------------------------------------------------------
def _ruff_available() -> bool:
    return shutil.which("ruff") is not None or (ROOT / ".venv" / "bin" / "ruff").exists()


def test_no_unused_imports_or_variables() -> None:
    """用 ruff 补扫 AST 名字统计覆盖不到的项：未使用 import / 未使用局部变量 / 未定义名。

    自建扫描器只管"定义有没有人用"，管不到"import 进来却没用到"。两者互补。
    ruff 属开发依赖（requirements-dev.txt），未安装时跳过而非失败。

    这里**不传 `--select`**：规则集由 pyproject.toml `[tool.ruff.lint]` 统一定义，
    避免"测试里一套规则、CI 里另一套"的配置漂移。
    """
    if not _ruff_available():
        pytest.skip("ruff 未安装（pip install -r requirements-dev.txt 后可启用）")

    ruff = shutil.which("ruff") or str(ROOT / ".venv" / "bin" / "ruff")
    proc = subprocess.run(
        [ruff, "check", "app", "scripts", "tests",
         "--no-cache", "--output-format", "concise"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "\nruff 检测到未使用 import / 变量 / 未定义名：\n"
        + (proc.stdout or proc.stderr)
    )


def test_deadcode_scan_cli_is_runnable() -> None:
    """CLI 入口可用（供开发者/CI 直接出报告）。"""
    proc = subprocess.run(
        [sys.executable, "scripts/deadcode_scan.py", "."],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    # 0 = 无发现；1 = 有发现（两者都说明脚本正常执行）
    assert proc.returncode in (0, 1), f"扫描脚本执行异常：{proc.stderr}"
    assert "未使用" in proc.stdout or "未发现" in proc.stdout


# ---------------------------------------------------------------------------
# 5. 严格模式（--strict）：把问题从「有没有人用」提升为「生产代码里有没有人用」
# ---------------------------------------------------------------------------
def _plant_prod_and_test(tmp_path: Path) -> None:
    """在临时仓库里造一对组合：生产函数只被测试引用。"""
    app = tmp_path / "app"
    app.mkdir()
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "prod.py").write_text(
        "def only_tests_use():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_prod.py").write_text(
        "from app.prod import only_tests_use\n"
        "\n"
        "\n"
        "def test_it():\n"
        "    assert only_tests_use() == 1\n",
        encoding="utf-8",
    )


def test_strict_mode_reports_test_only_production_api(tmp_path: Path) -> None:
    """只被 tests/ 引用的生产函数：默认模式看不见，严格模式必须报出来。

    这正是"仅被测试/工具引用"这一类隐蔽死代码——它在普通扫描里永远是
    "已使用"，只有把引用来源收缩到 app/ 内才会浮出水面。
    """
    _plant_prod_and_test(tmp_path)

    default_hits = [f for f in scan_findings(tmp_path) if f.name == "only_tests_use"]
    strict_hits = [f for f in scan_findings(tmp_path, strict=True)
                   if f.name == "only_tests_use"]

    assert default_hits == [], "默认模式本就把测试引用算作已使用"
    assert len(strict_hits) == 1, "严格模式漏报了仅被测试引用的生产函数"


def test_strict_mode_does_not_report_test_helpers(tmp_path: Path) -> None:
    """严格模式只报 app/ 内的定义：测试目录里互相引用的辅助函数属正常用法。

    不排除的话，测试辅助函数会成片涌入报告，把真正的信号淹掉。
    """
    _plant_prod_and_test(tmp_path)
    tests = tmp_path / "tests"
    (tests / "helpers.py").write_text(
        "def shared_helper():\n"
        "    return 2\n",
        encoding="utf-8",
    )
    (tests / "test_helper.py").write_text(
        "from tests.helpers import shared_helper\n"
        "\n"
        "\n"
        "def test_h():\n"
        "    assert shared_helper() == 2\n",
        encoding="utf-8",
    )

    names = {f.name for f in scan_findings(tmp_path, strict=True)}
    assert "shared_helper" not in names, "测试目录内的辅助函数不该被严格模式报出"


def test_strict_mode_keeps_script_only_dependencies(tmp_path: Path) -> None:
    """严格模式不影响依赖判定：只有 scripts/ 用到的库仍是真实依赖。

    若把「未使用依赖」也按严格口径算，删掉被脚本引用的库会让工具链直接崩——
    依赖的判定标准是"仓库里有没有人用"，不是"生产代码里有没有人用"。
    """
    app = tmp_path / "app"
    app.mkdir()
    (app / "__init__.py").write_text("", encoding="utf-8")
    (app / "prod.py").write_text("VALUE = 1\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "tool.py").write_text("import yaml\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("PyYAML>=6.0\n", encoding="utf-8")

    hits = [f for f in scan_findings(tmp_path, strict=True)
            if f.kind == "unused_dependency"]
    assert hits == [], f"严格模式误报了只被脚本使用的依赖：{hits}"

