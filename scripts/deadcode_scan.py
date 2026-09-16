"""未使用代码 / 依赖 静态扫描器（项目专用）。

为什么不直接用 vulture / ruff 就完事
------------------------------------
通用工具只看「显式调用点」，对本项目的三类约定必然误判：

1. **FastAPI 路由**：`@router.get("/x")` 注册，函数名永不出现在调用点 → 误报；
2. **LangGraph 节点**：除 `add_node("name", fn)` 外，节点名还以**字符串**形式出现在
   `workflow_graph.py`、`api/workflow.py` 的拓扑描述里 → 字符串引用要认；
3. **配置常量**：`config.py` 里定义，消费方用 `getattr(config, "X", default)` 读取
   （见 `app/core/model_router.py` 等），或干脆只在 `.env` 里出现 → 纯名字统计抓不到。

本模块在标准 AST 名字统计之上，额外做了：
    - 装饰器豁免（路由 / fixture / validator / property / abstractmethod …）
    - **字符串引用识别**（排除 docstring，避免注释里的名字把死代码"洗白"）
    - **`getattr(config, "X")` 与 `config.X` 双通道**配置消费识别
    - `__all__` / `__init__.py` re-export / 模型字段 / 枚举成员识别
    - 未引用**模块**检测（整文件无人 import）
    - 未使用**声明依赖**检测（requirements.txt ∩ 全仓 import）

输出既给人看（CLI 报表），也给测试用（`scan_findings()` 返回结构化数据）。

两种口径：默认 / `--strict`
---------------------------
默认口径问的是「仓库里有没有人用」，把 tests/ 与 scripts/ 的引用也算作使用。
这对 CI 门禁是对的——**但会系统性漏掉一类真实问题**：只被测试或工具脚本引用的
生产代码（典型如 `evaluator.filter_by_section`、`CHUNK_MIN_CHARS`：它们"有用"，
只是生产链路从不走），以及只被测试驱动的方法。

`--strict` 把引用来源收缩到 `app/` 内，问题变成「**生产代码**里有没有人用」，
这类隐蔽死代码就会浮出来。它刻意**不进 CI 门禁**：结论必然包含一批需要人工判断的
条目（例如只给 `scripts/` 用的度量工具就是合理的），当硬失败只会逼人堆豁免。
正确用法是定期人工巡检：`python scripts/deadcode_scan.py . --strict`。

已知局限（务必知悉）
--------------------
**引用判定按「裸名」统计，不做限定名解析**。这是为保持零依赖、零配置而做的
刻意权衡，代价见下：

- 方法名与高频内置 / 三方 API 撞名时会被误判为「已使用」。实测案例：给会话
  序号新增的 `MemoryRedis.get()` 全程零调用，但仓里大量的 `dict.get(...)` /
  `os.getenv(...)` 贡献了 `get` 这个名字，扫描器**漏报**（当时靠人工复核发现）。
- 同类高风险名：`get` / `set` / `save` / `load` / `run` / `update` / `close`。

因此：**本扫描器不是唯一裁判**。请配合 `ruff`（F401/F811 精确到绑定关系）
与人工复核调用点使用；数据访问层、工具类等撞名高发区尤其如此。

注意这**不影响门禁的可信度**：门禁的定位是「新增即拦截」，而新写的死代码
通常就在自己模块内，不会恰好撞上同文件的高频名。
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

#: 默认扫描根目录（相对仓库根）
DEFAULT_ROOTS = ("app", "scripts", "tests")

#: 「生产代码」的路径前缀。`--strict` 模式下只有这个前缀下的文件算作**引用来源**，
#: 也只有这个前缀下的定义会被报告。
PRODUCTION_PREFIX = "app/"

#: 不扫描的目录
EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".pytest_tmp", ".pytest_cache",
    "artifacts", "node_modules", ".git", ".idea", "vector_store",
    "memory_store", "logs", "data",
}

#: 装饰器前缀白名单 —— 带这些装饰器的函数/方法由框架隐式调用，不算未使用。
#: 语义完全对齐 vulture 的 `--ignore-decorators` 与 FastAPI 官方对 Depends 的说明。
EXEMPT_DECORATOR_PREFIXES = (
    "app.",            # FastAPI 应用级：app.get / app.middleware / app.exception_handler
    "router.",         # FastAPI APIRouter
    "pytest.",         # pytest.fixture / pytest.mark.*
    "field_validator", "model_validator", "validator",
    "computed_field",
    "property", "cached_property",
    "staticmethod", "classmethod",
    "abstractmethod", "abstractproperty",
    "contextmanager", "asynccontextmanager",
    "lru_cache", "cache",
    "on_event",
    "task",            # celery / dramatiq 风格任务
)

#: 配置常量消费的额外通道：这些调用形式的字符串参数视为"被引用"。
CONFIG_ACCESSOR_CALLS = ("getattr", "setattr", "hasattr")

#: requirements.txt 包名 → 实际 import 的顶层模块名（不一致时的显式映射）。
DIST_TO_IMPORT = {
    "python-dotenv": "dotenv",
    "python-multipart": "multipart",
    "langchain-text-splitters": "langchain_text_splitters",
    "langchain-openai": "langchain_openai",
    "langchain-community": "langchain_community",
    "langchain-core": "langchain_core",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "python-docx": "docx",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "sentence-transformers": "sentence_transformers",
}

#: 只在运行期通过命令行/框架隐式需要、代码内不 import 的依赖（deptry DEP002 的经典误报）。
#: 条目形如 (包名, 保留理由)。
RUNTIME_ONLY_DEPS: Dict[str, str] = {
    "uvicorn": "ASGI 服务器，由命令行 `uvicorn app.main:app` 启动，业务代码不 import",
    "python-multipart": "FastAPI 解析 multipart/form-data（文件上传）的运行时依赖，无 import 点",
    "pytest": "测试框架，由命令行调用",
    "pytest-asyncio": "pytest 插件，运行时按 entry point 自动加载，无 import 点",
    "httpx": "测试客户端与 httpx 传输层，部分场景由 langchain-openai 间接调用",
}

#: 名字层面的豁免正则（测试函数、脚本入口等）。
EXEMPT_NAME_PATTERNS = (
    "test_*",      # pytest 测试函数
    "Test*",       # pytest 测试类
    "main",        # 脚本入口
    "conftest",
)

#: 永远视为"已使用"的特殊名字
ALWAYS_USED = {
    "__init__", "__main__", "__all__", "__doc__", "__version__",
    "__enter__", "__exit__", "__call__", "__iter__", "__next__",
    "__len__", "__getitem__", "__setitem__", "__contains__",
    "__eq__", "__hash__", "__repr__", "__str__", "__bool__",
    "__aenter__", "__aexit__", "__post_init__",
}

#: 行内豁免标记：`# deadcode: ignore`（与 ruff 的 noqa、deptry 的 # deptry: ignore 同风格）
INLINE_IGNORE_RE = re.compile(r"#\s*(?:deadcode|vulture)\s*:\s*ignore", re.I)

#: 「死代码治理的元数据」文件：它们**不是使用方**。
#:
#: 这些文件里出现某个符号名，含义是"该名字被登记为豁免"，绝不能算作"被使用"——
#: 否则豁免清单会把它自己豁免的死代码洗白，导致条目刚登记就变成僵尸，
#: 而僵尸校验又会因此失败（踩过一次：三条豁免刚写完就报 stale）。
#: 注意这只把文件排除出「引用来源」，它们自身的定义仍会被正常检查。
REFERENCE_EXCLUDED_FILES = ("tests/deadcode_allowlist.py",)

#: 允许「解析基类属性」的可信标准库根模块（只 import 这些，绝不 import 第三方/业务模块，
#: 避免扫描器自身产生副作用）。用于识别"覆盖基类方法＝框架回调"。
TRUSTED_STDLIB_ROOTS = {
    "abc", "asyncio", "collections", "concurrent", "contextlib", "dataclasses",
    "enum", "http", "io", "json", "logging", "multiprocessing", "socketserver",
    "threading", "typing", "unittest", "weakref", "pathlib", "datetime",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Definition:
    """一处定义（函数 / 类 / 方法 / 常量 / 模块）。"""
    name: str
    qualname: str          # 含类前缀的限定名，如 LongTermMemory.update_user_profile
    kind: str              # function / async_function / class / method / constant / module
    file: str              # 仓库相对路径
    lineno: int
    decorators: List[str] = field(default_factory=list)
    is_dunder: bool = False
    is_private: bool = False
    in_class: Optional[str] = None
    exempt: bool = False           # 由装饰器等规则豁免
    exempt_reason: str = ""


@dataclass
class Finding:
    """一条未使用发现。"""
    kind: str              # unused_function / unused_class / unused_method / unused_constant /
                           # unreferenced_module / unused_dependency
    name: str
    file: str
    lineno: int
    qualname: str = ""
    detail: str = ""

    @property
    def key(self) -> str:
        """稳定标识，供 allowlist 匹配（不含行号，避免改文件就失配）。"""
        return f"{self.kind}:{self.file}:{self.qualname or self.name}"

    def __str__(self) -> str:
        return f"{self.file}:{self.lineno}: [{self.kind}] {self.qualname or self.name} — {self.detail}"


# ---------------------------------------------------------------------------
# AST 工具
# ---------------------------------------------------------------------------

def _decorator_name(node: ast.AST) -> str:
    """把装饰器表达式还原成点分名字（去掉调用参数）。"""
    if isinstance(node, ast.Call):
        node = node.func
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_exempt_decorator(dotted: str) -> bool:
    if dotted in EXEMPT_DECORATOR_PREFIXES:
        return True
    return any(dotted.startswith(p) for p in EXEMPT_DECORATOR_PREFIXES if p.endswith("."))


def _iter_module_strings(tree: ast.Module) -> Iterable[str]:
    """产出模块内所有「非 docstring」字符串常量。

    为什么必须排除 docstring：文档里顺口提一句某个函数名，不该把它"洗白"成
    已使用。所以只统计**真正参与运行时逻辑**的字符串（如 `add_node("x", fn)`、
    `getattr(config, "X")` 的参数）。

    实现要点（踩过的坑）：docstring 是「Module / FunctionDef / ClassDef 的
    body[0]」这个 *外层节点* 的属性，不能拿 Constant 去和它的直接父节点
    `ast.Expr` 比对——`Expr` 根本没有 `.body`，判断会恒为 False，导致排除
    静默失效。必须先按外层节点算出 docstring 的节点 id，再统一过滤。
    """
    docstring_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstring_ids.add(id(first.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstring_ids:
                yield node.value


def _has_inline_ignore(source_lines: Sequence[str], lineno: int) -> bool:
    if 1 <= lineno <= len(source_lines):
        return bool(INLINE_IGNORE_RE.search(source_lines[lineno - 1]))
    return False


# ---------------------------------------------------------------------------
# 单文件分析
# ---------------------------------------------------------------------------

class _FileAnalysis:
    """单文件的分析结果。"""

    def __init__(self, rel_path: str, source: str) -> None:
        self.rel_path = rel_path
        self.source = source
        self.lines = source.splitlines()
        self.tree = ast.parse(source, filename=rel_path)
        self.definitions: List[Definition] = []
        self.used_names: Set[str] = set()       # 所有 Name(Load) 标识符
        self.attr_names: Set[str] = set()       # 所有 Attribute.attr
        self.string_names: Set[str] = set()     # 非 docstring 字符串（按词切分后的 token）
        self.imported_modules: Set[str] = set()  # 顶层 import 的模块名
        self.imported_paths: Set[str] = set()    # 完整点分路径（含 from X import Y 的 X.Y）
        self.imported_names: Set[str] = set()    # from x import y 的 y
        self.import_aliases: Dict[str, str] = {}  # 本地名 → 完整点分路径（供基类解析）
        self.all_exports: Set[str] = set()       # __all__ 声明
        self.config_getattr: Set[str] = set()    # getattr(config, "X") 的 X
        self.override_names: Set[str] = set()    # 基类同名方法（框架回调，如 logging.Filter.filter）
        self.inline_ignored_lines: Set[int] = set()
        self._collect()

    # -- 收集 --

    def _collect(self) -> None:
        self._collect_imports_and_uses()
        self._collect_definitions(self.tree, prefix="")

    def _collect_imports_and_uses(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self.used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                self.attr_names.add(node.attr)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.imported_modules.add(alias.name.split(".")[0])
                    self.imported_paths.add(alias.name)
                    self.import_aliases[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imported_modules.add(node.module.split(".")[0])
                    self.imported_paths.add(node.module)
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.imported_names.add(local)
                    # `from app import config` 等价于导入了子模块 app.config——
                    # 漏掉这一步会把 config 这类模块误判为"无人 import"（踩过一次）。
                    if node.module and node.level == 0:
                        self.imported_paths.add(f"{node.module}.{alias.name}")
                        self.import_aliases[local] = f"{node.module}.{alias.name}"
                    elif node.module:
                        self.import_aliases[local] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.Call):
                self._maybe_config_access(node)

        for text in _iter_module_strings(self.tree):
            self.string_names.update(_identifier_tokens(text))

    def _maybe_config_access(self, node: ast.Call) -> None:
        """识别 getattr(config, "X") / hasattr(config, "X") 这种动态配置消费。"""
        func = node.func
        fname = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else ""
        )
        if fname not in CONFIG_ACCESSOR_CALLS or len(node.args) < 2:
            return
        target = node.args[0]
        target_name = target.id if isinstance(target, ast.Name) else (
            target.attr if isinstance(target, ast.Attribute) else ""
        )
        if target_name not in ("config", "settings", "CONFIG"):
            return
        second = node.args[1]
        if isinstance(second, ast.Constant) and isinstance(second.value, str):
            self.config_getattr.add(second.value)

    def _collect_definitions(self, body: ast.AST, prefix: str) -> None:
        """递归收集模块级与类级的定义。"""
        for node in getattr(body, "body", []):
            # __all__ 声明
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                        self._read_all(node.value)

            # 函数 / 类
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.definitions.append(self._mk_definition(node, prefix))
                if isinstance(node, ast.ClassDef):
                    # 类内部：方法 + 类属性
                    self._collect_class_body(node, prefix + node.name + ".")
                continue

            # 模块级常量（UPPER_CASE 或 annotations）
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for name in _target_names(node):
                    if prefix == "" and _looks_like_constant(name):
                        self.definitions.append(Definition(
                            name=name, qualname=name, kind="constant",
                            file=self.rel_path, lineno=node.lineno,
                            is_private=name.startswith("_"),
                        ))

            # 条件块（if TYPE_CHECKING: / try: 等）里的定义也要收集
            if isinstance(node, ast.If):
                self._collect_definitions(node, prefix)
                continue
            if isinstance(node, ast.Try):
                for block in (node.body, getattr(node, "orelse", []), getattr(node, "finalbody", [])):
                    self._collect_definitions(_Block(block), prefix)
                continue

    def _collect_class_body(self, cls: ast.ClassDef, prefix: str) -> None:
        overrides = self._base_class_members(cls)
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                d = self._mk_definition(node, prefix)
                # 覆盖标准库 / 框架基类的同名方法（如 logging.Filter.filter）由框架回调，
                # 不是死代码。按「基类是否真有这个属性」判定，比维护硬编码名单更通用。
                if not d.exempt and d.name in overrides:
                    d.exempt, d.exempt_reason = True, f"覆盖基类方法（框架回调，基类含 {d.name}）"
                self.definitions.append(d)
                if isinstance(node, ast.ClassDef):
                    self._collect_class_body(node, prefix + node.name + ".")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for name in _target_names(node):
                    if _looks_like_constant(name):
                        self.definitions.append(Definition(
                            name=name, qualname=prefix + name, kind="class_attr",
                            file=self.rel_path, lineno=node.lineno,
                            is_private=name.startswith("_"),
                            in_class=prefix.rstrip("."),
                        ))
            elif isinstance(node, ast.If):
                self._collect_class_body(_Block(node.body), prefix)  # type: ignore[arg-type]

    def _base_class_members(self, cls: ast.ClassDef) -> Set[str]:
        """解析基类能提供的属性名集合；只对可信的标准库模块做解析，避免副作用。"""
        members: Set[str] = set()
        for base in cls.bases:
            dotted = _decorator_name(base)
            if not dotted:
                continue
            # 本地别名还原：`import logging` → Filter 未别名时无法解析，只处理点分形式
            full = dotted
            root = dotted.split(".")[0]
            if root in self.import_aliases:
                full = self.import_aliases[root] + dotted[len(root):]
            module_name, _, attr_path = full.partition(".")
            if module_name not in TRUSTED_STDLIB_ROOTS or not attr_path:
                continue
            obj: Any = None
            try:
                module = importlib.import_module(module_name)
                obj = module
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
            except Exception:  # noqa: BLE001 —— 探测基类成员：导入失败/属性不存在都应跳过，不该中断扫描
                continue
            if obj is not None:
                members.update(dir(obj))
        return members

    def _mk_definition(self, node, prefix: str) -> Definition:
        decorators = [_decorator_name(d) for d in node.decorator_list]
        exempt = False
        reason = ""
        for dec in decorators:
            if _is_exempt_decorator(dec):
                exempt, reason = True, f"装饰器 @{dec}（框架隐式调用）"
                break
        kind = "class" if isinstance(node, ast.ClassDef) else (
            "method" if prefix else
            ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
        )
        if prefix and isinstance(node, ast.ClassDef):
            kind = "class"
        return Definition(
            name=node.name, qualname=prefix + node.name, kind=kind,
            file=self.rel_path, lineno=node.lineno, decorators=decorators,
            is_dunder=node.name.startswith("__") and node.name.endswith("__"),
            is_private=node.name.startswith("_"), in_class=prefix.rstrip(".") or None,
            exempt=exempt, exempt_reason=reason,
        )

    def _read_all(self, value: ast.AST) -> None:
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    self.all_exports.add(elt.value)


class _Block:
    """把一个语句列表伪装成有 `.body` 的节点，便于复用 `_collect_definitions`。"""

    def __init__(self, body: List[ast.stmt]) -> None:
        self.body = body


def _target_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(node, ast.Assign):
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                names.append(tgt.id)
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                names.extend(e.id for e in tgt.elts if isinstance(e, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.append(node.target.id)
    return names


def _looks_like_constant(name: str) -> bool:
    """UPPER_CASE 约定（含 _PREFIXED_UPPER）。"""
    stripped = name.lstrip("_")
    return bool(stripped) and stripped.upper() == stripped and any(c.isalpha() for c in stripped)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _identifier_tokens(text: str) -> Set[str]:
    """把字符串切成标识符 token（用于识别字符串形式的动态引用）。"""
    return set(_IDENT_RE.findall(text))


# ---------------------------------------------------------------------------
# 全仓分析
# ---------------------------------------------------------------------------

#: 「入口模块」：由进程启动器按**路径**加载，天然没有 import 点，不该被报未引用。
#:   - `__init__.py` 由解释器在导入包时加载；
#:   - `__main__.py` 由 `python -m pkg` 加载；
#:   - `main.py` 由 `uvicorn app.main:app` / `python app/main.py` 加载
#:     （本项目 Dockerfile CMD 与 README 都是这个用法）。
#: 严格模式下尤其重要：`app/main.py` 只被 `tests/test_dify_api.py` import，
#: 若不排除就会稳定误报成「生产代码里没人用」。
ENTRY_MODULE_NAMES = ("__init__.py", "__main__.py", "main.py")


class RepoAnalyzer:
    def __init__(
        self,
        root: Path,
        roots: Sequence[str] = DEFAULT_ROOTS,
        strict: bool = False,
    ) -> None:
        self.root = root
        self.roots = roots
        #: 严格模式：只把 `app/` 内的引用算作「已使用」，也只报告 `app/` 内的定义。
        #: 语义是把问题从「有没有人用」提升为「**生产代码**里有没有人用」。
        #: 默认关闭——因为严格模式的结论必然包含一批「仅被测试/工具引用」的条目，
        #: 需要逐条人工判断（可能是合理的：如只给 scripts/ 用的度量工具函数），
        #: 不适合直接当 CI 门禁。
        self.strict = strict
        self.files: Dict[str, _FileAnalysis] = {}
        self._load()

    def _is_production(self, rel: str) -> bool:
        return rel.startswith(PRODUCTION_PREFIX)

    def _iter_py_files(self) -> Iterable[Path]:
        for r in self.roots:
            base = self.root / r
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.py")):
                # 按「相对扫描根」的路径判排除，而不是绝对路径——
                # 否则扫描根落在 .pytest_tmp/ 之类目录下时会被整体跳过（踩过一次：
                # 自检用例把临时包建在 .pytest_tmp 内，扫描结果恒为空，测试假通过）。
                rel_parts = p.relative_to(self.root).parts
                if any(part in EXCLUDE_DIRS for part in rel_parts):
                    continue
                yield p

    def _load(self) -> None:
        for p in self._iter_py_files():
            rel = str(p.relative_to(self.root))
            try:
                src = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                self.files[rel] = _FileAnalysis(rel, src)
            except SyntaxError as exc:  # 语法错误单独暴露，不静默吞掉
                print(f"[WARN] 跳过无法解析的文件 {rel}: {exc}", file=sys.stderr)

    # -- 引用索引 --

    @property
    def reference_files(self) -> Iterable[Tuple[str, "_FileAnalysis"]]:
        """可作为「引用来源」的文件——排除死代码治理的元数据文件。

        `--strict` 下进一步只保留 `app/`：测试与工具脚本的引用不算数，
        这样"只被 tests/ 或 scripts/ 调用的生产 API"才会暴露出来。
        """
        for rel, fa in self.files.items():
            if rel in REFERENCE_EXCLUDED_FILES:
                continue
            if self.strict and not self._is_production(rel):
                continue
            yield rel, fa

    @property
    def all_used_names(self) -> Set[str]:
        used: Set[str] = set()
        for _rel, fa in self.reference_files:
            used |= fa.used_names
            used |= fa.attr_names
            used |= fa.string_names
            used |= fa.imported_names
            used |= fa.config_getattr
        return used

    @property
    def all_import_roots(self) -> Set[str]:
        """全仓被 import 的顶层模块名。

        **`--strict` 不影响本项**：它服务于「声明的依赖是否被用到」，
        而被 scripts/ 或 tests/ 用到的库仍然是真实依赖，按严格模式报成
        "未使用依赖"会导致误删（删了脚本就跑不起来）。
        """
        roots: Set[str] = set()
        for fa in self.files.values():
            roots |= fa.imported_modules
        return roots

    @property
    def all_import_paths(self) -> Set[str]:
        """被 import 的完整点分路径（含 `from app import config` 推出的 app.config）。

        `--strict` 下只看 `app/` 内部的 import：判定的问题是「这个模块在生产
        代码里可达吗」，而不是「仓库里有没有人 import 过它」。
        """
        paths: Set[str] = set()
        for _rel, fa in self.reference_files if self.strict else self.files.items():
            paths |= fa.imported_paths
        return paths

    # -- 死代码检测 --

    def scan_dead_code(self) -> List[Finding]:
        """未被引用的函数 / 类 / 方法 / 常量。

        `--strict` 下只报告 `app/` 内的定义：本模式要回答的是"生产代码里有谁
        没被生产代码用到"。测试目录里"只被其它测试用到"的辅助函数属于正常用法，
        报出来只会制造噪音，把真正的信号淹掉。
        """
        used = self.all_used_names
        findings: List[Finding] = []

        # 统计每个「裸名」被引用的次数（跨文件累计），用于区分"仅定义处出现"
        for rel, fa in self.files.items():
            if self.strict and not self._is_production(rel):
                continue
            for d in fa.definitions:
                if d.exempt or d.is_dunder or d.name in ALWAYS_USED:
                    continue
                if any(fnmatch.fnmatch(d.name, pat) for pat in EXEMPT_NAME_PATTERNS):
                    continue
                if _has_inline_ignore(fa.lines, d.lineno):
                    continue
                if d.name in used:
                    continue

                # Pydantic / TypedDict / dataclass 字段：同名会出现在构造处，已由 used 覆盖；
                # 到这里说明确实无人引用。
                kind = {
                    "function": "unused_function",
                    "async_function": "unused_function",
                    "class": "unused_class",
                    "method": "unused_method",
                    "class_attr": "unused_class_attr",
                    "constant": "unused_constant",
                }.get(d.kind, "unused_symbol")

                findings.append(Finding(
                    kind=kind, name=d.name, qualname=d.qualname,
                    file=rel, lineno=d.lineno,
                    detail=_describe(d),
                ))
        return sorted(findings, key=lambda f: (f.file, f.lineno))

    def scan_unreferenced_modules(self) -> List[Finding]:
        """检测「无人 import 的模块」——排除了包初始化文件与测试/脚本目录。"""
        findings: List[Finding] = []
        imported = self.all_import_paths

        # 补齐 langchain 风格的子包导入：`import app.rag.lexical` 会隐含引用各级包
        expanded = set()
        for path in imported:
            parts = path.split(".")
            for i in range(1, len(parts) + 1):
                expanded.add(".".join(parts[:i]))
        # 字符串形式的动态 import（如 importlib.import_module("app.rag.lexical")）
        for _rel, fa in self.reference_files:
            for name in fa.string_names:
                if name.startswith("app.") or name.startswith("scripts."):
                    expanded.add(name)
                    parts = name.split(".")
                    for i in range(1, len(parts) + 1):
                        expanded.add(".".join(parts[:i]))

        for rel, fa in self.files.items():
            p = Path(rel)
            if p.name in ENTRY_MODULE_NAMES:
                continue  # 入口模块由启动器按路径加载，不需要被 import
            if "tests" in p.parts or "scripts" in p.parts:
                continue  # 脚本/测试本身是入口，不需要被 import
            module = str(p.with_suffix("")).replace("/", ".")
            if module in expanded:
                continue
            if _has_inline_ignore(fa.lines, 1):
                continue
            findings.append(Finding(
                kind="unreferenced_module", name=p.name, qualname=module,
                file=rel, lineno=1,
                detail="整个模块无任何 import 引用点（也未被 __init__ re-export）",
            ))
        return sorted(findings, key=lambda f: f.file)

    def scan_dead_config(self, config_rel: str = "app/config.py") -> List[Finding]:
        """检测 config.py 中「定义了但无人消费」的配置项。

        消费通道覆盖：`config.X` 属性访问、`from app.config import X`、
        `getattr(config, "X")`、以及其它模块直接 `os.getenv("X")` 的字符串引用。

        **刻意不把「出现在 .env / .env.example」当作被消费**：
        配置文件里有这个键，只能说明"有人在配"，不能说明"代码在读"。
        恰恰相反——`定义在 config.py + 写在 .env + 全仓无读取点` 才是最危险的
        伪配置：运维照着 .env 调整它，行为毫无变化且不报错。
        （此前的实现把 .env 键当作已消费，导致 CHUNK_FALLBACK_OVERLAP_RATIO
        这类真死配置被漏报，已修正。）
        """
        fa = self.files.get(config_rel)
        if not fa:
            return []

        # 外部引用：所有文件里出现过的 Attribute.attr / Name / 字符串 token
        external: Set[str] = set()
        for rel, other in self.reference_files:
            if rel == config_rel:
                continue
            external |= other.attr_names
            external |= other.used_names
            external |= other.imported_names
            external |= other.string_names
            external |= other.config_getattr

        # config.py 自身的引用（如 BASE_DIR 被同文件 LOG_DIR 使用）。
        # 注意只取 Name/Attribute，**不取字符串**——否则 `X = _env("X", ...)` 的自引用
        # 会把每个配置项都"洗白"，死配置检测直接失效。
        external |= fa.used_names
        external |= fa.attr_names
        external |= fa.config_getattr

        findings: List[Finding] = []
        for d in fa.definitions:
            if d.kind != "constant" or d.is_dunder:
                continue
            if _has_inline_ignore(fa.lines, d.lineno):
                continue
            if d.name in external:
                continue
            findings.append(Finding(
                kind="dead_config", name=d.name, qualname=d.name,
                file=config_rel, lineno=d.lineno,
                detail="config.py 中定义，但全仓无 config.X / getattr / import 读取点"
                       "（.env 里配了也不算被消费）",
            ))
        return sorted(findings, key=lambda f: f.lineno)

    def scan_unused_dependencies(self, req_file: str = "requirements.txt") -> List[Finding]:
        """检测声明了但代码中从未 import 的第三方依赖。"""
        req_path = self.root / req_file
        if not req_path.exists():
            return []
        declared = _parse_requirements(req_path.read_text(encoding="utf-8"))
        import_roots = self.all_import_roots

        findings: List[Finding] = []
        for dist, _spec in declared:
            module = DIST_TO_IMPORT.get(dist, dist.replace("-", "_"))
            if module in import_roots or dist.replace("-", "_") in import_roots:
                continue
            if dist in RUNTIME_ONLY_DEPS:
                continue
            findings.append(Finding(
                kind="unused_dependency", name=dist, qualname=dist,
                file=req_file, lineno=0,
                detail="已在 requirements.txt 声明，但全仓无任何 import 点",
            ))
        return sorted(findings, key=lambda f: f.name)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _describe(d: Definition) -> str:
    where = {"method": "方法", "class": "类", "constant": "模块常量",
             "class_attr": "类属性", "function": "函数",
             "async_function": "异步函数"}.get(d.kind, d.kind)
    kind = "私有" if d.is_private else "公开"
    return f"{kind}{where}，全仓无调用点 / 属性访问 / 字符串引用"


def _parse_requirements(text: str) -> List[Tuple[str, str]]:
    """解析 requirements.txt，返回 [(规范化包名, 版本约束)]。"""
    out: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        if not m:
            continue
        out.append((m.group(1).lower(), m.group(2).strip()))
    return out


def default_roots_for(root: Path) -> List[str]:
    return [r for r in DEFAULT_ROOTS if (root / r).exists()]


def scan_findings(
    root: Path,
    roots: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> List[Finding]:
    """一站式入口：返回全部未使用发现（供测试与 CLI 复用）。

    Args:
        root: 仓库根目录。
        roots: 扫描根（默认 `app` / `scripts` / `tests`）。
        strict: 严格模式，只把 `app/` 内的引用算作「已使用」。
            会额外暴露"仅被测试或工具脚本引用的生产 API"。**不用于 CI 门禁**：
            其结论需要逐条人工判断（只给 scripts/ 用的度量工具是合理的），
            适合作为定期人工巡检而非硬失败。
    """
    analyzer = RepoAnalyzer(
        root,
        tuple(roots) if roots else tuple(default_roots_for(root)),
        strict=strict,
    )
    findings: List[Finding] = []
    findings += analyzer.scan_dead_code()
    findings += analyzer.scan_unreferenced_modules()
    findings += analyzer.scan_dead_config()
    findings += analyzer.scan_unused_dependencies()
    return findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_GROUPS = [
    ("未引用模块（unreferenced_module）", {"unreferenced_module"}),
    ("未使用函数 / 方法（unused_function/method）", {"unused_function", "unused_method"}),
    ("未使用类（unused_class）", {"unused_class"}),
    ("未使用常量 / 类属性（unused_constant/class_attr）", {"unused_constant", "unused_class_attr"}),
    ("死配置（dead_config）", {"dead_config"}),
    ("未使用依赖（unused_dependency）", {"unused_dependency"}),
]


def _build_parser() -> argparse.ArgumentParser:
    """命令行参数。

    兼容老用法：`python scripts/deadcode_scan.py .`（首个位置参数即仓库根）。
    """
    parser = argparse.ArgumentParser(
        prog="deadcode_scan.py",
        description="未使用代码 / 配置 / 依赖检测（懂本项目的三类隐式引用约定）",
    )
    parser.add_argument(
        "root", nargs="?", default=".",
        help="仓库根目录（默认当前目录）",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "严格模式：只把 app/ 内的引用算作「已使用」，只报告 app/ 内的定义。"
            "用于找出「仅被 tests/ 或 scripts/ 引用」的生产 API。"
            "结论需人工逐条判断，不适合直接当 CI 门禁"
        ),
    )
    return parser


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv if argv is not None else sys.argv[1:]))
    root = Path(args.root).resolve()
    findings = scan_findings(root, strict=args.strict)

    if args.strict:
        print(
            f"[严格模式] 只统计 {PRODUCTION_PREFIX} 内的引用；\n"
            "           下列条目意味着「生产代码里没人用」，但可能被测试或工具脚本使用，\n"
            "           需逐条判断：是补生产接线、还是连同调用方一起下线。\n"
        )

    if not findings:
        print("未发现未使用代码 / 依赖。")
        return 0

    print(f"共发现 {len(findings)} 项未使用问题：\n")
    seen: Set[int] = set()
    for title, kinds in _GROUPS:
        group = [f for f in findings if f.kind in kinds]
        group.sort(key=lambda f: (f.file, f.lineno))
        if not group:
            continue
        print(f"== {title}：{len(group)} 项 ==")
        for f in group:
            print(f"   {f}")
            seen.add(id(f))
        print()
    rest = [f for f in findings if id(f) not in seen]
    if rest:
        print(f"== 其他：{len(rest)} 项 ==")
        for f in sorted(rest, key=lambda f: (f.file, f.lineno)):
            print(f"   {f}")

    print(
        "\n提示：引用判定按裸名统计，`get`/`save`/`run` 等高频名撞名会漏报；"
        "请配合 ruff 与人工复核，勿把本工具当唯一裁判。"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
