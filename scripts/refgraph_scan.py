#!/usr/bin/env python3
"""引用图分析 —— 弥补死代码扫描器的「裸名撞名漏报」。

背景
----
`scripts/deadcode_scan.py` 判定一个定义是否被使用时，取的是全仓
「裸名 / 属性名 / 字符串 token / import 名」的**并集**。只要同名名字在
任何地方出现过，就认为它被用到了。后果：`app/core/prompts.py` 的
`render` 与 `app/static/gen_favicon.py` 的 `render` 撞名，
`prompts.render` 的存在感被 `gen_favicon.render` 完全掩盖 → 漏报。

本脚本改用**限定名解析**，为每个定义收集四类证据：

  own_ref      `X.attr` 解析出的限定名 == 本定义       → 强证据
  import_ref   `from <本模块> import <名字>`           → 强证据
  self_ref     同类内 `self.attr` / `cls.attr`         → 强证据（仅方法）
  borrowed     同名名字只被别的定义 / 标准库类型用着   → **弱证据**

判定：
  ALIVE   有任一强证据
  SUSPECT 只有弱证据 → 死代码就藏在这里（扫描器漏报区）
  DEAD    连弱证据都没有

接收者还原规则（置信度从高到低）：
  1. 模块别名（`from app.core import prompts` → `prompts.render` 解析为
     `app.core.prompts.render`）
  2. 实例变量：文件内 `x = MemoryStore()` / `def f(x: MemoryStore)` /
     `self._s = MemoryStore()` → `x.ttl` 解析为 `MemoryStore.ttl`
     （**仅当类名首字母大写**才视为构造，避免把 `decision = route_model(...)`
     误当成类）
  3. `self.` / `cls.` → 同类内方法
  4. 其余 → 「未知接收者」，原样列出供人工一秒判定

豁免规则**直接复用主扫描器**（装饰器 / dunder / 内联 ignore），避免两套
豁免逻辑各自漂移。

用法：
  python scripts/refgraph_scan.py .                # SUSPECT + DEAD
  python scripts/refgraph_scan.py . --all          # 连 ALIVE 一起看
  python scripts/refgraph_scan.py . --name render  # 追踪某个名字的全部引用点
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deadcode_scan as dcs  # 复用主扫描器的 AST 与豁免逻辑（依赖上面 sys.path 注入）

SKIP_PARTS = {".venv", "__pycache__", ".pytest_tmp", "artifacts", ".git"}


def module_path_of(rel: str) -> str:
    p = rel[:-3] if rel.endswith(".py") else rel
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


def receiver_expr(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{receiver_expr(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"{receiver_expr(node.func)}()"
    if isinstance(node, ast.Subscript):
        return f"{receiver_expr(node.value)}[...]"
    return type(node).__name__


class Ref:
    """一处 `X.attr` 引用点，以及 X 指向的候选目标。"""

    __slots__ = ("file", "lineno", "receiver", "classes", "module")

    def __init__(
        self,
        file: str,
        lineno: int,
        receiver: str,
        classes: frozenset[str] = frozenset(),
        module: str = "",
    ) -> None:
        self.file = file
        self.lineno = lineno
        self.receiver = receiver
        self.classes = classes      # 接收者可能是这些类（简单名）
        self.module = module        # 接收者是模块别名时，记录模块全路径

    @property
    def target(self) -> str:
        if self.module:
            return self.module
        if self.classes:
            return "{" + ", ".join(sorted(self.classes)) + "}"
        return f"<未解析:{self.receiver}>"


def _looks_like_class(name: str) -> bool:
    """首字母大写才当类：把 `decision = route_model(...)` 排除在构造之外。

    **必须剥掉前导下划线**——`_RouteStats` / `_RateLimitRetryModel` /
    `_cache_instance` 都是真实类名，`_` 开头不能当成普通函数。
    """
    stripped = name.lstrip("_")
    return bool(stripped) and stripped[0].isupper()


def _func_name(expr: ast.AST) -> str:
    """取调用/引用处的函数名（裸名或 `mod.func` 的末段）。"""
    if isinstance(expr, ast.Call):
        return _func_name(expr.func)
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    if isinstance(expr, ast.Await):
        return _func_name(expr.value)
    return ""


def unwrap(expr: ast.AST) -> ast.AST:
    """剥掉 await / 括号等不影响类型的包装。"""
    while isinstance(expr, (ast.Await,)):
        expr = expr.value
    return expr


def collect_func_returns(trees: dict[str, ast.AST]) -> dict[str, set[str]]:
    """函数名 → 可能返回的类名集合。

    目的是让 `x = get_vector_store()` 之后的 `x.add(...)` 能落到
    `MemoryVectorStore.add` / `ChromaVectorStore.add` 上——否则所有
    「工厂函数返回接口实现」的方法都会被误判为无人调用。

    会迭代到不动点：`_cache()` 返回 `get_embedding_cache()` 的调用结果，
    而后者返回 `EmbeddingCache` 实例 → 两层都能解析。
    """
    returns: dict[str, set[str]] = {}

    for _ in range(5):  # 足够的迭代层数；链式工厂一般不超过 2~3 层
        changed = False

        def resolve(expr: ast.AST, local: dict[str, set[str]]) -> set[str]:
            """单点的类别推断；func_returns 取上一轮的结果（不动点迭代）。"""
            expr = unwrap(expr)
            if isinstance(expr, ast.BoolOp):  # `a or b` / `a and b`
                out: set[str] = set()
                for v in expr.values:
                    out |= resolve(v, local)
                return out
            if isinstance(expr, ast.Name):
                if expr.id in local:
                    return set(local[expr.id])
                return {expr.id} if _looks_like_class(expr.id) else returns.get(expr.id, set())
            if isinstance(expr, ast.IfExp):  # `a if cond else b`
                return resolve(expr.body, local) | resolve(expr.orelse, local)
            if not isinstance(expr, ast.Call):
                return set()
            direct = _direct_class(expr)
            if direct:
                return {direct}
            fn = _func_name(expr)
            if fn in returns:
                return set(returns[fn])
            if fn in local:
                return set(local[fn])
            # `MemoryStore(...)` 经由变量传入的情况由 local / 注解覆盖
            return {fn} if _looks_like_class(fn) else set()

        for tree in trees.values():
            module_vars: dict[str, set[str]] = {}
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    c = _direct_class(node.value)
                    if c:
                        module_vars.setdefault(node.targets[0].id, set()).add(c)
                elif (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.annotation is not None
                ):
                    c = _annot_classes(node.annotation)
                    if c:
                        module_vars.setdefault(node.target.id, set()).update(c)

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                local: dict[str, set[str]] = dict(module_vars)
                found: set[str] = set()
                # 返回注解也是线索：`def f() -> EmbeddingCache`
                if node.returns is not None:
                    found |= _annot_classes(node.returns)
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Assign)
                        and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name)
                    ):
                        c = resolve(sub.value, local)
                        if c:
                            local.setdefault(sub.targets[0].id, set()).update(c)
                    elif isinstance(sub, ast.Return) and sub.value is not None:
                        found |= resolve(sub.value, local)
                if found:
                    before = len(returns.get(node.name, set()))
                    returns.setdefault(node.name, set()).update(found)
                    if len(returns[node.name]) != before:
                        changed = True

        if not changed:
            break

    return returns


def _annot_classes(expr: ast.AST) -> set[str]:
    """从类型注解里取类名：`Optional[MemoryStore]` / `"EmbeddingCache"`。"""
    if expr is None:
        return set()
    if isinstance(expr, ast.Name):
        return {expr.id} if _looks_like_class(expr.id) else set()
    if isinstance(expr, ast.Attribute):
        return {expr.attr} if _looks_like_class(expr.attr) else set()
    if isinstance(expr, ast.Subscript):
        return _annot_classes(expr.slice)
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        tail = expr.value.split(".")[-1]
        return {tail} if _looks_like_class(tail) else set()
    if isinstance(expr, ast.BinOp):  # `X | None`（PEP 604）
        return _annot_classes(expr.left) | _annot_classes(expr.right)
    return set()


def _direct_class(expr: ast.AST) -> str:
    """`MemoryVectorStore(...)` / `mod.Cls(...)` → 类名；否则空串。"""
    expr = unwrap(expr)
    if not isinstance(expr, ast.Call):
        return ""
    f = expr.func
    if isinstance(f, ast.Name) and _looks_like_class(f.id):
        return f.id
    if isinstance(f, ast.Attribute) and _looks_like_class(f.attr):
        return f.attr
    return ""


def build_aliases(tree: ast.AST) -> dict[str, str]:
    """本地名 → 模块全路径。"""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.level:
                continue
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def build_var_classes(
    tree: ast.AST,
    aliases: dict[str, str],
    func_returns: dict[str, set[str]],
) -> dict[str, set[str]]:
    """变量名 → 可能的类名集合。只认首字母大写的构造、类型注解、工厂返回值。

    误差方向是「多认成活的」——保守估计，不会凭空产生假阳性。
    """
    var_cls: dict[str, set[str]] = {}

    def bind(name: str, classes: set[str]) -> None:
        if classes:
            var_cls.setdefault(name, set()).update(classes)

    def class_of(expr: ast.AST) -> set[str]:
        expr = unwrap(expr)
        # `a or b` / `a and b`：两侧都可能是构造（`store or MemoryStore(...)`）
        if isinstance(expr, ast.BoolOp):
            out: set[str] = set()
            for v in expr.values:
                out |= class_of(v)
            return out
        if isinstance(expr, ast.IfExp):
            return class_of(expr.body) | class_of(expr.orelse)
        if isinstance(expr, ast.Call):
            direct = _direct_class(expr)
            if direct:
                return {direct}
            # 工厂函数：`x = get_vector_store()` → {MemoryVectorStore, ChromaVectorStore}
            fn = _func_name(expr)
            if fn in func_returns:
                return set(func_returns[fn])
            return set()
        if isinstance(expr, ast.Name):
            if expr.id in var_cls:
                return set(var_cls[expr.id])
            return {expr.id} if _looks_like_class(expr.id) else set()
        if isinstance(expr, ast.Attribute):
            return {expr.attr} if _looks_like_class(expr.attr) else set()
        return set()

    def annot_of(expr: ast.AST) -> set[str]:
        if expr is None:
            return set()
        if isinstance(expr, ast.Name) and _looks_like_class(expr.id):
            return {expr.id}
        if isinstance(expr, ast.Attribute) and _looks_like_class(expr.attr):
            return {expr.attr}
        if isinstance(expr, ast.Subscript):
            return annot_of(expr.slice)
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            tail = expr.value.split(".")[-1]
            return {tail} if _looks_like_class(tail) else set()
        return set()

    # 模块级变量优先（供函数内的 `return _instance` 推断）
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                bind(tgt.id, class_of(node.value))
            elif (
                isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
            ):
                bind(f"self.{tgt.attr}", class_of(node.value))
        elif isinstance(node, (ast.AnnAssign, ast.arg)):
            if isinstance(node, ast.AnnAssign):
                bind_target = node.target
                classes = annot_of(node.annotation)
            else:
                bind_target = ast.Name(id=node.arg)
                classes = annot_of(node.annotation)
            if isinstance(bind_target, ast.Name):
                bind(bind_target.id, classes)
            elif (
                isinstance(bind_target, ast.Attribute)
                and isinstance(bind_target.value, ast.Name)
                and bind_target.value.id == "self"
            ):
                bind(f"self.{bind_target.attr}", classes)

    return var_cls


def scan(root: Path):
    """返回 (名字 → 引用点列表, 模块 → 精确 import 名集合, 裸名 Load 表)。"""
    refs: dict[str, list[Ref]] = defaultdict(list)
    imports: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    bare: dict[str, list[tuple[str, int]]] = defaultdict(list)
    # `getattr(obj, "name")` 形式的动态读取——不限定 obj 是 config。
    # `app/rag/indexer.py` 就用 `getattr(store, "get_texts", None)` 取向量库能力，
    # 只认 config 会把这类真实使用判成死代码。
    dyn: dict[str, list[str]] = defaultdict(list)
    ranges: dict[tuple[str, int], tuple[int, int]] = {}

    # 第一遍：把所有文件解析成 AST（返回类型推断需要全仓视野）
    trees: dict[str, ast.AST] = {}
    sources: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        rel = str(p.relative_to(root))
        if not any(rel.startswith(r + "/") for r in dcs.DEFAULT_ROOTS):
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            trees[rel] = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        sources[rel] = src
        paths[rel] = p

    # 第二遍：推断「哪个函数会返回哪个类」，供工厂函数调用点解析
    func_returns = collect_func_returns(trees)

    for rel, tree in trees.items():
        aliases = build_aliases(tree)
        var_cls = build_var_classes(tree, aliases, func_returns)

        def visit(node: ast.AST, enclosing: str) -> None:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                bare[node.id].append((rel, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for a in node.names:
                    imports[node.module][a.name].append(rel)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr", "setattr")
                and len(node.args) >= 2
            ):
                second = node.args[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    dyn[second.value].append(rel)

            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    visit(child, node.name)
                return

            if isinstance(node, ast.Attribute):
                name = node.attr
                base = node.value
                if isinstance(base, ast.Name):
                    b = base.id
                    if b == "self":
                        refs[name].append(
                            Ref(rel, node.lineno, "self",
                                frozenset({enclosing}) if enclosing else frozenset())
                        )
                    elif b == "cls":
                        refs[name].append(
                            Ref(rel, node.lineno, "cls",
                                frozenset({enclosing}) if enclosing else frozenset())
                        )
                    elif b in var_cls:
                        refs[name].append(
                            Ref(rel, node.lineno, b, frozenset(var_cls[b]))
                        )
                    elif b in aliases:
                        refs[name].append(
                            Ref(rel, node.lineno, b, module=aliases[b])
                        )
                    else:
                        refs[name].append(Ref(rel, node.lineno, b))
                else:
                    recv = receiver_expr(base)
                    classes: frozenset[str] = frozenset()
                    if (
                        isinstance(base, ast.Attribute)
                        and isinstance(base.value, ast.Name)
                        and base.value.id == "self"
                    ):
                        classes = frozenset(var_cls.get(f"self.{base.attr}", set()))
                    elif isinstance(base, ast.Call):
                        fn = _func_name(base)
                        classes = frozenset(func_returns.get(fn, set()))
                    refs[name].append(Ref(rel, node.lineno, recv, classes))

            for child in ast.iter_child_nodes(node):
                visit(child, enclosing)

        visit(tree, "")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ranges[(rel, node.lineno)] = (
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno) or node.lineno,
                )

    return refs, imports, bare, ranges, dyn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--all", action="store_true", help="连 ALIVE 一起打印")
    ap.add_argument("--name", help="只追踪某个名字的全部引用点")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    refs, imports, bare, ranges, dyn = scan(root)

    if args.name:
        print(f"=== 名字 `{args.name}` 的全部属性访问点 ===")
        for r in sorted(refs.get(args.name, []), key=lambda x: (x.file, x.lineno)):
            print(f"  {r.file}:{r.lineno}  {r.receiver}.{args.name}  →  {r.target}")
        print(f"\n=== `{args.name}` 的裸名使用点 ===")
        for f, ln in sorted(bare.get(args.name, [])):
            print(f"  {f}:{ln}")
        print("\n=== 该名字的定义点 ===")
        for p in sorted(root.rglob("*.py")):
            if any(part in SKIP_PARTS for part in p.parts):
                continue
            rel = str(p.relative_to(root))
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) and node.name == args.name:
                    print(f"  {rel}:{node.lineno}  {module_path_of(rel)}.{node.name}")
        return

    analyzer = dcs.RepoAnalyzer(root)          # 默认口径：与门禁跑的完全一致
    scanner_used = analyzer.all_used_names     # 扫描器认为「有人用」的名字全集
    # getattr(config, "X", 默认值) 形式的动态读取——必须算作证据，
    # 否则 config 里所有走 getattr 兜底的项都会被误判成死配置。
    getattr_files: dict[str, set[str]] = defaultdict(set)
    for fa in analyzer.files.values():
        for name in fa.config_getattr:
            getattr_files[name].add(fa.rel_path)

    def is_prod(path: str) -> bool:
        return path.startswith(dcs.PRODUCTION_PREFIX)

    alive, test_only, suspect, dead = [], [], [], []

    for rel, fa in analyzer.files.items():
        if not rel.startswith("app/"):
            continue
        modpath = module_path_of(rel)
        for d in fa.definitions:
            if d.exempt or d.is_dunder or d.name in dcs.ALWAYS_USED:
                continue
            if any(fnmatch.fnmatch(d.name, pat) for pat in dcs.EXEMPT_NAME_PATTERNS):
                continue
            if dcs._has_inline_ignore(fa.lines, d.lineno):
                continue

            allrefs = refs.get(d.name, [])
            owner = d.qualname.rsplit(".", 2)[-2] if "." in d.qualname else ""

            own: list[Ref] = []
            borrowed: list[Ref] = []
            for r in allrefs:
                if r.module == modpath:                      # `mod.name` 精确指向
                    own.append(r)
                elif owner and owner in r.classes:            # `obj.name`，obj 可能是本类
                    own.append(r)
                else:
                    borrowed.append(r)

            imp_files = imports.get(modpath, {}).get(d.name, [])
            own_body = ranges.get((rel, d.lineno))
            bare_here = [
                (f, ln)
                for f, ln in bare.get(d.name, [])
                if not (own_body and f == rel and own_body[0] <= ln <= own_body[1])
            ]

            # 每条证据都标上来源目录，用于区分「生产在用」与「只有测试在用」
            sources: list[tuple[str, str]] = []   # (证据描述, 来源文件)
            for r in own:
                sources.append((f"指向本定义的引用 {r.receiver}.{d.name}", r.file))
            for f in imp_files:
                sources.append(("显式 from-import", f))
            for f in getattr_files.get(d.name, set()):
                sources.append(("getattr(config, …) 动态读取", f))
            for f in dyn.get(d.name, []):
                sources.append(("getattr(obj, …) 动态读取", f))
            for f, ln in bare_here:
                sources.append((f"裸名调用 {ln} 行", f))

            prod = [s for s in sources if is_prod(s[1])]
            entry = (f"{rel}:{d.lineno}", d.qualname, d.kind, borrowed, sources)

            if prod:
                alive.append((entry, prod))
            elif sources:
                test_only.append(entry)      # 只有 tests/ 或 scripts/ 在用
            elif d.name in scanner_used:
                suspect.append(entry)        # 扫描器说有人用，却没有任何证据指向它
            else:
                dead.append(entry)

    def show(items, title, with_sources=False):
        print(f"\n### {title}（{len(items)} 项）")
        for loc, qn, kind, borrowed, sources in items:
            print(f"  {loc:40s} {qn:36s} [{kind}]")
            if with_sources:
                for desc, f in sorted(sources, key=lambda s: s[1])[:6]:
                    print(f"        ← {f}  {desc}")
            seen: set[tuple[str, str]] = set()
            for r in sorted(borrowed, key=lambda x: (x.file, x.lineno)):
                key = (r.file, r.target)
                if key in seen:
                    continue
                seen.add(key)
                print(f"        × {r.file}:{r.lineno}  {r.receiver}.{qn.split('.')[-1]} → {r.target}")

    print("[限定名引用分析] 只覆盖 app/ 内定义；豁免规则复用主扫描器。")
    show(dead, "DEAD — 门禁也认为没人用（应与主扫描器默认模式一致）")
    show(suspect, "SUSPECT — 扫描器认为『有人用』，但无证据指向本定义（= 撞名漏报区）")
    show(
        test_only,
        "TEST-ONLY — 生产代码里没人用，只有 tests/ 或 scripts/ 在用",
        with_sources=True,
    )
    if args.all:
        print(f"\n### ALIVE — 生产代码在用（{len(alive)} 项）")
        for (loc, qn, kind, _b, _s), ev in alive:
            print(f"  {loc:40s} {qn:36s} [{kind}]  ← {'; '.join(desc for desc, _ in ev)}")


if __name__ == "__main__":
    sys.exit(main())
