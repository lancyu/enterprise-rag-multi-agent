#!/usr/bin/env python3
"""模块清单生成器：提取每个模块的 docstring 首段 + 对外定义（def/class）。

用途：为「全量模块盘点」提供结构化底稿，避免人工逐个打开文件。
输出：TSV/文本，按目录分组。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOTS = ("app", "scripts", "tests")
SKIP_PARTS = {".venv", "__pycache__", ".pytest_tmp", "artifacts"}


def first_line(text: str | None) -> str:
    if not text:
        return ""
    for line in text.strip().splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def analyze(path: Path) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    mod_doc = first_line(ast.get_docstring(tree))
    defs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = first_line(ast.get_docstring(node))
            defs.append(f"def {node.name}()  # {doc}")
        elif isinstance(node, ast.ClassDef):
            doc = first_line(ast.get_docstring(node))
            methods = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("_")
            ]
            defs.append(
                f"class {node.name}  # {doc}"
                + (f"  [方法: {', '.join(methods)}]" if methods else "")
            )
    return {
        "path": str(path),
        "lines": len(src.splitlines()),
        "doc": mod_doc,
        "defs": defs,
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files: list[Path] = []
    for r in ROOTS:
        base = root / r
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if any(part in SKIP_PARTS for part in p.parts):
                continue
            files.append(p)

    for p in files:
        try:
            info = analyze(p)
        except SyntaxError as exc:
            print(f"!! 语法错误 {p}: {exc}")
            continue
        print(f"\n### {info['path']}  ({info['lines']} 行)")
        if info["doc"]:
            print(f"    {info['doc']}")
        for d in info["defs"]:
            print(f"      - {d}")


if __name__ == "__main__":
    main()
