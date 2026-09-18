"""按校验器的结论回填文档行号（默认 dry-run，加 ``--write`` 才落盘）。

为什么需要它
------------
``verify_doc_linenos.py`` 是个**门禁**：代码一改行号就漂移，它会红。
但红了之后，43 处数字靠人肉逐个改，既慢又容易改错（这已是本项目第三次漂移）。
本脚本把这一步自动化。

为什么不做成 ``verify_doc_linenos.py --fix``
--------------------------------------------
校验器是门禁，往里加写能力会把「漏报」一起改进去。本脚本是**修复器**，
定位不同：它不自己推断任何行号，只把校验器**已经算出**的实际值写回文档。
两者共用同一份结论 —— 校验器说哪里不一致、实际值是多少，脚本就写什么，
因此不可能出现「修复器和校验器各自漂移」。

安全策略（fail-closed）
-----------------------
- 只处理能**唯一解析**出「旧值 → 新值」的问题；
- 旧值在该行出现**多次**时跳过（可能替换错位置）；
- 多候选区间（如 ``AST 实为 [(1,2), (5,6)]``）跳过；
- 所有跳过项**逐条列出**，交人工处理，绝不猜。
- 落盘前自动备份原文档到 ``artifacts/backup/doc-linenos-<时间戳>/``。

用法
----
    python scripts/fix_doc_linenos.py            # 只看会改什么（dry-run）
    python scripts/fix_doc_linenos.py --write    # 实际写入

退出码：0 = 没有待修项或已全部写入；1 = 存在无法自动修复的项。
"""
from __future__ import annotations

import pathlib
import re
import shutil
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from verify_doc_linenos import DEFAULT_DOC, verify

#: 「文档写 A-B，实际 1-C」——文件总行数（区间式）
_P_TOTAL_RANGE = re.compile(r"^L(\d+): .*? 文档写 (\d+)-(\d+)，实际 1-(\d+)$")
#: 「文档写第 X 行，AST 实为 [(C, D)]」——分区表里夹带的**单项**符号行号。
#: 与 _P_SYMBOL 的差别：这里声明的是单个数字（"第 X 行"），不是区间。
_P_BARE_SYMBOL = re.compile(r"^L(\d+): .*? 文档写第 (\d+) 行，AST 实为 \[\((\d+), \d+\)\]$")
#: 「文档写 A-B，AST 实为 [(C, D)]」——符号行号（含松散单元格、模块标题）
_P_SYMBOL = re.compile(r"^L(\d+): .*? 文档写 (\d+)-(\d+)，AST 实为 \[\((\d+), (\d+)\)\]$")
#: 「文档写 X，实际 Y」——附录裸数字 / 通用文件行数
_P_TOTAL_BARE = re.compile(r"^L(\d+): .*? 文档写 ([\d,]+)，实际 (\d+)$")
#: 散文引用漂移：「app/x.py:A-B 未精确命中任何符号…附近实际为 [(C, D)]」
_P_PROSE = re.compile(
    r"^L(\d+): (?:app/[\w/]+\.py):(\d+)-(\d+) 未精确命中任何符号.*?"
    r"附近实际为 \[\((\d+), (\d+)\)\]$"
)
#: 区域表末段没收尾：「区域表止于 X，但 app/x.py 共 Y 行」
_P_REGION_TAIL = re.compile(r"^L(\d+): 区域表止于 (\d+)，但 .*? 共 (\d+) 行")
#: 配置分区表（第 13 类）：「配置分区表第 N 行写 A-B，但源码横幅对应的是 C-D」
#: 加这一条之前，本脚本对这张 25 行的表**零覆盖** —— 而它是最容易整体错位的一张
#: （`app/config.py` 任何增删都会让它全部下移），每次只能照着校验器的报错手工重写。
_P_CONFIG_SECTION = re.compile(
    r"^L(\d+): 配置分区表第 \d+ 行写 (\d+)-(\d+)，但源码横幅对应的是 (\d+)-(\d+)$"
)


def _fmt_like(raw: str, value: int) -> str:
    """按原书写格式输出数字（原值带千分位就继续带）。"""
    return f"{value:,}" if "," in raw else str(value)


def parse(message: str) -> tuple[int, str, str] | None:
    """把一条问题翻译成 ``(行号, 旧文本, 新文本)``；无法唯一确定时返回 None。"""
    m = _P_REGION_TAIL.match(message)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)

    m = _P_PROSE.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}-{m.group(5)}"

    m = _P_SYMBOL.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}-{m.group(5)}"

    m = _P_BARE_SYMBOL.match(message)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)

    m = _P_CONFIG_SECTION.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}-{m.group(5)}"

    m = _P_TOTAL_RANGE.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"1-{m.group(4)}"

    m = _P_TOTAL_BARE.match(message)
    if m:
        return int(m.group(1)), m.group(2), _fmt_like(m.group(2), int(m.group(3)))

    m = re.match(r"^L(\d+): app/[\w]+/ 文档写 ([\d,]+) 行，实际 (\d+)$", message)
    if m:
        return int(m.group(1)), m.group(2), _fmt_like(m.group(2), int(m.group(3)))

    m = re.match(r"^L(\d+): app/ 总计 文档写 ([\d,]+) 行，实际 (\d+)$", message)
    if m:
        return int(m.group(1)), m.group(2), _fmt_like(m.group(2), int(m.group(3)))

    return None


def plan(doc: str) -> tuple[list[tuple[int, str, str]], list[str]]:
    """返回 ``(可自动修复项, 需人工处理的问题)``。"""
    fixes: list[tuple[int, str, str]] = []
    manual: list[str] = []
    for message in verify(doc):
        parsed = parse(message)
        if parsed:
            fixes.append(parsed)
        else:
            manual.append(message)
    return fixes, manual


def apply_fixes(lines: list[str], fixes: list[tuple[int, str, str]]) -> tuple[list[str], list[str]]:
    """逐行应用替换；旧值不唯一或找不到的项被拒绝（fail-closed）。"""
    rejected: list[str] = []
    by_line: dict[int, list[tuple[str, str]]] = {}
    for lineno, old, new in fixes:
        by_line.setdefault(lineno, []).append((old, new))

    out = list(lines)
    for lineno, pairs in sorted(by_line.items()):
        if not (1 <= lineno <= len(out)):
            rejected.append(f"L{lineno}: 行号越界，文档只有 {len(out)} 行")
            continue
        line = out[lineno - 1]
        for old, new in pairs:
            if old == new:
                continue
            hits = line.count(old)
            if hits == 0:
                rejected.append(f"L{lineno}: 找不到旧值 {old!r}（文档可能已被改动）")
                continue
            if hits > 1:
                rejected.append(f"L{lineno}: 旧值 {old!r} 在该行出现 {hits} 次，位置不唯一，跳过")
                continue
            line = line.replace(old, new, 1)
        out[lineno - 1] = line
    return out, rejected


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    doc = args[0] if args else DEFAULT_DOC
    path = pathlib.Path(doc)
    if not path.exists():
        print(f"找不到文档：{doc}")
        return 2

    fixes, manual = plan(doc)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fixed_lines, rejected = apply_fixes([ln.rstrip("\n") for ln in lines], fixes)
    skipped = manual + rejected

    print(f"待修 {len(fixes)} 处；无法自动修复 {len(skipped)} 处")
    for message in skipped:
        print("   ⚠️ ", message)

    if not write:
        print("\n（dry-run，未写入。加 --write 生效）")
        return 1 if skipped else 0

    if not fixes:
        print("无需改动。")
        return 1 if skipped else 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = pathlib.Path("artifacts") / "backup" / f"doc-linenos-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)
    print(f"\n原文档已备份：{backup_dir / path.name}")

    path.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")
    print(f"已写入 {doc}")

    remaining = verify(doc)
    if remaining:
        print(f"\n❌ 仍有 {len(remaining)} 处未修复：")
        for item in remaining:
            print("   ✗", item)
        return 1
    print("\n✅ 回填后再次校验：全部与源码 AST 一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
