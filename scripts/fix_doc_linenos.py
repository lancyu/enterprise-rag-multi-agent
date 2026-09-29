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
- 旧值在该行出现**多次**时跳过（可能替换错位置），**除非**报文里带了符号名：
  那就挑离那个名字最近的一次（见 ``_pick_occurrence``）—— 行号常常撞车
  （``ROUTE_RETRY_BUDGET`` 与 ``VERIFIER_MODE_CHOICES`` 都写着 193），
  唯一能区分它们的就是紧挨着的那个名字；
- 多候选区间（如 ``AST 实为 [(1,2), (5,6)]``）跳过；
- 所有跳过项**逐条列出**，交人工处理，绝不猜。
- 落盘前自动备份原文档到 ``artifacts/backup/doc-linenos-<时间戳>/``。

⚠️ 本脚本**不自己推断行号**，但有三类报文只给"哪里不对"、不给"对的是多少"
（点名符号跨度不符 / 匿名跨度不符 / 函数级区间漂移）。这三类靠
:class:`_Ctx` 从**源码 AST** 取真值（``collect_symbols``），
仍然是"校验器已经算过的那件事"，不是另起一套推断。

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

import prune_backups
from verify_doc_linenos import DEFAULT_DOC, collect_symbols, verify

#: 「文档写 A-B，实际 1-C」——文件总行数（区间式）
_P_TOTAL_RANGE = re.compile(r"^L(\d+): .*? 文档写 (\d+)-(\d+)，实际 1-(\d+)$")
#: 「文档写第 X 行，AST 实为 [(C, D)]」——分区表里夹带的**单项**符号行号。
#: 与 _P_SYMBOL 的差别：这里声明的是单个数字（"第 X 行"），不是区间。
#: 同时捕获 ``file::NAME`` —— 它是**消歧锚点**（同一行里两个 193 靠它区分）。
_P_BARE_SYMBOL = re.compile(
    r"^L(\d+): ([\w/\.]+)::(\w+) 文档写第 (\d+) 行，AST 实为 \[\((\d+), \d+\)\]$"
)
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

# ---------------------------------------------------------------------------
# 2026-09-24 补的三类：校验器的第 12/15 类新报文
# ---------------------------------------------------------------------------
# 这三类过去**只能手工改**：报文只说"哪里不对"，正确值写在别处。实测一次代码
# 改动就产生 14 条，逐条手改既慢又正是"改错一位看不出来"的场景 —— 而这三种
# 形状恰好是**代码一动就必然出现**的那批（函数上下移动 / 整表错位）。
#
# 它们的共同点是"真值可从源码 AST 直接取"，于是交给 `_Ctx` 去取，
# 本脚本仍然不做任何自己的推断。
#: 第 15 类（函数级区间）：`` `name` 文档写 A-B，AST 实为 file:S-E ``（可能是多个，``、`` 连接）
_P_R15_RANGE = re.compile(r"^L(\d+): `([^`]+)` 文档写 (\d+)-(\d+)，AST 实为 (.+)$")
#: 第 12 类（点名符号）：``FILE 的 `NAME` 指向 A-B，但该文件里没有符号恰好是这个跨度``
_P_R12_NAMED = re.compile(
    r"^L(\d+): ([\w/\.]+\.py) 的 `([^`]+)` 指向 (\d+)-(\d+)，但该文件里没有符号恰好是这个跨度$"
)
#: 第 12 类（匿名区间）：``FILE 里没有任何符号恰好跨 A-B`` —— 报文不带名字，
#: 改从**那一行的原文**里找（`effective_score_threshold`（`config.py:458-465`））。
_P_R12_ANON = re.compile(r"^L(\d+): ([\w/\.]+\.py) 里没有任何符号恰好跨 (\d+)-(\d+)$")

#: 从 ``AST 实为 app/x.py:140-224、tests/y.py:9-10`` 这类尾巴里拆出候选。
_R_SPAN_IN_TAIL = re.compile(r"([\w/\.]+\.py):(\d+)-(\d+)")
#: 反引号里的标识符（用于在文档行上找"这一处讲的是哪个符号"）。
_R_BACKTICK_NAME = re.compile(r"`([A-Za-z_][\w\.]*)`")


class _Ctx:
    """解析上述三类报文所需的两类事实：源码符号跨度 + 文档行原文。

    为什么需要它：那三类报文**只报错、不给正确值**（"没有任何符号恰好跨
    458-465"）。真值只有源码 AST 知道，所以由这里去取 —— 取法与校验器
    完全一致（同一个 ``collect_symbols``），不是第二套推断。

    取不到、或候选不唯一时**一律返回 None**（交人工）：这是本脚本的
    fail-closed 底线。宁可让人看一眼，也不能把行号写到另一个函数头上 ——
    后者的症状与改对了完全同形，下次校验还会"通过"。
    """

    def __init__(self, symbols: dict, lines: list[str]):
        self._symbols = symbols
        self._lines = lines

    def spans(self, file: str, name: str) -> list[tuple[int, int]]:
        """``file`` 里名为 ``name`` 的符号跨度（去重排序）；查不到返回空。"""
        return sorted(set(self._symbols.get(file, {}).get(name) or ()))

    def sole_symbol_on_line(self, lineno: int, file: str) -> str | None:
        """文档第 ``lineno`` 行上**唯一**属于 ``file`` 的符号名；0 个或多个返回 None。

        "多个就放弃"是刻意的：一行里出现两个候选时，无从判断这处区间说的是哪一个。
        """
        if not 1 <= lineno <= len(self._lines):
            return None
        bucket = self._symbols.get(file, {})
        found: list[str] = []
        for raw in _R_BACKTICK_NAME.findall(self._lines[lineno - 1]):
            short = raw.rsplit(".", 1)[-1]
            if short in bucket and short not in found:
                found.append(short)
        return found[0] if len(found) == 1 else None


def _fmt_like(raw: str, value: int) -> str:
    """按原书写格式输出数字（原值带千分位就继续带）。"""
    return f"{value:,}" if "," in raw else str(value)


def parse(message: str, ctx: _Ctx | None = None) -> tuple[int, str, str, str] | None:
    """把一条问题翻译成 ``(行号, 旧文本, 新文本, 锚点)``；无法唯一确定时返回 None。

    ``锚点`` 是"这一处讲的是哪个符号名"，只在旧值于该行出现多次时用来消歧，
    取不到就是空串（届时按旧规矩跳过）。
    """
    m = _P_REGION_TAIL.match(message)
    if m:
        return int(m.group(1)), m.group(2), m.group(3), m.group(2)

    m = _P_R15_RANGE.match(message)
    if m:
        cands = _R_SPAN_IN_TAIL.findall(m.group(5))
        if ctx is not None and len(cands) == 1:
            return (
                int(m.group(1)), f"{m.group(3)}-{m.group(4)}",
                f"{cands[0][1]}-{cands[0][2]}", m.group(2),
            )
        return None                       # 同名符号在多处定义 → 猜不得

    m = _P_R12_NAMED.match(message)
    if m:
        spans = ctx.spans(m.group(2), m.group(3)) if ctx else []
        if len(spans) == 1:
            return int(m.group(1)), f"{m.group(4)}-{m.group(5)}", f"{spans[0][0]}-{spans[0][1]}", m.group(3)
        return None

    m = _P_R12_ANON.match(message)
    if m:
        name = ctx.sole_symbol_on_line(int(m.group(1)), m.group(2)) if ctx else None
        spans = ctx.spans(m.group(2), name) if (ctx and name) else []
        if len(spans) == 1:
            return int(m.group(1)), f"{m.group(3)}-{m.group(4)}", f"{spans[0][0]}-{spans[0][1]}", name or ""
        return None

    m = _P_PROSE.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}-{m.group(5)}", ""

    m = _P_SYMBOL.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}-{m.group(5)}", ""

    m = _P_BARE_SYMBOL.match(message)
    if m:
        return int(m.group(1)), m.group(4), m.group(5), m.group(3)

    m = _P_CONFIG_SECTION.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"{m.group(4)}-{m.group(5)}", ""

    m = _P_TOTAL_RANGE.match(message)
    if m:
        return int(m.group(1)), f"{m.group(2)}-{m.group(3)}", f"1-{m.group(4)}", ""

    m = _P_TOTAL_BARE.match(message)
    if m:
        return int(m.group(1)), m.group(2), _fmt_like(m.group(2), int(m.group(3))), ""

    m = re.match(r"^L(\d+): app/[\w]+/ 文档写 ([\d,]+) 行，实际 (\d+)$", message)
    if m:
        return int(m.group(1)), m.group(2), _fmt_like(m.group(2), int(m.group(3))), ""

    m = re.match(r"^L(\d+): app/ 总计 文档写 ([\d,]+) 行，实际 (\d+)$", message)
    if m:
        return int(m.group(1)), m.group(2), _fmt_like(m.group(2), int(m.group(3))), ""

    return None


def plan(doc: str) -> tuple[list[tuple[int, str, str, str]], list[str]]:
    """返回 ``(可自动修复项, 需人工处理的问题)``。"""
    lines = pathlib.Path(doc).read_text(encoding="utf-8").splitlines()
    symbols, _ = collect_symbols("app")
    ctx = _Ctx(symbols, lines)
    fixes: list[tuple[int, str, str, str]] = []
    manual: list[str] = []
    for message in verify(doc):
        parsed = parse(message, ctx)
        if parsed:
            fixes.append(parsed)
        else:
            manual.append(message)
    return fixes, manual


def _find_anchor(line: str, anchor: str) -> int:
    """找 ``anchor`` 作为**独立标识符**出现的位置；找不到返回 -1。

    ⚠️ 不能直接用 ``line.find`` —— 实测就栽在这上面：锚点 ``DOCSTORE_STRATEGY``
    会先命中 ``DOCSTORE_STRATEGY_CHOICES`` 里的那一段（前缀吞并），于是
    "离锚点最近"变成了"离另一个符号最近"，把同一行上两个数字**换错了位**。
    （后果是原本只错一处变成错两处 —— 由校验器当场逮住，见 __main__ 的用法说明。）
    所以这里要求锚点两侧都不是标识符字符。
    """
    if not anchor:
        return -1
    start = 0
    while True:
        at = line.find(anchor, start)
        if at < 0:
            return -1
        before = line[at - 1] if at else ""
        after = line[at + len(anchor):at + len(anchor) + 1]
        ok_before = not (before.isalnum() or before == "_")
        ok_after = not (after.isalnum() or after == "_")
        if ok_before and ok_after:
            return at
        start = at + 1


def _pick_occurrence(line: str, old: str, anchor: str) -> int | None:
    """``old`` 在 ``line`` 里出现多次时，挑**离 anchor 最近**的那一次的下标。

    为什么需要：行号撞车是常态而非例外（``ROUTE_RETRY_BUDGET`` 与
    ``VERIFIER_MODE_CHOICES`` 在同一行都写着 193，而只有后者变了）。
    唯一能区分两者的，是紧跟数字的那个**符号名** —— 报文里本来就带着它。
    距离相同时返回 None（谁也说服不了谁），交人工。
    """
    at = _find_anchor(line, anchor)
    if at < 0:
        return None
    end = at + len(anchor)
    positions: list[int] = []
    cursor = line.find(old)
    while cursor >= 0:
        positions.append(cursor)
        cursor = line.find(old, cursor + 1)
    if not positions:
        return None
    ranked = sorted((abs(pos - end), pos) for pos in positions)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def apply_fixes(
    lines: list[str], fixes: list[tuple[int, str, str, str]]
) -> tuple[list[str], list[str]]:
    """逐行应用替换；旧值不唯一或找不到的项被拒绝（fail-closed）。"""
    rejected: list[str] = []
    by_line: dict[int, list[tuple[str, str, str]]] = {}
    for lineno, old, new, anchor in fixes:
        by_line.setdefault(lineno, []).append((old, new, anchor))

    out = list(lines)
    for lineno, pairs in sorted(by_line.items()):
        if not (1 <= lineno <= len(out)):
            rejected.append(f"L{lineno}: 行号越界，文档只有 {len(out)} 行")
            continue
        line = out[lineno - 1]
        for old, new, anchor in pairs:
            if old == new:
                continue
            hits = line.count(old)
            if hits == 0:
                rejected.append(f"L{lineno}: 找不到旧值 {old!r}（文档可能已被改动）")
                continue
            if hits > 1:
                at = _pick_occurrence(line, old, anchor)
                if at is None:
                    rejected.append(
                        f"L{lineno}: 旧值 {old!r} 在该行出现 {hits} 次，"
                        f"且锚点 {anchor!r} 无法唯一定位，跳过"
                    )
                    continue
                line = line[:at] + new + line[at + len(old):]
                continue
            line = line.replace(old, new, 1)
        out[lineno - 1] = line
    return out, rejected


def _prune_backups(root: pathlib.Path) -> None:
    """把自动快照收敛回保留上界之内（论证见 `prune_backups.py`）。

    这里是**收纳**，不是**前置**：回填已经写完了，清理失败不该把回填结果一起否掉，
    所以 fail-open。但失败必须打出来——"没清理"和"清理成功"在静默时长得一模一样，
    而下一次 `--write` 又会叠加一份，问题会一直藏到磁盘上多出几十份快照才被人看见。

    ⚠️ 必须显式传 ``apply=True``：`prune()` 默认是 dry-run。漏了这个参数的表现极难查——
    接线在、日志在、还打印了"已清掉 N 份"，**实际一份没动**，快照数每次照样 +1。
    （真踩过一次，是 `test_the_backfill_step_prunes_for_real` 把它钉住的。）
    """
    try:
        kept, dropped, dest = prune_backups.prune(root, apply=True)
    except (OSError, shutil.Error) as exc:  # 权限 / 占用 / 跨设备，都不该影响回填
        print(f"⚠️  旧快照清理失败（不影响本次回填）：{exc}")
        return
    if dest is not None:
        print(f"已清掉 {len(dropped)} 份旧快照 → {dest}")
    elif dropped:
        # 有候选、又是 apply=True，却没拿到废纸篓目录 —— 保留策略没真正执行。
        # 这里**绝不能**打印"已清掉"：把没做的事说成做了，比不说更难查。
        print(f"⚠️  {len(dropped)} 份快照超出上界但【未被移动】——保留策略没生效，请检查 prune()")
    else:
        print(f"快照共 {len(kept)} 份，在保留上界内")


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
    _prune_backups(backup_dir.parent)

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
