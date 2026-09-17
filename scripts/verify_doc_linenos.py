"""校验文档里的行号声明是否与源码一致。

为什么需要它
------------
`docs/project-introduction.md` 的核心价值是「每个模块在文件的第几行」，
但**代码一改行号就漂移**，而文档不会报错、只会悄悄变错。
靠人肉复核几百条声明不现实，所以把复核本身自动化。

它校验十一类声明
----------------
1. **符号行号**：表格里 ``| `func_name` | 120-145 |`` 形式的行，
   用 AST 取该符号真实的 ``lineno`` / ``end_lineno`` 比对。
   支持类方法（``Class.method``）与函数内嵌套函数。
2. **文件总行数（区间式）**：``| `app/foo.py` | **1-583** |`` 形式。
3. **文件总行数（附录裸数字式）**：``| `config.py` | 632 | `main.py` | 214 |``。
   附录 A/B 的「完整文件索引」用的是这种两列裸数字表。它与第 2 类写法不同，
   早期版本只认区间式，于是**整块附录从未被校验**——文档悄悄变错也没人知道。
4. **章节/全量小计（散文式）**：``### 3.2 `app/api/` — HTTP 接口层（1,478 行）``
   与 ``**应用代码 `app/`（12,868 行）**``。这类数字不在表格里，同样会悄悄变错。
5. **模块标题里的行号范围**：``#### 📍 `app/config.py`（1-612）``。
   早期只拿这一行定位「当前文件」，括号里的数字从不比对，于是它成了
   最大的一块盲区（`config.py` 实际 632 行却一直写着 612）。
6. **松散单元格里的符号行号**：``| `prompts.py` | `PROMPTS` 18 / `render` 103-112 |``。
   第二列不是纯区间，第 1 类的正则认不出，同样是盲区。
7. **散文引用**：正文里 ``app/foo.py:120-145`` 形式。**要求精确命中某个符号**，
   而不只是「不越界」——代码在中间插行后，旧范围往往仍然界内，
   却已指向另一个函数。实测一次重构后有 9 条这样静默漂移。
8. **通用文件行数声明**：`` `README.md`（362 行）``、``Dockerfile``（35 行）``，
   以及 §2.1 架构图里**没加反引号**的 ``app/config.py（583 行）``。
   这类横跨 .py / .md / .yml / .txt 且不在 `app/` 下，此前从未被覆盖。
9. **区域行号表（裸区间）**：``| 178-204 | 向量数据库配置 |`` 这种既无文件名
   也无符号名的行，靠最近的 `#### 📍 <文件>` 标题定位。校验「递增不重叠 +
   界内 + **末段收到文件最后一行**」——最后一条抓的就是「文件变长、表没跟着长」。
10. **散文引用（精确性）**：见第 7 类。它与「越界」是两个独立判据：
    越界是硬错，不精确是**静默漂移**，后者更常见也更难发现。
11. **不带文件名的符号引用**：表格里的 ``| `CHANNELS` | 70 |``（单数字）与散文里的
    ``（`_decide` 565-690）``。**前十类全都要求行里有文件名**，所以这两类整片落空。
    文件由最近的 `#### 📍 <文件>` 标题提供，判据同第 7 类（精确等于 AST 跨度）。
    实测：补上它之后声明数 522 → 557（**此后文档自身还会新增声明，总数以运行输出为准**），
    并立刻多抓出 2 处同型漂移（`retriever.py::_attach_parent_content`、
    `generator.py::estimate_confidence`）。

第 3～11 类都是**后来补齐的盲区**。共同的教训是：
校验器只覆盖了它「认得出」的写法，认不出的写法会静默通过，
于是「全部一致」这个结论本身是可疑的——**先把覆盖面补全，再去改文档**。
第 3～10 类补的都是「文档里**有**某种写法但没人校验」；
第 11 类不同，它补的是**一个结构性前提**：前十类都建立在「行里有文件名」之上，
而文档里有整整一类引用是靠小标题继承上下文的。**补覆盖面时别只看正则，
要看它隐含的前提。**
另外两条教训见 ``_SECTION_DIR_RE`` 与 ``_ROW_RE`` 的注释：
正则重名会让上下文继承静默失效（把正确声明报成错误）；
文件名模式漏掉 ``/`` 会让整行被静默跳过（把过期声明放过去）。
**报错和漏报，都要先怀疑校验器。**

附录 A/B 会引用 ``tests/`` 与 ``scripts/`` 下的文件，因此索引范围是
``app/ + tests/ + scripts/``（见 ``SOURCE_ROOTS``）。

上下文继承
----------
文档里为了可读性，表格中常写**裸文件名**（目录写在章节标题上），
例如标题 ``### 3.2 `app/api/` — HTTP 接口层`` 下的表格写 ``chat.py``。
本脚本会记住最近的章节标题里的目录，并据此把裸名解析成完整路径。

用法
----
    python scripts/verify_doc_linenos.py                    # 默认校验项目介绍文档
    python scripts/verify_doc_linenos.py path/to/doc.md     # 校验指定文档

退出码：0 = 全部一致；1 = 有不一致（可直接接进 CI）。
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

DEFAULT_DOC = "docs/project-introduction.md"
SOURCE_ROOT = "app"
#: 附录 A/B 也会声明 tests/ 与 scripts/ 下文件的总行数，一并索引才校验得到。
SOURCE_ROOTS = (SOURCE_ROOT, "tests", "scripts")

#: 章节标题里声明的目录，如 "### 3.2 `app/api/` — HTTP 接口层"
#: 这是**上下文继承**的基础：裸文件名靠它才知道该解析到哪个目录。
#: 注意别和下面的 `_SUBTOTAL_DIR_RE` 重名——踩过一次：重名会让本正则被覆盖，
#: 于是 current_dir 恒为 None，裸名解析退化成「按固定顺序猜」，
#: `rerank.py` 猜到 `providers/`、`__init__.py` 猜到 `app/`，
#: 校验器把 6 条**正确**的声明报成错误（校验器自己的缺陷被读成文档的错）。
_SECTION_DIR_RE = re.compile(r"^###\s+.*?`(app/[\w/]*/?)`")
#: 模块小标题，如 "#### 📍 `app/rag/retriever.py`（1-506）"
_MODULE_HEAD_RE = re.compile(r"^####\s+📍\s+(?:[^`]*?)`?([\w/\.]+\.py)`?")
#: 小标题里紧跟文件名的行号范围，如 "（1-612）"。
#: 一行可能写多个：§4.3 的标题同时带 `edges.py`（1-56）与 `workflow_graph.py`（1-104）。
#: 早期实现只拿它定位 current_file、**不校验这两个数字**，于是
#: `app/config.py`（1-612）能一直挂着错值（实际 632），而校验器仍报「全部一致」。
_MODULE_HEAD_RANGE_RE = re.compile(r"`([\w/\.]+\.py)`\s*（(\d+)\s*-\s*(\d+)）")
#: 表行：| `name` | 120-145 |  或  | ├ `name` | **120-145** |
#: 名字里必须允许 `/`：§3.1 顶层用的是全路径（`app/config.py`），
#: 早期模式只认 ``[A-Za-z_][\w\.]*``，于是把 ``app/config.py | **1-583**``
#: 里的 "app" 当成符号名、又因为没有当前文件而**整行跳过**——
#: 这条 1-583 明明早已过期（实际 632），却一直报「全部一致」。
_ROW_RE = re.compile(
    r"^\|\s*[├└]?\s*`([A-Za-z_][\w/\.]*)`[^|]*\|\s*\*{0,2}(\d+)\s*-\s*(\d+)\*{0,2}"
)
#: 附录里的两列裸数字表：| `config.py` | 632 | `main.py` | 214 |
#: 与 _ROW_RE 的 `1-583` 区间形式互斥——这里要求数字后面直接跟 `|`（不是 `-`）。
#: 末尾必须用**前瞻**而不是吃掉 `|`：附录是两列表，`| 214 | \`config.py\` | 612 |`
#: 若把分隔用的 `|` 也匹配掉，第二列就永远不会被检查到（自己踩过这个坑）。
_ROW_TOTAL_RE = re.compile(r"\|\s*`([\w][\w/\.]*\.py)`\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*(?=\|)")
#: 章节小计：``### 3.2 `app/api/` — HTTP 接口层（1,478 行）``
#: 同一行可能写多个目录（3.9 同时写了 tools 与 utils），故按出现顺序配对。
#: 名字必须与 `_SECTION_DIR_RE` 区分开（后者负责上下文继承，语义完全不同）。
_SUBTOTAL_DIR_RE = re.compile(r"`app/(\w+)/`")
_SECTION_NUM_RE = re.compile(r"（([\d,]+)\s*行）")
#: 全量小计：``**应用代码 `app/`（12,858 行）**`` 与表格里的 ``应用代码 | **12,858 行**``
_APP_TOTAL_RE = re.compile(r"应用代码[^0-9]*([\d,]+)\s*行")
#: 散文引用：`app/foo.py:120-145`
_PROSE_RE = re.compile(r"`?(app/[\w/]+\.py):(\d+)-(\d+)`?")
#: 「松散单元格」：``| `prompts.py` | `PROMPTS` 18 / `render` 103-112 |``。
#: 第二列不是纯区间（前面还有文字或符号名），`_ROW_RE` 匹配不到，长期是盲区。
_LOOSE_ROW_RE = re.compile(r"^\|\s*[├└]?\s*`([A-Za-z_][\w\.]*\.py)`\s*\|(.*)\|\s*$")
#: 松散单元格里的 `` `符号名` 120-145 `` 或 `` `常量名` 18 ``
_LOOSE_NAMED_RE = re.compile(r"`([A-Za-z_]\w*)`\s*(\d+)(?:\s*-\s*(\d+))?")
#: 第 11 类：**不带文件名的符号引用**。符号名与数字之间只允许空白或 `:`/`|`，
#: 文件由最近的 `#### 📍 <文件>` 标题提供。写法形如::
#:
#:     | `CHANNELS` | 70 | 包内别名……            ← 单数字，`_ROW_RE` 只认区间
#:     （`_build_raw_model` 195-223 显式写 `max_retries=0`）   ← 散文，且不带文件名
#:
#: 前十类**全都要求行里有文件名**（`_ROW_RE` 靠行首单元格、`_LOOSE_ROW_RE` 靠
#: `x.py` 单元格、`_PROSE_RE` 靠 `app/` 前缀），于是这类写法整片落空。
#: 注意符号名模式**故意不允许 `/` 和 `.`**：那样才能天然排除 `app/config.py` 这类
#: 文件名，不必再写一条负向断言。
_CTX_REF_RE = re.compile(r"`([A-Za-z_]\w*)`\s*[:\|]?\s*(\d+)(?:\s*[-–~]\s*(\d+))?")
#: 数字后面紧跟中文量词 → 那是计数不是行号（`` `SCENES` 5 次``）。
#: 含 `行`：宁可漏判「`x` 55 行」这种少见的行号写法，也不要误报「5 行」这种长度。
_QUANTIFIER_RE = re.compile(r"\s*(次|个|条|项|处|人|天|张|组|份|种|行)")
#: 纯区间单元格，如 ``**1-583**`` —— 这类交给 `_ROW_RE`，松散流程要跳过
_STRICT_TOTAL_RE = re.compile(r"^\s*\**\d+\s*-\s*\d+\**\s*$")
#: 「区域行号表」的裸区间行：``| 178-204 | 向量数据库配置（…）|``。
#: 这类行既没有文件名也没有符号名，只能靠最近的 `#### 📍 <文件>` 标题定位。
_BARE_RANGE_RE = re.compile(r"^\|\s*(\d+)\s*-\s*(\d+)\s*\|")
#: 通用文件行数声明：`` `README.md`（362 行）``、``Dockerfile``（35 行）``，
#: 以及 §2.1 架构图里**没加反引号**的 ``app/config.py（583 行）``。
#: 这类声明横跨 .py / .md / .yml / .txt，且不在 `app/` 下，此前完全没被覆盖。
_ANY_FILE_RE = re.compile(
    r"`?((?:[\w/\.\-]*[\w\-]\.(?:py|md|ya?ml|txt|toml|lock|sh|json))|Dockerfile|Makefile)"
    r"`?\s*（([\d,]+)\s*行"
)
#: 建通用文件索引时要跳过的目录（虚拟环境 / 缓存 / 数据 / 归档）
_SKIP_DIRS = {".venv", "__pycache__", ".git", "data", "logs", ".workbuddy",
              "_archive", "node_modules", ".pytest_cache", ".ruff_cache"}

_SUBPACKAGES = ("api", "core", "db", "graph", "memory",
                "providers", "rag", "tools", "utils")


def collect_symbols(root: str) -> tuple[dict, dict]:
    """扫描源码，返回 {文件: {符号名: [(起, 止), ...]}} 与 {文件: 总行数}。

    同名符号（不同类各有一个 ``search``）会存成列表；
    同时额外登记 ``Class.method`` 形式的限定名，便于精确定位。
    """
    symbols: dict = {}
    totals: dict = {}

    for path in sorted(pathlib.Path(root).rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        source = path.read_text(encoding="utf-8")
        key = str(path)
        totals[key] = len(source.splitlines())
        bucket = symbols.setdefault(key, {})

        def walk(node: ast.AST, prefix: str = "") -> None:
            for child in node.body:  # type: ignore[attr-defined]
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    span = (child.lineno, child.end_lineno)
                    bucket.setdefault(child.name, []).append(span)
                    bucket.setdefault(prefix + child.name, []).append(span)
                    walk(child, prefix + child.name + ".")
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    # 模块级 / 类级常量也要能被校验。
                    #
                    # 为什么补这一段：文档里存在 `ROUTE_TARGETS | 60-66` 这类
                    # 「常量行号」声明，而最初的校验器只索引函数与类，于是把
                    # 正确的声明误报成「找不到符号」——**校验器的盲区会被误读成
                    # 文档的错误**。宁可先把校验器补全，再去改文档。
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            bucket.setdefault(target.id, []).append(
                                (child.lineno, child.end_lineno)
                            )
                        elif isinstance(target, ast.Attribute):
                            bucket.setdefault(target.attr, []).append(
                                (child.lineno, child.end_lineno)
                            )

        walk(ast.parse(source))

    return symbols, totals


def collect_any_files() -> dict[str, list[pathlib.Path]]:
    """索引仓库里的**所有**普通文件，返回 {文件名: [路径, ...]}。

    用于校验 ``README.md（362 行）`` / ``Dockerfile（35 行）`` 这类声明。
    同名文件会有多个候选，调用方据此判歧义。
    """
    found: dict[str, list[pathlib.Path]] = {}
    for path in sorted(pathlib.Path(".").rglob("*")):
        if not path.is_file():
            continue
        if _SKIP_DIRS & set(path.parts):
            continue
        found.setdefault(path.name, []).append(path)
    return found


def make_resolver(totals: dict):
    """构造「裸文件名 → 完整路径」的解析器（按章节目录优先）。"""

    def resolve(name: str, current_dir: str | None) -> str | None:
        if name in totals:
            return name
        candidates = [f"{current_dir}{name}" if current_dir else None, f"app/{name}"]
        candidates += [f"app/{pkg}/{name}" for pkg in _SUBPACKAGES]
        for cand in candidates:
            if cand and cand in totals:
                return cand
        return None

    return resolve


def verify(doc_path: str) -> list[str]:
    """返回不一致清单；空列表表示全部通过。"""
    symbols: dict = {}
    totals: dict = {}
    for root in SOURCE_ROOTS:
        root_symbols, root_totals = collect_symbols(root)
        symbols.update(root_symbols)
        totals.update(root_totals)
    resolve = make_resolver(totals)
    any_files = collect_any_files()
    lines = pathlib.Path(doc_path).read_text(encoding="utf-8").splitlines()

    current_dir: str | None = None
    current_file: str | None = None
    problems: list[str] = []
    counted = 0

    for lineno, line in enumerate(lines, 1):
        m = _SECTION_DIR_RE.match(line)
        if m:
            current_dir = m.group(1) if m.group(1).endswith("/") else m.group(1) + "/"
            continue

        m = _MODULE_HEAD_RE.match(line)
        if m:
            current_file = resolve(m.group(1), current_dir)
            # 标题括号里的 `（1-612）` 同样是行号声明，同样会漂移，必须一起校验。
            for hm in _MODULE_HEAD_RANGE_RE.finditer(line):
                hname, hstart, hend = hm.group(1), int(hm.group(2)), int(hm.group(3))
                key = resolve(hname, current_dir)
                if not key:
                    problems.append(f"L{lineno}: 无法解析文件 `{hname}`")
                    continue
                counted += 1
                if (hstart, hend) != (1, totals[key]):
                    problems.append(
                        f"L{lineno}: {key} 文档写 {hstart}-{hend}，实际 1-{totals[key]}"
                    )
            continue

        m = _ROW_RE.match(line)
        if not m:
            continue
        name, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        counted += 1

        if name.endswith(".py"):                     # --- 文件总行数 ---
            key = resolve(name, current_dir)
            if not key:
                problems.append(f"L{lineno}: 无法解析文件 `{name}`")
            elif (start, end) != (1, totals[key]):
                problems.append(
                    f"L{lineno}: {key} 文档写 {start}-{end}，实际 1-{totals[key]}"
                )
            continue

        if not current_file:                         # --- 符号行号 ---
            continue
        if end > totals[current_file]:
            problems.append(
                f"L{lineno}: {current_file}::{name} 结束行 {end} 超出文件总行数 {totals[current_file]}"
            )
            continue
        spans = symbols.get(current_file, {}).get(name)
        if not spans:
            problems.append(f"L{lineno}: {current_file} 中找不到符号 `{name}`（文档写 {start}-{end}）")
        elif (start, end) not in spans:
            problems.append(
                f"L{lineno}: {current_file}::{name} 文档写 {start}-{end}，AST 实为 {sorted(set(spans))}"
            )

    # --- 附录里的裸数字式文件总行数 ---
    # 用 resolve(name, None)：附录路径相对仓库根书写（`db/x.py`、`tests/x.py`），
    # 不能继承正文的章节目录，否则会被前一个 `### … \`app/xxx/\`` 标题带偏。
    for lineno, line in enumerate(lines, 1):
        for m in _ROW_TOTAL_RE.finditer(line):
            name, declared = m.group(1), int(m.group(2))
            key = resolve(name, None)
            if not key:
                problems.append(f"L{lineno}: 无法解析文件 `{name}`")
                continue
            counted += 1
            if declared != totals[key]:
                problems.append(f"L{lineno}: {key} 文档写 {declared}，实际 {totals[key]}")

    # --- 章节小计 / 全量小计（目录维度）---
    # 这两类是**散文里的数字**，不在表格里，同样会悄悄变错。
    def dir_lines(d: str) -> int:
        return sum(
            len(p.read_text(encoding="utf-8").splitlines())
            for p in pathlib.Path(SOURCE_ROOT, d).rglob("*.py")
        )

    for lineno, line in enumerate(lines, 1):
        dirs = _SUBTOTAL_DIR_RE.findall(line)
        nums = _SECTION_NUM_RE.findall(line)
        if dirs and len(dirs) == len(nums):
            for d, raw in zip(dirs, nums):
                if not pathlib.Path(SOURCE_ROOT, d).is_dir():
                    problems.append(f"L{lineno}: 引用了不存在的目录 app/{d}/")
                    continue
                counted += 1
                actual = dir_lines(d)
                if int(raw.replace(",", "")) != actual:
                    problems.append(f"L{lineno}: app/{d}/ 文档写 {raw} 行，实际 {actual}")

        m = _APP_TOTAL_RE.search(line)
        if m:
            counted += 1
            actual = sum(
                len(p.read_text(encoding="utf-8").splitlines())
                for p in pathlib.Path(SOURCE_ROOT).rglob("*.py")
            )
            if int(m.group(1).replace(",", "")) != actual:
                problems.append(f"L{lineno}: app/ 总计 文档写 {m.group(1)} 行，实际 {actual}")

    # --- 松散单元格里的符号行号 ---
    # 形如 ``| `prompts.py` | `PROMPTS` 18 / `render` 103-112 |``：
    # 第二列不是纯区间，`_ROW_RE` 认不出，这一段同样是盲区
    # （实测就藏着 `render` 103-112 vs 实际 109-118）。
    #
    # 只在「该符号确实存在于该文件」时才比对，找不到就跳过——
    # 散文里的普通词可能恰好和某个符号同名，宁可漏判不可误判。
    loose_dir: str | None = None
    for lineno, line in enumerate(lines, 1):
        m = _SECTION_DIR_RE.match(line)
        if m:
            loose_dir = m.group(1) if m.group(1).endswith("/") else m.group(1) + "/"
            continue
        m = _LOOSE_ROW_RE.match(line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        if _STRICT_TOTAL_RE.match(rest):          # 已由 _ROW_RE 覆盖
            continue
        key = resolve(name, loose_dir)
        if not key:
            continue
        for sm in _LOOSE_NAMED_RE.finditer(rest):
            sym = sm.group(1)
            start = int(sm.group(2))
            end = int(sm.group(3)) if sm.group(3) else start
            spans = symbols.get(key, {}).get(sym)
            if not spans:
                continue
            counted += 1
            if not any(s <= start and end <= e for s, e in spans):
                problems.append(
                    f"L{lineno}: {key}::{sym} 文档写 {start}-{end}，AST 实为 {sorted(set(spans))}"
                )

    # --- 不带文件名的符号引用（上下文继承到最近的 `#### 📍 <文件>` 标题）---
    # 第 11 类，也是**最后一块被发现的大盲区**：前十类全都要求行里有文件名，
    # 所以 ``| `CHANNELS` | 70 |``（单数字）与散文里的 ``（`_decide` 565-690）``
    # 谁都不管。实测：校验器一路报「522 条全部一致」时，这两类里累计藏着 8 处错值
    # （6 处由一次性脚本先发现并修好，另 2 处要等本类补上才现身）。
    #
    # 判据与第 7 类一致——**须精确等于符号的 AST 跨度**，而不是「不越界」。
    # 三个**刻意的漏判**，都是为了不误报（沿用第 6 类的取向：宁可漏判不可误判）：
    #   ① 该文件里找不到同名符号 → 跳过（散文词可能恰好撞名）；
    #   ② 一行里符号与数字分成两组写（`` `fit` / `_load_idf` | 98-107 / 72-81 ``）
    #      → 左右配对必然错位，整行弃判（这是实测过的误报来源）；
    #   ③ 数字后紧跟中文量词 → 计数不是行号。
    ctx_dir: str | None = None
    ctx_file: str | None = None
    for lineno, line in enumerate(lines, 1):
        m = _SECTION_DIR_RE.match(line)
        if m:
            ctx_dir = m.group(1) if m.group(1).endswith("/") else m.group(1) + "/"
            continue
        m = _MODULE_HEAD_RE.match(line)
        if m:
            ctx_file = resolve(m.group(1), ctx_dir)
            continue
        if not ctx_file:
            continue
        # 已由前面各类覆盖的写法跳过，避免同一处被报两遍
        if (_ROW_RE.match(line) or _LOOSE_ROW_RE.match(line)
                or _ROW_TOTAL_RE.search(line) or _ANY_FILE_RE.search(line)):
            continue
        total = totals.get(ctx_file)
        if total is None:
            continue
        matches = list(_CTX_REF_RE.finditer(line))
        for idx, mm in enumerate(matches):
            sym = mm.group(1)
            spans = symbols.get(ctx_file, {}).get(sym)
            if not spans:                                   # ① 找不到同名符号
                continue
            if idx == 0:
                preceding = line[: mm.start()]
            else:
                preceding = line[matches[idx - 1].end(): mm.start()]
            if "`" in preceding:                            # ② 两组分开写
                continue
            start = int(mm.group(2))
            end = int(mm.group(3)) if mm.group(3) else start
            if not 1 <= start <= end <= total:              # 越界当计数看，不算声明
                continue
            if _QUANTIFIER_RE.match(line[mm.end():]):       # ③ 计数不是行号
                continue
            counted += 1
            if mm.group(3):
                if (start, end) not in spans:
                    problems.append(
                        f"L{lineno}: {ctx_file}::{sym} 文档写 {start}-{end}，"
                        f"AST 实为 {sorted(set(spans))}"
                    )
            elif start not in {s for s, _ in spans}:
                problems.append(
                    f"L{lineno}: {ctx_file}::{sym} 文档写第 {start} 行，"
                    f"AST 实为 {sorted(set(spans))}"
                )

    # --- 通用文件行数声明（含非 Python、含架构图里的无反引号写法）---
    # 解析不出候选或候选有歧义时**直接跳过**：宁可漏判不可误判。
    if any_files:
        for lineno, line in enumerate(lines, 1):
            for m in _ANY_FILE_RE.finditer(line):
                name = m.group(1)
                declared = int(m.group(2).replace(",", ""))
                pool = any_files.get(pathlib.Path(name).name, [])
                # 先精确匹配完整相对路径（"README.md" 应指根目录那份，
                # 而不是 `artifacts/backup/README.md`）；精确匹配不唯一才退化为后缀匹配。
                exact = [p for p in pool if str(p) == name]
                cands = exact or [p for p in pool if str(p).endswith(name)]
                if len(cands) != 1:
                    continue                     # 找不到或歧义 → 跳过
                target = cands[0]
                counted += 1
                actual = len(target.read_text(encoding="utf-8", errors="ignore").splitlines())
                if declared != actual:
                    problems.append(
                        f"L{lineno}: {target} 文档写 {declared}，实际 {actual}"
                    )

    # --- 区域行号表（裸区间）---
    # 形如 ``| 178-204 | 向量数据库配置 |``：既无文件名也无符号名，
    # 只能靠最近的 `#### 📍 <文件>` 标题定位。
    #
    # 这类表此前**完全在校验范围之外**：config.py 从 612 长到 632 行后，
    # 表末行仍停在 583，没有任何机制会发现。
    # 可稳定校验的不变量有三条：
    #   ① 每段都在文件界内；② 区间递增不重叠；③ **末段必须收到文件最后一行**。
    # 第 ③ 条正是用来抓「文件变长、表没跟着长」的。
    region_dir: str | None = None
    region_file: str | None = None
    region_run: list[tuple[int, int, int]] = []

    def check_region_run() -> None:
        nonlocal counted
        # 少于 3 段的多半不是区域表，别误判
        if len(region_run) < 3 or not region_file:
            return
        total = totals.get(region_file)
        if total is None:
            return
        prev_end = 0
        for rl, start, end in region_run:
            counted += 1
            if not (1 <= start <= end <= total):
                problems.append(
                    f"L{rl}: 区间 {start}-{end} 越界（{region_file} 共 {total} 行）"
                )
            elif start <= prev_end:
                problems.append(f"L{rl}: 区间 {start}-{end} 与前一段重叠，区域表应递增")
            prev_end = max(prev_end, end)
        last_line, _, last_end = region_run[-1]
        if last_end != total:
            problems.append(
                f"L{last_line}: 区域表止于 {last_end}，但 {region_file} 共 {total} 行"
                "（没覆盖到文件末尾，多半是文件变长后没更新）"
            )

    for lineno, line in enumerate(lines, 1):
        sm = _SECTION_DIR_RE.match(line)
        if sm:
            check_region_run()
            region_run = []
            region_dir = sm.group(1) if sm.group(1).endswith("/") else sm.group(1) + "/"
            continue
        hm = _MODULE_HEAD_RE.match(line)
        if hm:
            check_region_run()
            region_run = []
            region_file = resolve(hm.group(1), region_dir)
            continue
        rm = _BARE_RANGE_RE.match(line)
        if rm:
            region_run.append((lineno, int(rm.group(1)), int(rm.group(2))))
            continue
        if line.lstrip().startswith("|"):      # 表格结束，结算本段
            check_region_run()
            region_run = []
    check_region_run()

    for lineno, line in enumerate(lines, 1):         # --- 散文引用 ---
        for m in _PROSE_RE.finditer(line):
            path, start, end = m.group(1), int(m.group(2)), int(m.group(3))
            if path not in totals:
                problems.append(f"L{lineno}: 引用了不存在的文件 {path}")
                continue
            counted += 1
            if not (1 <= start <= end <= totals[path]):
                problems.append(
                    f"L{lineno}: {path}:{start}-{end} 越界（文件共 {totals[path]} 行）"
                )
                continue
            # 只校验「不越界」是不够的——这是第 10 类，也是最隐蔽的一类。
            #
            # 代码在中间插了几十行之后，一个引用旧函数范围的 `foo.py:120-160`
            # **仍然在文件界内**，于是静默通过，却已经指向了另一个函数。
            # 实测一次重构后，本文档 24 条散文引用里有 9 条这样悄悄漂移
            # （节点④ 指向 model_router 的 526-653，实际主入口已移到 709-848）。
            # 因此这里要求引用**精确等于**某个符号的 (lineno, end_lineno)，
            # 或等于整文件范围 (1, 总行数)。
            if (start, end) == (1, totals[path]):
                continue
            spans = [sp for ss in symbols.get(path, {}).values() for sp in ss]
            if (start, end) not in spans:
                near = sorted(
                    {sp for sp in spans if abs(sp[0] - start) <= 15},
                    key=lambda s: abs(s[0] - start),
                )[:2]
                hint = f"，附近实际为 {near}" if near else ""
                problems.append(
                    f"L{lineno}: {path}:{start}-{end} 未精确命中任何符号"
                    f"（多半是代码改动后漂移）{hint}"
                )

    print(f"校验 {doc_path}：共 {counted} 条行号声明")
    return problems


def main() -> int:
    doc = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOC
    if not pathlib.Path(doc).exists():
        print(f"找不到文档：{doc}")
        return 2

    problems = verify(doc)
    if problems:
        print(f"\n❌ 发现 {len(problems)} 处不一致：")
        for item in problems:
            print("   ✗", item)
        print("\n提示：代码改动后行号会漂移，请按实际位置修正文档。")
        return 1

    print("\n✅ 全部与源码 AST 一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
