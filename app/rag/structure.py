"""RAG 五层架构 · L1.5 结构解析层。

职责边界
--------
位于 L1（数据准备）与 L2（索引/切分）之间：**只负责「看懂文档的层级」，
不负责切分**。输出章节树与每个节点的 `heading_path`，供 T3-2 结构感知切分消费。

设计要点
--------
1. **纯正则、模型无关**：全部锚点来自 `config.STRUCTURE_*_PATTERNS`，
   新增文档格式改配置即可，不改代码（遵循「不写死任何值」原则）。
2. **永不抛错**：解析失败一律降级为 `structure="flat"`，由调用方回退递归切分。
   这是 L1.5 的硬约束 —— 结构解析是**增强**，绝不能阻断主链路。
3. **行首锚定**：只用 `^`（配合 `re.MULTILINE`）匹配行首，避免正文里的
   「第三章 所述…」被误判为标题。宁可漏判，不可误判。

层级定义
--------
    L1 章  第三章 假期管理 / 一、账号与密码
    L2 节  3.2 年休假 / Q8：... / ## 6 知识库管理 API
    L3 条  1、第一步 / （1）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app import config
from app.utils.logger import logger

# 结构类型
HIERARCHICAL = "hierarchical"
FLAT = "flat"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """一个章节节点。"""

    level: int                      # 1=章 2=节 3=条
    title: str                      # 标题原文（已 strip）
    line: int                       # 标题所在行号（0-based）
    heading_path: str = ""          # 「第三章 假期管理 > 3.2 年休假」
    parent: Optional["Section"] = field(default=None, repr=False)

    @property
    def path_segments(self) -> List[str]:
        return [s for s in self.heading_path.split(" > ") if s]


@dataclass
class SectionTree:
    """一篇文档的结构解析结果。"""

    structure: str                          # hierarchical | flat
    sections: List[Section] = field(default_factory=list)
    doc_title: str = ""
    doc_version: str = ""
    doc_effective_date: str = ""
    doc_updated_date: str = ""

    @property
    def is_hierarchical(self) -> bool:
        return self.structure == HIERARCHICAL

    def section_at(self, line: int) -> Optional[Section]:
        """给定行号，返回它所属的最深层节点。"""
        hit = None
        for s in self.sections:
            if s.line <= line:
                hit = s
            else:
                break
        return hit

    def path_at(self, line: int) -> str:
        """给定行号，返回它所属的 heading_path（无归属则返回文档标题）。"""
        s = self.section_at(line)
        if s is None:
            return self.doc_title
        return f"{self.doc_title} > {s.heading_path}" if self.doc_title else s.heading_path


# ---------------------------------------------------------------------------
# 正则编译（惰性 + 缓存，避免每篇文档重复编译）
# ---------------------------------------------------------------------------
_PATTERN_CACHE: dict = {}


def _compile(key: str, patterns: str) -> Optional[re.Pattern]:
    """把「|」分隔的多模式串编译为单个正则（行首锚定 + 多行模式）。"""
    if not patterns or not patterns.strip():
        return None
    cache_key = (key, patterns)
    if cache_key in _PATTERN_CACHE:
        return _PATTERN_CACHE[cache_key]

    parts = [p.strip() for p in patterns.split("|") if p.strip()]
    if not parts:
        return None
    try:
        compiled = re.compile("|".join(parts), re.MULTILINE)
    except re.error as exc:  # 配置写错时降级，不阻断
        logger.warning("结构锚点正则编译失败（%s）：%s，该层级将被忽略", key, exc)
        compiled = None

    _PATTERN_CACHE[cache_key] = compiled
    return compiled


def _chapter_re() -> Optional[re.Pattern]:
    return _compile("chapter", config.STRUCTURE_CHAPTER_PATTERNS)


def _section_re() -> Optional[re.Pattern]:
    return _compile("section", config.STRUCTURE_SECTION_PATTERNS)


def _item_re() -> Optional[re.Pattern]:
    return _compile("item", config.STRUCTURE_ITEM_PATTERNS)


def _md_re() -> Optional[re.Pattern]:
    return _compile("md", config.STRUCTURE_MD_PATTERNS)


# ---------------------------------------------------------------------------
# 文档头信息抽取（供上下文头与元数据使用）
# ---------------------------------------------------------------------------
_TITLE_RE = re.compile(r"^\s*(.{2,60}?)\s*[·•|\-—–]{1}\s*(.{2,40})\s*$")
_VERSION_RE = re.compile(r"版本[:：]?\s*([A-Za-z0-9._]+)")
_EFFECTIVE_RE = re.compile(r"生效日期[:：]?\s*(\d{4}-\d{1,2}-\d{1,2})")
_UPDATED_RE = re.compile(r"更新日期[:：]?\s*(\d{4}-\d{1,2}-\d{1,2})")


def extract_doc_meta(text: str) -> dict:
    """从文档首部抽取标题 / 版本 / 生效日期 / 更新日期。

    只看前 10 行，且**跳过被识别为章节锚点的行**，避免把「第三章」当成文档标题。
    抽不到一律返回空串，不抛错。
    """
    meta = {"doc_title": "", "doc_version": "",
            "doc_effective_date": "", "doc_updated_date": ""}
    lines = text.splitlines()[:10]

    for line in lines:
        raw = line.strip()
        # Markdown 标题行去掉前导 # 作为标题值；
        # 但锚点判定仍用原文（md 锚点本身依赖前导 #，剥掉就判不出来了）
        stripped = raw.lstrip("#").strip()
        if not raw:
            continue
        # 跳过纯分隔线（==== ---- 之类）
        if re.fullmatch(r"[==\-—_*\s]{3,}", raw):
            continue
        if not meta["doc_title"]:
            # 「公司名 · 文档名」形式优先；否则取第一个非空非锚点行
            m = _TITLE_RE.match(stripped)
            if m:
                meta["doc_title"] = f"{m.group(1).strip()} · {m.group(2).strip()}"
                continue
            # 一级 Markdown 标题（# 开头）本身就是文档名，允许作为标题
            is_h1 = bool(_md_re() and re.match(r"^#[^#]", raw))
            if is_h1 or not _is_anchor_line(raw):
                meta["doc_title"] = stripped[:60]
                continue
        if not meta["doc_version"]:
            m = _VERSION_RE.search(raw)
            if m:
                meta["doc_version"] = m.group(1)
        if not meta["doc_effective_date"]:
            m = _EFFECTIVE_RE.search(raw)
            if m:
                meta["doc_effective_date"] = m.group(1)
        if not meta["doc_updated_date"]:
            m = _UPDATED_RE.search(raw)
            if m:
                meta["doc_updated_date"] = m.group(1)

    return meta


def _is_anchor_line(line: str) -> bool:
    """该行是否命中任一结构锚点（用于排除文档头误判）。"""
    for rx in (_chapter_re(), _md_re(), _section_re(), _item_re()):
        if rx and rx.match(line):
            return True
    return False


# ---------------------------------------------------------------------------
# 核心解析
# ---------------------------------------------------------------------------
def _detect_level(line: str) -> Optional[tuple]:
    """判断一行是否标题，返回 (level, title)；非标题返回 None。"""
    # 顺序：章 > Markdown 标题 > 节 > 条
    rx = _chapter_re()
    if rx and rx.match(line):
        return 1, line.strip()

    rx = _md_re()
    if rx:
        m = rx.match(line)
        if m:
            hashes = len(m.group(0).strip())
            return min(hashes, 3), line.strip().lstrip("#").strip()

    rx = _section_re()
    if rx and rx.match(line):
        return 2, line.strip()

    rx = _item_re()
    if rx and rx.match(line):
        return 3, line.strip()

    return None


def parse_document(text: str, doc_title: str = "") -> SectionTree:
    """解析单篇文档的结构，输出 SectionTree。

    Args:
        text: 文档正文
        doc_title: 文档标题（留空则尝试从正文首部抽取）

    Returns:
        SectionTree；**永不抛错** —— 任何异常都会降级为 flat 结构。
    """
    try:
        meta = extract_doc_meta(text)
        title = doc_title or meta.get("doc_title", "")

        sections: List[Section] = []
        stack: List[Section] = []     # 维护当前层级栈，用于拼 heading_path

        for idx, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            hit = _detect_level(line)
            if hit is None:
                continue

            level, raw_title = hit

            # 弹出层级 >= 当前层级的节点（找父节点）
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1] if stack else None

            seg = raw_title
            heading_path = f"{parent.heading_path} > {seg}" if parent else seg

            node = Section(
                level=level, title=raw_title, line=idx,
                heading_path=heading_path, parent=parent,
            )
            sections.append(node)
            stack.append(node)

        # 判定结构类型：没有 L1/L2 锚点 → flat
        has_major = any(s.level <= 2 for s in sections)
        structure = HIERARCHICAL if has_major else FLAT

        if structure == FLAT and config.STRUCTURE_FLAT_FALLBACK:
            logger.debug("文档无章节结构，降级 flat：%s", title or "(无标题)")

        return SectionTree(
            structure=structure,
            sections=sections,
            doc_title=title,
            doc_version=meta.get("doc_version", ""),
            doc_effective_date=meta.get("doc_effective_date", ""),
            doc_updated_date=meta.get("doc_updated_date", ""),
        )

    except Exception as exc:  # noqa: BLE001 —— 结构解析绝不阻断主链路
        logger.warning("结构解析失败，降级为 flat：%s", exc)
        return SectionTree(structure=FLAT, sections=[], doc_title=doc_title)


def detect_structure(text: str) -> str:
    """只判定结构类型（不建树），供快速分支使用。"""
    return parse_document(text).structure


def split_by_section(text: str) -> List[dict]:
    """按章节切分为「(heading_path, 正文)」片段序列。

    T3-2 结构感知切分的输入准备。每个片段包含：
        - heading_path: 章节路径
        - level: 层级
        - content: 该章节到下一章节之间的正文（含标题行本身）
        - start_line / end_line

    无结构文档返回单个片段（heading_path=""），由调用方走递归降级。
    """
    tree = parse_document(text)
    lines = text.splitlines()

    if not tree.sections:
        return [{
            "heading_path": tree.doc_title, "level": 0,
            "content": text, "start_line": 0, "end_line": len(lines),
        }]

    blocks: List[dict] = []
    for i, sec in enumerate(tree.sections):
        start = sec.line
        end = tree.sections[i + 1].line if i + 1 < len(tree.sections) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        if not content:
            continue
        blocks.append({
            "heading_path": sec.heading_path,
            "level": sec.level,
            "title": sec.title,
            "content": content,
            "start_line": start,
            "end_line": end,
        })

    # 文档开头在第一个标题之前的内容（如文档头/前言）
    head = "\n".join(lines[:tree.sections[0].line]).strip()
    if head:
        blocks.insert(0, {
            "heading_path": tree.doc_title, "level": 0, "title": tree.doc_title,
            "content": head, "start_line": 0, "end_line": tree.sections[0].line,
        })

    return blocks
