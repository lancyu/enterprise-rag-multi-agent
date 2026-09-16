"""L1.5 结构解析层单测（T2-1）。

验证目标：
1. 五类锚点（章 / 一级 / 小节 / Q&A / Markdown）都能识别
2. 无结构文档正确降级为 flat
3. heading_path 层级嵌套正确
4. **永不抛错**（异常输入一律降级，绝不阻断主链路）
5. 真实语料上的分类符合预期
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag.structure import (  # noqa: E402
    FLAT,
    HIERARCHICAL,
    SectionTree,
    detect_structure,
    extract_doc_meta,
    parse_document,
    split_by_section,
)


# ---------------------------------------------------------------------------
# 1. 锚点识别
# ---------------------------------------------------------------------------
def test_chapter_anchor():
    """「第三章 假期管理」识别为 L1 章。"""
    tree = parse_document("第三章 假期管理\n年假有 10 天。\n3.2 年休假\n细则……")
    levels = {s.level for s in tree.sections}
    assert 1 in levels, f"未识别到章级锚点：{[s.title for s in tree.sections]}"
    assert tree.structure == HIERARCHICAL


def test_top_level_anchor():
    """「一、账号与密码」识别为 L1 章。"""
    tree = parse_document("一、账号与密码\n1.1 初始账号\n账号由 IT 开通。")
    assert tree.structure == HIERARCHICAL
    assert tree.sections[0].level == 1
    assert tree.sections[0].title == "一、账号与密码"


def test_section_anchor():
    """「3.2 年休假」识别为 L2 节。"""
    tree = parse_document("第三章 假期管理\n3.2 年休假\n年假 10 天。")
    secs = [s for s in tree.sections if s.level == 2]
    assert secs, "未识别到小节锚点"
    assert secs[0].title.startswith("3.2")


def test_qa_anchor():
    """「Q8：...」识别为 L2 节（FAQ 文档）。"""
    tree = parse_document("Q8：如何重置密码？\n访问自助门户重置。\nQ9：如何申请 VPN？\n在 OA 提交。")
    secs = [s for s in tree.sections if s.title.startswith("Q")]
    assert len(secs) == 2, f"Q&A 锚点识别数不符：{[s.title for s in tree.sections]}"
    assert tree.structure == HIERARCHICAL


def test_markdown_anchor():
    """Markdown 标题按 # 数量定级。"""
    tree = parse_document("# 标题一\n正文\n## 6 知识库管理 API\n正文\n### 6.1 接口\n正文")
    h1 = [s for s in tree.sections if s.level == 1]
    h2 = [s for s in tree.sections if s.level == 2]
    assert h1 and h2, f"Markdown 定级失败：{[(s.level, s.title) for s in tree.sections]}"
    assert h2[0].title == "6 知识库管理 API"   # 已剥离前导 #


# ---------------------------------------------------------------------------
# 2. flat 降级
# ---------------------------------------------------------------------------
def test_flat_document_detected():
    """无任何章节锚点的文档 → flat。"""
    text = "【账号与密码】\n- 初始密码：短信发送\n- 锁定：连续输错 5 次\n【网络】\n- VPN：OA 申请"
    tree = parse_document(text)
    assert tree.structure == FLAT, "无结构文档未降级为 flat"
    assert detect_structure(text) == FLAT


def test_flat_still_has_title():
    """flat 文档仍应抽到文档标题（供上下文头使用）。"""
    tree = parse_document("星河科技 · 速查表\n- 项目 A：找张三\n- 项目 B：找李四")
    assert tree.structure == FLAT
    assert "速查表" in tree.doc_title


def test_split_by_section_flat_returns_single_block():
    """flat 文档切分应退化为单个整块，由调用方走递归降级。"""
    blocks = split_by_section("无结构内容\n只有普通段落\n没有标题")
    assert len(blocks) == 1
    assert blocks[0]["level"] == 0


# ---------------------------------------------------------------------------
# 3. heading_path 层级
# ---------------------------------------------------------------------------
def test_heading_path_nesting():
    """heading_path 应逐层拼接。"""
    text = (
        "第三章 假期管理\n"
        "3.1 法定节假日\n"
        "春节放假 7 天。\n"
        "3.2 年休假\n"
        "工龄满 1 年享 5 天。\n"
        "第四章 考勤\n"
        "4.1 打卡\n"
        "每日打卡两次。"
    )
    tree = parse_document(text)
    paths = [s.heading_path for s in tree.sections]
    assert "第三章 假期管理 > 3.2 年休假" in paths, f"路径拼接错误：{paths}"
    assert "第四章 考勤 > 4.1 打卡" in paths, f"跨章路径错误：{paths}"
    # 章与节必须都在
    assert any(p == "第三章 假期管理" for p in paths)


def test_path_at_line():
    """path_at 能定位行号所属章节。"""
    lines = ["一、账号与密码", "1.1 初始账号", "正文", "二、VPN", "2.1 申请", "正文"]
    tree = parse_document("\n".join(lines))
    assert "一、账号与密码" in tree.path_at(2)
    assert "二、VPN" in tree.path_at(5)


# ---------------------------------------------------------------------------
# 4. 健壮性：永不抛错
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_input", [
    "",                       # 空串
    "\n\n\n",                 # 纯换行
    "===",                    # 纯分隔线
    "🎉🎉 emoji 标题\n内容",    # emoji
    "a" * 5000,               # 超长单行
    "\x00\x01 控制字符\n第三章 测试",  # 控制字符
])
def test_never_raises(bad_input):
    """异常输入一律降级，绝不抛错（L1.5 硬约束）。"""
    tree = parse_document(bad_input)
    assert isinstance(tree, SectionTree)
    assert tree.structure in (HIERARCHICAL, FLAT)


def test_broken_pattern_config_degrades_gracefully(monkeypatch):
    """锚点正则配置写错时，应告警并降级而非崩溃。"""
    from app import config

    monkeypatch.setattr(config, "STRUCTURE_CHAPTER_PATTERNS", "第[未闭合(")
    monkeypatch.setattr(config, "STRUCTURE_SECTION_PATTERNS", "")
    # 不应抛异常
    tree = parse_document("第三章 测试\n内容")
    assert isinstance(tree, SectionTree)


# ---------------------------------------------------------------------------
# 5. 真实语料
# ---------------------------------------------------------------------------
def test_corpus_classification():
    """真实语料上：扁平速查表应判 flat，其余多为 hierarchical。"""
    from app.rag.prepare import prepare_documents
    from app.utils.doc_loader import load_all_documents

    prepared = prepare_documents(load_all_documents())
    assert prepared, "语料为空，无法验证"

    result = {}
    for d in prepared:
        name = d.metadata.get("file_name", "?")
        result[name] = parse_document(d.page_content).structure

    # 系统权限与审批速查是无标题的扁平列表，必须判 flat
    flat_docs = [n for n, s in result.items() if s == FLAT]
    assert any("速查" in n for n in flat_docs), (
        f"速查表未被识别为 flat，实际分类：{result}"
    )

    # 制度类文档应有层级
    hierarchical = [n for n, s in result.items() if s == HIERARCHICAL]
    assert len(hierarchical) >= 8, f"层级文档过少：{result}"


def test_doc_meta_extraction():
    """文档头能抽出版本 / 更新日期。"""
    text = "星河科技有限公司 · 员工手册\n版本：V3.2 生效日期：2026-01-01\n\n第一章 入职\n正文"
    meta = extract_doc_meta(text)
    assert meta["doc_version"] == "V3.2"
    assert meta["doc_effective_date"] == "2026-01-01"
    assert "员工手册" in meta["doc_title"]


def test_doc_title_not_confused_with_chapter():
    """文档标题不应被误取为章节标题。"""
    text = "第一章 总则\n本制度适用于全体员工。\n第二章 细则\n……"
    meta = extract_doc_meta(text)
    assert not meta["doc_title"].startswith("第一章"), (
        f"文档标题误取为章节：{meta['doc_title']!r}"
    )
