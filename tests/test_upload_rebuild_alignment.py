#!/usr/bin/env python3
"""同一个文件走"上传入库"与"重建入库"必须得到同一份文本（P0-3）。

守什么
------
本服务有两条入库路径：

- ``POST /knowledge/upload-file``  —— 用户当场传一个文件，增量入库；
- ``POST /knowledge/rebuild``      —— 一键重建，扫 ``data/`` 全量重灌。

它们曾经**各自挑解析器**：上传路径自己 ``import PyPDFLoader``，而那是
``app/utils/doc_loader._load_pdf`` 三级降级（pdfplumber → pypdf → PyPDFLoader）
里**最弱的一级**；重建路径走完整的三级降级。于是同一个 PDF：

    pdfplumber 抽出来：[表格 1] | 项目 | 周期 | ... （Markdown 表格，有结构）
    PyPDFLoader 抽出来：项目 周期 ...            （表格塌成一行行）

两份文本不同 → 切分不同 → 检索命中的片段不同。更糟的是**重建会悄悄改掉**
早先上传那份文档的内容：用户没动任何文件，同一个问题的答案却变了 ——
这类"入库的真相取决于你从哪个口进的库"没有任何一处会报错。

怎么守才不是恒真
----------------
拿 ``data/`` 里那份真实 PDF，**真的走一遍两条路径**，再逐字比对：

- 重建侧：``load_all_documents(data_dir)``（重建实际调用的入口）；
- 上传侧：真发一个 multipart 请求给端点，截获它交给 ``add_document`` 的正文。

比对的是"最终入库的那段文本"，不是"某函数被调用了" —— 这样重构实现不会误红，
但把解析器换回去一定红。
"""
import ast
from pathlib import Path

import pytest

from app import config
from app.utils.doc_loader import (
    SUPPORTED_SUFFIX,
    _SUFFIX_LOADERS,
    load_all_documents,
    load_file,
)

_DOC_LOADER = Path(__file__).resolve().parents[1] / "app" / "utils" / "doc_loader.py"

#: 语料里唯一那份 PDF（也是唯一带表格的文档，正好能放大两级解析器的差异）。
_PDF = config.DATA_DIR / "星河智能云平台产品白皮书.pdf"


def _rebuild_text(path: Path) -> str:
    """重建路径：走 ``load_all_documents``，取该文件拼出的正文。"""
    docs = [
        d for d in load_all_documents(path.parent)
        if Path(str(d.metadata.get("source", ""))) == path
    ]
    assert docs, f"重建路径没读到 {path.name}，用例失去意义"
    return "\n".join(d.page_content for d in docs)


def _upload_text(monkeypatch, path: Path) -> str:
    """上传路径：真发 multipart 请求，截获端点交给 ``add_document`` 的正文。"""
    from fastapi.testclient import TestClient

    import app.api.knowledge as kb_mod
    from app.main import app

    captured: dict = {}

    def _fake_add_document(file_name: str, content: str) -> int:
        captured["file_name"] = file_name
        captured["content"] = content
        return 0

    monkeypatch.setattr(kb_mod, "add_document", _fake_add_document)

    client = TestClient(app)
    resp = client.post(
        "/knowledge/upload-file",
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
    )
    assert resp.status_code == 200, f"上传失败：{resp.text[:400]}"
    assert captured, "端点没有走到入库那一步，用例失去意义"
    return captured["content"]


@pytest.mark.skipif(not _PDF.exists(), reason="语料里没有 PDF，无法比对两级解析器")
def test_upload_and_rebuild_parse_the_same_pdf_identically(monkeypatch):
    """同一份 PDF，两条入库路径的正文必须**逐字相同**。

    退回验证：把上传端点改回 ``PyPDFLoader(str(tmp_path)).load()``，本用例变红
    （pdfplumber 会产出 ``[表格 1]`` 与 Markdown 管道符，PyPDFLoader 产不出）。
    """
    rebuilt = _rebuild_text(_PDF)
    uploaded = _upload_text(monkeypatch, _PDF)

    assert uploaded == rebuilt, (
        "上传路径与重建路径解析出的文本不一致 —— 同一个 PDF 会有两份内容，\n"
        f"重建还会悄悄覆盖早先上传的那份。\n"
        f"  上传侧 {len(uploaded)} 字，前 120 字：{uploaded[:120]!r}\n"
        f"  重建侧 {len(rebuilt)} 字，前 120 字：{rebuilt[:120]!r}"
    )
    # 顺带证明"对齐"不是"两边都退化成了最弱那级"：pdfplumber 认得表格结构。
    assert "[表格 1]" in rebuilt, "两侧都退化成纯文本抽取了，那不是对齐，是双双降级"


@pytest.mark.skipif(not _PDF.exists(), reason="语料里没有 PDF，无法比对两级解析器")
def test_upload_goes_through_the_shared_loader(monkeypatch):
    """上传端点必须经过 ``doc_loader.load_file``，不能自己 new 一个 loader。

    与上一条的分工：上面守"文本一样"，这条守"走的是同一个东西"。
    只守前者的话，有人把两侧都改成 PyPDFLoader 也能过 —— 那就把表格识别能力
    整体丢掉了，而这不是对齐、是倒退。
    """
    from fastapi.testclient import TestClient

    import app.api.knowledge as kb_mod
    from app.main import app

    calls: list = []
    original = kb_mod.load_file

    def _spy(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(kb_mod, "load_file", _spy)
    monkeypatch.setattr(kb_mod, "add_document", lambda *a, **kw: 0)

    client = TestClient(app)
    resp = client.post(
        "/knowledge/upload-file",
        files={"file": (_PDF.name, _PDF.read_bytes(), "application/pdf")},
    )
    assert resp.status_code == 200, f"上传失败：{resp.text[:400]}"
    assert len(calls) == 1, f"上传路径没有（或不止一次）经过 load_file：{calls}"


def test_unsupported_suffix_is_rejected_loudly(tmp_path):
    """未知后缀要**报错**，不能静默退化成按纯文本读。

    静默退化的症状是"文档入库了但检索不到内容"：一个拼错的后缀被当成 txt 读，
    索引里躺着一堆乱码片段，而日志干干净净。
    """
    bogus = tmp_path / "手册.docx"
    bogus.write_bytes(b"irrelevant")
    with pytest.raises(ValueError) as exc:
        load_file(bogus)
    assert "docx" in str(exc.value)


def test_supported_suffix_is_derived_from_the_loader_map():
    """``SUPPORTED_SUFFIX`` 必须是分发表的**派生值**，不能另写一份字面量集合。

    历史上白名单在三处各写一遍（doc_loader 的 set、upload 端点的不带点集合、
    prepare.py 的元组），加一种格式只改其中一处时，另外两条路径会静默拒绝它。

    ⚠️ 为什么这里必须看 AST 而不是比大小：``{".pdf", ".md", ".markdown", ".txt"}``
    与 ``set(_SUFFIX_LOADERS)`` **当前取值完全相等**，`==` 恒真。只有"是不是派生的"
    这个结构事实能区分它们 —— 而"又一次各写一遍"正是要防的东西。
    （本用例第一版写的就是 `==`，退回验证时没变红。）
    """
    tree = ast.parse(_DOC_LOADER.read_text(encoding="utf-8"))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "SUPPORTED_SUFFIX" for t in node.targets)
    ]
    assert len(assignments) == 1, "SUPPORTED_SUFFIX 应当只有一个模块级赋值"
    value = assignments[0].value
    assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == "set", (
        "SUPPORTED_SUFFIX 写成了字面量集合（或别的构造方式），应从 _SUFFIX_LOADERS 派生"
    )
    assert isinstance(value.args[0], ast.Name) and value.args[0].id == "_SUFFIX_LOADERS"
    # 派生关系在运行时也必须成立
    assert SUPPORTED_SUFFIX == set(_SUFFIX_LOADERS)


@pytest.mark.parametrize(
    "suffix, raw, expected",
    [
        (".txt", "纯文本文档正文".encode(), "纯文本文档正文"),
        (".md", "# 标题\n\n正文".encode(), "标题"),
        (".markdown", "# 标题\n\n正文".encode(), "标题"),
    ],
)
def test_every_declared_suffix_can_actually_be_parsed(tmp_path, suffix, raw, expected):
    """分发表里的每个后缀都要真能跑通 —— 防止加了 key 却忘了实现，或写错扩展名。"""
    assert suffix in SUPPORTED_SUFFIX
    path = tmp_path / f"sample{suffix}"
    path.write_bytes(raw)
    docs = load_file(path)
    assert docs, f"{suffix} 解析出 0 段"
    assert expected in docs[0].page_content


def test_blank_content_is_dropped_by_the_shared_entry(tmp_path):
    """空白段落的过滤放在共享入口里，不放在调用方。

    放在调用方时上传路径就会漏掉它 —— 于是"文件里夹一段空白"变成两条路径文本
    不同，而这是最难查的一种不同（只在那一个文件上、只在重建之后才显形）。
    """
    blank = tmp_path / "只有空行.md"
    blank.write_text("\n\n   \n\t\n", encoding="utf-8")
    assert load_file(blank) == []

    normal = tmp_path / "有内容.md"
    normal.write_text("有内容\n", encoding="utf-8")
    assert len(load_file(normal)) == 1
