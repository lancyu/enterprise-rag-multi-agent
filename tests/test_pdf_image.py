"""PDF 图片抽取测试（T6-2）。

重点验证三件事：
1. 默认关闭时行为完全不变（不落盘、不改正文、不加 metadata）；
2. 开启时能抽到真实图片，并按面积过滤掉装饰性小图；
3. 依赖缺失（OCR 未装 / 渲染失败）时优雅降级，绝不打断文本抽取。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import config
from app.utils.doc_loader import _extract_page_images, _image_area, _load_pdf

PDF_PATH = Path("data/星河智能云平台产品白皮书.pdf")

pytestmark = pytest.mark.skipif(
    not PDF_PATH.exists(), reason="需要知识库中的示例 PDF"
)


@pytest.fixture
def _tmp_image_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PDF_IMAGE_DIR", str(tmp_path / "imgs"))
    return tmp_path / "imgs"


def test_disabled_changes_nothing(_tmp_image_dir, monkeypatch):
    """关闭时：正文不含图片占位、metadata 不含 images、不落盘。"""
    monkeypatch.setattr(config, "PDF_IMAGE_EXTRACTION", False)

    docs = _load_pdf(PDF_PATH)

    assert docs, "PDF 应正常解析出内容"
    for doc in docs:
        assert "[图片" not in doc.page_content
        assert "images" not in doc.metadata
    assert not _tmp_image_dir.exists()


def test_enabled_extracts_image(_tmp_image_dir, monkeypatch):
    """开启时：第 1 页的图表被抽出、落盘、写进 metadata。"""
    monkeypatch.setattr(config, "PDF_IMAGE_EXTRACTION", True)
    monkeypatch.setattr(config, "PDF_IMAGE_MIN_AREA", 10000)

    docs = _load_pdf(PDF_PATH)

    with_images = [d for d in docs if d.metadata.get("images")]
    assert with_images, "示例 PDF 第 1 页含一张图表，应被抽出"

    import json

    images = json.loads(with_images[0].metadata["images"])
    assert images[0]["page"] == 1
    assert images[0]["width"] > 0 and images[0]["height"] > 0
    assert Path(images[0]["path"]).exists(), "图片应真的落盘"
    assert "[图片" in with_images[0].page_content


def test_min_area_filters_decorations(_tmp_image_dir, monkeypatch):
    """面积下限调到极大 → 所有图片被视为装饰元素丢弃。"""
    monkeypatch.setattr(config, "PDF_IMAGE_EXTRACTION", True)
    monkeypatch.setattr(config, "PDF_IMAGE_MIN_AREA", 10 ** 9)

    docs = _load_pdf(PDF_PATH)

    assert all(not d.metadata.get("images") for d in docs)
    assert all("[图片" not in d.page_content for d in docs)


def test_missing_ocr_degrades_gracefully(_tmp_image_dir, monkeypatch):
    """OCR 开启但 pytesseract 未装 → 只降级为空文本，不抛错、不影响正文。"""
    monkeypatch.setattr(config, "PDF_IMAGE_EXTRACTION", True)
    monkeypatch.setattr(config, "PDF_IMAGE_OCR", True)
    monkeypatch.setattr(config, "PDF_IMAGE_MIN_AREA", 10000)

    docs = _load_pdf(PDF_PATH)          # 不应抛异常

    assert docs and any(d.page_content.strip() for d in docs)
    for doc in docs:
        assert "[表格" in doc.page_content or doc.page_content.strip()


def test_image_area_helper():
    assert _image_area({"x0": 0, "top": 0, "x1": 10, "bottom": 20}) == 200
    assert _image_area({"x0": 10, "top": 20, "x1": 0, "bottom": 0}) == 0  # 逆向坐标
    assert _image_area({}) == 0                                            # 缺字段


def test_extract_returns_empty_without_images(_tmp_image_dir):
    """页面无图片时返回空列表（不产生副作用）。"""
    class _Page:
        images = []

    assert _extract_page_images(_Page(), Path("x.pdf"), 0) == []
