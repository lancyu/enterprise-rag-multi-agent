"""企业文档加载器 —— 支持 PDF / Markdown / TXT 等多种格式。

自动扫描知识库目录，按后缀匹配加载器，解析后统一标准化输出，为向量化提供干净数据源。

PDF 解析策略（按能力从强到弱自动降级，永不抛错打断主链路）：
    1. pdfplumber  —— 纯 Python，能识别表格并以 Markdown pipe table 形式输出；
                     开启 PDF_IMAGE_EXTRACTION 时还能抽取内嵌图片（T6-2）；
    2. pypdf       —— 纯 Python，仅做文本提取（表格退化为线性文本）；
    3. PyPDFLoader —— langchain_community 兜底，行为与历史实现一致。
"""
import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from app import config
from app.utils.logger import logger

SUPPORTED_SUFFIX = {".pdf", ".md", ".markdown", ".txt"}


def _table_to_markdown(table: Optional[List[List[Optional[str]]]]) -> str:
    """把 pdfplumber 提取的二维表转成 Markdown pipe table。

    pdfplumber 的单元格可能为 None 或含换行；统一清洗后再拼接，
    避免 Markdown 渲染错位。
    """
    if not table:
        return ""
    cleaned: List[List[str]] = []
    max_cols = 0
    for row in table:
        cells = [(c or "").strip().replace("\n", " ").replace("|", "\\|") for c in row]
        cleaned.append(cells)
        max_cols = max(max_cols, len(cells))
    if max_cols == 0:
        return ""
    # 补齐列数，确保 Markdown 渲染整齐
    padded = [r + [""] * (max_cols - len(r)) for r in cleaned]
    header = padded[0]
    body = padded[1:] if len(padded) > 1 else [[""]] * 1
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * max_cols) + "|",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def _image_area(img: dict) -> float:
    """图片面积（pt²）。用于过滤装饰性的小图标与分隔线。"""
    try:
        w = max(0.0, float(img["x1"]) - float(img["x0"]))
        h = max(0.0, float(img["bottom"]) - float(img["top"]))
        return w * h
    except Exception:  # noqa: BLE001
        return 0.0


def _ocr_image(path: Path) -> str:
    """对图片做 OCR，返回识别文本。

    三重降级（任一不满足就返回空串，绝不抛错）：
    1. 开关未开 → 不调；
    2. pytesseract / PIL 未装 → 跳过；
    3. 系统 tesseract 缺失或语言包缺失 → 跳过。
    中文识别需要 `chi_sim` 语言包，缺它会退到英文，故显式指定 `chi_sim+eng`。
    """
    try:
        import pytesseract
        from PIL import Image

        with Image.open(str(path)) as im:
            return (pytesseract.image_to_string(im, lang="chi_sim+eng") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.info("图片 OCR 不可用，跳过（文本抽取不受影响）：%s", exc)
        return ""


def _extract_page_images(page, file_path: Path, page_idx: int) -> List[dict]:
    """抽取单页内嵌图片：按面积过滤 → 渲染落盘 → 可选 OCR。

    返回图片信息列表；任一步失败只降级不抛错（与 PDF 解析的三级降级一致）。
    """
    raw_images = getattr(page, "images", None) or []
    if not raw_images:
        return []

    min_area = getattr(config, "PDF_IMAGE_MIN_AREA", 10000)
    do_ocr = getattr(config, "PDF_IMAGE_OCR", False)
    dpi = getattr(config, "PDF_IMAGE_DPI", 150)
    out_dir = Path(getattr(config, "PDF_IMAGE_DIR", "artifacts/pdf_images")) / file_path.stem

    results: List[dict] = []
    for img in raw_images:
        area = _image_area(img)
        if area < min_area:
            continue                      # 装饰性小图，丢弃
        info: dict = {
            "page": page_idx + 1,
            "width": round(float(img["x1"]) - float(img["x0"]), 1),
            "height": round(float(img["bottom"]) - float(img["top"]), 1),
            "path": "",
            "ocr_text": "",
        }
        # 渲染落盘：失败也不影响文本抽取，只丢图片文件
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"p{page_idx + 1}_img{len(results) + 1}.png"
            page.crop((img["x0"], img["top"], img["x1"], img["bottom"])).to_image(
                resolution=dpi
            ).save(str(dest), format="PNG")
            info["path"] = str(dest)
        except Exception as exc:  # noqa: BLE001
            logger.info("图片渲染失败，仅记录位置信息：%s", exc)

        if do_ocr and info["path"]:
            info["ocr_text"] = _ocr_image(Path(info["path"]))
        results.append(info)
    return results


def _load_pdf_pdfplumber(file_path: Path) -> List[Document]:
    """用 pdfplumber 提取每页正文 + 表格，表格以 Markdown 形式追加到该页内容末尾。

    开启 PDF_IMAGE_EXTRACTION 时，额外抽取内嵌图片（T6-2）：
    图片以 `[图片 N]` 占位描述拼进正文，完整信息（路径 / 尺寸 / OCR 文本）
    存进 metadata，便于后续按图检索或人工核对。
    """
    import pdfplumber

    extract_images = getattr(config, "PDF_IMAGE_EXTRACTION", False)
    docs: List[Document] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for idx, page in enumerate(pdf.pages):
            parts: List[str] = []
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables):
                md = _table_to_markdown(table)
                if md:
                    parts.append(f"[表格 {t_idx + 1}]\n{md}")

            images: List[dict] = []
            if extract_images:
                try:
                    images = _extract_page_images(page, file_path, idx)
                except Exception:  # noqa: BLE001
                    logger.exception("图片抽取失败，跳过（文本抽取不受影响）：%s", file_path.name)
                for i_idx, img in enumerate(images, start=1):
                    # 有 OCR 文本就带上，没有就只标位置与尺寸 —— 宁可信息少，不要编造
                    desc = f"[图片 {i_idx}：第 {img['page']} 页"
                    if img.get("ocr_text"):
                        desc += f"，OCR：{img['ocr_text']}"
                    else:
                        desc += f"，{img['width']:.0f}x{img['height']:.0f}pt，未 OCR]"
                    parts.append(desc)

            content = "\n\n".join(parts).strip()
            if content:
                meta = {"source": str(file_path), "page": idx + 1}
                if images:
                    # 序列化存储：向量库 metadata 只保证基本类型兼容
                    meta["images"] = json.dumps(images, ensure_ascii=False)
                docs.append(Document(page_content=content, metadata=meta))
    return docs


def _load_pdf_pypdf(file_path: Path) -> List[Document]:
    """pypdf 仅做文本提取（表格会丢失结构但保留字面内容）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    docs: List[Document] = []
    for idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": idx + 1},
                )
            )
    return docs


def _load_pdf(file_path: Path) -> List[Document]:
    """PDF 加载入口：按能力优先级尝试，失败/缺失即降级，绝不抛错。"""
    try:
        return _load_pdf_pdfplumber(file_path)
    except ImportError:
        logger.info("pdfplumber 未安装，降级到 pypdf 解析 PDF：%s", file_path.name)
    except Exception:  # noqa: BLE001 —— 三级降级：任何解析异常都该落到下一级，不挑异常类型
        logger.exception("pdfplumber 解析失败，降级到 pypdf：%s", file_path.name)
    try:
        return _load_pdf_pypdf(file_path)
    except ImportError:
        logger.info("pypdf 未安装，降级到 PyPDFLoader：%s", file_path.name)
    except Exception:  # noqa: BLE001 —— 同上，最后一级之前的兜底
        logger.exception("pypdf 解析失败，降级到 PyPDFLoader：%s", file_path.name)
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(file_path)).load()


def _load_markdown(file_path: Path) -> List[Document]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": str(file_path)})]


def _load_text(file_path: Path) -> List[Document]:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(file_path), encoding="utf-8").load()


def load_all_documents(data_dir: Path | None = None) -> List[Document]:
    """递归扫描知识库目录，加载所有支持格式的文档。"""
    if data_dir is None:
        data_dir = config.DATA_DIR

    all_docs: List[Document] = []
    if not data_dir.exists():
        logger.warning("知识库目录不存在，无文档可加载：%s", data_dir)
        return all_docs

    for file_path in sorted(data_dir.rglob("*")):
        if file_path.is_dir() or file_path.name.startswith("."):
            continue
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIX:
            continue
        try:
            if suffix == ".pdf":
                docs = _load_pdf(file_path)
            elif suffix in {".md", ".markdown"}:
                docs = _load_markdown(file_path)
            else:
                docs = _load_text(file_path)

            for doc in docs:
                doc.metadata.setdefault("source", str(file_path))
                doc.metadata["file_name"] = file_path.name
            # 过滤空白内容
            docs = [d for d in docs if d.page_content and d.page_content.strip()]
            all_docs.extend(docs)
            logger.info("已加载文档：%s（%d 段）", file_path.name, len(docs))
        except Exception:  # noqa: BLE001
            logger.exception("文档加载失败：%s", file_path)

    logger.info("文档加载完成，共 %d 条记录", len(all_docs))
    return all_docs


def load_text_content(file_name: str, content: str) -> List[Document]:
    """把接口上传的文本内容包装为 Document。"""
    return [Document(page_content=content, metadata={"source": file_name, "file_name": file_name})]


def list_data_files(data_dir: Path | None = None) -> List[dict]:
    """列出知识库目录中的原始文档（用于前端列表展示）。"""
    if data_dir is None:
        data_dir = config.DATA_DIR
    if not data_dir.exists():
        return []
    files = []
    for file_path in sorted(data_dir.rglob("*")):
        if file_path.is_dir() or file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIX:
            continue
        files.append(
            {
                "file_name": file_path.name,
                "size": file_path.stat().st_size,
                "suffix": file_path.suffix.lower(),
                "modified": int(file_path.stat().st_mtime),
            }
        )
    return files
