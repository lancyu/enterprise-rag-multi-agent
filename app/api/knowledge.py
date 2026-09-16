"""知识库管理接口 —— 列表 / 上传 / 删除 / 语义检索 / 索引重建。"""
import asyncio
import threading

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app import config
from app.core.rag_engine import add_document, build_index, delete_document, get_stats, search
from app.utils.doc_loader import list_data_files
from app.utils.logger import logger
from app.utils.validator import KnowledgeSearchRequest, KnowledgeUploadRequest, sanitize_filename

router = APIRouter(prefix="/knowledge", tags=["知识库管理"])

# 重建索引会先清空整个向量库再重灌，期间所有问答都会命中空库。
# 并发重建会让两次清空/重灌互相踩踏，故串行化并拒绝重入。
# 注意：这是**单实例**锁；多副本部署需要改用 Redis 分布式锁，
# 与限流（core/rate_limit.py）面临的是同一类问题。
_REBUILD_LOCK = threading.Lock()


def _human_size(n: int) -> str:
    """把字节数渲染成易读单位（避免 1KB 显示成「0MB」这种没用的提示）。"""
    if n >= 1024 * 1024:
        value, unit = n / (1024 * 1024), "MB"
    else:
        value, unit = n / 1024, "KB"
    return f"{value:.0f}{unit}" if value >= 10 else f"{value:.1f}{unit}"


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """分块读取并在超出上限时立即中断。

    不能先 `await upload.read()` 再判断大小 —— 那样超限文件已经完整进内存，
    服务早就被打爆了。边读边累计才能真正挡住大文件。
    """
    pieces: list = []
    total = 0
    while True:
        piece = await upload.read(65536)
        if not piece:
            break
        total += len(piece)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大，上限 {_human_size(limit)}（可用 MAX_UPLOAD_BYTES 调整）",
            )
        pieces.append(piece)
    return b"".join(pieces)


@router.get("/list")
async def knowledge_list() -> dict:
    """获取知识库文档列表（含向量库分片统计）。"""
    stats = get_stats()
    return {
        "code": 0,
        "files": list_data_files(),
        "sources": stats["sources"],
        "total_chunks": stats["total_chunks"],
        "config": stats["config"],
    }


@router.post("/upload")
async def knowledge_upload(req: KnowledgeUploadRequest) -> dict:
    """上传文本形式的知识库文档（JSON 接口，供前端与第三方系统调用）。"""
    try:
        # 入库含 embedding 调用，是同步耗时操作，放进线程避免阻塞事件循环
        chunks = await asyncio.to_thread(add_document, req.file_name, req.content)
        return {"code": 0, "file_name": req.file_name, "chunks": chunks, "total_chunks": get_stats()["total_chunks"]}
    except Exception:  # noqa: BLE001
        logger.exception("文档上传失败")
        raise HTTPException(status_code=500, detail="文档入库失败，请查看服务日志定位原因")


@router.post("/upload-file")
async def knowledge_upload_file(request: Request, file: UploadFile = File(...)) -> dict:
    """以 multipart 形式上传 PDF / Markdown / TXT 文档。"""
    filename = sanitize_filename(file.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="文件名无效")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"pdf", "md", "markdown", "txt"}:
        raise HTTPException(status_code=400, detail="仅支持 PDF / Markdown / TXT 格式")

    # 第一道闸：Content-Length 预检，超限请求连 body 都不读
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，上限 {_human_size(config.MAX_UPLOAD_BYTES)}",
        )
    # 第二道闸：边读边累计，防止客户端谎报 Content-Length
    raw = await _read_limited(file, config.MAX_UPLOAD_BYTES)

    try:
        if suffix == "pdf":
            # PyPDFLoader 仅支持文件路径，先落盘到临时文件再解析
            import tempfile
            from pathlib import Path

            from langchain_community.document_loaders import PyPDFLoader

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = Path(tmp.name)
            try:
                docs = PyPDFLoader(str(tmp_path)).load()
            finally:
                tmp_path.unlink(missing_ok=True)
            content = "\n".join(d.page_content for d in docs)
        else:
            content = raw.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        # 不回显解析异常原文（可能含临时文件路径等内部细节），详情已入日志
        logger.exception("文件解析失败：%s", filename)
        raise HTTPException(status_code=400, detail="文件解析失败，请确认文件未损坏且格式正确")

    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")

    # 超限截断必须**显式告知**，否则用户会以为整篇都入库了，
    # 之后检索不到后半部分内容时将无从排查。
    max_chars = config.MAX_DOC_CONTENT_CHARS
    truncated = len(content) > max_chars
    if truncated:
        logger.warning("文档 %s 内容 %d 字超过上限 %d，已截断", filename, len(content), max_chars)

    chunks = add_document(filename, content[:max_chars])
    return {
        "code": 0,
        "file_name": filename,
        "chunks": chunks,
        "truncated": truncated,
        "total_chunks": get_stats()["total_chunks"],
    }


@router.delete("/{file_name}")
async def knowledge_delete(file_name: str) -> dict:
    """删除指定文档的全部向量片段。"""
    removed = delete_document(file_name)
    return {"code": 0, "file_name": file_name, "removed": removed, "total_chunks": get_stats()["total_chunks"]}


@router.post("/search")
async def knowledge_search(req: KnowledgeSearchRequest) -> dict:
    """纯语义检索测试（不经过大模型，用于调试检索质量）。"""
    hits = search(req.query, top_k=req.top_k)
    return {
        "code": 0,
        "query": req.query,
        "hits": [
            {"content": h["content"], "source": h["source"].split("/")[-1], "score": h["score"], "fallback": h["fallback"]}
            for h in hits
        ],
    }


@router.post("/rebuild")
async def knowledge_rebuild() -> dict:
    """一键重建向量索引（串行执行，且期间拒绝重入）。"""
    if not _REBUILD_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="索引重建进行中，请稍后重试")
    try:
        # 重建是同步且耗时的（全量重新 embedding），放进线程避免阻塞事件循环
        result = await asyncio.to_thread(build_index)
        return {"code": 0, **result}
    except Exception:  # noqa: BLE001
        logger.exception("索引重建失败")
        raise HTTPException(status_code=500, detail="索引重建失败，请查看服务日志定位原因")
    finally:
        _REBUILD_LOCK.release()
