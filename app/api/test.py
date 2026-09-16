"""服务自测接口 —— 全量 7 项检测 / 快速 3 项健康检测。"""
import asyncio

from fastapi import APIRouter, Query

from app.core.self_check import get_health, run_self_check
from app.utils.logger import logger

router = APIRouter(prefix="/test", tags=["服务自测"])


@router.get("/all")
async def test_all() -> dict:
    """执行全量服务测试（含深度检查：真实调用 LLM / Embedding）。"""
    report = await asyncio.to_thread(run_self_check, deep=True)
    logger.info("全量自测完成：%d/%d 通过", report["passed"], report["total"])
    return {"code": 0, **report}


@router.get("/quick")
async def test_quick() -> dict:
    """执行 3 项快速健康检测。"""
    report = await asyncio.to_thread(run_self_check, only_fast=True)
    return {"code": 0, **report}


@router.get("/health")
async def test_health(verbose: bool = Query(default=False)) -> dict:
    """轻量健康检查（可选返回完整配置快照）。"""
    data = await asyncio.to_thread(get_health)
    if not verbose:
        data.pop("config", None)
    return {"code": 0, **data}
