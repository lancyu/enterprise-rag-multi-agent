"""记忆系统管理接口 —— 查看、编辑、蒸馏长期记忆。

把记忆暴露成可读写接口，是刻意为之的设计：
    自动记忆最大的风险不是"记错"，而是"记了什么用户不知道"。
    用户必须能看到助手记住了什么、能手动修正、能一键蒸馏。
    这也是 nanobot 提供 /dream、/dream-log 等命令的出发点。
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.memory import get_long_term_memory, get_memory_stats, run_dream
from app.memory.store import MemoryStore
from app.utils.logger import logger

router = APIRouter(prefix="/memory", tags=["记忆系统"])

#: SOUL.md 是全局共享的人格设定。上限用于防误操作（一次 PUT 写坏全站人格），
#: 不是性能考虑——KB 级文本对读取毫无压力。
SOUL_MAX_CHARS = 8000


class FactsRequest(BaseModel):
    """手动写入长期事实。"""

    facts: list = Field(..., description="事实条目列表")
    user_id: str = Field("default", max_length=64)


class SoulRequest(BaseModel):
    """覆盖写入全局人格设定（SOUL.md）。"""

    content: str = Field(
        ...,
        max_length=SOUL_MAX_CHARS,
        description="Markdown 文本，**覆盖式**写入（非追加）。传空串等于清除人格设定",
    )


@router.put("/soul")
def memory_write_soul(req: SoulRequest) -> dict:
    """覆盖写入全局人格设定（SOUL.md）。**要求开启入站鉴权**。

    为什么这个端点比其它记忆接口多一道闸门：

    ``PUT /memory/{user_id}/facts``、``DELETE /memory/{user_id}`` 影响面是**单个
    用户**；SOUL.md 是全局共享的，会被拼进**每个人**的 ``memory_context``。
    "一次改掉全站人格"的能力不应该存在于一个整体不设防的服务上，所以：

    - ``AUTH_ENABLED=false`` 时直接 403（fail-closed）。服务默认不开启鉴权是为了
      本地开发的零配置体验，但这个理由不适用于全局破坏面操作。响应体里写明
      如何开启，避免调用方以为是接口写错了。
    - 一旦开启鉴权，请求已由 ``app.main`` 的中间件校验过 API Key，本端点无需
      重复校验。

    写入是原子的（``MemoryStore._write_file``），不会留下半截内容。
    """
    if not config.AUTH_ENABLED:
        raise HTTPException(
            status_code=403,
            detail=(
                "该接口会修改全局人格设定（影响所有用户），仅在开启入站鉴权后可用。"
                "请设置 AUTH_ENABLED=true 与 AUTH_API_KEY，或直接编辑 SOUL.md。"
            ),
        )

    content = req.content
    store = MemoryStore()
    store.write_soul(content)
    logger.info("SOUL.md 已更新：覆盖写入 %d 字", len(content))
    return {
        "code": 0,
        "message": "全局人格设定已更新" if content.strip() else "全局人格设定已清空",
        "chars": len(content),
        "empty": not content.strip(),
    }


@router.get("/{user_id}/stats")
def memory_stats(user_id: str) -> dict:
    """记忆系统状态：归档条目数、待蒸馏数量、各开关状态。"""
    return {"code": 0, **get_memory_stats(user_id)}


@router.get("/{user_id}")
def memory_detail(user_id: str) -> dict:
    """查看用户的全部长期记忆内容（可人工审阅与修正）。"""
    ltm = get_long_term_memory(user_id)
    return {
        "code": 0,
        "user_id": user_id,
        "files": ltm.export(),
        "facts": ltm.get_facts(),
        "recent_history": ltm.recent_history(limit=10),
        "stats": ltm.stats(),
    }


@router.post("/{user_id}/dream")
def memory_dream(user_id: str, batch_size: Optional[int] = None) -> dict:
    """手动触发一次记忆蒸馏（把对话归档沉淀为长期知识）。

    默认不自动执行：蒸馏需要真实的大模型调用（一次完整往返），
    交由调用方在低峰期或按需触发。
    """
    try:
        result = run_dream(user_id, batch_size=batch_size)
        return {"code": 0, "message": "蒸馏完成", **result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("记忆蒸馏失败")
        raise HTTPException(status_code=500, detail=f"蒸馏失败：{exc}")


@router.post("/{user_id}/facts")
def memory_add_facts(user_id: str, req: FactsRequest) -> dict:
    """手动写入长期事实（自动去重）。"""
    facts = [str(f).strip() for f in (req.facts or []) if str(f).strip()]
    if not facts:
        raise HTTPException(status_code=400, detail="facts 不能为空")
    added = get_long_term_memory(user_id).add_facts(facts)
    return {"code": 0, "message": f"新增 {added} 条事实", "added": added}


@router.delete("/{user_id}")
def memory_clear(user_id: str) -> dict:
    """清空该用户的长期事实与用户画像（归档保留，便于追溯）。"""
    ltm = get_long_term_memory(user_id)
    ltm.clear_facts()
    ltm.store.write_user("# 用户画像\n")
    logger.info("已清空用户长期记忆：%s", user_id)
    return {"code": 0, "message": f"用户 {user_id} 的长期记忆已清空"}
