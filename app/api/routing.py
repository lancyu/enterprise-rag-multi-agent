"""意图路由预演接口 —— **纯预演，不进入任何子 Agent，不产生任何副作用。**

为什么必须有这个接口（而不是"看日志就行"）
------------------------------------------
混合路由是一个**可判错的粗分类器**，判错的排障成本极高：
"这句话为什么走了工具通道"如果只能靠读日志反推，就永远无法回答
"是分低了，还是阈值配错了"。所以这里返回的是:

1. **完整候选得分表**（含第二名与差距），不是单一标签；
2. **本次生效的阈值**（含 embedding 模式）——没有它，看到 ``low_floor``
   也不知道该怪分还是怪配置。

预演接口与生产链路**互相隔离**：
- 它不写 ``request_ctx`` 里的业务状态、不调用任何子 Agent、不写库；
- 它可以在生产流量上做**影子调用**（同一句话两条路各跑一次，只记不改），
  这正是 Phase 0 要拿"旧判定 / 新判定 / 人工标注"三元对照表的方式（文档 §8.1）。

之所以安全：路由漏斗本身是**只读**的（唯一的外部依赖是 embedding 与一次
灰区模型调用），所以预演不会污染任何东西。

Phase 0 的定位：**本接口是唯一读混合路由配置的地方**，
生产链路仍走 ``router_agent.route_query`` 的一次模型调用。
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app import config
from app.core.routing import match_intent
from app.core.routing.catalog import CHANNELS, CHANNEL_TARGETS
from app.core.routing.router import describe_catalog
from app.utils.validator import MAX_QUERY_LENGTH, sanitize_text

router = APIRouter(prefix="/routing", tags=["意图路由"])


class IntentPreviewRequest(BaseModel):
    """预演请求。"""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="待判定的用户原话")
    explain: bool = Field(
        True,
        description="是否返回完整候选得分表。**标定阈值时必须为 true**——只看标签无法调参",
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        v = sanitize_text(v.strip())
        if not v:
            raise ValueError("query 不能为空或仅包含非法字符")
        return v


def _thresholds() -> dict:
    """本次生效的全部阈值与**它们为什么是这个值**。

    把 embedding 模式一并返回：语义层的两个阈值是随模式自适应的
    （本地哈希与真实神经向量的余弦尺度差一个数量级），
    不写明模式，跨环境比对同一个灰区原因会得出完全相反的结论。
    """
    from app.providers.embeddings import get_embedding_mode

    return {
        "embedding_mode": get_embedding_mode(),
        "lexical_floor": config.ROUTE_LEXICAL_FLOOR,
        "lexical_margin": config.ROUTE_LEXICAL_MARGIN,
        "semantic_floor": config.effective_route_semantic_floor(),
        "semantic_margin": config.effective_route_semantic_margin(),
        "budget_ms": config.ROUTE_BUDGET_MS,
        "embed_timeout_ms": config.ROUTE_EMBED_TIMEOUT_MS,
        "arbitration_timeout_ms": config.ROUTE_ARBITRATION_TIMEOUT_MS,
    }


@router.post("/intent-preview")
def intent_preview(req: IntentPreviewRequest) -> dict:
    """判定一句话会走哪个通道，并给出**完整依据**。

    ``source`` 取值：``anchor``（层① 确定性）/ ``lexical``（词面直接通过）
    / ``fused``（词面+语义融合通过）/ ``arbitration``（灰区 LLM 仲裁）
    / ``fallback``（兜底）。

    ⚠️ ``gray_reason`` 非空**不等于**故障：它表示"这一次没能拿到足够证据"，
    与 ``degraded``（本该做的事没做成）是两件事，前端与指标都不要混用。
    """
    decision = match_intent(req.query)
    payload = decision.to_dict(explain=req.explain)
    payload["thresholds"] = _thresholds()
    if decision.channel not in CHANNELS or not CHANNEL_TARGETS.get(decision.channel):
        # 理论上不可达（通道是闭集、启动期已校验过映射）。留着是因为
        # "预演接口本身出错"比"路由判错"更难被发现——它没有下游会报错。
        payload["warning"] = f"通道 {decision.channel!r} 没有对应的图节点"
    return {"code": 0, **payload}


@router.get("/catalog")
def routing_catalog() -> dict:
    """当前生效的意图目录快照。

    存在的理由是**可对照**：一次误判发生时，最需要回答的是
    "这条规则当时长什么样"。规则写在代码里、随版本变化，事后翻日志是翻不到的。
    """
    specs = describe_catalog()
    return {
        "code": 0,
        "channels": list(CHANNELS),
        "channel_targets": dict(CHANNEL_TARGETS),
        "specs": specs,
        "thresholds": _thresholds(),
    }
