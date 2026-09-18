"""Dify 兼容接口 —— 把本项目作为 Dify 的「外部知识库」接入。

背景
----
Dify 的「知识检索」节点支持接入外部知识库：它按固定契约
`POST {base_url}/retrieval` 调用外部服务。本项目实现该契约后即可被 Dify
直接调用，好处是**文档不用再往 Dify 传一份** —— 切分策略、检索权重、
拒答阈值仍然由本项目掌控，Dify 只负责编排与生成。

规范要点（docs.dify.ai · External Knowledge API）
------------------------------------------------
- 认证：Dify 把配置的 API Key 以 `Authorization: Bearer {key}` 透传，
  **校验逻辑由本项目实现**，Dify 自己不校验。
- 请求：`{knowledge_id, query, retrieval_setting:{top_k, score_threshold}, metadata_condition?}`
- 响应：`{"records": [{"content", "score", "title", "metadata"}]}`
  - `metadata` 必须是对象，**不能为 null**（null 会导致 Dify 检索流程报错）
  - `score` 必须是 0~1 —— 这是最大的坑，见 `_normalize_score`
- 错误码（约定，非强制）：1001 无效 Authorization 格式 / 1002 认证失败 / 2001 知识库不存在

路由
----
同时注册 `/retrieval` 与 `/dify/retrieval`：Dify 只要求「API Base URL + /retrieval」，
两种填法都能work，避免用户猜该填哪个。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import config
from app.db.vector_db import get_vector_store
from app.rag import retriever
from app.utils.auth_header import api_key_matches, bearer_token
from app.utils.doc_loader import file_name
from app.utils.logger import logger, preview

router = APIRouter(tags=["Dify 兼容接口"])

# Dify 约定错误码
ERR_AUTH_FORMAT = 1001
ERR_AUTH_FAILED = 1002
ERR_KB_NOT_FOUND = 2001
ERR_INTERNAL = 5001


class RetrievalSetting(BaseModel):
    top_k: int = Field(default=3, ge=1, le=50)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalRequest(BaseModel):
    knowledge_id: str = ""
    query: str = ""
    retrieval_setting: RetrievalSetting = Field(default_factory=RetrievalSetting)
    metadata_condition: Optional[Dict[str, Any]] = None


def _err(error_code: int, error_msg: str, status: int = 200) -> JSONResponse:
    """按 Dify 约定返回错误。

    默认用 **200 + error_code** 而不是 4xx：Dify 先看 HTTP 状态码，非 200
    只在界面上抛一个没有上下文的通用错误；返回 200 + `error_msg` 才能把具体
    原因（例如「服务端未配置 DIFY_API_KEY」）直接显示给用户，排查成本低得多。
    """
    return JSONResponse(
        status_code=status,
        content={"error_code": error_code, "error_msg": error_msg},
    )


def _normalize_score(fused: float) -> float:
    """把 RRF 融合分归一化到 0~1。

    ⚠️ 这是接 Dify 最容易踩、也最难排查的坑：
    Dify 的 `score_threshold` 语义是 **0~1 的绝对相关度**，而本项目内部的
    `fused` 是 RRF 分，量级只有 `(w_dense + w_lex) / (RRF_K + 1)` ≈ 0.016。
    若直接透传，Dify 里只要把分数阈值调到 0.02 以上，所有结果都会被过滤掉，
    表现为「知识库明明有内容，Dify 却永远检索不到」。

    做法：除以 RRF 的理论上限。该上限只在 `app.rag.retriever.rrf_upper_bound`
    定义一次——此前这里与 `app/rag/generator.py` 各算一遍同一个公式，改权重时
    只要漏改一处，两边的归一化口径就不再一致。
    与项目内置信度计算口径一致，因此归一化后的分数跨查询、跨配置可比。
    """
    upper = retriever.rrf_upper_bound()
    if upper <= 0:
        return 0.0
    return max(0.0, min(1.0, float(fused) / upper))


#: Dify 外部知识库**对外暴露**的 metadata 字段白名单。
#:
#: 必须显式声明，而不是把内部 metadata 全量倒给外部——这些字段既供 Dify 侧
#: 的 `metadata_condition` 过滤，也供其前端做引用展示，属于对外契约。
#: 曾经只声明 6 个字段，导致 Dify 按 `doc_version` / `file_type` / 生效日期
#: 等配置过滤时，`_match_conditions` 取值为空串 → `is` 比较恒假 → 记录被
#: 静默丢弃、返回空结果且不报错（表现为「知识库明明有这份文档，Dify 却
#: 永远检索不到」）。新增字段时请同步 tests/test_dify_api.py 的字段集断言。
_DIFY_PAYLOAD_FIELDS = (
    "chapter",
    "section",
    "heading_path",
    "chunk_index",
    # —— 文档级业务字段：过滤 / 时效判断 / 引用展示 ——
    "doc_title",
    "doc_version",
    "doc_effective_date",
    "doc_updated_date",
    "file_type",
    "structure",
    "page",
)


def _build_payload_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """按对外白名单构造 Dify 的 `records[].metadata`。

    两个字段不走白名单：
    - `source` 可能缺席，兜底为 `""`；
    - `file_name` 由 `source` 路径切出，不依赖上游是否写了该字段。

    最后剔除空值：Dify 要求 metadata 必须是对象，且空串/哨兵值会让其过滤
    界面出现无意义的可选项。
    """
    meta = dict(meta or {})
    source = str(meta.get("source", ""))
    payload: Dict[str, Any] = {"source": source, "file_name": file_name(source)}
    for field in _DIFY_PAYLOAD_FIELDS:
        # chunk_index / page 的缺席哨兵是 -1，其余字段是空串
        payload[field] = meta.get(field, -1 if field in ("chunk_index", "page") else "")
    return {k: v for k, v in payload.items() if v not in ("", None, -1)}


def _match_conditions(meta: Dict[str, Any], cond: Optional[Dict[str, Any]]) -> bool:
    """Dify `metadata_condition` 的最小可用实现。

    只实现最常用的运算符；**遇到不支持的运算符一律放行**（返回 True）而不报错：
    宁可少过滤，也不能让整次检索因为一个花哨条件而失败。
    """
    if not cond:
        return True
    conditions = cond.get("conditions") or []
    if not conditions:
        return True
    logical = str(cond.get("logical_operator") or "and").lower()

    results: List[bool] = []
    for c in conditions:
        name = c.get("name") or ""
        op = str(c.get("comparison_operator") or "").lower()
        val = c.get("value")
        actual = str(meta.get(name, ""))
        target = "" if val is None else str(val)
        values = val if isinstance(val, list) else [target]

        if op == "is":
            ok = actual == target
        elif op == "is not":
            ok = actual != target
        elif op == "contains":
            ok = target in actual
        elif op == "not contains":
            ok = target not in actual
        elif op == "start with":
            ok = actual.startswith(target)
        elif op == "end with":
            ok = actual.endswith(target)
        elif op == "in":
            ok = actual in values
        elif op == "not in":
            ok = actual not in values
        elif op == "empty":
            ok = actual == ""
        elif op == "not empty":
            ok = actual != ""
        else:
            ok = True          # 不支持的运算符：放行
        results.append(ok)

    return any(results) if logical == "or" else all(results)


def _check_auth(authorization: Optional[str]) -> Optional[JSONResponse]:
    """校验 Bearer 鉴权，通过返回 None，失败返回错误响应。"""
    expected = config.DIFY_API_KEY
    if not expected:
        logger.error("Dify 检索被拒绝：服务端未配置 DIFY_API_KEY")
        return _err(ERR_AUTH_FAILED, "服务端未配置 DIFY_API_KEY，请在 .env 中配置后重启服务")

    if not authorization or not authorization.strip():
        return _err(ERR_AUTH_FORMAT, "缺少 Authorization 请求头，期望 'Bearer {API_KEY}' 格式")

    # 解析委托 app.utils.auth_header（全项目唯一实现处）。
    # 这里原先自己写 `split(None, 1)`，它按**任意空白**切，于是
    # `"Bearer\txxx"` 在本端点被接受、在入站中间件被拒绝 —— 同一个请求两条入口两种结论。
    token = bearer_token(authorization)
    if not token:
        return _err(ERR_AUTH_FORMAT, "Invalid Authorization header format. Expected 'Bearer {API_KEY}'.")

    # 定长安全比较（避免了计时侧信道），fail-closed 也一并由它负责
    if not api_key_matches(token, expected):
        return _err(ERR_AUTH_FAILED, "Authorization failed. Please check your API key.")
    return None


async def _do_retrieval(req: RetrievalRequest, authorization: Optional[str]) -> JSONResponse:
    """Dify 外部知识库检索的实际实现（两个路由共用）。"""
    auth_err = _check_auth(authorization)
    if auth_err:
        return auth_err

    kid = (req.knowledge_id or "").strip()
    if not kid:
        return _err(ERR_KB_NOT_FOUND, "knowledge_id 不能为空（请在 Dify 外部知识库配置里填写「外部知识库 ID」）")
    expected_kid = config.DIFY_KNOWLEDGE_ID
    if expected_kid and kid != expected_kid:
        return _err(ERR_KB_NOT_FOUND, f"知识库不存在：{kid}（服务端期望 {expected_kid}）")

    query = (req.query or "").strip()
    if not query:
        return JSONResponse(content={"records": []})   # 空查询 = 无结果，不是错误

    top_k = req.retrieval_setting.top_k
    threshold = req.retrieval_setting.score_threshold
    try:
        # 多召回一些再按归一化分过滤，避免阈值过滤后凑不满 top_k
        hits = retriever.retrieve(query, top_k=max(top_k * 2, 10))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dify 检索失败：query=%s", preview(query, 50))
        return _err(ERR_INTERNAL, f"检索失败：{exc}", status=500)

    records: List[Dict[str, Any]] = []
    for hit in hits:
        score = _normalize_score(float(hit.get("fused", 0.0)))
        if score < threshold:
            continue
        meta = dict(hit.get("metadata") or {})
        source = str(meta.get("source", ""))

        payload_meta = _build_payload_meta(meta)
        if not _match_conditions(payload_meta, req.metadata_condition):
            continue

        records.append({
            # 父子索引开启时用父块，让 Dify 拿到的上下文更完整
            "content": hit.get("parent_content") or hit.get("content", ""),
            "score": round(score, 4),
            "title": file_name(source) or "unknown",
            "metadata": payload_meta or {},
        })
        if len(records) >= top_k:
            break

    logger.info("Dify 检索 | knowledge_id=%s | query=%s | 返回 %d 条", kid, preview(query, 30), len(records))
    return JSONResponse(content={"records": records})


@router.post("/retrieval")
async def dify_retrieval_root(
    req: RetrievalRequest,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    """Dify 外部知识库检索（根路径，对应 Dify 里填 `http://host:port`）。"""
    return await _do_retrieval(req, authorization)


@router.post("/dify/retrieval")
async def dify_retrieval(
    req: RetrievalRequest,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    """Dify 外部知识库检索（显式前缀，对应 Dify 里填 `http://host:port/dify`）。"""
    return await _do_retrieval(req, authorization)


@router.get("/dify/openapi.json")
async def dify_openapi(request: Request) -> dict:
    """返回 OpenAPI schema，供 Dify「自定义工具」导入。

    外部知识库契约解决的是「知识检索节点」；若想在 Dify 里把本项目当成一个
    可被 Agent **自主调用的工具**，则需要这份 schema（Dify 自定义工具要求
    OpenAPI 格式）。两条路指向同一个 `/retrieval`，鉴权方式也一致。
    """
    base = str(request.base_url).rstrip("/")
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "企业知识引擎 · 检索",
            "version": "1.0.0",
            "description": "从企业知识库检索与问题相关的文档片段（章节感知切分 + 混合召回）。",
        },
        "servers": [{"url": base}],
        "paths": {
            "/retrieval": {
                "post": {
                    "summary": "检索知识库",
                    "operationId": "retrieval",
                    "description": "输入自然语言问题，返回最相关的文档片段。",
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": {"type": "string", "description": "检索问题"},
                                        "knowledge_id": {"type": "string", "description": "知识库 ID"},
                                        "top_k": {"type": "integer", "default": 3, "description": "返回条数"},
                                        "score_threshold": {
                                            "type": "number", "default": 0.0,
                                            "description": "相关度下限（0~1，已归一化）",
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "检索结果",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "records": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "content": {"type": "string"},
                                                        "score": {"type": "number"},
                                                        "title": {"type": "string"},
                                                        "metadata": {"type": "object"},
                                                    },
                                                },
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            }
        },
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}
        },
    }


@router.get("/dify/info")
async def dify_info() -> dict:
    """接入自检：返回端点配置状态（**不含密钥**），在 Dify 连通前先自查。"""
    try:
        chunks = get_vector_store().count()
    except Exception:  # noqa: BLE001
        chunks = -1
    return {
        "enabled": bool(config.DIFY_API_KEY),
        "endpoints": ["/retrieval", "/dify/retrieval"],
        "openapi_for_custom_tool": "/dify/openapi.json",
        "auth": "Authorization: Bearer {DIFY_API_KEY}",
        "knowledge_id": config.DIFY_KNOWLEDGE_ID or "（未限制，接受任意非空值）",
        "expects_knowledge_id": bool(config.DIFY_KNOWLEDGE_ID),
        "index_chunks": chunks,
        # 提醒：Dify 的 score_threshold 是 0~1，本项目内部 RRF 分量级约 0.016，已做归一化
        "score_range": "0~1（已对 RRF 融合分做归一化）",
        "docs": "docs/dify-integration.md",
    }
