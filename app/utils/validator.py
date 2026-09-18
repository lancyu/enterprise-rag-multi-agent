"""请求校验器 —— 输入安全校验 & 参数合法性检查。

防御点：
1. 注入过滤：清除 <script> / {{ }} / <% %> 等模板与脚本注入片段
2. 参数长度限制：防止超大参数压垮服务
3. 文件名安全清洗：禁止隐藏文件与非法字符写入知识库
"""
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app import config

# ------------------------------------------------------------
# 校验规则常量
# ------------------------------------------------------------
MAX_QUERY_LENGTH = 2000

#: 单篇文档内容长度上限。**数字的唯一来源是 :data:`app.config.MAX_DOC_CONTENT_CHARS`。**
#:
#: 这里原先硬编码 ``50_000``，与 config 那份是两个数字：改一处忘一处时，
#: 两个上传入口会对同一份文档给出**不同**的结果，而且两边都不报错。
#:
#: ⚠️ 数字统一了，**语义并没有统一，也不该统一**：
#: 本模块服务的是 JSON 接口（调用方是程序）→ 超限**拒收**，它能改；
#: 文件上传接口（调用方是人）→ 超限**截断**并回 ``truncated``，报错等于整篇不收。
#: 详见 config 里该配置项上的说明。
MAX_DOC_CONTENT = config.MAX_DOC_CONTENT_CHARS

FORBIDDEN_PATTERNS: List[re.Pattern] = [
    re.compile(r"<script.*?>", re.IGNORECASE),
    re.compile(r"</script>", re.IGNORECASE),
    re.compile(r"\{\{.*?\}\}"),
    re.compile(r"<%[^>]*%>"),
    re.compile(r"javascript:", re.IGNORECASE),
]

SESSION_ID_PATTERN = r"[a-zA-Z0-9_-]+"


def sanitize_text(text: str) -> str:
    """过滤潜在注入片段。"""
    for pattern in FORBIDDEN_PATTERNS:
        text = pattern.sub("", text)
    return text


def sanitize_filename(name: str) -> str:
    """文件名安全清洗：仅保留安全字符，禁止隐藏文件。"""
    name = re.sub(r"[^\w.\-]", "_", name, flags=re.UNICODE)
    if name.startswith("."):
        name = "_" + name
    return name


class ChatRequest(BaseModel):
    """对话请求模型（含校验）"""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="用户提问")
    chat_history: list = Field(default_factory=list, description="多轮对话历史")
    session_id: Optional[str] = Field(default=None, max_length=64, description="会话标识")
    # 用户标识：用于隔离长期记忆。不传时按 default 处理，
    # 未做用户体系的部署场景下不影响原有行为。
    user_id: Optional[str] = Field(default=None, max_length=64, description="用户标识（隔离长期记忆）")
    # 来源白名单：按部门/来源做知识隔离。传入后检索只返回 source 在该列表内的片段。
    # 由调用方（鉴权中间件/网关）按用户权限注入；不传表示不过滤（默认行为）。
    allowed_sources: Optional[List[str]] = Field(
        default=None, description="来源白名单（按来源/部门隔离知识），不传表示不过滤"
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        v = sanitize_text(v.strip())
        if not v:
            raise ValueError("query 不能为空或仅包含非法字符")
        return v

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: Optional[str]) -> Optional[str]:
        if v and not re.fullmatch(SESSION_ID_PATTERN, v):
            raise ValueError("user_id 只能包含字母、数字、下划线和连字符")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v and not re.fullmatch(SESSION_ID_PATTERN, v):
            raise ValueError("session_id 只能包含字母、数字、下划线和连字符")
        return v

    @field_validator("allowed_sources")
    @classmethod
    def validate_allowed_sources(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if len(cleaned) > 256:
            raise ValueError("allowed_sources 最多 256 项")
        return cleaned


class KnowledgeUploadRequest(BaseModel):
    """知识文档上传请求"""

    file_name: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    rebuild: bool = Field(default=True, description="上传后是否增量入库")

    @field_validator("file_name")
    @classmethod
    def clean_filename(cls, v: str) -> str:
        return sanitize_filename(v)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if len(v) > MAX_DOC_CONTENT:
            raise ValueError(f"单篇文档内容不能超过 {MAX_DOC_CONTENT} 字符")
        return sanitize_text(v)


class KnowledgeSearchRequest(BaseModel):
    """知识库语义检索请求"""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        v = sanitize_text(v.strip())
        if not v:
            raise ValueError("query 不能为空")
        return v


class WorkflowExecuteRequest(BaseModel):
    """手动触发工作流请求"""

    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    session_id: Optional[str] = Field(default=None, max_length=64)
    # 会话标识：只用于隔离长期记忆，与"谁在提问"无关。不传按 default 处理。
    user_id: Optional[str] = Field(
        default=None, max_length=64, description="会话标识（隔离长期记忆）"
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        return sanitize_text(v.strip())

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: Optional[str]) -> Optional[str]:
        # 与 ChatRequest 同一套字符约束：user_id 会参与记忆目录的路径拼接，
        # 不允许出现空白/引号等会让路径静默失配的字符。
        if v and not re.fullmatch(SESSION_ID_PATTERN, v):
            raise ValueError("user_id 只能包含字母、数字、下划线和连字符")
        return v
