"""大模型（LLM）提供者 —— 真实模型优先，无 Key 时降级为本地 Mock 模型。

真实模式：ChatOpenAI 对接任意 OpenAI 兼容服务（豆包 / 混元 / DeepSeek / 通义 / OpenAI / 本地 vLLM 等）。
Mock 模式：内置规则引擎，可在完全离线环境下跑通「意图识别 → 检索 → 生成」全链路，
          便于开发自测与 CI，配置 Key 后自动切换，业务代码零改动。

为什么移除了自造的限流重试包装器
--------------------------------
旧实现有一个 ``_RateLimitRetryModel``：命中 429 时按指数退避等待并重试。它是为
**当时的免费档账号**（RPM 极低，连续两次调用要隔 20 秒）写的，代价有两块，
都是实打实的，**且与账号是否限流无关**：

1. ``bind_tools`` 落到 ``BaseChatModel`` 的默认实现（``raise NotImplementedError``）
   ——**底层 ChatOpenAI 本来支持的 function calling，被这层壳整个挡住了**。
   能力不是"没选"，是被自己加的包装闷死的；
2. ``_stream`` 里"未产出内容就回退整段生成"的逻辑与 SDK 自身重试叠加，
   把 ``LLM_TIMEOUT=30`` 实际放大到 101s（有 trace 实证：用户看到的是"一直加载"）。

所以它被整体移除：模型调用**直连 ``ChatOpenAI``**，function calling 走**原生实现**。
少一层包装，就少一处"同一语义两处各定义一遍"的土壤。

⚠️ **重试策略是"不重试"，不是"交回 SDK"**：``_build_raw_model`` 显式传
``max_retries=0``。理由已从"省配额"换成**给出可控的最坏延迟**——SDK 默认重试 2 次
会把 30s 超时放大到上面的 101s，那正是用户看到的"一直加载"。失败改为**快速暴露**，
交由上层降级（检索失败诚实作答、生成失败转人工）。

唯一保留的是 ``_is_rate_limit``：它是**纯判断函数**（零成本、无副作用），
自检模块仍用它区分「限流导致的失败」与「真正的配置错误」——这个区分与具体账号无关，
任何供应商都可能返回 429。换回限流模型时，它立刻又完全有用。
"""
import re
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import config
from app.utils.logger import logger

# LangSmith 可观测接入：必须在「第一次 LLM 调用」前设置环境变量，故在 provider 导入时
# 执行（幂等，未配置 Key 时静默跳过）。见 app/core/observability.py。
from app.core.observability import setup_langsmith

setup_langsmith()

# ============================================================
# Mock 规则库
# ============================================================
_UNKNOWN_KEYWORDS = ["写首诗", "讲个笑话", "股票", "彩票", "预测明天", "天气怎么样"]

_INTENT_RULES: List[tuple] = [
    ("年假", "knowledge"),
    ("报销", "knowledge"),
    ("考勤", "knowledge"),
    ("请假", "knowledge"),
    ("加班", "knowledge"),
    ("密码", "knowledge"),
    ("vpn", "knowledge"),
    ("制度", "knowledge"),
    ("流程", "knowledge"),
    ("手册", "knowledge"),
    ("怎么", "knowledge"),
    ("如何", "knowledge"),
    ("什么是", "knowledge"),
]

_CONTEXT_PATTERN = re.compile(r"参考内容[:：]\s*(.*?)\n用户问题[:：]\s*(.*?)(?:\n对话历史[:：]|$)", re.S)
_QUERY_PATTERN = re.compile(r"用户提问[:：]\s*(.*?)$", re.S)


def _mock_intent(text: str) -> str:
    lowered = text.lower()
    for pattern, intent in _INTENT_RULES:
        if re.search(pattern, lowered):
            return intent
    if any(k in lowered for k in _UNKNOWN_KEYWORDS):
        return "unknown"
    if len(text.strip()) < 4:
        return "unknown"
    return "knowledge"


def _mock_answer(prompt: str) -> str:
    """基于检索内容生成诚实、可溯源的回答（不编造）。"""
    match = _CONTEXT_PATTERN.search(prompt)
    context = match.group(1).strip() if match else ""
    query = match.group(2).strip() if match else ""

    blocks = [b.strip() for b in re.split(r"\n(?=\[片段)", context) if b.strip()]
    if not blocks and context:
        blocks = [context]

    if not blocks:
        return (
            "未在知识库中检索到与您问题相关的内容，为避免误导，我不做推测回答。\n"
            "建议：① 换用更具体的关键词重试；② 在「知识库管理」中补充相关文档；③ 转接人工客服。\n"
            "（当前为本地 Mock 模型，配置 LLM_API_KEY 后将由大模型生成归纳式回答）"
        )

    lines = [f"关于「{query}」，知识库中检索到以下规定：", ""]
    for i, block in enumerate(blocks[:3], 1):
        m = re.match(r"^\[片段\d+\s*·\s*来源：([^·\]]+?)\s*(?:·[^]]*)?\]\s*", block)
        if m:
            source, body = m.group(1).strip(), block[m.end():]
        else:
            source, body = None, block
        body = body if len(body) <= 300 else body[:300] + "…"
        lines.append(f"{i}. 【{source}】" if source else f"{i}.")
        lines.append(f"   {body}")
    lines.append("")
    lines.append("（当前为本地 Mock 模型：以上为检索原文摘录，未做二次归纳；配置 LLM_API_KEY 后将由大模型生成整合式回答）")
    return "\n".join(lines)


class MockChatModel(BaseChatModel):
    """本地规则模型：接口与 ChatModel 完全一致，可无缝替换 ChatOpenAI。

    **不支持 function calling**（未实现 ``bind_tools``）。这是刻意的：
    Mock 的定位是"在完全离线时跑通链路"，不是"模拟智能"。若让它用规则假装
    抽参，测试会以为工具链路验证过了，实际验证的却是 Mock 里的规则——
    真正要测的 function calling 接入反而被掩盖。

    Mock 模式下需要工具调用能力的测试，请注入一个返回固定 ``tool_calls``
    的假模型（见 tests/test_agent.py）。

    注意 `app/core/agent.py` 对这一点有明确处理：捕获 NotImplementedError 后
    退化为「无条件检索 + 受控生成」，保证离线 / CI 环境整条链路仍可跑通。
    """

    model_name: str = "mock-rule-engine"

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(
            m.content for m in messages if isinstance(m, (HumanMessage, SystemMessage)) and isinstance(m.content, str)
        )
        if "knowledge / tool / unknown" in prompt:
            # 关键：只对用户实际提问做分类，避免 prompt 里的规则示例词干扰判定
            m = _QUERY_PATTERN.search(prompt)
            text = _mock_intent(m.group(1).strip() if m else prompt)
        else:
            text = _mock_answer(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


# 按「实际模型名」缓存实例，而非按档位：两档配置成同一模型时复用同一个实例，既省内存，
# 也避免内部状态出现两份不一致的副本。
_chat_models: dict = {}
_chat_mode: Optional[str] = None


def _resolve_model_name() -> str:
    """当前使用的模型名。

    这里曾经是「档位 → 模型名」的映射（flash / pro）。档位机制已随
    「改用不限流模型 + function calling」删除：只有一个模型时，映射表、
    阈值标定、会话 pin 全部失去落点。现在直接返回全局配置的模型名。
    """
    return config.LLM_MODEL_NAME


def _is_rate_limit(exc: Exception) -> bool:
    """判断异常是否为限流（HTTP 429）。兼容 openai / httpx / 原生异常等多种形态。

    采用三层精确判定（避免裸子串误判 request id / content hash 里的 429）：
        1. status_code == 429（最可靠）
        2. 异常类型名含 RateLimit
        3. 文本仅在「错误码语境」下匹配 429，或出现 rate limit / too many requests 字样

    注意：它现在**只用于判断，不再驱动任何重试**（自造的重试包装器已移除）。
    保留它是为了自检模块能区分「限流导致的失败」与「真正的配置错误」——
    前者可跳过、后者必须暴露。
    """
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    if "RateLimit" in type(exc).__name__:
        return True
    text = str(exc).lower()
    if re.search(r"rate.?limit|too many requests", text):
        return True
    return re.search(r"(?:status|code|error)[^\d]{0,12}429", text) is not None


def _build_raw_model(model_name: Optional[str] = None):
    """构建 OpenAI 兼容 ChatModel（全部参数来自配置，无任何模型特定逻辑）。

    ``max_retries=0`` 是**刻意保留**的：SDK 默认对超时/连接错误静默重试 2 次，
    而本项目的策略是**快速失败、交给上层降级**（检索失败就诚实回答，生成失败
    转人工），而不是让用户在原地多等。SDK 的静默重试会把 ``LLM_TIMEOUT``
    成倍放大，且失败时看不到真实原因。
    """
    from langchain_openai import ChatOpenAI

    # provider 特定参数（OpenAI 官方协议无此字段）经 extra_body 透传。
    # 见 config.LLM_DISABLE_THINKING 的说明：推理模型的思考阶段会让首个可见字
    # 延迟约 10 倍，并与正文共享 max_tokens 预算导致截断，故支持一键关闭。
    extra_body = {"thinking": {"type": "disabled"}} if config.LLM_DISABLE_THINKING else None

    return ChatOpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=model_name or config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        max_tokens=config.LLM_MAX_TOKENS,
        # 见上方说明：快速失败，不在 SDK 层静默重试。
        max_retries=0,
        # 多数 OpenAI 兼容服务（Moonshot 等）不支持流式携带 stream_options，
        # 默认关闭以保证兼容；如需 token 用量统计可在 .env 打开 LLM_STREAM_USAGE。
        stream_usage=config.LLM_STREAM_USAGE,
        extra_body=extra_body,
    )


def _build_real_model(model_name: Optional[str] = None):
    """构建真实模型。曾经这里包一层限流重试，现已移除（见模块 docstring）。

    保留这个函数名而不是让调用方直接用 ``_build_raw_model``：两者语义不同
    （"真实模型" vs "未加工模型"），将来若再加通用包装，改这一处即可。
    """
    return _build_raw_model(model_name=model_name)


def get_chat_model() -> BaseChatModel:
    """获取大模型实例（真实模型优先，失败自动降级）。

    返回值原生支持 ``bind_tools``（真实模式为 ChatOpenAI），工具选择与参数填充
    走标准 function calling，不再需要任何自造包装——这正是把架构改为
    「以 function calling 为核心」的前提条件。

    已不再接受 ``tier`` 参数：档位机制（flash / pro）随动态路由一并删除。
    """
    global _chat_mode
    model_name = _resolve_model_name()

    if model_name in _chat_models:
        return _chat_models[model_name]

    if config.USE_REAL_LLM:
        model = _build_real_model(model_name=model_name)
        if config.LLM_HEALTH_CHECK:
            try:
                model.invoke("ping")
            except Exception as exc:  # noqa: BLE001
                logger.warning("大模型启动校验失败，降级为本地 Mock 模型：%s", exc)
                _chat_models[model_name] = MockChatModel()
                _chat_mode = "mock"
                logger.info("大模型初始化完成：mode=mock（离线规则引擎）")
                return _chat_models[model_name]
        _chat_models[model_name] = model
        _chat_mode = "real"
        logger.info(
            "大模型初始化成功：mode=real model=%s base_url=%s（LLM_HEALTH_CHECK=%s）",
            model_name, config.LLM_BASE_URL, config.LLM_HEALTH_CHECK,
        )
        return _chat_models[model_name]

    _chat_models[model_name] = MockChatModel()
    _chat_mode = "mock"
    logger.info("大模型初始化完成：mode=mock（离线规则引擎）")
    return _chat_models[model_name]


def get_llm_mode() -> str:
    if not _chat_models:
        get_chat_model()
    return _chat_mode or "unknown"


def reset_chat_model() -> None:
    """清空模型实例缓存（用于配置变更后的热切换）。"""
    global _chat_mode
    _chat_models.clear()
    _chat_mode = None
