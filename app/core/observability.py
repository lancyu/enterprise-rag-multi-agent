"""LangSmith 可观测接入（可选）—— 把 LLM 调用自动 trace 到 LangSmith 平台。

LangSmith 是 LangChain 官方可观测平台，接入后每次 LLM 调用（输入输出、token 用量、
耗时）自动上报，可在 Web 控制台按 trace 回放、统计成本与延迟。

接入原理（业界标准做法）：
    设置环境变量 LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT 后，
    LangChain 的 ChatOpenAI 等模型调用会透传 trace 到 LangSmith，业务代码零改动。
    关键约束：必须在「第一次 LLM 调用」之前设置，故在 llm_factory 导入时执行。

本模块是可选依赖：未配置 Key 或未安装 langsmith 时静默跳过，绝不拖垮主链路。
"""
import os

from app.utils.logger import logger

_setup_done = False


def setup_langsmith() -> None:
    """按配置设置 LangSmith 环境变量（幂等，进程内只需执行一次）。"""
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    from app import config

    if not config.LANGSMITH_ENABLED or not config.LANGSMITH_API_KEY:
        return
    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
        if config.LANGSMITH_ENDPOINT:
            os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT
        logger.info("LangSmith 追踪已启用：project=%s", config.LANGSMITH_PROJECT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LangSmith 初始化失败，本次进程不上报 trace：%s", exc)
