"""全链路日志配置 —— 控制台 + 文件双输出，便于线上问题溯源。"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from app import config

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(trace_id)s%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

config.LOG_DIR.mkdir(parents=True, exist_ok=True)


class TraceIdFilter(logging.Filter):
    """把当前上下文的 trace_id 注入每条日志（无 trace 时留空）。

    只在这里读取一次 contextvars，避免各模块自行拼接 trace 前缀。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from app.core.tracing import get_trace_id

            trace_id = get_trace_id()
        except Exception:  # noqa: BLE001
            trace_id = None
        record.trace_id = f"[trace={trace_id}] " if trace_id else ""
        return True


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("enterprise-bot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    trace_filter = TraceIdFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    console.addFilter(trace_filter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        config.LOG_DIR / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(trace_filter)
    logger.addHandler(file_handler)

    return logger


logger = _build_logger()


def preview(text: str, n: int = 40) -> str:
    """把可能含 PII 的文本截断为日志安全预览（如用户 query）。

    日志是可观测性数据，但明文记录完整 query 仍属隐私泄露风险。统一截断到
    前 n 个字符并标注省略号；短于 n 则原样返回。所有记录用户原始输入的日志点
    都应经此函数处理（评估 P0-3 子项：日志 query 脱敏）。
    """
    if not text:
        return ""
    text = str(text)
    if len(text) <= n:
        return text
    return text[:n] + "…"
