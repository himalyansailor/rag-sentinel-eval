"""Structured logging configuration (see docs/LLD.md §12)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_NOISY_LOGGERS = ("httpx", "httpcore", "chromadb", "urllib3", "posthog", "watchdog")


def _stderr_logger_factory(*_: Any) -> structlog.PrintLogger:
    """Resolve ``sys.stderr`` when a logger is assembled, not once at configure time.

    Hosts such as pytest, Typer's CliRunner or Streamlit swap ``sys.stderr``; holding on to
    the original stream would later write to a closed file.
    """
    return structlog.PrintLogger(file=sys.stderr)


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog (and the stdlib root logger) to write to stderr.

    Args:
        level: Minimum log level name.
        json_logs: Emit one JSON object per line (CI / log shippers) instead of pretty output.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric_level, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=_stderr_logger_factory,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a lazy structlog logger carrying the module name.

    The logger must stay lazy: calling ``.bind()`` at import time would freeze structlog's
    *default* configuration (stdout, all levels) before :func:`configure_logging` runs.
    """
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(module=name)
    return logger
