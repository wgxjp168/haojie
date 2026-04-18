"""Structured logging setup using structlog.

Call :func:`configure_logging` once at process start. All modules then
use ``structlog.get_logger(__name__)`` to obtain a logger.
"""
from __future__ import annotations

import logging
import sys

import structlog

from config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure stdlib logging and structlog for the application."""

    log_level = getattr(logging, settings.log_level, logging.INFO)

    # stdlib root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silence chatty third-parties in production
    if settings.is_production:
        for name in ("urllib3", "botocore", "asyncio"):
            logging.getLogger(name).setLevel(logging.WARNING)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
