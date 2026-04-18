"""Structured logging for the gateway."""
from __future__ import annotations

import logging
import sys

import structlog

from gateway.config import GatewaySettings


def configure_gateway_logging(settings: GatewaySettings) -> None:
    level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    if settings.is_production:
        for name in ("urllib3", "botocore", "asyncio", "httpx"):
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
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
