"""Request-level middleware: request id, timing, metrics."""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from gateway.core.logging import get_logger
from gateway.core.metrics import GW_REQUEST_DURATION, GW_REQUESTS_TOTAL

logger = get_logger(__name__)

_EXCLUDED_PATHS = ("/health", "/metrics", "/docs", "/redoc", "/openapi.json")


def _path_template(path: str) -> str:
    """Collapse IDs so we don't explode Prometheus label cardinality."""
    parts = path.split("/")
    out = []
    for p in parts:
        if not p:
            out.append(p)
            continue
        # UUID
        if len(p) == 36 and p.count("-") == 4:
            out.append(":id")
        # numeric
        elif p.isdigit():
            out.append(":id")
        else:
            out.append(p)
    return "/".join(out)


class GatewayMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()

        method = request.method
        path_tmpl = _path_template(request.url.path)
        excluded = any(request.url.path.startswith(p) for p in _EXCLUDED_PATHS)

        try:
            response: Response = await call_next(request)
        except Exception:
            logger.exception(
                "gateway.unhandled_error",
                method=method,
                path=path_tmpl,
                request_id=request_id,
            )
            raise

        duration = time.perf_counter() - started
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{duration * 1_000:.2f}"

        if not excluded:
            user_type = "unknown"
            principal = getattr(request.state, "principal", None)
            if principal is not None:
                user_type = principal.user_type.value
            GW_REQUESTS_TOTAL.labels(
                method=method,
                path=path_tmpl,
                status=str(response.status_code),
                user_type=user_type,
            ).inc()
            GW_REQUEST_DURATION.labels(
                method=method, path=path_tmpl
            ).observe(duration)

        logger.info(
            "gateway.request",
            method=method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration * 1_000, 2),
            request_id=request_id,
        )
        return response
