"""Request logging and metrics middleware."""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.logging import get_logger
from core.metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id and timing to every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        method = request.method
        path = request.url.path
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "http.unhandled_error", method=method, path=path, request_id=request_id
            )
            raise
        duration = time.perf_counter() - started
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{duration * 1_000:.2f}"

        # Record metrics (exclude docs / metrics / health to reduce cardinality).
        if not path.startswith(("/docs", "/redoc", "/openapi", "/metrics", "/health")):
            HTTP_REQUESTS_TOTAL.labels(
                method=method, path=path, status=str(response.status_code)
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
                duration
            )

        logger.info(
            "http.request",
            method=method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration * 1_000, 2),
            request_id=request_id,
        )
        return response
