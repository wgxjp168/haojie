"""Gateway FastAPI app entry-point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from gateway.api import admin_routes, auth_routes, proxy_routes, ws_routes
from gateway.auth.jwt_manager import reset_jwt_manager
from gateway.auth.token_blacklist import reset_token_blacklist
from gateway.circuit_breaker.breaker import reset_circuit_breaker_registry
from gateway.config import GatewaySettings, get_gateway_settings
from gateway.core.exceptions import GatewayError
from gateway.core.logging import configure_gateway_logging, get_logger
from gateway.core.metrics import GATEWAY_REGISTRY
from gateway.monitoring.middleware import GatewayMetricsMiddleware
from gateway.rate_limiting.strategies import build_rate_limiter, set_rate_limiter
from gateway.routing.forwarder import get_forwarder, reset_forwarder
from gateway.websocket.manager import get_ws_manager, reset_ws_manager

settings: GatewaySettings = get_gateway_settings()
configure_gateway_logging(settings)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "gateway.starting",
        env=settings.env.value,
        version=settings.version,
        port=settings.port,
    )
    # Build Redis-backed rate limiter if Redis is configured.
    set_rate_limiter(await build_rate_limiter(settings))
    # Open the upstream HTTP pool.
    await get_forwarder().start()
    yield
    await get_forwarder().close()


app = FastAPI(
    title="ILBuyAI Gateway",
    description="Part 2/10 — access layer for the ILBuyAI procurement system.",
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id", "x-process-time-ms"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(GatewayMetricsMiddleware)

# Routes
app.include_router(auth_routes.router)
app.include_router(ws_routes.router)
app.include_router(admin_routes.router)
app.include_router(proxy_routes.router)


# --- error handlers ---


@app.exception_handler(GatewayError)
async def _gateway_error_handler(request: Request, exc: GatewayError):
    payload = {
        "error": exc.to_dict(),
        "request_id": getattr(request.state, "request_id", None),
    }
    headers = {}
    if exc.code.value == "RATE_LIMITED":
        retry = exc.details.get("retry_after")
        if retry is not None:
            headers["Retry-After"] = str(int(retry))
    return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)


# --- system routes ---


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.service_name,
        "version": settings.version,
        "env": settings.env.value,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "auth": "/auth",
            "proxy": "/api/v1",
            "ws": "/ws",
            "admin": "/admin",
        },
    }


@app.get("/health", tags=["health"])
async def health():
    manager = get_ws_manager()
    stats = manager.stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "service": settings.service_name,
        "version": settings.version,
        "ws": stats,
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    if not settings.metrics_enabled:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    return Response(
        generate_latest(GATEWAY_REGISTRY), media_type=CONTENT_TYPE_LATEST
    )


# Expose test helpers on the app state for integration tests.
app.state._test_reset = [
    reset_jwt_manager,
    reset_token_blacklist,
    reset_circuit_breaker_registry,
    reset_forwarder,
    reset_ws_manager,
]
