"""FastAPI application entry-point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from api.middleware import RequestContextMiddleware
from api.v1 import api_router
from cache.manager import get_cache_manager
from config.settings import get_settings
from core.exceptions import InputProcessorError
from core.logging import configure_logging, get_logger
from managers.input_manager import get_input_manager
from managers.session_manager import get_session_manager

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("app.starting", env=settings.env.value, version=settings.version)
    warnings = settings.production_safety_check()
    if warnings:
        for w in warnings:
            logger.warning("app.production_warning", warning=w)

    cache = get_cache_manager()
    await cache.initialize()

    sessions = get_session_manager()
    sessions.start_cleanup_task()

    # Eagerly construct the input manager so any provider config blows up now.
    manager = get_input_manager()
    app.state.input_manager = manager

    logger.info("app.started")
    yield

    # Shutdown
    logger.info("app.stopping")
    await sessions.stop_cleanup_task()
    await manager.close()
    await cache.close()
    logger.info("app.stopped")


app = FastAPI(
    title="ILBuyAI Multi-Modal Input Processor",
    description=(
        "Part 1/10 of the ILBuyAI intelligent procurement decision system. "
        "Accepts text, image, link, and voice input, normalises it and emits "
        "a structured payload consumed by the downstream AI decision hub."
    ),
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# --- middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id", "x-process-time-ms"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(RequestContextMiddleware)

# --- routers ---
app.include_router(api_router, prefix="/api/v1")


# --- error handlers ---


@app.exception_handler(InputProcessorError)
async def _domain_error_handler(request: Request, exc: InputProcessorError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.to_dict(), "request_id": getattr(request.state, "request_id", None)},
    )


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
            "api_v1": "/api/v1",
        },
    }


@app.get("/health", tags=["health"])
async def health():
    manager = get_input_manager()
    report = await manager.health_check()
    return JSONResponse(
        status_code=status.HTTP_200_OK if report["overall"] else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "healthy" if report["overall"] else "degraded",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "components": report["components"],
        },
    )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    if not settings.metrics_enabled:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
