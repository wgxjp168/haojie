"""Hub FastAPI app — Part 3 orchestrator entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from hub.api.routes import router as hub_router
from hub.config import get_hub_settings
from hub.core.exceptions import HubError
from hub.core.logging import configure_logging, get_logger
from hub.core.metrics import HUB_REGISTRY
from hub.service import get_hub_service, reset_hub_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_hub_settings()
    configure_logging(settings.log_level, as_json=settings.is_production)
    log = get_logger(__name__)
    log.info(
        "hub.service.starting",
        env=settings.env.value,
        port=settings.port,
        intent_mode=settings.intent_mode.value,
        decision_mode=settings.decision_mode.value,
        llm_mode=settings.llm_mode.value,
        report_mode=settings.report_mode.value,
    )

    service = get_hub_service()
    app.state.service = service
    try:
        yield
    finally:
        log.info("hub.service.shutting_down")
        reset_hub_service()


def create_app() -> FastAPI:
    settings = get_hub_settings()
    app = FastAPI(
        title="ILBuyAI Decision Hub",
        version=settings.version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.include_router(hub_router)

    @app.exception_handler(HubError)
    async def hub_error_handler(request: Request, exc: HubError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.get("/health", tags=["meta"])
    def root_health() -> dict:
        service = get_hub_service()
        health = service.health()
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "stages": [s.model_dump() for s in health.stages],
            "uptime_seconds": health.uptime_seconds,
        }

    if settings.metrics_enabled:
        @app.get("/metrics", tags=["meta"])
        def metrics() -> Response:
            payload = generate_latest(HUB_REGISTRY)
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
