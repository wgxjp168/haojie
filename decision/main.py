"""Decision-engine FastAPI app — Part 3.2 of ILBuyAI."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from decision.api.routes import router as decision_router
from decision.config import get_decision_settings
from decision.core.exceptions import DecisionError
from decision.core.logging import configure_logging, get_logger
from decision.core.metrics import DECISION_REGISTRY
from decision.service import get_decision_service, reset_decision_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_decision_settings()
    configure_logging(settings.log_level, as_json=settings.is_production)
    log = get_logger(__name__)
    log.info(
        "decision.service.starting",
        env=settings.env.value,
        strategy=settings.strategy.value,
        port=settings.port,
    )

    service = get_decision_service()
    app.state.service = service
    try:
        yield
    finally:
        log.info("decision.service.shutting_down")
        reset_decision_service()


def create_app() -> FastAPI:
    settings = get_decision_settings()
    app = FastAPI(
        title="ILBuyAI Decision Engine",
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
    app.include_router(decision_router)

    @app.exception_handler(DecisionError)
    async def decision_error_handler(
        request: Request, exc: DecisionError
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.get("/health", tags=["meta"])
    def root_health() -> dict:
        service = get_decision_service()
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "strategy": service.effective_strategy,
            "uptime_seconds": round(service.uptime_seconds, 3),
        }

    if settings.metrics_enabled:
        @app.get("/metrics", tags=["meta"])
        def metrics() -> Response:
            payload = generate_latest(DECISION_REGISTRY)
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
