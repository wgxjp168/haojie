"""Intent service FastAPI app — Part 3.1 of ILBuyAI."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from intent.api.routes import router as intent_router
from intent.config import get_intent_settings
from intent.core.exceptions import IntentError
from intent.core.logging import configure_logging, get_logger
from intent.core.metrics import INTENT_REGISTRY
from intent.service import get_intent_service, reset_intent_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_intent_settings()
    configure_logging(settings.log_level, as_json=settings.is_production)
    log = get_logger(__name__)
    log.info(
        "intent.service.starting",
        env=settings.env.value,
        backend=settings.backend.value,
        port=settings.port,
    )

    # Warm singletons (this will preload the transformer if configured).
    service = get_intent_service()
    app.state.service = service
    try:
        yield
    finally:
        log.info("intent.service.shutting_down")
        reset_intent_service()


def create_app() -> FastAPI:
    settings = get_intent_settings()
    app = FastAPI(
        title="ILBuyAI Intent Service",
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
    app.include_router(intent_router)

    @app.exception_handler(IntentError)
    async def intent_error_handler(request: Request, exc: IntentError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.get("/health", tags=["meta"])
    def root_health() -> dict:
        service = get_intent_service()
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "backend": service.classifier.effective_backend,
            "uptime_seconds": round(service.uptime_seconds, 3),
        }

    if settings.metrics_enabled:
        @app.get("/metrics", tags=["meta"])
        def metrics() -> Response:
            payload = generate_latest(INTENT_REGISTRY)
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
