"""LLM service FastAPI app — Part 3.3 of ILBuyAI."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llm.api.routes import router as llm_router
from llm.config import get_llm_settings
from llm.core.exceptions import LLMError
from llm.core.logging import configure_logging, get_logger
from llm.core.metrics import LLM_REGISTRY
from llm.service import get_llm_service, reset_llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_llm_settings()
    configure_logging(settings.log_level, as_json=settings.is_production)
    log = get_logger(__name__)
    log.info(
        "llm.service.starting",
        env=settings.env.value,
        strategy=settings.default_strategy.value,
        default_model=settings.default_model_id,
        port=settings.port,
    )

    service = get_llm_service()
    app.state.service = service
    log.info(
        "llm.service.providers_ready",
        providers=service.providers_available,
    )
    try:
        yield
    finally:
        log.info("llm.service.shutting_down")
        reset_llm_service()


def create_app() -> FastAPI:
    settings = get_llm_settings()
    app = FastAPI(
        title="ILBuyAI LLM Service",
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
    app.include_router(llm_router)

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.get("/health", tags=["meta"])
    def root_health() -> dict:
        service = get_llm_service()
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "default_strategy": settings.default_strategy.value,
            "providers_available": service.providers_available,
            "uptime_seconds": round(service.uptime_seconds, 3),
        }

    if settings.metrics_enabled:
        @app.get("/metrics", tags=["meta"])
        def metrics() -> Response:
            payload = generate_latest(LLM_REGISTRY)
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
