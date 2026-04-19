"""Report service FastAPI app — Part 3.4 of ILBuyAI."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from report.api.routes import router as report_router
from report.config import get_report_settings
from report.core.exceptions import ReportError
from report.core.logging import configure_logging, get_logger
from report.core.metrics import REPORT_REGISTRY
from report.service import get_report_service, reset_report_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_report_settings()
    configure_logging(settings.log_level, as_json=settings.is_production)
    log = get_logger(__name__)
    log.info(
        "report.service.starting",
        env=settings.env.value,
        default_format=settings.default_format.value,
        default_audience=settings.default_audience.value,
        port=settings.port,
    )

    service = get_report_service()
    app.state.service = service
    log.info(
        "report.service.backends_ready",
        backends=service.available_backends,
    )
    try:
        yield
    finally:
        log.info("report.service.shutting_down")
        reset_report_service()


def create_app() -> FastAPI:
    settings = get_report_settings()
    app = FastAPI(
        title="ILBuyAI Report Service",
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
    app.include_router(report_router)

    @app.exception_handler(ReportError)
    async def report_error_handler(
        request: Request, exc: ReportError
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.get("/health", tags=["meta"])
    def root_health() -> dict:
        service = get_report_service()
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "storage_backends": service.available_backends,
            "uptime_seconds": round(service.uptime_seconds, 3),
        }

    if settings.metrics_enabled:
        @app.get("/metrics", tags=["meta"])
        def metrics() -> Response:
            payload = generate_latest(REPORT_REGISTRY)
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
