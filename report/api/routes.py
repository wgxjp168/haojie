"""Report REST endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from report.api.dependencies import require_api_key, service_dep, settings_dep
from report.catalog import TEMPLATE_REGISTRY
from report.config import ReportFormat, ReportSettings
from report.schemas import (
    BatchReportRequest,
    BatchReportResponse,
    HealthResponse,
    ReportRequest,
    ReportResponse,
    StoredReport,
    TemplateInfo,
)
from report.service import ReportService

router = APIRouter(prefix="/api/v1", tags=["report"])


@router.post(
    "/reports/generate",
    response_model=ReportResponse,
    summary="Render a decision report in the requested format.",
    dependencies=[Depends(require_api_key)],
)
def generate(
    payload: ReportRequest,
    service: ReportService = Depends(service_dep),
) -> ReportResponse:
    return service.generate(payload)


@router.post(
    "/reports/generate/batch",
    response_model=BatchReportResponse,
    summary="Render a batch of reports.",
    dependencies=[Depends(require_api_key)],
)
def generate_batch(
    payload: BatchReportRequest,
    service: ReportService = Depends(service_dep),
) -> BatchReportResponse:
    return service.generate_batch(payload)


@router.get(
    "/reports/templates/list",
    response_model=List[TemplateInfo],
    summary="List the available report templates.",
)
def list_templates() -> List[TemplateInfo]:
    supported_formats = [f.value for f in ReportFormat]
    return [
        TemplateInfo(
            template_id=meta.template_id,
            audience=meta.audience.value,
            formats=supported_formats,
            description=meta.description,
        )
        for meta in TEMPLATE_REGISTRY.values()
    ]


@router.get(
    "/reports/health",
    response_model=HealthResponse,
    summary="Detailed health for the report service.",
)
def health(
    service: ReportService = Depends(service_dep),
    settings: ReportSettings = Depends(settings_dep),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        default_format=settings.default_format.value,
        default_audience=settings.default_audience.value,
        storage_backends=service.available_backends,
        cache_ready=service.cache_ready,
        storage_circuit_states=service.circuit_states,
        uptime_seconds=round(service.uptime_seconds, 3),
    )


# NOTE: path-parameter routes are declared last so that ``/reports/health`` and
# ``/reports/templates/list`` match before ``/reports/{report_id}``.
@router.get(
    "/reports/{report_id}",
    response_model=StoredReport,
    summary="Retrieve a previously-stored report by id.",
    dependencies=[Depends(require_api_key)],
)
def get_report(
    report_id: str,
    backend: Optional[str] = Query(
        None, description="Pin to a specific storage backend."
    ),
    service: ReportService = Depends(service_dep),
) -> StoredReport:
    return service.get_report(report_id, backend=backend)


@router.delete(
    "/reports/{report_id}",
    summary="Remove a stored report from every configured backend.",
    dependencies=[Depends(require_api_key)],
)
def delete_report(
    report_id: str,
    service: ReportService = Depends(service_dep),
) -> dict:
    outcomes = service.delete_report(report_id)
    return {"report_id": report_id, "removed": outcomes}
