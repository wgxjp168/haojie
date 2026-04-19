"""Hub REST endpoints — thin wrapper around ``HubService``."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from hub.api.dependencies import require_api_key, service_dep
from hub.schemas import (
    BatchPipelineRequest,
    BatchPipelineResponse,
    HealthResponse,
    PipelineRequest,
    PipelineResponse,
)
from hub.service import HubService

router = APIRouter(prefix="/api/v1", tags=["hub"])


@router.post(
    "/hub/pipeline",
    response_model=PipelineResponse,
    summary="Run the full AI decision-hub pipeline for one request.",
    dependencies=[Depends(require_api_key)],
)
async def run_pipeline(
    payload: PipelineRequest,
    service: HubService = Depends(service_dep),
) -> PipelineResponse:
    return await service.run_pipeline(payload)


@router.post(
    "/hub/pipeline/batch",
    response_model=BatchPipelineResponse,
    summary="Run a batch of pipeline requests concurrently.",
    dependencies=[Depends(require_api_key)],
)
async def run_batch(
    payload: BatchPipelineRequest,
    service: HubService = Depends(service_dep),
) -> BatchPipelineResponse:
    return await service.run_batch(payload)


@router.get(
    "/hub/health",
    response_model=HealthResponse,
    summary="Aggregate downstream-service health.",
)
def health(service: HubService = Depends(service_dep)) -> HealthResponse:
    return service.health()
