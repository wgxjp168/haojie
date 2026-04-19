"""Decision REST endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from decision.api.dependencies import require_api_key, service_dep, settings_dep
from decision.catalog import ACTION_REGISTRY
from decision.config import DecisionSettings
from decision.schemas import (
    ActionInfo,
    BatchDecisionRequest,
    BatchDecisionResponse,
    DecisionRequest,
    DecisionResponse,
    HealthResponse,
)
from decision.service import DecisionService

router = APIRouter(prefix="/api/v1", tags=["decision"])


@router.post(
    "/decisions/evaluate",
    response_model=DecisionResponse,
    summary="Evaluate a single procurement decision.",
    dependencies=[Depends(require_api_key)],
)
def evaluate(
    payload: DecisionRequest,
    service: DecisionService = Depends(service_dep),
) -> DecisionResponse:
    return service.decide(payload)


@router.post(
    "/decisions/evaluate/batch",
    response_model=BatchDecisionResponse,
    summary="Evaluate a batch of procurement decisions.",
    dependencies=[Depends(require_api_key)],
)
def evaluate_batch(
    payload: BatchDecisionRequest,
    service: DecisionService = Depends(service_dep),
) -> BatchDecisionResponse:
    return service.decide_batch(payload)


@router.get(
    "/decisions/actions",
    response_model=List[ActionInfo],
    summary="List the supported decision actions.",
)
def list_actions() -> List[ActionInfo]:
    return [
        ActionInfo(
            id=a.id,
            label_zh=a.label_zh,
            label_en=a.label_en,
            description=a.description,
            audiences=list(a.audiences),
            severity=a.severity,
            terminal=a.terminal,
        )
        for a in ACTION_REGISTRY.values()
    ]


@router.get(
    "/decisions/health",
    response_model=HealthResponse,
    summary="Detailed health for the decision service.",
)
def health(
    service: DecisionService = Depends(service_dep),
    settings: DecisionSettings = Depends(settings_dep),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        strategy=service.effective_strategy,
        model_loaded=service.model_loaded,
        cache_ready=service.cache_ready,
        circuit_state=service.circuit_state,
        uptime_seconds=round(service.uptime_seconds, 3),
    )
