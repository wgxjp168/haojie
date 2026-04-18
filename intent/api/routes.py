"""Intent REST endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from intent.api.dependencies import require_api_key, service_dep, settings_dep
from intent.config import IntentSettings
from intent.schemas import (
    BatchIntentRequest,
    BatchIntentResponse,
    HealthResponse,
    IntentInfo,
    IntentRequest,
    IntentResponse,
)
from intent.service import IntentService
from intent.taxonomy import INTENT_REGISTRY

router = APIRouter(prefix="/api/v1", tags=["intent"])


@router.post(
    "/intents/predict",
    response_model=IntentResponse,
    summary="Classify a single query into an intent.",
    dependencies=[Depends(require_api_key)],
)
def predict(
    payload: IntentRequest,
    service: IntentService = Depends(service_dep),
) -> IntentResponse:
    return service.predict(payload)


@router.post(
    "/intents/predict/batch",
    response_model=BatchIntentResponse,
    summary="Classify a batch of queries.",
    dependencies=[Depends(require_api_key)],
)
def predict_batch(
    payload: BatchIntentRequest,
    service: IntentService = Depends(service_dep),
) -> BatchIntentResponse:
    return service.predict_batch(payload)


@router.get(
    "/intents",
    response_model=List[IntentInfo],
    summary="List the supported intent taxonomy.",
)
def list_intents() -> List[IntentInfo]:
    return [
        IntentInfo(
            id=defn.id,
            label_zh=defn.label_zh,
            label_en=defn.label_en,
            description=defn.description,
            audiences=list(defn.audiences),
        )
        for defn in INTENT_REGISTRY.values()
    ]


@router.get("/intents/health", response_model=HealthResponse, summary="Service health")
def health(
    service: IntentService = Depends(service_dep),
    settings: IntentSettings = Depends(settings_dep),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        backend=service.classifier.effective_backend,
        model_loaded=service.model_loaded,
        cache_ready=service.cache_ready,
        circuit_state=service.circuit_state,
        uptime_seconds=round(service.uptime_seconds, 3),
    )
