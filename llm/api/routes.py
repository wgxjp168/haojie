"""LLM REST endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from llm.api.dependencies import require_api_key, service_dep, settings_dep
from llm.catalog import MODEL_CATALOG
from llm.config import LLMSettings
from llm.schemas import (
    BatchCompletionRequest,
    BatchCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    ModelInfoPublic,
)
from llm.service import LLMService

router = APIRouter(prefix="/api/v1", tags=["llm"])


@router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Chat completion — routes across configured providers.",
    dependencies=[Depends(require_api_key)],
)
async def complete(
    payload: CompletionRequest,
    service: LLMService = Depends(service_dep),
) -> CompletionResponse:
    return await service.complete(payload)


@router.post(
    "/completions/batch",
    response_model=BatchCompletionResponse,
    summary="Batch chat completion.",
    dependencies=[Depends(require_api_key)],
)
async def complete_batch(
    payload: BatchCompletionRequest,
    service: LLMService = Depends(service_dep),
) -> BatchCompletionResponse:
    return await service.complete_batch(payload)


@router.get(
    "/models",
    response_model=List[ModelInfoPublic],
    summary="List the configured model catalog.",
)
def list_models(
    service: LLMService = Depends(service_dep),
) -> List[ModelInfoPublic]:
    available = set(service.providers_available)
    return [
        ModelInfoPublic(
            model_id=m.model_id,
            provider=m.provider,
            context_window=m.context_window,
            max_output_tokens=m.max_output_tokens,
            price_per_1k_prompt_usd=m.price_per_1k_prompt_usd,
            price_per_1k_completion_usd=m.price_per_1k_completion_usd,
            available=m.provider in available,
            description=m.description,
        )
        for m in MODEL_CATALOG.values()
    ]


@router.get(
    "/completions/health",
    response_model=HealthResponse,
    summary="Detailed health for the LLM service.",
)
def health(
    service: LLMService = Depends(service_dep),
    settings: LLMSettings = Depends(settings_dep),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        default_strategy=settings.default_strategy.value,
        default_model_id=settings.default_model_id,
        providers_available=service.providers_available,
        cache_ready=service.cache_ready,
        circuit_states=service.circuit_states,
        uptime_seconds=round(service.uptime_seconds, 3),
    )
