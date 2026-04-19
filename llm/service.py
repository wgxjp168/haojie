"""Public LLM service — composes router + cache + provider registry."""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from typing import Dict, List, Optional

from llm.cache import LLMCache
from llm.catalog import MODEL_CATALOG
from llm.circuit_breaker import CircuitBreakerRegistry
from llm.config import LLMSettings, get_llm_settings
from llm.core.exceptions import NoProviderAvailableError, ValidationLLMError
from llm.core.logging import get_logger
from llm.core.metrics import LLM_BATCH_SIZE
from llm.providers import BaseProvider, StubProvider
from llm.providers.anthropic import AnthropicProvider
from llm.providers.google import GoogleProvider
from llm.providers.openai import AzureOpenAIProvider, OpenAIProvider
from llm.rate_limiter import RateLimiterRegistry
from llm.router import LLMRouter
from llm.schemas import (
    BatchCompletionRequest,
    BatchCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)

log = get_logger(__name__)


def _build_default_providers(settings: LLMSettings) -> Dict[str, BaseProvider]:
    creds = settings.provider_credentials()
    providers: Dict[str, BaseProvider] = {
        "stub": StubProvider(),
        "openai": OpenAIProvider(credentials=creds["openai"]),
        "anthropic": AnthropicProvider(credentials=creds["anthropic"]),
        "google": GoogleProvider(credentials=creds["google"]),
        "azure": AzureOpenAIProvider(credentials=creds["azure"]),
    }
    return providers


class LLMService:
    """High-level facade over the LLM router."""

    def __init__(
        self,
        *,
        settings: Optional[LLMSettings] = None,
        providers: Optional[Dict[str, BaseProvider]] = None,
        router: Optional[LLMRouter] = None,
        cache: Optional[LLMCache] = None,
    ) -> None:
        self.settings = settings or get_llm_settings()
        self.providers = providers or _build_default_providers(self.settings)

        if router is not None:
            self.router = router
        else:
            self.router = LLMRouter(
                settings=self.settings,
                providers=self.providers,
                breakers=CircuitBreakerRegistry(
                    failure_threshold=self.settings.cb_failure_threshold,
                    recovery_seconds=self.settings.cb_recovery_seconds,
                ),
                rate_limiters=RateLimiterRegistry(
                    rate_per_minute=self.settings.rate_limit_per_minute,
                    burst=self.settings.rate_limit_burst,
                ),
            )

        if cache is not None:
            self.cache = cache
        elif self.settings.cache_enabled:
            self.cache = LLMCache(
                memory_capacity=self.settings.memory_cache_size,
                redis_url=self.settings.redis_url,
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
        else:
            self.cache = LLMCache(memory_capacity=0, redis_url=None, ttl_seconds=0)

        self._started_at = time.time()

    # ---- public api ----

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    @property
    def cache_ready(self) -> bool:
        return bool(getattr(self.cache, "ready", False))

    @property
    def providers_available(self) -> List[str]:
        return [name for name, p in self.providers.items() if p.available]

    @property
    def circuit_states(self) -> Dict[str, str]:
        return self.router.breakers.all_states()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.perf_counter()
        self._validate_request(request)

        cache_key = self._cache_key(request) if self.settings.cache_enabled else None
        cached = self.cache.get(cache_key) if cache_key else None
        if cached is not None:
            response = CompletionResponse.model_validate(cached)
            response.cached = True
            response.request_id = request.request_id or response.request_id
            response.session_id = request.session_id or response.session_id
            response.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            return response

        response = await self.router.run(request)

        if cache_key:
            self.cache.set(cache_key, response.model_dump(mode="json"))
        return response

    async def complete_batch(
        self, batch: BatchCompletionRequest
    ) -> BatchCompletionResponse:
        start = time.perf_counter()
        if len(batch.items) > self.settings.max_batch_size:
            raise ValidationLLMError(
                f"batch size {len(batch.items)} exceeds max_batch_size "
                f"{self.settings.max_batch_size}",
                details={"max": self.settings.max_batch_size},
            )
        LLM_BATCH_SIZE.observe(len(batch.items))

        # Run all items concurrently; aggregate successes + per-item errors.
        coros = [self._complete_safe(item) for item in batch.items]
        results = await asyncio.gather(*coros)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return BatchCompletionResponse(
            results=results, total=len(results), latency_ms=latency_ms
        )

    async def _complete_safe(self, request: CompletionRequest) -> CompletionResponse:
        try:
            return await self.complete(request)
        except NoProviderAvailableError as exc:
            return self._error_response(request, "NO_PROVIDER_AVAILABLE", exc.message)
        except ValidationLLMError as exc:
            return self._error_response(request, "VALIDATION_ERROR", exc.message)
        except Exception as exc:  # noqa: BLE001
            log.error("llm.service.batch_item_failed", error=str(exc))
            return self._error_response(request, "INTERNAL_ERROR", str(exc))

    # ---- internals ----

    def _validate_request(self, request: CompletionRequest) -> None:
        if request.model_id and request.model_id not in MODEL_CATALOG:
            raise ValidationLLMError(
                f"unknown model id {request.model_id!r}",
                details={"model_id": request.model_id},
            )
        if request.target_model_ids:
            unknown = [
                mid for mid in request.target_model_ids if mid not in MODEL_CATALOG
            ]
            if unknown:
                raise ValidationLLMError(
                    f"unknown model ids in target_model_ids: {unknown}",
                    details={"unknown": unknown},
                )

    def _cache_key(self, request: CompletionRequest) -> Optional[str]:
        payload = {
            "messages": [m.model_dump() for m in request.normalized_messages()],
            "model_id": request.model_id or self.settings.default_model_id,
            "temperature": request.temperature if request.temperature is not None else self.settings.default_temperature,
            "top_p": request.top_p if request.top_p is not None else self.settings.default_top_p,
            "max_tokens": request.max_tokens if request.max_tokens is not None else self.settings.max_completion_tokens,
            "strategy": request.strategy or self.settings.default_strategy.value,
            "target_model_ids": request.target_model_ids,
            "stop": request.stop,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _error_response(
        self, request: CompletionRequest, code: str, message: str
    ) -> CompletionResponse:
        return CompletionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            content="",
            provider="none",
            model_id=request.model_id or self.settings.default_model_id,
            strategy=request.strategy or self.settings.default_strategy.value,
            finish_reason="error",
            confidence=0.0,
            confident=False,
            cost_usd=0.0,
            latency_ms=0.0,
            attempts=[],
            cached=False,
            metadata={"error_code": code, "error_message": message},
        )


# -------- singleton helpers (used by FastAPI lifespan) --------

_service_lock = threading.Lock()
_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = LLMService()
    return _service_instance


def reset_llm_service() -> None:
    global _service_instance
    with _service_lock:
        _service_instance = None
