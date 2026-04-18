"""Public intent service: composes preprocessing, classification, cache."""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence

from intent.cache import IntentCache
from intent.classifier import IntentClassifier
from intent.config import IntentSettings, get_intent_settings
from intent.core.exceptions import ValidationIntentError
from intent.core.logging import get_logger
from intent.core.metrics import INTENT_BATCH_SIZE
from intent.preprocessor import NormalizedText, TextNormalizer
from intent.schemas import (
    BatchIntentRequest,
    BatchIntentResponse,
    IntentCandidate,
    IntentRequest,
    IntentResponse,
)
from intent.taxonomy import label_of

log = get_logger(__name__)


class IntentService:
    """Entry point used by FastAPI routes *and* other Python callers."""

    def __init__(
        self,
        *,
        settings: Optional[IntentSettings] = None,
        classifier: Optional[IntentClassifier] = None,
        normalizer: Optional[TextNormalizer] = None,
        cache: Optional[IntentCache] = None,
    ) -> None:
        self.settings = settings or get_intent_settings()
        self.normalizer = normalizer or TextNormalizer(
            max_length=self.settings.max_text_length,
            min_length=self.settings.min_text_length,
            default_language=self.settings.default_language,
            supported_languages=self.settings.supported_languages,
        )
        self.classifier = classifier or IntentClassifier(settings=self.settings)
        if cache is not None:
            self.cache = cache
        elif self.settings.cache_enabled:
            self.cache = IntentCache(
                memory_capacity=self.settings.memory_cache_size,
                redis_url=self.settings.redis_url,
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
        else:
            self.cache = IntentCache(memory_capacity=0, redis_url=None, ttl_seconds=0)
        self._started_at = time.time()

    # ---- public api ----

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    @property
    def model_loaded(self) -> bool:
        tf = self.classifier.transformer
        return bool(tf and tf.loaded)

    @property
    def cache_ready(self) -> bool:
        return bool(getattr(self.cache, "ready", False))

    @property
    def circuit_state(self) -> str:
        return self.classifier.breaker.state_label

    def predict(self, request: IntentRequest) -> IntentResponse:
        start = time.perf_counter()
        normalized = self._normalize(request)
        cache_key = self._cache_key(normalized, request)

        cached = self.cache.get(cache_key) if cache_key else None
        if cached is not None:
            response = IntentResponse.model_validate(cached)
            response.cached = True
            response.request_id = request.request_id or response.request_id
            response.session_id = request.session_id or response.session_id
            response.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            return response

        outcome = self.classifier.classify(
            normalized,
            user_type=request.user_type,
            top_k=request.top_k or self.settings.top_k,
        )

        candidates = [
            IntentCandidate(
                intent=intent_id,
                label=label_of(intent_id, normalized.language),
                confidence=round(float(score), 4),
            )
            for intent_id, score in outcome.candidates
        ]
        top = candidates[0]

        response = IntentResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            top_intent=top.intent,
            top_label=top.label,
            top_confidence=top.confidence,
            confident=top.confidence >= self.settings.confident_threshold,
            candidates=candidates,
            backend=outcome.backend,
            language=normalized.language,
            latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
            cached=False,
        )
        if cache_key:
            self.cache.set(cache_key, response.model_dump(mode="json"))
        return response

    def predict_batch(self, batch: BatchIntentRequest) -> BatchIntentResponse:
        start = time.perf_counter()
        INTENT_BATCH_SIZE.observe(len(batch.items))

        # First pass: normalise + cache lookup.
        pending_indexes: List[int] = []
        pending_items: List[NormalizedText] = []
        pending_requests: List[IntentRequest] = []
        pending_cache_keys: List[Optional[str]] = []
        results: List[Optional[IntentResponse]] = [None] * len(batch.items)

        for i, item in enumerate(batch.items):
            try:
                normalized = self._normalize(item)
            except ValidationIntentError as exc:
                # Surface a per-item failure as a low-confidence "other"
                # so one bad item doesn't nuke the whole batch. We log at
                # warning level and attach details in metadata.
                log.warning(
                    "intent.service.batch_item_invalid",
                    index=i,
                    error=exc.message,
                )
                results[i] = IntentResponse(
                    request_id=item.request_id,
                    session_id=item.session_id,
                    top_intent="other",
                    top_label=label_of("other", self.settings.default_language),
                    top_confidence=0.0,
                    confident=False,
                    candidates=[
                        IntentCandidate(
                            intent="other",
                            label=label_of("other", self.settings.default_language),
                            confidence=0.0,
                        )
                    ],
                    backend="validation-error",
                    language=self.settings.default_language,
                    latency_ms=0.0,
                    cached=False,
                )
                continue

            key = self._cache_key(normalized, item)
            cached = self.cache.get(key) if key else None
            if cached is not None:
                resp = IntentResponse.model_validate(cached)
                resp.cached = True
                resp.request_id = item.request_id or resp.request_id
                resp.session_id = item.session_id or resp.session_id
                results[i] = resp
            else:
                pending_indexes.append(i)
                pending_items.append(normalized)
                pending_requests.append(item)
                pending_cache_keys.append(key)

        # Second pass: classify the cache misses as one call.
        if pending_items:
            # All batch items share the first-request's top_k/user_type for
            # batching. We already validated items individually.
            first = pending_requests[0]
            outcomes = self.classifier.classify_batch(
                pending_items,
                user_type=first.user_type,
                top_k=first.top_k or self.settings.top_k,
            )
            for idx, normalized, req, outcome, key in zip(
                pending_indexes, pending_items, pending_requests, outcomes, pending_cache_keys
            ):
                candidates = [
                    IntentCandidate(
                        intent=intent_id,
                        label=label_of(intent_id, normalized.language),
                        confidence=round(float(score), 4),
                    )
                    for intent_id, score in outcome.candidates
                ]
                top = candidates[0]
                response = IntentResponse(
                    request_id=req.request_id,
                    session_id=req.session_id,
                    top_intent=top.intent,
                    top_label=top.label,
                    top_confidence=top.confidence,
                    confident=top.confidence >= self.settings.confident_threshold,
                    candidates=candidates,
                    backend=outcome.backend,
                    language=normalized.language,
                    latency_ms=outcome.latency_ms,
                    cached=False,
                )
                if key:
                    self.cache.set(key, response.model_dump(mode="json"))
                results[idx] = response

        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        # All slots should be filled.
        final: List[IntentResponse] = [r for r in results if r is not None]
        return BatchIntentResponse(results=final, total=len(final), latency_ms=latency_ms)

    # ---- internals ----

    def _normalize(self, request: IntentRequest) -> NormalizedText:
        if request.language is not None:
            self.normalizer.ensure_language_supported(request.language)
        return self.normalizer.normalize(request.text, language=request.language)

    def _cache_key(
        self, normalized: NormalizedText, request: IntentRequest
    ) -> Optional[str]:
        if not self.settings.cache_enabled:
            return None
        # Include user_type + top_k in the key so audience-biased rules
        # and truncated candidate lists don't return stale data.
        suffix = f"|{request.user_type or 'any'}|{request.top_k or self.settings.top_k}"
        return normalized.cache_key + ":" + hex(abs(hash(suffix)))[-8:]


# -------- singleton helpers (used by FastAPI lifespan) --------

_service_lock = threading.Lock()
_service_instance: Optional[IntentService] = None


def get_intent_service() -> IntentService:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = IntentService()
    return _service_instance


def reset_intent_service() -> None:
    global _service_instance
    with _service_lock:
        _service_instance = None
