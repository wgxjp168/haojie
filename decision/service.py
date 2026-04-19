"""Public DecisionService — composes engine + cache.

This is the single integration point used by the FastAPI routes *and*
any in-process consumer. Symmetric with ``intent.service.IntentService``.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from decision.cache import DecisionCache
from decision.catalog import (
    ACTION_REGISTRY,
    FALLBACK_ACTION,
    label_of,
    next_steps_of,
)
from decision.config import DecisionSettings, get_decision_settings
from decision.core.exceptions import ValidationDecisionError
from decision.core.logging import get_logger
from decision.core.metrics import (
    DECISION_ACTION_TOTAL,
    DECISION_BATCH_SIZE,
    DECISION_REVIEW_TOTAL,
    DECISION_RISK,
)
from decision.engine import DecisionEngine, EngineDecision
from decision.features import FeatureExtractor, FeatureVector
from decision.schemas import (
    BatchDecisionRequest,
    BatchDecisionResponse,
    DecisionCandidate,
    DecisionRequest,
    DecisionResponse,
)

log = get_logger(__name__)


class DecisionService:
    """High-level facade over the decision engine."""

    def __init__(
        self,
        *,
        settings: Optional[DecisionSettings] = None,
        engine: Optional[DecisionEngine] = None,
        cache: Optional[DecisionCache] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.settings = settings or get_decision_settings()
        self.extractor = extractor or FeatureExtractor(
            default_language=self.settings.default_language
        )
        self.engine = engine or DecisionEngine(
            settings=self.settings, extractor=self.extractor
        )
        if cache is not None:
            self.cache = cache
        elif self.settings.cache_enabled:
            self.cache = DecisionCache(
                memory_capacity=self.settings.memory_cache_size,
                redis_url=self.settings.redis_url,
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
        else:
            self.cache = DecisionCache(
                memory_capacity=0, redis_url=None, ttl_seconds=0
            )
        self._started_at = time.time()

    # ----- public api -----

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    @property
    def model_loaded(self) -> bool:
        ml = self.engine.ml
        return bool(ml and ml.ready)

    @property
    def cache_ready(self) -> bool:
        return bool(getattr(self.cache, "ready", False))

    @property
    def circuit_state(self) -> str:
        return self.engine.breaker.state_label

    @property
    def effective_strategy(self) -> str:
        return self.engine.effective_strategy

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        start = time.perf_counter()
        self._validate_request(request)
        features = self.extractor.extract(request)
        cache_key = self._cache_key(features, request) if self.settings.cache_enabled else None

        cached = self.cache.get(cache_key) if cache_key else None
        if cached is not None:
            response = DecisionResponse.model_validate(cached)
            response.cached = True
            response.request_id = request.request_id or response.request_id
            response.session_id = request.session_id or response.session_id
            response.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            return response

        decision = self.engine.evaluate(
            features, strategy_override=request.strategy_override
        )
        response = self._materialize(request, features, decision)
        if cache_key:
            self.cache.set(cache_key, response.model_dump(mode="json"))
        return response

    def decide_batch(self, batch: BatchDecisionRequest) -> BatchDecisionResponse:
        start = time.perf_counter()
        if len(batch.items) > self.settings.max_batch_size:
            raise ValidationDecisionError(
                f"batch size {len(batch.items)} exceeds max_batch_size "
                f"{self.settings.max_batch_size}",
                details={"max": self.settings.max_batch_size},
            )
        DECISION_BATCH_SIZE.observe(len(batch.items))

        results: List[Optional[DecisionResponse]] = [None] * len(batch.items)
        pending_idx: List[int] = []
        pending_features: List[FeatureVector] = []
        pending_keys: List[Optional[str]] = []
        overrides: List[Optional[str]] = []

        for i, item in enumerate(batch.items):
            try:
                self._validate_request(item)
            except ValidationDecisionError as exc:
                results[i] = self._validation_response(item, exc)
                continue

            features = self.extractor.extract(item)
            key = (
                self._cache_key(features, item) if self.settings.cache_enabled else None
            )
            cached = self.cache.get(key) if key else None
            if cached is not None:
                resp = DecisionResponse.model_validate(cached)
                resp.cached = True
                resp.request_id = item.request_id or resp.request_id
                resp.session_id = item.session_id or resp.session_id
                results[i] = resp
                continue
            pending_idx.append(i)
            pending_features.append(features)
            pending_keys.append(key)
            overrides.append(item.strategy_override)

        if pending_features:
            # If overrides are mixed, fall back to per-item evaluation.
            uniq = set(overrides)
            if len(uniq) == 1:
                shared = uniq.pop()
                outcomes = self.engine.evaluate_batch(
                    pending_features, strategy_override=shared
                )
            else:
                outcomes = [
                    self.engine.evaluate(f, strategy_override=ov)
                    for f, ov in zip(pending_features, overrides)
                ]
            for idx, features, decision, key in zip(
                pending_idx, pending_features, outcomes, pending_keys
            ):
                request = batch.items[idx]
                resp = self._materialize(request, features, decision)
                if key:
                    self.cache.set(key, resp.model_dump(mode="json"))
                results[idx] = resp

        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        final = [r for r in results if r is not None]
        return BatchDecisionResponse(
            results=final, total=len(final), latency_ms=latency_ms
        )

    # ----- internals -----

    def _validate_request(self, request: DecisionRequest) -> None:
        if request.product.unit_price is not None and request.product.unit_price < 0:
            raise ValidationDecisionError("unit_price must be non-negative")
        if request.product.quantity is not None and request.product.quantity < 1:
            raise ValidationDecisionError("quantity must be >= 1")
        if request.buyer.budget is not None and request.buyer.budget < 0:
            raise ValidationDecisionError("budget must be non-negative")
        if (
            request.language
            and request.language not in self.settings.supported_languages
        ):
            # Soft-validate: surface hint via metadata but don't reject —
            # the rule strategy's bilingual rationale is best-effort.
            log.warning(
                "decision.service.unsupported_language",
                language=request.language,
            )

    def _cache_key(
        self, features: FeatureVector, request: DecisionRequest
    ) -> Optional[str]:
        suffix = f"|{request.strategy_override or 'auto'}"
        return features.cache_key() + ":" + hex(abs(hash(suffix)))[-8:]

    def _materialize(
        self,
        request: DecisionRequest,
        features: FeatureVector,
        decision: EngineDecision,
    ) -> DecisionResponse:
        outcome = decision.outcome
        # Map (action_id, conf) -> DecisionCandidate.
        candidates: List[DecisionCandidate] = []
        for action_id, conf in outcome.candidates[: 1 + max(0, self.settings.top_k_alternatives)]:
            if action_id not in ACTION_REGISTRY:
                continue
            candidates.append(
                DecisionCandidate(
                    action=action_id,
                    label=label_of(action_id, features.language),
                    confidence=round(float(conf), 4),
                )
            )
        if not candidates:
            candidates = [
                DecisionCandidate(
                    action=FALLBACK_ACTION,
                    label=label_of(FALLBACK_ACTION, features.language),
                    confidence=0.0,
                )
            ]

        top = candidates[0]
        risk = float(outcome.risk_score)
        requires_review = self._needs_review(features, top.action, risk)

        next_steps = outcome.next_steps or next_steps_of(top.action, features.language)

        DECISION_RISK.observe(risk)
        DECISION_ACTION_TOTAL.labels(action=top.action).inc()
        if requires_review:
            DECISION_REVIEW_TOTAL.labels(reason=self._review_reason(features, risk)).inc()

        meta = dict(outcome.metadata or {})
        if decision.fallback_reasons:
            meta["fallback_reasons"] = decision.fallback_reasons

        return DecisionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            action=top.action,
            label=top.label,
            confidence=top.confidence,
            confident=top.confidence >= self.settings.confident_threshold,
            risk_score=round(risk, 4),
            requires_review=requires_review,
            rationale=outcome.rationale or "no rationale produced",
            next_steps=next_steps,
            candidates=candidates,
            strategy=decision.actual_strategy,
            language=features.language,
            latency_ms=decision.latency_ms,
            cached=False,
            metadata=meta,
        )

    def _needs_review(self, features: FeatureVector, action: str, risk: float) -> bool:
        if risk >= self.settings.risk_review_threshold:
            return True
        if (
            features.total_amount > 0
            and features.total_amount >= self.settings.require_human_review_above
        ):
            return True
        if action == "escalate_to_human":
            return True
        if action == "approve_purchase" and features.total_amount > self.settings.max_auto_approve_amount:
            return True
        return False

    def _review_reason(self, features: FeatureVector, risk: float) -> str:
        if (
            features.total_amount > 0
            and features.total_amount >= self.settings.require_human_review_above
        ):
            return "high_value"
        if risk >= self.settings.risk_review_threshold:
            return "risk_threshold"
        return "policy"

    def _validation_response(
        self, request: DecisionRequest, exc: ValidationDecisionError
    ) -> DecisionResponse:
        lang = request.language or self.settings.default_language
        return DecisionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            action=FALLBACK_ACTION,
            label=label_of(FALLBACK_ACTION, lang),
            confidence=0.0,
            confident=False,
            risk_score=0.0,
            requires_review=False,
            rationale=f"validation error: {exc.message}",
            next_steps=next_steps_of(FALLBACK_ACTION, lang),
            candidates=[
                DecisionCandidate(
                    action=FALLBACK_ACTION,
                    label=label_of(FALLBACK_ACTION, lang),
                    confidence=0.0,
                )
            ],
            strategy="validation-error",
            language=lang,
            latency_ms=0.0,
            cached=False,
            metadata={"validation_error": exc.message, "details": exc.details},
        )


# -------- singleton helpers (used by FastAPI lifespan) --------

_service_lock = threading.Lock()
_service_instance: Optional[DecisionService] = None


def get_decision_service() -> DecisionService:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DecisionService()
    return _service_instance


def reset_decision_service() -> None:
    global _service_instance
    with _service_lock:
        _service_instance = None
