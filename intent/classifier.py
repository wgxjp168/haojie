"""High-level intent classifier that composes rules + transformer.

``IntentClassifier`` is the layer that picks a backend, applies the
circuit breaker, and produces the final ranked candidate list. It does NOT
know about HTTP, auth or caching — those belong to ``IntentService``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from intent.circuit_breaker import CircuitBreaker
from intent.config import IntentBackend, IntentSettings
from intent.core.exceptions import CircuitOpenError, ModelFailedError, ModelNotReadyError
from intent.core.logging import get_logger
from intent.core.metrics import (
    INTENT_CONFIDENCE,
    INTENT_LABEL_TOTAL,
    INTENT_PREDICTION_DURATION,
    INTENT_REQUESTS_TOTAL,
)
from intent.model import TransformerIntentModel, TransformerPrediction
from intent.preprocessor import NormalizedText
from intent.rules import RuleBasedClassifier, RuleMatch
from intent.taxonomy import FALLBACK_INTENT, INTENT_IDS

log = get_logger(__name__)


@dataclass
class ClassificationOutcome:
    """Result for one input — backend-agnostic."""

    candidates: List[Tuple[str, float]]  # [(intent_id, prob), ...] descending, top-k
    backend: str
    latency_ms: float

    @property
    def top_intent(self) -> str:
        return self.candidates[0][0]

    @property
    def top_confidence(self) -> float:
        return float(self.candidates[0][1])


class IntentClassifier:
    """Compose rule-based + transformer backends according to settings."""

    def __init__(
        self,
        *,
        settings: IntentSettings,
        rules: Optional[RuleBasedClassifier] = None,
        transformer: Optional[TransformerIntentModel] = None,
        breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.settings = settings
        self.rules = rules or RuleBasedClassifier()
        self.transformer = transformer
        self.breaker = breaker or CircuitBreaker(
            failure_threshold=settings.cb_failure_threshold,
            recovery_seconds=settings.cb_recovery_seconds,
        )

        if settings.backend in (IntentBackend.TRANSFORMER, IntentBackend.ENSEMBLE):
            if self.transformer is None:
                self.transformer = TransformerIntentModel(
                    model_name_or_path=settings.model_name_or_path,
                    cache_dir=settings.model_cache_dir,
                    max_seq_length=settings.model_max_seq_length,
                    device=settings.model_device,
                    batch_size=settings.model_batch_size,
                )
            if settings.preload_model:
                try:
                    self.transformer.ensure_loaded()
                except ModelNotReadyError:
                    if not settings.allow_rules_fallback:
                        raise
                    log.warning("intent.classifier.preload_failed_fallback_rules")

    # ---- public api ----

    @property
    def effective_backend(self) -> str:
        """What backend will actually run given current state."""
        if self.settings.backend == IntentBackend.RULES:
            return "rules"
        if self.transformer is None:
            return "rules"
        if self.breaker.state_label == "open":
            return "rules"
        if self.transformer.loaded:
            return self.settings.backend.value
        # Not loaded yet but can try.
        return self.settings.backend.value

    def classify(
        self,
        normalized: NormalizedText,
        *,
        user_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> ClassificationOutcome:
        k = top_k or self.settings.top_k
        start = time.perf_counter()

        if self.settings.backend == IntentBackend.RULES or self.transformer is None:
            outcome = self._classify_rules(normalized, user_type=user_type, top_k=k)
        else:
            try:
                self.breaker.ensure_allowed()
                tf_preds = self._run_transformer([normalized.clean], top_k=k)
                outcome = self._combine(
                    normalized,
                    tf_pred=tf_preds[0],
                    user_type=user_type,
                    top_k=k,
                )
                self.breaker.record_success()
            except CircuitOpenError:
                log.warning("intent.classifier.circuit_open_fallback")
                outcome = self._classify_rules(normalized, user_type=user_type, top_k=k)
            except (ModelNotReadyError, ModelFailedError) as exc:
                self.breaker.record_failure()
                if not self.settings.allow_rules_fallback:
                    raise
                log.warning("intent.classifier.transformer_failed_fallback", error=str(exc))
                outcome = self._classify_rules(normalized, user_type=user_type, top_k=k)

        latency_ms = (time.perf_counter() - start) * 1000.0
        outcome = ClassificationOutcome(
            candidates=outcome.candidates,
            backend=outcome.backend,
            latency_ms=round(latency_ms, 3),
        )
        self._emit_metrics(outcome)
        return outcome

    def classify_batch(
        self,
        normalized_items: Sequence[NormalizedText],
        *,
        user_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[ClassificationOutcome]:
        k = top_k or self.settings.top_k
        if not normalized_items:
            return []

        # Only the transformer backend benefits from batching; the rule
        # classifier is fast enough to call per-item.
        if (
            self.settings.backend == IntentBackend.RULES
            or self.transformer is None
            or self.breaker.state_label == "open"
        ):
            return [
                self.classify(item, user_type=user_type, top_k=k) for item in normalized_items
            ]

        start = time.perf_counter()
        try:
            self.breaker.ensure_allowed()
            tf_preds = self._run_transformer(
                [n.clean for n in normalized_items], top_k=k
            )
            self.breaker.record_success()
            out: List[ClassificationOutcome] = []
            for item, pred in zip(normalized_items, tf_preds):
                combined = self._combine(item, tf_pred=pred, user_type=user_type, top_k=k)
                out.append(combined)
        except CircuitOpenError:
            log.warning("intent.classifier.batch_circuit_open_fallback")
            return [
                self.classify(item, user_type=user_type, top_k=k) for item in normalized_items
            ]
        except (ModelNotReadyError, ModelFailedError) as exc:
            self.breaker.record_failure()
            if not self.settings.allow_rules_fallback:
                raise
            log.warning("intent.classifier.batch_transformer_failed_fallback", error=str(exc))
            return [
                self.classify(item, user_type=user_type, top_k=k) for item in normalized_items
            ]

        latency_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(normalized_items))
        for o in out:
            o.latency_ms = round(latency_ms, 3)
            self._emit_metrics(o)
        return out

    # ---- internals ----

    def _run_transformer(
        self, texts: Sequence[str], *, top_k: int
    ) -> List[TransformerPrediction]:
        assert self.transformer is not None
        return self.transformer.predict(texts, top_k=top_k)

    def _classify_rules(
        self,
        normalized: NormalizedText,
        *,
        user_type: Optional[str],
        top_k: int,
    ) -> ClassificationOutcome:
        matches: List[RuleMatch] = self.rules.classify(
            normalized.clean,
            language=normalized.language,
            user_type=user_type,
            top_k=top_k,
        )
        candidates = [(m.intent, float(m.score)) for m in matches]
        if not candidates:
            candidates = [(FALLBACK_INTENT, 0.0)]
        return ClassificationOutcome(candidates=candidates, backend="rules", latency_ms=0.0)

    def _combine(
        self,
        normalized: NormalizedText,
        *,
        tf_pred: TransformerPrediction,
        user_type: Optional[str],
        top_k: int,
    ) -> ClassificationOutcome:
        tf_scores = tf_pred.scores or [(FALLBACK_INTENT, 0.0)]
        tf_top_intent, tf_top_conf = tf_scores[0]

        if self.settings.backend == IntentBackend.TRANSFORMER:
            return ClassificationOutcome(
                candidates=tf_scores[:top_k],
                backend="transformer",
                latency_ms=0.0,
            )

        # Ensemble: only consult the rules when the transformer isn't confident.
        if tf_top_conf >= self.settings.low_confidence_threshold:
            return ClassificationOutcome(
                candidates=tf_scores[:top_k],
                backend="ensemble:transformer",
                latency_ms=0.0,
            )

        rule_match = self.rules.classify_best(
            normalized.clean,
            language=normalized.language,
            user_type=user_type,
        )
        if rule_match.intent != FALLBACK_INTENT and rule_match.score > tf_top_conf:
            # Rules win: re-rank candidates.
            blended: List[Tuple[str, float]] = [(rule_match.intent, rule_match.score)]
            for intent, score in tf_scores:
                if intent != rule_match.intent:
                    blended.append((intent, score))
                if len(blended) >= top_k:
                    break
            return ClassificationOutcome(
                candidates=blended[:top_k],
                backend="ensemble:rules",
                latency_ms=0.0,
            )

        # Transformer still preferred even though low-confidence.
        return ClassificationOutcome(
            candidates=tf_scores[:top_k],
            backend="ensemble:transformer-lowconf",
            latency_ms=0.0,
        )

    def _emit_metrics(self, outcome: ClassificationOutcome) -> None:
        backend_root = outcome.backend.split(":", 1)[0]
        INTENT_REQUESTS_TOTAL.labels(backend=backend_root, status="ok").inc()
        INTENT_PREDICTION_DURATION.labels(backend=backend_root).observe(outcome.latency_ms / 1000.0)
        INTENT_CONFIDENCE.labels(backend=backend_root).observe(outcome.top_confidence)
        if outcome.top_intent in INTENT_IDS:
            INTENT_LABEL_TOTAL.labels(intent=outcome.top_intent).inc()
