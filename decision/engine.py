"""Decision engine orchestrator.

Composes feature extraction + strategy selection + circuit breaker. Knows
nothing about HTTP / auth / caching — those live in ``DecisionService``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from decision.circuit_breaker import CircuitBreaker
from decision.config import DecisionSettings, DecisionStrategyKind
from decision.core.exceptions import (
    CircuitOpenError,
    ModelFailedError,
    ModelNotReadyError,
    UnsupportedStrategyError,
)
from decision.core.logging import get_logger
from decision.core.metrics import (
    DECISION_CONFIDENCE,
    DECISION_DURATION,
    DECISION_FALLBACK_TOTAL,
    DECISION_REQUESTS_TOTAL,
)
from decision.features import FeatureExtractor, FeatureVector
from decision.strategies.base import BaseStrategy, StrategyOutcome
from decision.strategies.ensemble import EnsembleStrategy
from decision.strategies.ml import MLBasedStrategy
from decision.strategies.rules import RuleBasedStrategy

log = get_logger(__name__)


@dataclass
class EngineDecision:
    """The full result of one engine evaluation (per request)."""

    outcome: StrategyOutcome
    features: FeatureVector
    latency_ms: float
    actual_strategy: str
    fallback_reasons: List[str] = field(default_factory=list)


class DecisionEngine:
    """The orchestrator. Stateful (holds strategies) but thread-safe."""

    def __init__(
        self,
        *,
        settings: DecisionSettings,
        rules: Optional[RuleBasedStrategy] = None,
        ml: Optional[MLBasedStrategy] = None,
        ensemble: Optional[EnsembleStrategy] = None,
        breaker: Optional[CircuitBreaker] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.settings = settings
        self.extractor = extractor or FeatureExtractor(
            default_language=settings.default_language
        )
        self.rules = rules or RuleBasedStrategy(settings=settings)

        # ML strategy is optional. We construct it lazily so callers using
        # only the rule strategy never pay the import cost.
        if ml is not None:
            self.ml: Optional[MLBasedStrategy] = ml
        elif settings.strategy in {DecisionStrategyKind.ML, DecisionStrategyKind.ENSEMBLE}:
            self.ml = MLBasedStrategy(settings=settings)
        else:
            self.ml = None

        if ensemble is not None:
            self.ensemble: Optional[EnsembleStrategy] = ensemble
        elif settings.strategy == DecisionStrategyKind.ENSEMBLE:
            self.ensemble = EnsembleStrategy(
                settings=settings, rules=self.rules, ml=self.ml
            )
        else:
            self.ensemble = None

        self.breaker = breaker or CircuitBreaker(
            failure_threshold=settings.cb_failure_threshold,
            recovery_seconds=settings.cb_recovery_seconds,
        )

        if settings.preload_model and self.ml is not None:
            try:
                self.ml.warmup()
            except (ModelNotReadyError, ModelFailedError) as exc:
                if not settings.allow_rules_fallback:
                    raise
                log.warning(
                    "decision.engine.preload_failed_fallback_rules",
                    error=str(exc),
                )

    # ----- public api -----

    @property
    def effective_strategy(self) -> str:
        kind = self.settings.strategy
        if kind == DecisionStrategyKind.RULES:
            return "rules"
        if kind == DecisionStrategyKind.ML:
            if self.ml is None:
                return "rules"
            if self.breaker.state_label == "open":
                return "rules"
            return "ml"
        # ensemble
        if self.ml is None or self.breaker.state_label == "open":
            return "ensemble:rules"
        return "ensemble"

    def evaluate(
        self,
        features: FeatureVector,
        *,
        strategy_override: Optional[str] = None,
    ) -> EngineDecision:
        start = time.perf_counter()
        strategy_kind, strategy = self._resolve_strategy(strategy_override)
        fallback_reasons: List[str] = []

        try:
            if isinstance(strategy, MLBasedStrategy):
                self.breaker.ensure_allowed()
                try:
                    outcome = strategy.evaluate(features)
                    self.breaker.record_success()
                except (ModelNotReadyError, ModelFailedError) as exc:
                    self.breaker.record_failure()
                    if not self.settings.allow_rules_fallback:
                        raise
                    log.warning(
                        "decision.engine.ml_failed_falling_back_to_rules",
                        error=str(exc),
                    )
                    DECISION_FALLBACK_TOTAL.labels(
                        from_strategy="ml", to_strategy="rules", reason=type(exc).__name__
                    ).inc()
                    outcome = self.rules.evaluate(features)
                    strategy_kind = "rules_fallback"
                    fallback_reasons.append(type(exc).__name__)
            elif isinstance(strategy, EnsembleStrategy):
                outcome = strategy.evaluate(features)
                if outcome.metadata.get("ml_fallback"):
                    fallback_reasons.append(str(outcome.metadata["ml_fallback"]))
            else:
                outcome = strategy.evaluate(features)
        except CircuitOpenError as exc:
            log.warning("decision.engine.circuit_open_falling_back", error=str(exc))
            DECISION_FALLBACK_TOTAL.labels(
                from_strategy=strategy_kind, to_strategy="rules", reason="CircuitOpen"
            ).inc()
            outcome = self.rules.evaluate(features)
            strategy_kind = "rules_fallback"
            fallback_reasons.append("CircuitOpen")

        latency_ms = (time.perf_counter() - start) * 1000.0
        decision = EngineDecision(
            outcome=outcome,
            features=features,
            latency_ms=round(latency_ms, 3),
            actual_strategy=strategy_kind,
            fallback_reasons=fallback_reasons,
        )
        self._emit_metrics(decision)
        return decision

    def evaluate_batch(
        self,
        batch: Sequence[FeatureVector],
        *,
        strategy_override: Optional[str] = None,
    ) -> List[EngineDecision]:
        if not batch:
            return []
        # We always evaluate per-item so each item's fallback / metrics
        # are independently tracked. Strategies that benefit from batching
        # implement ``evaluate_batch`` internally; the orchestrator's
        # batch path delegates so the rules vs ml decision is uniform.
        kind, strategy = self._resolve_strategy(strategy_override)
        start = time.perf_counter()
        try:
            if isinstance(strategy, MLBasedStrategy):
                self.breaker.ensure_allowed()
                try:
                    outcomes = strategy.evaluate_batch(batch)
                    self.breaker.record_success()
                except (ModelNotReadyError, ModelFailedError) as exc:
                    self.breaker.record_failure()
                    if not self.settings.allow_rules_fallback:
                        raise
                    log.warning(
                        "decision.engine.batch_ml_failed_falling_back_to_rules",
                        error=str(exc),
                    )
                    DECISION_FALLBACK_TOTAL.labels(
                        from_strategy="ml", to_strategy="rules", reason=type(exc).__name__
                    ).inc()
                    outcomes = self.rules.evaluate_batch(batch)
                    kind = "rules_fallback"
            else:
                outcomes = strategy.evaluate_batch(batch)
        except CircuitOpenError:
            DECISION_FALLBACK_TOTAL.labels(
                from_strategy=kind, to_strategy="rules", reason="CircuitOpen"
            ).inc()
            outcomes = self.rules.evaluate_batch(batch)
            kind = "rules_fallback"

        elapsed_each = ((time.perf_counter() - start) * 1000.0) / max(1, len(batch))

        results: List[EngineDecision] = []
        for features, outcome in zip(batch, outcomes):
            fb_reasons: List[str] = []
            if outcome.metadata.get("ml_fallback"):
                fb_reasons.append(str(outcome.metadata["ml_fallback"]))
            decision = EngineDecision(
                outcome=outcome,
                features=features,
                latency_ms=round(elapsed_each, 3),
                actual_strategy=kind,
                fallback_reasons=fb_reasons,
            )
            self._emit_metrics(decision)
            results.append(decision)
        return results

    # ----- internals -----

    def _resolve_strategy(
        self, override: Optional[str]
    ) -> "tuple[str, BaseStrategy]":
        kind = (override or self.settings.strategy.value).lower()

        if kind == "rules":
            return "rules", self.rules
        if kind == "ml":
            if self.ml is None:
                raise UnsupportedStrategyError(
                    "ml strategy requested but is not configured",
                    details={"strategy": kind},
                )
            return "ml", self.ml
        if kind == "ensemble":
            if self.ensemble is None:
                # Build an ad-hoc ensemble (rules-only fallback) instead
                # of raising. This way ``strategy_override='ensemble'``
                # behaves predictably even on a rules-only deployment.
                ensemble = EnsembleStrategy(
                    settings=self.settings, rules=self.rules, ml=self.ml
                )
                self.ensemble = ensemble
            return "ensemble", self.ensemble
        raise UnsupportedStrategyError(
            f"unknown strategy {kind!r}",
            details={"supported": ["rules", "ml", "ensemble"]},
        )

    def _emit_metrics(self, decision: EngineDecision) -> None:
        strategy_root = decision.actual_strategy.split(":", 1)[0]
        DECISION_REQUESTS_TOTAL.labels(strategy=strategy_root, status="ok").inc()
        DECISION_DURATION.labels(strategy=strategy_root).observe(
            decision.latency_ms / 1000.0
        )
        DECISION_CONFIDENCE.labels(strategy=strategy_root).observe(
            decision.outcome.top_confidence
        )
