"""Ensemble strategy: blend rules + ML with graceful degradation.

The ensemble:

1. Always evaluates the rule strategy (it's cheap and always available).
2. If the ML strategy is ready and not low-confidence, blends ML votes
   with the rule votes using weighted-sum.
3. If the ML strategy fails, the rule outcome is used as-is and we record
   ``metadata['ml_fallback'] = reason``.

Weights come from ``DecisionSettings.ensemble_rules_weight`` and
``ensemble_ml_weight``; they need not sum to 1 because we re-normalise
after blending.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from decision.catalog import FALLBACK_ACTION, next_steps_of
from decision.config import DecisionSettings
from decision.core.exceptions import (
    CircuitOpenError,
    ModelFailedError,
    ModelNotReadyError,
)
from decision.core.logging import get_logger
from decision.features import FeatureVector
from decision.strategies.base import BaseStrategy, StrategyOutcome
from decision.strategies.ml import MLBasedStrategy
from decision.strategies.rules import RuleBasedStrategy

log = get_logger(__name__)


class EnsembleStrategy(BaseStrategy):
    """Weighted blend of rules + ML with rule fallback."""

    name = "ensemble"

    def __init__(
        self,
        *,
        settings: DecisionSettings,
        rules: RuleBasedStrategy,
        ml: Optional[MLBasedStrategy],
    ) -> None:
        self.settings = settings
        self.rules = rules
        self.ml = ml
        self._w_rules = max(0.0, float(settings.ensemble_rules_weight))
        self._w_ml = max(0.0, float(settings.ensemble_ml_weight))
        if self._w_rules + self._w_ml <= 0:
            raise ValueError("ensemble weights must sum to > 0")

    @property
    def ready(self) -> bool:
        return True  # rule path always works

    def warmup(self) -> None:
        if self.ml is not None:
            try:
                self.ml.warmup()
            except (ModelNotReadyError, ModelFailedError) as exc:
                log.warning("decision.ensemble.warmup_ml_failed", error=str(exc))

    def evaluate(self, features: FeatureVector) -> StrategyOutcome:
        rule_outcome = self.rules.evaluate(features)
        if self.ml is None:
            return self._tag_only_rules(rule_outcome, reason="ml_not_configured")

        try:
            ml_outcome = self.ml.evaluate(features)
        except (CircuitOpenError, ModelNotReadyError, ModelFailedError) as exc:
            log.warning(
                "decision.ensemble.ml_failed_falling_back",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return self._tag_only_rules(rule_outcome, reason=type(exc).__name__)

        # If ML is too unsure we still trust the rules.
        if ml_outcome.top_confidence < self.settings.low_confidence_threshold:
            return self._blend(rule_outcome, ml_outcome, ml_low_confidence=True)
        return self._blend(rule_outcome, ml_outcome, ml_low_confidence=False)

    def evaluate_batch(self, batch: Sequence[FeatureVector]) -> List[StrategyOutcome]:
        if not batch:
            return []
        rule_outcomes = self.rules.evaluate_batch(batch)
        if self.ml is None:
            return [
                self._tag_only_rules(o, reason="ml_not_configured")
                for o in rule_outcomes
            ]

        try:
            ml_outcomes = self.ml.evaluate_batch(batch)
        except (CircuitOpenError, ModelNotReadyError, ModelFailedError) as exc:
            log.warning(
                "decision.ensemble.batch_ml_failed_falling_back",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return [
                self._tag_only_rules(o, reason=type(exc).__name__)
                for o in rule_outcomes
            ]

        out: List[StrategyOutcome] = []
        for features, rule_o, ml_o in zip(batch, rule_outcomes, ml_outcomes):
            low_conf = ml_o.top_confidence < self.settings.low_confidence_threshold
            out.append(self._blend(rule_o, ml_o, ml_low_confidence=low_conf, features=features))
        return out

    # ----- internals -----

    def _tag_only_rules(self, rule_outcome: StrategyOutcome, *, reason: str) -> StrategyOutcome:
        meta = dict(rule_outcome.metadata)
        meta["ml_fallback"] = reason
        return StrategyOutcome(
            candidates=rule_outcome.candidates,
            rationale=rule_outcome.rationale,
            strategy="ensemble:rules",
            risk_score=rule_outcome.risk_score,
            metadata=meta,
            next_steps=rule_outcome.next_steps,
        )

    def _blend(
        self,
        rule_outcome: StrategyOutcome,
        ml_outcome: StrategyOutcome,
        *,
        ml_low_confidence: bool,
        features: Optional[FeatureVector] = None,
    ) -> StrategyOutcome:
        # If ML is low-confidence we shrink its weight to 1/3 of nominal
        # so the rules dominate but ML still adds shape to the ranking.
        w_rules = self._w_rules
        w_ml = self._w_ml * (0.33 if ml_low_confidence else 1.0)

        scores: Dict[str, float] = {}
        for action_id, conf in rule_outcome.candidates:
            scores[action_id] = scores.get(action_id, 0.0) + w_rules * conf
        for action_id, conf in ml_outcome.candidates:
            scores[action_id] = scores.get(action_id, 0.0) + w_ml * conf

        if not scores:
            scores = {FALLBACK_ACTION: 0.0}

        # Re-normalise so the top score is 1.0 — keeps the contract that
        # confidence is bounded by [0, 1] without losing relative order.
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        max_score = ordered[0][1] or 1.0
        normalised: List[Tuple[str, float]] = [
            (a, round(min(1.0, s / max_score), 4)) for a, s in ordered
        ]

        # Risk: weighted average of underlying risks plus a low-conf penalty.
        risk = (
            (rule_outcome.risk_score * w_rules + ml_outcome.risk_score * w_ml)
            / max(1e-6, w_rules + w_ml)
        )
        if ml_low_confidence:
            risk = min(1.0, risk + 0.05)

        rationale = (
            f"rules: {rule_outcome.rationale} | ml: {ml_outcome.rationale}"
        )
        meta = {
            "rule_top": rule_outcome.candidates[0][0] if rule_outcome.candidates else None,
            "ml_top": ml_outcome.candidates[0][0] if ml_outcome.candidates else None,
            "ml_top_confidence": ml_outcome.top_confidence,
            "ml_low_confidence": ml_low_confidence,
            "weights": {"rules": w_rules, "ml": w_ml},
        }

        # Resolve language for next_steps from features if available;
        # otherwise inherit from the rule outcome's next_steps directly.
        if features is not None:
            next_steps = next_steps_of(normalised[0][0], features.language)
        else:
            next_steps = (
                rule_outcome.next_steps
                if normalised[0][0] == rule_outcome.top_action
                else ml_outcome.next_steps
            )

        return StrategyOutcome(
            candidates=normalised,
            rationale=rationale,
            strategy=(
                "ensemble:rules_dominant" if ml_low_confidence else "ensemble:blended"
            ),
            risk_score=round(risk, 4),
            metadata=meta,
            next_steps=next_steps,
        )
