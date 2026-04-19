"""ML-based decision strategy (optional).

Two backends are supported:

* ``sklearn`` — a pickled ``sklearn`` estimator with ``predict_proba`` and
  a ``classes_`` attribute matching the action catalog.
* ``dummy``   — a deterministic logistic-style model that uses only the
  built-in ``math`` module. It exists so that the test suite can exercise
  the ML code path without optional dependencies.

A ``torch`` backend stub is included for forward-compatibility but is
*not* enabled by default and falls through to ``dummy`` when torch isn't
importable.

Failure modes are explicit: when the underlying model cannot be loaded
or inference fails, we raise ``ModelNotReadyError`` / ``ModelFailedError``
and let the orchestrator decide whether to fall back to the rule
strategy.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from decision.catalog import ACTION_IDS, ACTION_REGISTRY, FALLBACK_ACTION, next_steps_of
from decision.config import DecisionSettings
from decision.core.exceptions import ModelFailedError, ModelNotReadyError
from decision.core.logging import get_logger
from decision.core.metrics import DECISION_MODEL_LOAD_EVENTS
from decision.features import FeatureVector
from decision.strategies.base import BaseStrategy, StrategyOutcome

log = get_logger(__name__)


# ----------------------------------------------------------------- backends


class _BaseModel:
    """Internal protocol every ML backend implements."""

    classes_: Tuple[str, ...] = ()

    def predict_proba(self, batch: List[List[float]]) -> List[List[float]]:
        raise NotImplementedError


class _DummyModel(_BaseModel):
    """A deterministic logistic-style model — no third-party deps.

    The dummy model assigns weights based on a few common-sense priors so
    that calling ``evaluate`` produces non-degenerate, reproducible output
    for tests.
    """

    classes_ = (
        "approve_purchase",
        "reject_purchase",
        "request_more_info",
        "request_quote",
        "negotiate_terms",
        "bulk_order",
        "add_to_cart",
        "proceed_to_checkout",
        "wait_for_promotion",
        "recommend_alternatives",
        "compare_alternatives",
        "show_product_details",
        "check_inventory",
        "escalate_to_human",
        "contact_supplier",
        "cancel_request",
        "initiate_return",
        "defer_decision",
        "no_action",
    )

    def predict_proba(self, batch: List[List[float]]) -> List[List[float]]:
        results: List[List[float]] = []
        for row in batch:
            scores = self._score_row(row)
            results.append(_softmax(scores))
        return results

    def _score_row(self, row: List[float]) -> List[float]:
        # Feature order matches ``MLBasedStrategy._encode``.
        (
            intent_conf,
            unit_price,
            quantity,
            total_amount,
            budget,
            ratio,
            discount,
            supplier_rating,
            history_orders,
            return_ratio,
            buyer_risk,
            lead_time,
            in_stock,
            promotion_active,
            has_alternatives,
            has_budget,
            has_product,
            is_b2b,
            is_b2c,
            urgency_high,
            urgency_low,
        ) = row

        scores: Dict[str, float] = {a: 0.0 for a in self.classes_}
        scores["approve_purchase"] += 1.5 * (1.0 - min(ratio, 1.0)) + 0.5 * in_stock
        scores["reject_purchase"] += 1.0 * max(0.0, ratio - 1.0) + 0.5 * buyer_risk
        scores["request_more_info"] += 1.0 * (1.0 - intent_conf)
        scores["request_quote"] += 1.5 * is_b2b + 0.4 * (1.0 if total_amount > 1000 else 0.0)
        scores["bulk_order"] += 1.5 * is_b2b + 0.05 * min(quantity, 100)
        scores["negotiate_terms"] += 0.7 * is_b2b
        scores["add_to_cart"] += 1.0 * is_b2c + 0.4 * in_stock
        scores["proceed_to_checkout"] += 0.8 * is_b2c + 0.4 * urgency_high
        scores["wait_for_promotion"] += 0.6 * is_b2c + 0.5 * urgency_low * (1.0 - promotion_active)
        scores["recommend_alternatives"] += 0.6 * has_alternatives + 0.5 * (1.0 - in_stock)
        scores["compare_alternatives"] += 0.4 * has_alternatives
        scores["show_product_details"] += 0.4 * has_product
        scores["check_inventory"] += 0.4 * (1.0 - in_stock)
        scores["escalate_to_human"] += 1.0 * buyer_risk + 0.5 * (1.0 if total_amount > 50000 else 0.0)
        scores["contact_supplier"] += 0.4 * is_b2b * (1.0 - supplier_rating / 5.0)
        scores["cancel_request"] += 0.3
        scores["initiate_return"] += 0.2 + 0.4 * return_ratio
        scores["defer_decision"] += 0.3 * (1.0 - intent_conf)
        scores["no_action"] += 0.2

        return [scores[c] for c in self.classes_]


class _SklearnModel(_BaseModel):
    """Wraps a pickled scikit-learn estimator."""

    def __init__(self, path: str) -> None:
        try:
            import joblib  # type: ignore
        except ImportError:  # pragma: no cover - optional
            try:
                import pickle as _pickle

                with open(path, "rb") as fh:
                    estimator = _pickle.load(fh)
            except Exception as exc:
                raise ModelNotReadyError(
                    f"failed to load sklearn model from {path}: {exc}",
                    details={"path": path},
                ) from exc
        else:
            try:
                estimator = joblib.load(path)
            except Exception as exc:
                raise ModelNotReadyError(
                    f"failed to load sklearn model from {path}: {exc}",
                    details={"path": path},
                ) from exc

        if not hasattr(estimator, "predict_proba"):
            raise ModelNotReadyError(
                "sklearn model must expose predict_proba",
                details={"path": path},
            )
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            raise ModelNotReadyError(
                "sklearn model is missing classes_ attribute",
                details={"path": path},
            )
        unknown = [c for c in classes if c not in ACTION_REGISTRY]
        if unknown:
            raise ModelNotReadyError(
                "sklearn model classes are not a subset of the action catalog",
                details={"unknown": unknown[:5]},
            )

        self._estimator = estimator
        self.classes_ = tuple(str(c) for c in classes)

    def predict_proba(self, batch: List[List[float]]) -> List[List[float]]:
        try:
            probs = self._estimator.predict_proba(batch)
        except Exception as exc:  # pragma: no cover - depends on model
            raise ModelFailedError(
                f"sklearn predict_proba failed: {exc}"
            ) from exc
        return [list(map(float, row)) for row in probs]


# ------------------------------------------------------------ helpers


def _softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


# ------------------------------------------------------------ strategy


class MLBasedStrategy(BaseStrategy):
    """ML-driven strategy that produces a probability distribution over actions."""

    name = "ml"

    def __init__(
        self,
        *,
        settings: DecisionSettings,
        model: Optional[_BaseModel] = None,
    ) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[Exception] = None
        self._loaded_at: Optional[float] = None
        self._model: Optional[_BaseModel] = model
        if model is not None:
            self._loaded = True
            self._loaded_at = time.time()

    # ----- public api -----

    @property
    def ready(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def model_loaded(self) -> bool:
        return self.ready

    @property
    def load_error(self) -> Optional[Exception]:
        return self._load_error

    def warmup(self) -> None:
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            DECISION_MODEL_LOAD_EVENTS.labels(event="attempt").inc()
            try:
                self._model = self._load_model()
                self._loaded = True
                self._load_error = None
                self._loaded_at = time.time()
                DECISION_MODEL_LOAD_EVENTS.labels(event="success").inc()
                log.info(
                    "decision.model.loaded",
                    kind=self.settings.ml_model_kind,
                    path=self.settings.ml_model_path,
                    classes=len(self._model.classes_),
                )
            except ModelNotReadyError:
                DECISION_MODEL_LOAD_EVENTS.labels(event="failure").inc()
                self._load_error = self._load_error  # propagate
                raise
            except Exception as exc:  # noqa: BLE001
                DECISION_MODEL_LOAD_EVENTS.labels(event="failure").inc()
                self._load_error = exc
                log.error("decision.model.load_failed", error=str(exc))
                raise ModelNotReadyError(
                    f"failed to load ML model: {exc}",
                    details={"kind": self.settings.ml_model_kind},
                ) from exc

    def evaluate(self, features: FeatureVector) -> StrategyOutcome:
        outcomes = self.evaluate_batch([features])
        return outcomes[0]

    def evaluate_batch(self, batch: Sequence[FeatureVector]) -> List[StrategyOutcome]:
        if not batch:
            return []
        self.ensure_loaded()
        assert self._model is not None
        encoded = [self._encode(f) for f in batch]
        try:
            probs = self._model.predict_proba(encoded)
        except ModelFailedError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelFailedError(f"ml inference failed: {exc}") from exc

        outcomes: List[StrategyOutcome] = []
        classes = self._model.classes_
        for features, row in zip(batch, probs):
            pairs: List[Tuple[str, float]] = []
            for action_id, p in zip(classes, row):
                if action_id in ACTION_REGISTRY:
                    pairs.append((action_id, float(p)))
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            if not pairs:
                pairs = [(FALLBACK_ACTION, 0.0)]
            top_action, top_conf = pairs[0]
            risk = self._risk_for(features, top_action, top_conf)
            outcomes.append(
                StrategyOutcome(
                    candidates=pairs,
                    rationale=self._rationale(features, top_action, top_conf),
                    strategy=self.name,
                    risk_score=round(risk, 4),
                    metadata={
                        "model_kind": self.settings.ml_model_kind,
                        "loaded_at": self._loaded_at,
                    },
                    next_steps=next_steps_of(top_action, features.language),
                )
            )
        return outcomes

    # ----- internals -----

    def _load_model(self) -> _BaseModel:
        kind = (self.settings.ml_model_kind or "dummy").lower()
        path = self.settings.ml_model_path

        if kind == "dummy":
            return _DummyModel()
        if kind == "sklearn":
            if not path:
                raise ModelNotReadyError(
                    "ml_model_path is required for sklearn backend",
                    details={"kind": kind},
                )
            return _SklearnModel(path)
        if kind == "torch":  # pragma: no cover - optional
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise ModelNotReadyError(
                    "torch is not installed; install pytorch to enable the torch backend",
                ) from exc
            # A real torch wrapper would live here; for now we degrade.
            log.warning(
                "decision.model.torch_backend_not_implemented_using_dummy"
            )
            return _DummyModel()
        raise ModelNotReadyError(
            f"unknown ml_model_kind: {kind!r}", details={"kind": kind}
        )

    def _encode(self, features: FeatureVector) -> List[float]:
        return [
            features.intent_confidence,
            features.unit_price,
            features.quantity,
            features.total_amount,
            features.budget,
            features.price_to_budget_ratio,
            features.discount_pct,
            features.supplier_rating,
            features.history_orders,
            features.return_ratio,
            features.buyer_risk_score,
            features.lead_time_days,
            features.in_stock,
            features.promotion_active,
            features.has_alternatives,
            features.has_budget,
            features.has_product,
            1.0 if features.user_type == "b2b" else 0.0,
            1.0 if features.user_type == "b2c" else 0.0,
            1.0 if features.urgency == "high" else 0.0,
            1.0 if features.urgency == "low" else 0.0,
        ]

    def _risk_for(self, features: FeatureVector, action: str, conf: float) -> float:
        # Risk grows with low confidence, high spend, and high buyer risk.
        risk = (1.0 - conf) * 0.4
        risk += min(0.4, features.total_amount / max(1.0, self.settings.require_human_review_above) * 0.4)
        risk += features.buyer_risk_score * 0.2
        if action in {"approve_purchase", "bulk_order"}:
            risk += 0.05
        return min(1.0, risk)

    def _rationale(self, features: FeatureVector, action: str, conf: float) -> str:
        is_zh = features.language.lower().startswith("zh")
        if is_zh:
            return f"模型 ({self.settings.ml_model_kind}) 以置信度 {conf:.2f} 选择 {action}"
        return f"Model ({self.settings.ml_model_kind}) chose {action} with confidence {conf:.2f}"


__all__ = [
    "MLBasedStrategy",
    "_BaseModel",
    "_DummyModel",
    "_SklearnModel",
]
