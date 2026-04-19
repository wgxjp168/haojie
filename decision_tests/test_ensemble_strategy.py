from decision.config import DecisionSettings
from decision.core.exceptions import ModelFailedError, ModelNotReadyError
from decision.features import FeatureExtractor
from decision.schemas import (
    BuyerContext,
    DecisionRequest,
    IntentSignal,
    ProductContext,
)
from decision.strategies.base import StrategyOutcome
from decision.strategies.ensemble import EnsembleStrategy
from decision.strategies.ml import MLBasedStrategy
from decision.strategies.rules import RuleBasedStrategy


def _features(**over):
    user_type = over.pop("user_type", "b2b")
    intent = over.pop("intent", "request_quote")
    conf = over.pop("intent_confidence", 0.85)
    language = over.pop("language", "en")
    req = DecisionRequest(
        intent=IntentSignal(intent=intent, confidence=conf),
        buyer=BuyerContext(user_type=user_type, budget=over.pop("budget", None)),
        product=ProductContext(
            unit_price=over.pop("unit_price", None),
            quantity=over.pop("quantity", None),
            in_stock=over.pop("in_stock", None),
        ),
        language=language,
    )
    return FeatureExtractor().extract(req)


class _StubMl(MLBasedStrategy):
    """ML stub returning a deterministic outcome; used to exercise blending logic."""

    def __init__(self, *, settings, candidates):
        super().__init__(settings=settings)
        self._stub_candidates = candidates
        # Mark as loaded so ensure_loaded is a no-op.
        self._loaded = True

    def evaluate(self, features):
        return StrategyOutcome(
            candidates=list(self._stub_candidates),
            rationale="stub ml",
            strategy="ml",
            risk_score=0.1,
            next_steps=[],
        )

    def evaluate_batch(self, batch):
        return [self.evaluate(f) for f in batch]


def _settings():
    return DecisionSettings(
        ensemble_rules_weight=0.5,
        ensemble_ml_weight=0.5,
        low_confidence_threshold=0.4,
    )


def test_ensemble_without_ml_uses_only_rules():
    s = _settings()
    e = EnsembleStrategy(settings=s, rules=RuleBasedStrategy(settings=s), ml=None)
    out = e.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert out.top_action == "request_quote"
    assert out.metadata.get("ml_fallback") == "ml_not_configured"
    assert out.strategy == "ensemble:rules"


def test_ensemble_blends_rules_and_ml():
    s = _settings()
    rules = RuleBasedStrategy(settings=s)
    ml = _StubMl(
        settings=s,
        candidates=[("approve_purchase", 0.9), ("request_quote", 0.05)],
    )
    e = EnsembleStrategy(settings=s, rules=rules, ml=ml)
    out = e.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert out.top_action in {"request_quote", "approve_purchase"}
    assert out.strategy == "ensemble:blended"
    assert "ml_top" in out.metadata


def test_low_ml_confidence_marks_rules_dominant():
    s = _settings()
    rules = RuleBasedStrategy(settings=s)
    ml = _StubMl(
        settings=s,
        candidates=[("approve_purchase", 0.1), ("no_action", 0.05)],
    )
    e = EnsembleStrategy(settings=s, rules=rules, ml=ml)
    out = e.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert out.strategy == "ensemble:rules_dominant"
    assert out.metadata["ml_low_confidence"] is True


def test_ml_failure_falls_back_to_rules():
    s = _settings()
    rules = RuleBasedStrategy(settings=s)

    class _Boom(MLBasedStrategy):
        def __init__(self):
            super().__init__(settings=s)
            self._loaded = True

        def evaluate(self, features):
            raise ModelFailedError("kaboom")

    e = EnsembleStrategy(settings=s, rules=rules, ml=_Boom())
    out = e.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert out.strategy == "ensemble:rules"
    assert out.metadata["ml_fallback"] == "ModelFailedError"


def test_ml_not_ready_falls_back_to_rules():
    s = _settings()
    rules = RuleBasedStrategy(settings=s)

    class _NotReady(MLBasedStrategy):
        def __init__(self):
            super().__init__(settings=s)
            self._loaded = True

        def evaluate(self, features):
            raise ModelNotReadyError("not ready")

    e = EnsembleStrategy(settings=s, rules=rules, ml=_NotReady())
    out = e.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert out.strategy == "ensemble:rules"


def test_zero_weights_raises():
    import pytest

    # Settings-level validator catches this before EnsembleStrategy ever sees it.
    with pytest.raises(Exception):
        DecisionSettings(ensemble_rules_weight=0.0, ensemble_ml_weight=0.0)


def test_evaluate_batch_returns_one_per_input():
    s = _settings()
    rules = RuleBasedStrategy(settings=s)
    ml = _StubMl(settings=s, candidates=[("approve_purchase", 0.8)])
    e = EnsembleStrategy(settings=s, rules=rules, ml=ml)
    feats = [
        _features(intent="request_quote", user_type="b2b"),
        _features(intent="place_order", user_type="b2c", in_stock=True),
    ]
    out = e.evaluate_batch(feats)
    assert len(out) == 2
