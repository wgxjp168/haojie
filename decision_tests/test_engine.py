import pytest

from decision.config import DecisionSettings, DecisionStrategyKind
from decision.core.exceptions import (
    ModelFailedError,
    ModelNotReadyError,
    UnsupportedStrategyError,
)
from decision.engine import DecisionEngine
from decision.features import FeatureExtractor
from decision.schemas import (
    BuyerContext,
    DecisionRequest,
    IntentSignal,
    ProductContext,
)
from decision.strategies.ml import MLBasedStrategy


def _features(**over):
    user_type = over.pop("user_type", "b2b")
    intent = over.pop("intent", "request_quote")
    conf = over.pop("intent_confidence", 0.85)
    req = DecisionRequest(
        intent=IntentSignal(intent=intent, confidence=conf),
        buyer=BuyerContext(user_type=user_type),
        product=ProductContext(
            unit_price=over.pop("unit_price", None),
            quantity=over.pop("quantity", None),
            in_stock=over.pop("in_stock", None),
        ),
        language=over.pop("language", "en"),
    )
    return FeatureExtractor().extract(req)


def test_rules_only_engine():
    engine = DecisionEngine(settings=DecisionSettings(strategy=DecisionStrategyKind.RULES))
    decision = engine.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert decision.outcome.top_action == "request_quote"
    assert decision.actual_strategy == "rules"


def test_ml_only_engine_with_dummy():
    engine = DecisionEngine(
        settings=DecisionSettings(
            strategy=DecisionStrategyKind.ML, ml_model_kind="dummy"
        )
    )
    decision = engine.evaluate(_features())
    assert decision.outcome.candidates


def test_ensemble_engine():
    engine = DecisionEngine(
        settings=DecisionSettings(
            strategy=DecisionStrategyKind.ENSEMBLE, ml_model_kind="dummy"
        )
    )
    decision = engine.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert "ensemble" in decision.actual_strategy or decision.actual_strategy == "rules_fallback"


def test_strategy_override_to_rules():
    engine = DecisionEngine(
        settings=DecisionSettings(
            strategy=DecisionStrategyKind.ENSEMBLE, ml_model_kind="dummy"
        )
    )
    decision = engine.evaluate(_features(), strategy_override="rules")
    assert decision.actual_strategy == "rules"


def test_strategy_override_invalid_raises():
    engine = DecisionEngine(settings=DecisionSettings(strategy=DecisionStrategyKind.RULES))
    with pytest.raises(UnsupportedStrategyError):
        engine.evaluate(_features(), strategy_override="bogus")


def test_strategy_override_ml_when_not_configured_raises():
    engine = DecisionEngine(settings=DecisionSettings(strategy=DecisionStrategyKind.RULES))
    with pytest.raises(UnsupportedStrategyError):
        engine.evaluate(_features(), strategy_override="ml")


def test_ml_failure_falls_back_to_rules_when_allowed():
    settings = DecisionSettings(
        strategy=DecisionStrategyKind.ML,
        ml_model_kind="dummy",
        allow_rules_fallback=True,
        cb_failure_threshold=10,
    )

    class _Boom(MLBasedStrategy):
        def __init__(self):
            super().__init__(settings=settings)
            self._loaded = True

        def evaluate(self, features):
            raise ModelFailedError("boom")

        def evaluate_batch(self, batch):
            raise ModelFailedError("boom")

    engine = DecisionEngine(settings=settings, ml=_Boom())
    decision = engine.evaluate(_features())
    assert decision.actual_strategy == "rules_fallback"
    assert "ModelFailedError" in decision.fallback_reasons


def test_ml_failure_propagates_when_fallback_disabled():
    settings = DecisionSettings(
        strategy=DecisionStrategyKind.ML,
        ml_model_kind="dummy",
        allow_rules_fallback=False,
        cb_failure_threshold=10,
    )

    class _Boom(MLBasedStrategy):
        def __init__(self):
            super().__init__(settings=settings)
            self._loaded = True

        def evaluate(self, features):
            raise ModelFailedError("boom")

    engine = DecisionEngine(settings=settings, ml=_Boom())
    with pytest.raises(ModelFailedError):
        engine.evaluate(_features())


def test_circuit_opens_after_repeated_ml_failures():
    settings = DecisionSettings(
        strategy=DecisionStrategyKind.ML,
        ml_model_kind="dummy",
        allow_rules_fallback=True,
        cb_failure_threshold=2,
        cb_recovery_seconds=10,
    )

    class _Boom(MLBasedStrategy):
        def __init__(self):
            super().__init__(settings=settings)
            self._loaded = True

        def evaluate(self, features):
            raise ModelFailedError("boom")

        def evaluate_batch(self, batch):
            raise ModelFailedError("boom")

    engine = DecisionEngine(settings=settings, ml=_Boom())
    engine.evaluate(_features())
    engine.evaluate(_features())
    # Now circuit should be open and subsequent calls bypass ML.
    decision = engine.evaluate(_features())
    assert decision.actual_strategy == "rules_fallback"
    assert engine.breaker.state_label == "open"


def test_evaluate_batch_returns_one_per_input():
    engine = DecisionEngine(settings=DecisionSettings(strategy=DecisionStrategyKind.RULES))
    feats = [_features(), _features(intent="cancel_order")]
    decisions = engine.evaluate_batch(feats)
    assert len(decisions) == 2


def test_effective_strategy_label():
    engine_rules = DecisionEngine(settings=DecisionSettings(strategy=DecisionStrategyKind.RULES))
    assert engine_rules.effective_strategy == "rules"

    engine_ens = DecisionEngine(
        settings=DecisionSettings(
            strategy=DecisionStrategyKind.ENSEMBLE, ml_model_kind="dummy"
        )
    )
    assert engine_ens.effective_strategy in {"ensemble", "ensemble:rules"}


def test_preload_failure_falls_back_when_allowed():
    settings = DecisionSettings(
        strategy=DecisionStrategyKind.ML,
        ml_model_kind="bogus",  # will fail to load
        allow_rules_fallback=True,
        preload_model=True,
    )
    # Should not raise.
    DecisionEngine(settings=settings)


def test_preload_failure_raises_when_not_allowed():
    settings = DecisionSettings(
        strategy=DecisionStrategyKind.ML,
        ml_model_kind="bogus",
        allow_rules_fallback=False,
        preload_model=True,
    )
    with pytest.raises(ModelNotReadyError):
        DecisionEngine(settings=settings)
