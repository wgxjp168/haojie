import pytest

from decision.config import DecisionSettings
from decision.core.exceptions import ModelNotReadyError
from decision.features import FeatureExtractor
from decision.schemas import (
    BuyerContext,
    DecisionRequest,
    IntentSignal,
    ProductContext,
)
from decision.strategies.ml import MLBasedStrategy, _DummyModel


def _features(**over):
    user_type = over.pop("user_type", "b2c")
    intent = over.pop("intent", "search_product")
    conf = over.pop("intent_confidence", 0.7)
    urgency = over.pop("urgency", "medium")
    language = over.pop("language", "en")
    req = DecisionRequest(
        intent=IntentSignal(intent=intent, confidence=conf),
        buyer=BuyerContext(
            user_type=user_type,
            budget=over.pop("budget", None),
            history_orders=over.pop("history_orders", 0),
            history_returns=over.pop("history_returns", 0),
            risk_score=over.pop("buyer_risk", None),
        ),
        product=ProductContext(
            unit_price=over.pop("unit_price", None),
            quantity=over.pop("quantity", None),
            in_stock=over.pop("in_stock", None),
            supplier_rating=over.pop("supplier_rating", None),
            promotion_active=over.pop("promotion_active", None),
            has_alternatives=over.pop("has_alternatives", None),
        ),
        urgency=urgency,
        language=language,
    )
    return FeatureExtractor().extract(req)


def test_dummy_model_loads_lazily():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="dummy"))
    assert not s.ready
    s.ensure_loaded()
    assert s.ready
    assert s.model_loaded


def test_dummy_strategy_evaluates_b2b_quote():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="dummy"))
    out = s.evaluate(_features(intent="request_quote", user_type="b2b", unit_price=2000.0, quantity=1))
    assert out.candidates
    actions = [a for a, _ in out.candidates]
    assert "request_quote" in actions


def test_dummy_strategy_b2c_emits_b2c_actions():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="dummy"))
    out = s.evaluate(_features(user_type="b2c", in_stock=True))
    actions = [a for a, _ in out.candidates]
    assert any(a in {"add_to_cart", "proceed_to_checkout", "wait_for_promotion"} for a in actions)


def test_warmup_loads_model():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="dummy"))
    s.warmup()
    assert s.ready


def test_unknown_kind_raises_model_not_ready():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="bogus"))
    with pytest.raises(ModelNotReadyError):
        s.ensure_loaded()


def test_sklearn_kind_requires_path():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="sklearn"))
    with pytest.raises(ModelNotReadyError):
        s.ensure_loaded()


def test_dummy_model_predict_proba_returns_distribution():
    m = _DummyModel()
    probs = m.predict_proba([[0.0] * 21])
    assert len(probs) == 1
    row = probs[0]
    # Softmax should sum to 1.0 (within float error).
    assert abs(sum(row) - 1.0) < 1e-6


def test_evaluate_batch_returns_one_per_input():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="dummy"))
    feats = [
        _features(intent="search_product"),
        _features(intent="place_order", user_type="b2c", unit_price=100.0, quantity=1, in_stock=True),
    ]
    outcomes = s.evaluate_batch(feats)
    assert len(outcomes) == 2
    for o in outcomes:
        assert o.candidates
        assert 0.0 <= o.top_confidence <= 1.0


def test_risk_grows_with_total_amount():
    s = MLBasedStrategy(
        settings=DecisionSettings(
            ml_model_kind="dummy",
            max_auto_approve_amount=5_000,
            require_human_review_above=10_000,
        )
    )
    s.warmup()
    low = s.evaluate(_features(intent="place_order", unit_price=10.0, quantity=1, in_stock=True))
    high = s.evaluate(_features(intent="place_order", unit_price=10_000.0, quantity=1, in_stock=True))
    assert high.risk_score >= low.risk_score


def test_metadata_records_model_kind():
    s = MLBasedStrategy(settings=DecisionSettings(ml_model_kind="dummy"))
    out = s.evaluate(_features())
    assert out.metadata.get("model_kind") == "dummy"
