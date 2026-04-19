import pytest
from pydantic import ValidationError

from decision.schemas import (
    BatchDecisionRequest,
    BuyerContext,
    DecisionRequest,
    IntentSignal,
    ProductContext,
)


def _req(**over):
    return DecisionRequest(
        intent=IntentSignal(intent=over.pop("intent", "search_product"), confidence=0.7),
        buyer=BuyerContext(**over.pop("buyer", {"user_type": "b2c"})),
        product=ProductContext(**over.pop("product", {})),
        **over,
    )


def test_intent_must_be_non_empty():
    with pytest.raises(ValidationError):
        IntentSignal(intent="   ", confidence=0.5)


def test_intent_confidence_bounded():
    with pytest.raises(ValidationError):
        IntentSignal(intent="x", confidence=1.5)


def test_unit_price_negative_rejected():
    with pytest.raises(ValidationError):
        ProductContext(unit_price=-1.0)


def test_quantity_zero_rejected():
    with pytest.raises(ValidationError):
        ProductContext(quantity=0)


def test_supplier_rating_bounded():
    with pytest.raises(ValidationError):
        ProductContext(supplier_rating=10.0)


def test_buyer_budget_negative_rejected():
    with pytest.raises(ValidationError):
        BuyerContext(user_type="b2c", budget=-1.0)


def test_strategy_override_validated():
    with pytest.raises(ValidationError):
        DecisionRequest(
            intent=IntentSignal(intent="x", confidence=0.5),
            strategy_override="bogus",
        )


def test_strategy_override_lowercased():
    req = DecisionRequest(
        intent=IntentSignal(intent="x", confidence=0.5),
        strategy_override="ENSEMBLE",
    )
    assert req.strategy_override == "ensemble"


def test_batch_must_be_non_empty():
    with pytest.raises(ValidationError):
        BatchDecisionRequest(items=[])


def test_batch_size_capped():
    with pytest.raises(ValidationError):
        BatchDecisionRequest(items=[_req() for _ in range(257)])


def test_total_amount_property():
    p = ProductContext(unit_price=10.0, quantity=3)
    assert p.total_amount == 30.0


def test_total_amount_when_missing_returns_none():
    p = ProductContext(unit_price=10.0)
    assert p.total_amount is None


def test_decision_request_defaults():
    req = DecisionRequest(intent=IntentSignal(intent="search_product", confidence=0.5))
    assert req.urgency == "medium"
    assert req.buyer.user_type == "b2c"


def test_user_type_validated():
    with pytest.raises(ValidationError):
        BuyerContext(user_type="b2x")
