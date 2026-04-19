import pytest

from decision.config import DecisionSettings, DecisionStrategyKind
from decision.core.exceptions import ValidationDecisionError
from decision.schemas import (
    BatchDecisionRequest,
    BuyerContext,
    DecisionRequest,
    IntentSignal,
    ProductContext,
)
from decision.service import DecisionService


def _service(**over):
    settings = DecisionSettings(
        strategy=over.pop("strategy", DecisionStrategyKind.RULES),
        ml_model_kind="dummy",
        cache_enabled=over.pop("cache_enabled", True),
        memory_cache_size=8,
        redis_url=None,
        **over,
    )
    return DecisionService(settings=settings)


def _req(**over):
    return DecisionRequest(
        intent=IntentSignal(
            intent=over.pop("intent", "request_quote"),
            confidence=over.pop("intent_confidence", 0.85),
        ),
        buyer=BuyerContext(
            user_type=over.pop("user_type", "b2b"),
            budget=over.pop("budget", None),
            risk_score=over.pop("buyer_risk", None),
        ),
        product=ProductContext(
            unit_price=over.pop("unit_price", None),
            quantity=over.pop("quantity", None),
            in_stock=over.pop("in_stock", None),
        ),
        urgency=over.pop("urgency", "medium"),
        language=over.pop("language", "en"),
        **over,
    )


def test_basic_decide_b2b_quote():
    svc = _service()
    resp = svc.decide(_req(intent="request_quote", user_type="b2b"))
    assert resp.action == "request_quote"
    assert resp.confident is True
    assert resp.strategy == "rules"
    assert resp.next_steps


def test_b2c_place_order_proceeds_to_checkout():
    svc = _service()
    resp = svc.decide(_req(intent="place_order", user_type="b2c"))
    assert resp.action == "proceed_to_checkout"


def test_caches_repeated_requests():
    svc = _service()
    r1 = svc.decide(_req(intent="request_quote", user_type="b2b"))
    r2 = svc.decide(_req(intent="request_quote", user_type="b2b"))
    assert r1.action == r2.action
    assert r2.cached is True


def test_cache_disabled_never_marks_cached():
    svc = _service(cache_enabled=False)
    r1 = svc.decide(_req(intent="request_quote", user_type="b2b"))
    r2 = svc.decide(_req(intent="request_quote", user_type="b2b"))
    assert r1.cached is False
    assert r2.cached is False


def test_strategy_override_per_request():
    svc = _service(strategy=DecisionStrategyKind.ENSEMBLE)
    resp = svc.decide(_req(strategy_override="rules"))
    assert resp.strategy == "rules"


def test_high_value_requires_human_review():
    svc = _service(
        max_auto_approve_amount=5_000.0,
        require_human_review_above=10_000.0,
    )
    resp = svc.decide(_req(
        intent="bulk_order",
        user_type="b2b",
        unit_price=20_000.0,
        quantity=1,
    ))
    assert resp.requires_review is True
    assert resp.action == "escalate_to_human"


def test_validation_rejects_negative_price():
    svc = _service()
    with pytest.raises(Exception):
        # Pydantic will raise during request construction.
        _req(unit_price=-5.0, quantity=1)


def test_service_uptime_positive():
    svc = _service()
    assert svc.uptime_seconds >= 0


def test_circuit_state_initially_closed():
    svc = _service()
    assert svc.circuit_state == "closed"


def test_decide_batch_basic():
    svc = _service()
    batch = BatchDecisionRequest(items=[
        _req(intent="request_quote", user_type="b2b"),
        _req(intent="cancel_order"),
        _req(intent="return_refund"),
    ])
    resp = svc.decide_batch(batch)
    assert resp.total == 3
    actions = [r.action for r in resp.results]
    assert "request_quote" in actions
    assert "cancel_request" in actions
    assert "initiate_return" in actions


def test_decide_batch_respects_max_batch_size():
    svc = _service(max_batch_size=2)
    batch = BatchDecisionRequest(items=[_req() for _ in range(3)])
    with pytest.raises(ValidationDecisionError):
        svc.decide_batch(batch)


def test_decide_batch_with_mixed_overrides():
    svc = _service(strategy=DecisionStrategyKind.ENSEMBLE)
    batch = BatchDecisionRequest(items=[
        _req(intent="request_quote", user_type="b2b", strategy_override="rules"),
        _req(intent="cancel_order", strategy_override="ensemble"),
    ])
    resp = svc.decide_batch(batch)
    assert resp.total == 2


def test_response_carries_request_and_session_ids():
    svc = _service()
    resp = svc.decide(_req(intent="request_quote", user_type="b2b", request_id="r-1", session_id="s-1"))
    assert resp.request_id == "r-1"
    assert resp.session_id == "s-1"


def test_metadata_includes_matched_rules():
    svc = _service()
    resp = svc.decide(_req(intent="request_quote", user_type="b2b"))
    matched = resp.metadata.get("matched_rules") or []
    assert matched


def test_low_confidence_marks_not_confident():
    svc = _service(confident_threshold=0.99)
    resp = svc.decide(_req(intent="search_product", intent_confidence=0.6, user_type="b2c"))
    assert resp.confident is False


def test_zh_label_for_zh_request():
    svc = _service()
    resp = svc.decide(_req(intent="request_quote", user_type="b2b", language="zh-CN"))
    assert any("\u4e00" <= ch <= "\u9fff" for ch in resp.label)


def test_unknown_action_replaced_with_fallback():
    # Sanity: unknown action ids never propagate (they get filtered).
    svc = _service()
    resp = svc.decide(_req(intent="this_intent_doesnt_exist"))
    # Even unknown intent should still produce a valid action id.
    assert resp.action in {a for a in __import__('decision.catalog', fromlist=['ACTION_REGISTRY']).ACTION_REGISTRY}


def test_complaint_escalates_and_flags_review():
    svc = _service()
    resp = svc.decide(_req(intent="complaint"))
    assert resp.action == "escalate_to_human"
    assert resp.requires_review is True


def test_negative_budget_rejected_by_validator():
    with pytest.raises(Exception):
        _req(budget=-5.0)


def test_response_has_localised_next_steps_zh():
    svc = _service()
    resp = svc.decide(_req(intent="request_quote", user_type="b2b", language="zh-CN"))
    assert resp.next_steps
    assert any("\u4e00" <= ch <= "\u9fff" for ch in "".join(resp.next_steps))
