from decision.config import DecisionSettings
from decision.features import FeatureExtractor
from decision.schemas import (
    BuyerContext,
    DecisionRequest,
    IntentSignal,
    ProductContext,
)
from decision.strategies.rules import RuleBasedStrategy


def _settings(**over):
    return DecisionSettings(**over)


def _features(**over):
    base_intent = over.pop("intent", "search_product")
    base_conf = over.pop("intent_confidence", 0.7)
    user_type = over.pop("user_type", "b2c")
    urgency = over.pop("urgency", "medium")
    language = over.pop("language", "en")
    buyer = BuyerContext(
        user_type=user_type,
        budget=over.pop("budget", None),
        history_orders=over.pop("history_orders", 0),
        history_returns=over.pop("history_returns", 0),
        risk_score=over.pop("buyer_risk", None),
    )
    product = ProductContext(
        unit_price=over.pop("unit_price", None),
        quantity=over.pop("quantity", None),
        in_stock=over.pop("in_stock", None),
        supplier_rating=over.pop("supplier_rating", None),
        promotion_active=over.pop("promotion_active", None),
        has_alternatives=over.pop("has_alternatives", None),
    )
    req = DecisionRequest(
        intent=IntentSignal(intent=base_intent, confidence=base_conf),
        buyer=buyer,
        product=product,
        urgency=urgency,
        language=language,
        **over,
    )
    return FeatureExtractor().extract(req)


def test_b2b_request_quote_picks_request_quote():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="request_quote", user_type="b2b"))
    assert out.top_action == "request_quote"
    assert out.top_confidence >= 0.5


def test_b2b_bulk_picks_bulk_order():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="bulk_order", user_type="b2b"))
    assert out.top_action == "bulk_order"


def test_b2c_place_order_proceeds_to_checkout():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="place_order", user_type="b2c"))
    assert out.top_action == "proceed_to_checkout"


def test_complaint_escalates_to_human():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="complaint"))
    assert out.top_action == "escalate_to_human"


def test_cancel_intent_emits_cancel_request():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="cancel_order"))
    assert out.top_action == "cancel_request"


def test_return_intent_emits_initiate_return():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="return_refund"))
    assert out.top_action == "initiate_return"


def test_over_budget_rejects():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(
        intent="place_order",
        user_type="b2c",
        budget=100.0,
        unit_price=300.0,
        quantity=1,
        in_stock=True,
    ))
    # Should reject because total / budget = 3.0 >= 1.5
    assert out.top_action == "reject_purchase"
    assert out.risk_score > 0.0


def test_high_value_escalates_to_human():
    s = RuleBasedStrategy(
        settings=_settings(
            max_auto_approve_amount=5_000.0,
            require_human_review_above=10_000.0,
        )
    )
    out = s.evaluate(_features(
        intent="bulk_order",
        user_type="b2b",
        unit_price=100_000.0,
        quantity=1,
        in_stock=True,
    ))
    assert out.top_action == "escalate_to_human"


def test_auto_approve_within_ceiling():
    s = RuleBasedStrategy(settings=_settings(
        max_auto_approve_amount=10_000.0,
        require_human_review_above=50_000.0,
    ))
    out = s.evaluate(_features(
        intent="place_order",
        user_type="b2c",
        unit_price=500.0,
        quantity=2,
        in_stock=True,
    ))
    assert "approve_purchase" in {c[0] for c in out.candidates}


def test_unknown_intent_falls_back_to_request_more_info_or_no_action():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="unknown_intent_id", intent_confidence=0.6))
    assert out.top_action in {"request_more_info", "no_action"}


def test_low_intent_confidence_requests_more_info():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="search_product", intent_confidence=0.2))
    assert "request_more_info" in {c[0] for c in out.candidates}


def test_oos_with_alternatives_recommends_alternatives():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(
        intent="search_product", in_stock=False, has_alternatives=True
    ))
    assert "recommend_alternatives" in {c[0] for c in out.candidates}


def test_zh_rationale_when_language_zh():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="request_quote", user_type="b2b", language="zh-CN"))
    # Chinese rationale should contain CJK characters
    assert any("\u4e00" <= ch <= "\u9fff" for ch in out.rationale)


def test_metadata_records_matched_rules():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="request_quote", user_type="b2b"))
    matched = out.metadata.get("matched_rules", [])
    assert any(m["action"] == "request_quote" for m in matched)


def test_audience_filtering_drops_b2c_only_for_b2b_user():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(
        intent="search_product",
        user_type="b2b",
        in_stock=True,
    ))
    assert "add_to_cart" not in {c[0] for c in out.candidates}


def test_high_buyer_risk_escalates_when_purchasing():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(
        intent="place_order",
        user_type="b2c",
        buyer_risk=0.9,
        unit_price=100.0,
        quantity=1,
        in_stock=True,
    ))
    assert "escalate_to_human" in {c[0] for c in out.candidates}


def test_negotiate_intent_for_b2b():
    s = RuleBasedStrategy(settings=_settings())
    out = s.evaluate(_features(intent="negotiate_terms", user_type="b2b"))
    assert out.top_action == "negotiate_terms"
