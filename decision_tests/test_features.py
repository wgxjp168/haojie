from decision.features import FeatureExtractor
from decision.schemas import (
    BuyerContext,
    DecisionRequest,
    IntentSignal,
    ProductContext,
)


def _request(**overrides):
    base = dict(
        intent=IntentSignal(intent="search_product", confidence=0.7),
        buyer=BuyerContext(user_type="b2c"),
        product=ProductContext(),
    )
    base.update(overrides)
    return DecisionRequest(**base)


def test_extract_minimum_request():
    f = FeatureExtractor().extract(_request())
    assert f.intent == "search_product"
    assert f.intent_confidence == 0.7
    assert f.user_type == "b2c"
    assert f.in_stock == 0.5  # unknown encodes to 0.5


def test_extract_total_amount_from_unit_price_and_qty():
    req = _request(product=ProductContext(unit_price=100.0, quantity=3, in_stock=True))
    f = FeatureExtractor().extract(req)
    assert f.total_amount == 300.0
    assert f.in_stock == 1.0


def test_extract_with_budget_ratio():
    req = _request(
        buyer=BuyerContext(user_type="b2b", budget=1000.0),
        product=ProductContext(unit_price=500.0, quantity=3),
    )
    f = FeatureExtractor().extract(req)
    assert f.budget == 1000.0
    assert f.has_budget == 1.0
    assert round(f.price_to_budget_ratio, 2) == 1.5


def test_extract_clips_extreme_ratio():
    req = _request(
        buyer=BuyerContext(user_type="b2c", budget=1.0),
        product=ProductContext(unit_price=100_000_000.0, quantity=1),
    )
    f = FeatureExtractor().extract(req)
    assert f.price_to_budget_ratio <= 10.0


def test_extract_return_ratio():
    req = _request(buyer=BuyerContext(user_type="b2c", history_orders=10, history_returns=4))
    f = FeatureExtractor().extract(req)
    assert round(f.return_ratio, 2) == 0.4


def test_extract_zero_orders_returns_zero_ratio():
    req = _request(buyer=BuyerContext(user_type="b2c", history_orders=0, history_returns=2))
    f = FeatureExtractor().extract(req)
    assert f.return_ratio == 0.0


def test_extract_language_default_when_missing():
    f = FeatureExtractor(default_language="zh-CN").extract(_request())
    assert f.language == "zh-CN"


def test_extract_language_uses_request_then_buyer_then_default():
    req = _request(language="en", buyer=BuyerContext(user_type="b2c", preferred_language="zh-CN"))
    f = FeatureExtractor(default_language="zh-CN").extract(req)
    assert f.language == "en"


def test_cache_key_stable_for_same_inputs():
    e = FeatureExtractor()
    a = e.extract(_request(product=ProductContext(unit_price=10.0, quantity=2, in_stock=True)))
    b = e.extract(_request(product=ProductContext(unit_price=10.0, quantity=2, in_stock=True)))
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_with_inputs():
    e = FeatureExtractor()
    a = e.extract(_request(product=ProductContext(unit_price=10.0, quantity=2)))
    b = e.extract(_request(product=ProductContext(unit_price=20.0, quantity=2)))
    assert a.cache_key() != b.cache_key()


def test_affordability_helper():
    e = FeatureExtractor()
    aff_unknown = e.extract(_request())
    assert aff_unknown.affordability == 0.5

    cheap = e.extract(_request(
        buyer=BuyerContext(user_type="b2c", budget=1000.0),
        product=ProductContext(unit_price=100.0, quantity=1),
    ))
    assert cheap.affordability >= 0.5

    expensive = e.extract(_request(
        buyer=BuyerContext(user_type="b2c", budget=100.0),
        product=ProductContext(unit_price=1000.0, quantity=1),
    ))
    assert expensive.affordability == 0.0
