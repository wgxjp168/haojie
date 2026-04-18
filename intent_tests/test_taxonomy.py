from intent.taxonomy import (
    FALLBACK_INTENT,
    INTENT_IDS,
    INTENT_REGISTRY,
    get_intent,
    intents_for_audience,
    label_of,
)


def test_taxonomy_has_expected_size():
    assert len(INTENT_IDS) == len(INTENT_REGISTRY)
    assert len(INTENT_IDS) >= 15
    assert FALLBACK_INTENT in INTENT_REGISTRY


def test_every_intent_has_both_labels():
    for defn in INTENT_REGISTRY.values():
        assert defn.label_zh
        assert defn.label_en
        assert defn.description
        assert defn.audiences


def test_b2b_only_and_b2c_only_intents_exist():
    b2b_ids = {d.id for d in intents_for_audience("b2b")}
    b2c_ids = {d.id for d in intents_for_audience("b2c")}
    # request_quote / bulk_order are B2B-only
    assert "request_quote" in b2b_ids
    assert "request_quote" not in b2c_ids


def test_label_of_returns_language():
    assert label_of("search_product", "zh-CN") == "商品搜索"
    assert label_of("search_product", "en") == "Product search"


def test_get_intent_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_intent("no_such_intent")
