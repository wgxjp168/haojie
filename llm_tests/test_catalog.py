import pytest

from llm.catalog import (
    MODEL_CATALOG,
    MODEL_IDS,
    STUB_MODEL_ID,
    cheapest_model,
    get_model,
    models_for_provider,
)


def test_stub_model_present():
    assert STUB_MODEL_ID in MODEL_CATALOG


def test_get_model_unknown_raises():
    with pytest.raises(KeyError):
        get_model("nope")


def test_get_model_returns_matching():
    model = get_model("stub-small")
    assert model.provider == "stub"


def test_models_for_provider():
    stub_models = models_for_provider("stub")
    assert len(stub_models) >= 1
    assert all(m.provider == "stub" for m in stub_models)


def test_ids_unique():
    assert len(MODEL_IDS) == len(set(MODEL_IDS))


def test_cheapest_model_is_a_stub_or_cheap():
    # The stub models are priced at 0 so they should be the cheapest.
    cheapest = cheapest_model()
    assert cheapest.price_per_1k_prompt_usd == 0.0


def test_estimated_cost_usd():
    m = get_model("gpt-4o")
    cost = m.estimated_cost_usd(prompt_tokens=1000, completion_tokens=500)
    assert cost > 0
    # 1K prompt * 0.005 + 0.5K completion * 0.015 = 0.0125
    assert abs(cost - (0.005 + 0.5 * 0.015)) < 1e-6


def test_stub_cost_is_zero():
    m = get_model("stub-small")
    assert m.estimated_cost_usd(100, 100) == 0.0
