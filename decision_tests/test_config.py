import pytest

from decision.config import DecisionEnvironment, DecisionSettings, DecisionStrategyKind


def test_defaults_load():
    s = DecisionSettings()
    assert s.service_name == "ilbuyai-decision"
    assert s.port == 8003
    assert s.strategy in {
        DecisionStrategyKind.RULES,
        DecisionStrategyKind.ML,
        DecisionStrategyKind.ENSEMBLE,
    }


def test_csv_split_for_lists():
    s = DecisionSettings(api_keys="k1, k2 , k3")
    assert s.api_keys == ["k1", "k2", "k3"]


def test_default_language_must_be_supported():
    with pytest.raises(Exception):
        DecisionSettings(default_language="ja-JP")


def test_ensemble_weights_must_sum_positive():
    with pytest.raises(Exception):
        DecisionSettings(ensemble_rules_weight=0.0, ensemble_ml_weight=0.0)


def test_review_ceiling_above_floor():
    with pytest.raises(Exception):
        DecisionSettings(
            max_auto_approve_amount=10_000.0,
            require_human_review_above=5_000.0,
        )


def test_is_production_flag():
    s = DecisionSettings(env=DecisionEnvironment.PRODUCTION)
    assert s.is_production is True

    s2 = DecisionSettings(env=DecisionEnvironment.DEVELOPMENT)
    assert s2.is_production is False


def test_threshold_bounds():
    with pytest.raises(Exception):
        DecisionSettings(confident_threshold=1.5)
    with pytest.raises(Exception):
        DecisionSettings(low_confidence_threshold=-0.1)
