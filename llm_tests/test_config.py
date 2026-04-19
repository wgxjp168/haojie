import pytest

from llm.config import LLMEnvironment, LLMSettings, LLMStrategy


def test_defaults_load():
    s = LLMSettings()
    assert s.service_name == "ilbuyai-llm"
    assert s.port == 8004
    assert s.default_model_id == "stub-small"


def test_strategy_enum():
    assert LLMStrategy("fallback") == LLMStrategy.FALLBACK


def test_fallback_chain_csv_env(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_CHAIN", "stub-small, stub-fast")
    s = LLMSettings()
    assert s.fallback_chain == ["stub-small", "stub-fast"]


def test_fallback_chain_must_not_be_empty():
    with pytest.raises(Exception):
        LLMSettings(fallback_chain=[])


def test_thresholds_must_be_ordered():
    with pytest.raises(Exception):
        LLMSettings(confident_threshold=0.3, low_confidence_threshold=0.8)


def test_api_keys_csv_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEYS", "k1, k2,k3")
    s = LLMSettings()
    assert s.api_keys == ["k1", "k2", "k3"]


def test_temperature_bounds():
    with pytest.raises(Exception):
        LLMSettings(default_temperature=3.0)


def test_is_production():
    s = LLMSettings(env=LLMEnvironment.PRODUCTION)
    assert s.is_production is True


def test_provider_credentials_helper():
    s = LLMSettings(openai_api_key="abc", anthropic_api_key="xyz")
    creds = s.provider_credentials()
    assert creds["openai"]["api_key"] == "abc"
    assert creds["anthropic"]["api_key"] == "xyz"
    assert creds["google"]["api_key"] is None
