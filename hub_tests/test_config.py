import pytest

from hub.config import ClientMode, HubEnvironment, HubSettings


def test_defaults_load():
    s = HubSettings()
    assert s.service_name == "ilbuyai-hub"
    assert s.port == 8006
    assert s.intent_mode in (ClientMode.EMBEDDED, ClientMode.HTTP)


def test_stage_modes_configurable(monkeypatch):
    monkeypatch.setenv("HUB_INTENT_MODE", "http")
    monkeypatch.setenv("HUB_REPORT_MODE", "embedded")
    s = HubSettings()
    assert s.intent_mode == ClientMode.HTTP
    assert s.report_mode == ClientMode.EMBEDDED


def test_csv_split_for_lists(monkeypatch):
    monkeypatch.setenv("HUB_API_KEYS", "k1 , k2,k3")
    s = HubSettings()
    assert s.api_keys == ["k1", "k2", "k3"]


def test_default_language_must_be_supported():
    with pytest.raises(Exception):
        HubSettings(default_language="fr")


def test_default_user_type_validated():
    with pytest.raises(Exception):
        HubSettings(default_user_type="b2x")


def test_is_production_flag():
    s = HubSettings(env=HubEnvironment.PRODUCTION)
    assert s.is_production is True


def test_port_bounds():
    with pytest.raises(Exception):
        HubSettings(port=70_000)
