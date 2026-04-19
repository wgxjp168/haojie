import pytest

from report.config import ReportAudience, ReportEnvironment, ReportFormat, ReportSettings, StorageBackend


def test_defaults_load():
    s = ReportSettings()
    assert s.service_name == "ilbuyai-report"
    assert s.port == 8005
    assert s.default_format == ReportFormat.MARKDOWN


def test_csv_split_for_lists(monkeypatch):
    monkeypatch.setenv("REPORT_API_KEYS", "k1, k2 ,k3")
    s = ReportSettings()
    assert s.api_keys == ["k1", "k2", "k3"]


def test_storage_backends_csv(monkeypatch):
    monkeypatch.setenv("REPORT_STORAGE_BACKENDS", "memory,local")
    s = ReportSettings()
    assert s.storage_backends == ["memory", "local"]


def test_storage_backends_unknown_rejected():
    with pytest.raises(Exception):
        ReportSettings(storage_backends=["bogus"])


def test_default_language_must_be_supported():
    with pytest.raises(Exception):
        ReportSettings(default_language="ja-JP")


def test_is_production_flag():
    s = ReportSettings(env=ReportEnvironment.PRODUCTION)
    assert s.is_production is True
    s2 = ReportSettings(env=ReportEnvironment.DEVELOPMENT)
    assert s2.is_production is False


def test_port_bounds():
    with pytest.raises(Exception):
        ReportSettings(port=70_000)


def test_backend_enum_values():
    assert {b.value for b in StorageBackend} == {"memory", "local", "s3"}


def test_audience_enum_values():
    assert {a.value for a in ReportAudience} == {"executive", "technical", "customer"}
