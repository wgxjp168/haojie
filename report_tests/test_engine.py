import pytest

from report.config import ReportSettings
from report.core.exceptions import (
    StorageFailedError,
    StorageUnavailableError,
    UnknownAudienceError,
    UnknownFormatError,
    UnknownTemplateError,
)
from report.engine import ReportEngine
from report.storage import MemoryStorageBackend
from report.storage.base import BaseStorageBackend


def _engine(**over):
    storages = over.pop("storages", None)
    settings = ReportSettings(
        storage_backends=over.pop("storage_backends", ["memory"]),
        redis_url=None,
        **over,
    )
    return ReportEngine(
        settings=settings,
        storages=storages if storages is not None else [MemoryStorageBackend()],
    )


def test_engine_renders_markdown(sample_request):
    engine = _engine()
    resp = engine.render(sample_request)
    assert resp.format == "markdown"
    assert resp.content.startswith("# ")
    assert resp.content_type.startswith("text/markdown")


def test_engine_generates_id_when_missing(sample_request):
    sample_request.request_id = None
    engine = _engine()
    resp = engine.render(sample_request)
    assert resp.report_id.startswith("rpt_")


def test_engine_defaults_to_settings(sample_request):
    sample_request.format = None
    sample_request.audience = None
    sample_request.language = None
    engine = _engine()
    resp = engine.render(sample_request)
    assert resp.format == "markdown"  # from test env
    assert resp.audience == "technical"


def test_engine_rejects_unknown_format(sample_request):
    engine = _engine()
    sample_request.format = "pdf"  # not registered
    # schemas.field_validator catches this earlier, so we test the engine
    # path directly by faking a renderer lookup: remove the registered one.
    engine.renderers.pop("markdown")
    sample_request.format = "markdown"
    with pytest.raises(UnknownFormatError):
        engine.render(sample_request)


def test_engine_rejects_unknown_template(sample_request):
    engine = _engine()
    sample_request.template_id = "nope"
    with pytest.raises(UnknownTemplateError):
        engine.render(sample_request)


def test_engine_storage_fan_out_writes_to_all(sample_request):
    m1 = MemoryStorageBackend()
    m2 = MemoryStorageBackend()
    # Both backends share the same .name, so we rename to expose both.
    m2.name = "memory2"
    engine = _engine(storages=[m1, m2])
    resp = engine.render(sample_request)
    assert len(resp.storage_results) == 2
    assert all(r.success for r in resp.storage_results)
    assert m1.exists(resp.report_id)
    assert m2.exists(resp.report_id)


def test_engine_skips_storage_when_store_false(sample_request):
    sample_request.store = False
    m = MemoryStorageBackend()
    engine = _engine(storages=[m])
    resp = engine.render(sample_request)
    assert resp.storage_results == []
    assert not m.exists(resp.report_id)


class _BoomBackend(BaseStorageBackend):
    name = "boom"

    @property
    def available(self) -> bool:
        return True

    def put(self, *args, **kwargs):
        raise StorageFailedError("disk full")

    def get(self, report_id):  # pragma: no cover - not exercised
        raise NotImplementedError

    def exists(self, report_id) -> bool:
        return False

    def delete(self, report_id) -> bool:
        return False


def test_engine_records_storage_failure_without_raising(sample_request):
    engine = _engine(storages=[_BoomBackend()])
    resp = engine.render(sample_request)
    assert len(resp.storage_results) == 1
    assert resp.storage_results[0].success is False
    assert resp.storage_results[0].error


def test_engine_circuit_opens_after_repeated_failures(sample_request):
    settings = ReportSettings(
        storage_backends=["memory"],
        cb_failure_threshold=2,
        cb_recovery_seconds=10,
        redis_url=None,
    )
    engine = ReportEngine(settings=settings, storages=[_BoomBackend()])

    engine.render(sample_request)
    engine.render(sample_request)
    # Third call: breaker is open so the backend is skipped without another put.
    resp = engine.render(sample_request)
    assert resp.storage_results[0].success is False
    assert engine.breakers.get("boom").state_label == "open"


class _UnavailableBackend(BaseStorageBackend):
    name = "down"

    @property
    def available(self) -> bool:
        return False

    def put(self, *args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("put called on unavailable backend")

    def get(self, report_id):  # pragma: no cover
        raise NotImplementedError

    def exists(self, report_id) -> bool:
        return False

    def delete(self, report_id) -> bool:
        return False


def test_engine_skips_unavailable_backend(sample_request):
    engine = _engine(storages=[_UnavailableBackend()])
    resp = engine.render(sample_request)
    assert resp.storage_results[0].success is False
    assert "unavailable" in (resp.storage_results[0].error or "").lower()
