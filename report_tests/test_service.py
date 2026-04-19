import pytest

from report.config import ReportAudience, ReportFormat, ReportSettings
from report.core.exceptions import (
    PayloadTooLargeError,
    ReportNotFoundError,
    ValidationReportError,
)
from report.schemas import BatchReportRequest, ReportRequest
from report.service import ReportService
from report.storage import LocalFilesystemBackend, MemoryStorageBackend


def _service(**over):
    storages = over.pop("storages", None)
    settings = ReportSettings(
        storage_backends=over.pop("storage_backends", ["memory"]),
        cache_enabled=over.pop("cache_enabled", True),
        memory_cache_size=over.pop("memory_cache_size", 16),
        redis_url=None,
        **over,
    )
    return ReportService(
        settings=settings,
        storages=storages if storages is not None else [MemoryStorageBackend()],
    )


def test_service_renders_and_stores(sample_request):
    svc = _service()
    resp = svc.generate(sample_request)
    assert resp.report_id
    assert resp.content
    assert resp.cached is False
    assert any(r.success for r in resp.storage_results)


def test_service_caches_repeated_requests(sample_request):
    svc = _service()
    r1 = svc.generate(sample_request)
    r2 = svc.generate(sample_request)
    assert r1.content == r2.content
    assert r2.cached is True


def test_service_cache_disabled_never_caches(sample_request):
    svc = _service(cache_enabled=False)
    r1 = svc.generate(sample_request)
    r2 = svc.generate(sample_request)
    assert r1.cached is False
    assert r2.cached is False


def test_service_retrieves_stored_report(sample_request):
    svc = _service()
    resp = svc.generate(sample_request)
    fetched = svc.get_report(resp.report_id)
    assert fetched.report_id == resp.report_id
    assert fetched.content == resp.content


def test_service_retrieves_with_backend_pin(sample_request):
    svc = _service()
    resp = svc.generate(sample_request)
    fetched = svc.get_report(resp.report_id, backend="memory")
    assert fetched.backend == "memory"


def test_service_retrieve_unknown_backend_rejected(sample_request):
    svc = _service()
    svc.generate(sample_request)
    with pytest.raises(ValidationReportError):
        svc.get_report("anything", backend="does-not-exist")


def test_service_retrieve_missing_raises(sample_request):
    svc = _service()
    with pytest.raises(ReportNotFoundError):
        svc.get_report("no-such-report")


def test_service_delete_report(sample_request):
    svc = _service()
    resp = svc.generate(sample_request)
    outcomes = svc.delete_report(resp.report_id)
    assert outcomes["memory"] is True
    with pytest.raises(ReportNotFoundError):
        svc.get_report(resp.report_id)


def test_service_payload_too_large(sample_request):
    svc = _service(max_payload_bytes=1024)
    # Balloon the LLM narrative to exceed the limit.
    sample_request.llm.content = "x" * 5_000
    with pytest.raises(PayloadTooLargeError):
        svc.generate(sample_request)


def test_service_batch_generation(sample_request):
    svc = _service()
    batch = BatchReportRequest(items=[sample_request, sample_request.model_copy()])
    resp = svc.generate_batch(batch)
    assert resp.total == 2
    assert all(r.content for r in resp.results)


def test_service_batch_caps_size(sample_request):
    svc = _service(max_batch_size=1)
    batch = BatchReportRequest(items=[sample_request, sample_request.model_copy()])
    with pytest.raises(ValidationReportError):
        svc.generate_batch(batch)


def test_service_batch_tolerates_item_errors(sample_request):
    svc = _service()
    bad = sample_request.model_copy()
    bad.template_id = "no_such_template"
    batch = BatchReportRequest(items=[sample_request, bad])
    resp = svc.generate_batch(batch)
    assert resp.total == 2
    # First one succeeds, second one carries an error envelope.
    assert resp.results[0].content
    assert resp.results[1].metadata.get("error_code") == "UNKNOWN_TEMPLATE"


def test_service_local_storage_roundtrip(sample_request, temp_dir):
    svc = _service(
        storages=[LocalFilesystemBackend(base_dir=temp_dir)],
        storage_backends=["local"],
    )
    resp = svc.generate(sample_request)
    fetched = svc.get_report(resp.report_id)
    assert fetched.content == resp.content
    assert fetched.backend == "local"


def test_service_available_backends_exposes_memory(sample_request):
    svc = _service()
    assert "memory" in svc.available_backends


def test_service_circuit_states_empty_initially(sample_request):
    svc = _service()
    # Breakers are created lazily; none yet.
    assert svc.circuit_states == {}


def test_service_english_rendering(sample_request):
    sample_request.language = "en"
    sample_request.format = "markdown"
    svc = _service()
    resp = svc.generate(sample_request)
    assert "Procurement Decision Report" in resp.content
