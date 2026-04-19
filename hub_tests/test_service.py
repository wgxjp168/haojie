"""End-to-end tests for ``HubService`` with embedded downstreams."""
from __future__ import annotations

import pytest

from hub.config import HubSettings
from hub.core.exceptions import ValidationHubError
from hub.schemas import (
    BatchPipelineRequest,
    PipelineBuyer,
    PipelineOptions,
    PipelineProduct,
    PipelineRequest,
)
from hub.service import HubService


def _service(**over) -> HubService:
    settings = HubSettings(
        max_batch_size=over.pop("max_batch_size", 8),
        pipeline_timeout_seconds=over.pop("pipeline_timeout_seconds", 15.0),
        **over,
    )
    return HubService(settings=settings)


@pytest.mark.asyncio
async def test_service_runs_pipeline_end_to_end(sample_b2b_request):
    svc = _service()
    resp = await svc.run_pipeline(sample_b2b_request)
    assert resp.overall_status == "success"
    assert resp.intent and resp.intent.status == "success"
    assert resp.decision and resp.decision.status == "success"
    assert resp.report and resp.report.status == "success"
    assert resp.intent.result["top_intent"] == "inquire_price"
    assert resp.decision.result["action"] == "request_quote"
    assert "采购决策报告" in resp.report.result["content"]


@pytest.mark.asyncio
async def test_service_runs_pipeline_with_llm(sample_b2b_request):
    sample_b2b_request.options = PipelineOptions(
        include_llm=True, include_report=True, report_format="markdown"
    )
    svc = _service()
    resp = await svc.run_pipeline(sample_b2b_request)
    assert resp.llm and resp.llm.status == "success"
    assert resp.llm.result["provider"] == "stub"
    # LLM narrative should end up in the report.
    assert resp.report and resp.report.status == "success"


@pytest.mark.asyncio
async def test_service_b2c_flow():
    svc = _service()
    req = PipelineRequest(
        text="有什么手机推荐？",
        user_type="b2c",
        buyer=PipelineBuyer(user_type="b2c", budget=5000),
        product=PipelineProduct(
            product_id="phone-1",
            name="智能手机",
            unit_price=3000,
            quantity=1,
            in_stock=True,
        ),
        options=PipelineOptions(include_llm=False, include_report=True),
    )
    resp = await svc.run_pipeline(req)
    assert resp.overall_status == "success"
    assert resp.intent.result["top_intent"] in {"search_product", "get_recommendation"}


@pytest.mark.asyncio
async def test_service_rejects_unsupported_language():
    svc = _service()
    req = PipelineRequest(text="hi", language="fr")
    with pytest.raises(ValidationHubError):
        await svc.run_pipeline(req)


@pytest.mark.asyncio
async def test_service_health_exposes_stages():
    svc = _service()
    health = svc.health()
    stages = {s.stage for s in health.stages}
    assert stages == {"intent", "decision", "llm", "report"}
    for s in health.stages:
        assert s.mode == "embedded"
        assert s.available is True


@pytest.mark.asyncio
async def test_service_batch_run(sample_b2b_request):
    svc = _service()
    batch = BatchPipelineRequest(
        items=[sample_b2b_request, sample_b2b_request.model_copy()]
    )
    resp = await svc.run_batch(batch)
    assert resp.total == 2
    assert all(r.overall_status == "success" for r in resp.results)


@pytest.mark.asyncio
async def test_service_batch_caps_size(sample_b2b_request):
    svc = _service(max_batch_size=1)
    batch = BatchPipelineRequest(
        items=[sample_b2b_request, sample_b2b_request.model_copy()]
    )
    with pytest.raises(ValidationHubError):
        await svc.run_batch(batch)


@pytest.mark.asyncio
async def test_service_partial_when_report_disabled_only(sample_b2b_request):
    sample_b2b_request.options = PipelineOptions(include_llm=False, include_report=False)
    svc = _service()
    resp = await svc.run_pipeline(sample_b2b_request)
    assert resp.report is None
    # Without report, success is still success (intent + decision succeeded).
    assert resp.overall_status == "success"


@pytest.mark.asyncio
async def test_service_forwards_request_id(sample_b2b_request):
    svc = _service()
    resp = await svc.run_pipeline(sample_b2b_request)
    assert resp.request_id == "hub-test-001"
    assert resp.session_id == "sess-001"


@pytest.mark.asyncio
async def test_service_intent_latency_recorded(sample_b2b_request):
    svc = _service()
    resp = await svc.run_pipeline(sample_b2b_request)
    assert resp.intent.latency_ms > 0
    assert resp.decision.latency_ms > 0
    assert resp.report.latency_ms > 0
    assert resp.total_latency_ms >= (
        resp.intent.latency_ms + resp.decision.latency_ms
    )
