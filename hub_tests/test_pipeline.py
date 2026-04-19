"""Pipeline orchestrator tests with fake stage clients.

Focus on the control flow: does the orchestrator sequence stages, feed
downstream payloads, skip stages when upstream fails, and compute the
right overall status.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from hub.clients.base import BaseStageClient, StageCallResult
from hub.config import ClientMode, HubSettings
from hub.pipeline import PipelineOrchestrator
from hub.schemas import (
    PipelineBuyer,
    PipelineOptions,
    PipelineProduct,
    PipelineRequest,
)


class _FakeClient(BaseStageClient):
    """Stage client that records calls and returns a canned result."""

    stage = "fake"

    def __init__(
        self,
        stage: str,
        *,
        result: Dict[str, Any] = None,
        fail: bool = False,
        error_code: str = "UPSTREAM_FAILED",
        error_message: str = "boom",
    ) -> None:
        self.stage = stage
        # Minimal breaker etc.
        super().__init__(
            mode=ClientMode.EMBEDDED,
            base_url="",
            timeout_seconds=5.0,
            max_retries=0,
        )
        self.calls: List[Dict[str, Any]] = []
        self._result = result or {}
        self._fail = fail
        self._err_code = error_code
        self._err_msg = error_message

    async def call(self, payload: Dict[str, Any]) -> StageCallResult:  # type: ignore[override]
        self.calls.append(payload)
        if self._fail:
            return StageCallResult(
                success=False,
                latency_ms=1.0,
                mode=self.mode.value,
                error_code=self._err_code,
                error_message=self._err_msg,
            )
        return StageCallResult(
            success=True, latency_ms=1.0, mode=self.mode.value, result=dict(self._result)
        )

    async def _call_embedded(self, payload):  # pragma: no cover
        raise NotImplementedError

    @property
    def _http_path(self) -> str:  # pragma: no cover
        return "/"


def _build(
    *, intent=None, decision=None, llm=None, report=None, settings=None
):
    settings = settings or HubSettings()
    return PipelineOrchestrator(
        settings=settings,
        intent_client=intent
        or _FakeClient(
            "intent",
            result={"top_intent": "inquire_price", "top_confidence": 0.9, "top_label": "询价"},
        ),
        decision_client=decision
        or _FakeClient(
            "decision",
            result={
                "action": "request_quote",
                "confidence": 0.85,
                "risk_score": 0.1,
                "requires_review": False,
                "rationale": "B 端询价",
                "next_steps": ["生成 RFQ"],
                "candidates": [],
                "strategy": "rules",
                "metadata": {"matched_rules": []},
            },
        ),
        llm_client=llm
        or _FakeClient(
            "llm",
            result={
                "content": "因为询价。",
                "model_id": "stub-small",
                "provider": "stub",
                "confidence": 0.9,
            },
        ),
        report_client=report
        or _FakeClient(
            "report",
            result={
                "report_id": "rpt-1",
                "format": "markdown",
                "audience": "technical",
                "template_id": "technical_detail",
                "language": "zh-CN",
                "content": "# report",
                "content_type": "text/markdown",
                "size_bytes": 12,
                "checksum": "abc",
                "render_latency_ms": 0.1,
                "storage_results": [],
                "cached": False,
            },
        ),
    )


@pytest.mark.asyncio
async def test_pipeline_happy_path(sample_b2b_request):
    orch = _build()
    resp = await orch.run(sample_b2b_request)
    assert resp.overall_status == "success"
    assert resp.stage_sequence == ["intent", "decision", "report"]
    assert resp.intent.status == "success"
    assert resp.decision.status == "success"
    assert resp.report.status == "success"
    assert resp.llm is None  # include_llm=False in sample


@pytest.mark.asyncio
async def test_pipeline_runs_llm_when_opted_in(sample_b2b_request):
    sample_b2b_request.options = PipelineOptions(include_llm=True)
    orch = _build()
    resp = await orch.run(sample_b2b_request)
    assert resp.llm is not None and resp.llm.status == "success"
    assert "llm" in resp.stage_sequence


@pytest.mark.asyncio
async def test_pipeline_passes_intent_into_decision(sample_b2b_request):
    intent_client = _FakeClient(
        "intent",
        result={"top_intent": "request_quote", "top_confidence": 0.95, "top_label": "请求报价"},
    )
    decision_client = _FakeClient(
        "decision",
        result={
            "action": "request_quote",
            "confidence": 0.9,
            "risk_score": 0.05,
            "requires_review": False,
            "rationale": "b2b 请求报价",
            "next_steps": ["RFQ"],
        },
    )
    orch = _build(intent=intent_client, decision=decision_client)
    await orch.run(sample_b2b_request)

    assert len(decision_client.calls) == 1
    payload = decision_client.calls[0]
    assert payload["intent"]["intent"] == "request_quote"
    assert payload["intent"]["confidence"] == 0.95
    assert payload["buyer"]["user_type"] == "b2b"
    assert payload["product"]["unit_price"] == 500.0


@pytest.mark.asyncio
async def test_pipeline_skips_decision_when_intent_fails(sample_b2b_request):
    intent_client = _FakeClient("intent", fail=True)
    decision_client = _FakeClient("decision", result={})
    orch = _build(intent=intent_client, decision=decision_client)
    resp = await orch.run(sample_b2b_request)
    assert resp.intent.status == "failed"
    assert resp.decision.status == "skipped"
    assert decision_client.calls == []


@pytest.mark.asyncio
async def test_pipeline_report_still_runs_when_llm_disabled(sample_b2b_request):
    orch = _build()
    resp = await orch.run(sample_b2b_request)
    assert resp.report is not None and resp.report.status == "success"


@pytest.mark.asyncio
async def test_pipeline_partial_status_when_report_fails(sample_b2b_request):
    report_client = _FakeClient("report", fail=True)
    orch = _build(report=report_client)
    resp = await orch.run(sample_b2b_request)
    assert resp.overall_status == "partial"
    assert resp.report.status == "failed"


@pytest.mark.asyncio
async def test_pipeline_fails_when_everything_fails(sample_b2b_request):
    sample_b2b_request.options = PipelineOptions(include_llm=True)
    orch = _build(
        intent=_FakeClient("intent", fail=True),
        decision=_FakeClient("decision", fail=True),
        llm=_FakeClient("llm", fail=True),
        report=_FakeClient("report", fail=True),
    )
    resp = await orch.run(sample_b2b_request)
    assert resp.overall_status == "failed"
    assert resp.intent.status == "failed"
    assert resp.decision.status == "skipped"  # intent failed first
    assert resp.llm.status == "failed"
    assert resp.report.status == "failed"


@pytest.mark.asyncio
async def test_pipeline_respects_include_report_false(sample_b2b_request):
    sample_b2b_request.options = PipelineOptions(include_report=False)
    report_client = _FakeClient("report", result={})
    orch = _build(report=report_client)
    resp = await orch.run(sample_b2b_request)
    assert resp.report is None
    assert report_client.calls == []


@pytest.mark.asyncio
async def test_pipeline_forwards_llm_narrative_into_report(sample_b2b_request):
    sample_b2b_request.options = PipelineOptions(include_llm=True, include_report=True)
    llm_client = _FakeClient(
        "llm",
        result={"content": "narrative", "model_id": "m1", "provider": "stub", "confidence": 0.7},
    )
    report_client = _FakeClient(
        "report",
        result={
            "report_id": "x",
            "format": "markdown",
            "audience": "technical",
            "template_id": "technical_detail",
            "language": "zh-CN",
            "content": "#",
            "content_type": "text/markdown",
            "size_bytes": 1,
            "checksum": "x",
            "render_latency_ms": 0.0,
            "storage_results": [],
            "cached": False,
        },
    )
    orch = _build(llm=llm_client, report=report_client)
    await orch.run(sample_b2b_request)
    assert len(report_client.calls) == 1
    assert report_client.calls[0]["llm"]["content"] == "narrative"


@pytest.mark.asyncio
async def test_pipeline_populates_stage_sequence(sample_b2b_request):
    sample_b2b_request.options = PipelineOptions(include_llm=True, include_report=True)
    orch = _build()
    resp = await orch.run(sample_b2b_request)
    assert resp.stage_sequence == ["intent", "decision", "llm", "report"]


@pytest.mark.asyncio
async def test_pipeline_records_warnings_on_failure(sample_b2b_request):
    orch = _build(llm=_FakeClient("llm", fail=True), report=_FakeClient("report", fail=True))
    sample_b2b_request.options = PipelineOptions(include_llm=True, include_report=True)
    resp = await orch.run(sample_b2b_request)
    assert any("llm failed" in w for w in resp.warnings)
    assert any("report failed" in w for w in resp.warnings)
