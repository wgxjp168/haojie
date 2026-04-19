"""Shared fixtures + env setup for report-service tests."""
from __future__ import annotations

import os
import tempfile

# Force deterministic settings *before* any report module is imported.
os.environ.setdefault("REPORT_ENV", "testing")
os.environ.setdefault("REPORT_DEFAULT_FORMAT", "markdown")
os.environ.setdefault("REPORT_DEFAULT_AUDIENCE", "technical")
os.environ.setdefault("REPORT_STORAGE_BACKENDS", "memory")
os.environ.setdefault("REPORT_CACHE_ENABLED", "true")
os.environ.setdefault("REPORT_REDIS_URL", "")
os.environ.setdefault("REPORT_API_KEY_REQUIRED", "false")
os.environ.setdefault("REPORT_METRICS_ENABLED", "true")
os.environ.setdefault("REPORT_MEMORY_CACHE_SIZE", "256")
os.environ.setdefault("REPORT_MAX_BATCH_SIZE", "16")

import pytest  # noqa: E402

from report.config import reset_report_settings  # noqa: E402
from report.service import reset_report_service  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_report_singletons():
    yield
    reset_report_settings()
    reset_report_service()


@pytest.fixture
def fresh_service():
    reset_report_settings()
    reset_report_service()
    from report.service import ReportService

    return ReportService()


@pytest.fixture
def sample_request():
    """Well-formed ReportRequest used as a baseline across tests."""
    from report.schemas import (
        BuyerSummary,
        DecisionCandidateSlice,
        DecisionSlice,
        IntentSlice,
        LLMSlice,
        ProductSummary,
        ReportRequest,
    )

    return ReportRequest(
        request_id="sample-req-001",
        format="markdown",
        audience="technical",
        language="zh-CN",
        buyer=BuyerSummary(user_id="u123", user_type="b2b", budget=100_000.0),
        product=ProductSummary(
            product_id="P100", name="办公椅", unit_price=500.0, quantity=50
        ),
        intent=IntentSlice(
            intent="request_quote",
            label="请求报价",
            confidence=0.91,
            backend="rules",
        ),
        decision=DecisionSlice(
            action="request_quote",
            label="请求报价",
            confidence=0.88,
            confident=True,
            risk_score=0.15,
            requires_review=False,
            rationale="B 端采购员请求报价",
            next_steps=["生成 RFQ", "发送给采购员审核"],
            candidates=[
                DecisionCandidateSlice(
                    action="request_quote", label="请求报价", confidence=0.88
                )
            ],
            strategy="rules",
            matched_rules=[
                {"name": "b2b_request_quote", "action": "request_quote"}
            ],
        ),
        llm=LLMSlice(
            content="建议请求正式报价。",
            model_id="stub-small",
            provider="stub",
            confidence=0.83,
        ),
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d
