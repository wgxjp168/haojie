"""Embedded-mode tests for every stage client.

We drive the downstream services via their in-process ``Service``
singletons — no network involved, but exercising the full request/
response validation paths.
"""
from __future__ import annotations

import pytest

from hub.clients import DecisionClient, IntentClient, LLMClient, ReportClient
from hub.config import ClientMode


def _client_kwargs():
    return dict(
        mode=ClientMode.EMBEDDED,
        base_url="http://unused",
        timeout_seconds=5.0,
        max_retries=0,
        retry_backoff_seconds=0.0,
        failure_threshold=3,
        recovery_seconds=1,
    )


@pytest.mark.asyncio
async def test_intent_client_embedded():
    client = IntentClient(**_client_kwargs())
    result = await client.call(
        {"text": "这个多少钱？", "language": "zh-CN", "user_type": "b2c"}
    )
    assert result.success is True
    assert result.mode == "embedded"
    assert result.result["top_intent"] == "inquire_price"


@pytest.mark.asyncio
async def test_intent_client_embedded_rejects_empty_text():
    client = IntentClient(**_client_kwargs())
    result = await client.call({"text": ""})
    assert result.success is False
    assert result.error_code in {"UPSTREAM_FAILED", "INTERNAL_ERROR"}


@pytest.mark.asyncio
async def test_decision_client_embedded():
    client = DecisionClient(**_client_kwargs())
    result = await client.call(
        {
            "intent": {
                "intent": "request_quote",
                "confidence": 0.9,
                "label": "请求报价",
            },
            "buyer": {"user_type": "b2b"},
            "product": {"unit_price": 500.0, "quantity": 50, "in_stock": True},
        }
    )
    assert result.success is True
    assert result.result["action"] == "request_quote"


@pytest.mark.asyncio
async def test_llm_client_embedded():
    client = LLMClient(**_client_kwargs())
    result = await client.call(
        {
            "messages": [
                {"role": "user", "content": "Explain the decision briefly."}
            ],
            "model_id": "stub-small",
            "strategy": "single",
        }
    )
    assert result.success is True
    assert result.result["provider"] == "stub"


@pytest.mark.asyncio
async def test_llm_client_embedded_rejects_missing_prompt():
    client = LLMClient(**_client_kwargs())
    result = await client.call({})
    assert result.success is False


@pytest.mark.asyncio
async def test_report_client_embedded():
    client = ReportClient(**_client_kwargs())
    result = await client.call(
        {
            "request_id": "test-report-1",
            "format": "markdown",
            "audience": "technical",
            "language": "zh-CN",
            "intent": {"intent": "inquire_price", "confidence": 0.9},
            "decision": {
                "action": "request_quote",
                "confidence": 0.85,
                "risk_score": 0.1,
                "requires_review": False,
                "rationale": "B 端询价",
                "next_steps": ["生成 RFQ"],
            },
        }
    )
    assert result.success is True
    assert result.result["format"] == "markdown"
    assert "采购决策报告" in result.result["content"]


@pytest.mark.asyncio
async def test_report_client_embedded_requires_something_to_render():
    client = ReportClient(**_client_kwargs())
    result = await client.call({"format": "markdown"})
    assert result.success is False
