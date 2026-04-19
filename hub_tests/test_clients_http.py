"""HTTP-mode tests for stage clients using ``httpx.MockTransport``."""
from __future__ import annotations

import pytest

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from hub.clients import IntentClient
from hub.config import ClientMode


pytestmark = pytest.mark.skipif(httpx is None, reason="httpx not installed")


def _http_client(
    *,
    base_url="http://intent.svc",
    max_retries=0,
    failure_threshold=3,
    recovery_seconds=1,
):
    return IntentClient(
        mode=ClientMode.HTTP,
        base_url=base_url,
        timeout_seconds=5.0,
        max_retries=max_retries,
        retry_backoff_seconds=0.0,
        failure_threshold=failure_threshold,
        recovery_seconds=recovery_seconds,
    )


@pytest.mark.asyncio
async def test_http_client_successful_call():
    client = _http_client()
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == "/api/v1/intents/predict"
        return httpx.Response(
            200,
            json={
                "top_intent": "inquire_price",
                "top_confidence": 0.8,
                "backend": "rules",
            },
        )

    client._transport = httpx.MockTransport(handler)
    result = await client.call({"text": "hi"})
    assert result.success is True
    assert result.mode == "http"
    assert result.result["top_intent"] == "inquire_price"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_http_client_retries_on_5xx():
    client = _http_client(max_retries=2)
    state = {"count": 0}

    def handler(request):
        state["count"] += 1
        if state["count"] < 3:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(
            200, json={"top_intent": "x", "top_confidence": 0.1}
        )

    client._transport = httpx.MockTransport(handler)
    result = await client.call({"text": "hi"})
    assert result.success is True
    assert state["count"] == 3


@pytest.mark.asyncio
async def test_http_client_fails_fast_on_4xx():
    client = _http_client(max_retries=3)
    state = {"count": 0}

    def handler(request):
        state["count"] += 1
        return httpx.Response(422, text="bad request")

    client._transport = httpx.MockTransport(handler)
    result = await client.call({"text": "hi"})
    assert result.success is False
    assert result.error_code == "UPSTREAM_FAILED"
    assert state["count"] == 1  # no retry on 4xx


@pytest.mark.asyncio
async def test_http_client_opens_circuit_after_failures():
    client = _http_client(max_retries=0, failure_threshold=2)

    def handler(request):
        return httpx.Response(500, text="boom")

    client._transport = httpx.MockTransport(handler)
    r1 = await client.call({"text": "one"})
    r2 = await client.call({"text": "two"})
    # Breaker should now be open — third call short-circuits.
    r3 = await client.call({"text": "three"})

    assert r1.success is False and r2.success is False
    assert r3.success is False
    assert r3.error_code == "UPSTREAM_UNAVAILABLE"
    assert client.circuit_state == "open"


@pytest.mark.asyncio
async def test_http_client_sends_api_key_header():
    client = _http_client()
    client.api_key = "secret-key-xyz"
    received = {}

    def handler(request):
        received["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(
            200, json={"top_intent": "x", "top_confidence": 0.1}
        )

    client._transport = httpx.MockTransport(handler)
    await client.call({"text": "hi"})
    assert received["api_key"] == "secret-key-xyz"


@pytest.mark.asyncio
async def test_http_client_timeout_returns_upstream_timeout():
    client = _http_client(max_retries=0)

    def handler(request):
        raise httpx.TimeoutException("simulated", request=request)

    client._transport = httpx.MockTransport(handler)
    result = await client.call({"text": "hi"})
    assert result.success is False
    assert result.error_code == "UPSTREAM_TIMEOUT"
