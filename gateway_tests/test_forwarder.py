"""Upstream forwarder tests using httpx MockTransport."""
from __future__ import annotations

import httpx
import pytest

from gateway.auth.user_types import GatewayPrincipal, GatewayRole, GatewayUserType
from gateway.circuit_breaker.breaker import CircuitBreakerRegistry, CircuitState
from gateway.config import get_gateway_settings
from gateway.core.exceptions import BadGatewayError, CircuitOpenError
from gateway.routing.forwarder import UpstreamForwarder


def _mock_client(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(
        transport=transport, base_url="http://upstream.invalid"
    )


@pytest.mark.asyncio
async def test_forwarder_success_and_header_propagation():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["x_user_id"] = request.headers.get("x-user-id")
        return httpx.Response(200, json={"ok": True})

    registry = CircuitBreakerRegistry(get_gateway_settings())
    fwd = UpstreamForwarder(breakers=registry)
    fwd._client = _mock_client(handler)

    principal = GatewayPrincipal(
        user_id="u1",
        user_type=GatewayUserType.B2B_PURCHASER,
        roles=[GatewayRole.B2B_PURCHASER],
        username="alice",
    )
    resp = await fwd.forward(
        "POST",
        "/api/v1/inputs/text",
        headers={"content-type": "application/json"},
        json={"text": "hello"},
        principal=principal,
    )
    assert resp.status_code == 200
    assert captured["path"] == "/api/v1/inputs/text"
    assert captured["method"] == "POST"
    assert captured["x_user_id"] == "u1"


@pytest.mark.asyncio
async def test_forwarder_retries_5xx():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(502)
        return httpx.Response(200, json={"ok": True})

    registry = CircuitBreakerRegistry(get_gateway_settings())
    fwd = UpstreamForwarder(breakers=registry)
    fwd._client = _mock_client(handler)

    resp = await fwd.forward("GET", "/anything")
    assert resp.status_code == 200
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_forwarder_opens_circuit_after_repeated_failures():
    def always_fail(request):
        return httpx.Response(500, json={"oops": True})

    registry = CircuitBreakerRegistry(get_gateway_settings())
    fwd = UpstreamForwarder(breakers=registry)
    fwd._client = _mock_client(always_fail)

    # Settings default failure_threshold=5; repeated calls should trip it.
    for _ in range(10):
        try:
            await fwd.forward("GET", "/boom")
        except (BadGatewayError, CircuitOpenError):
            pass

    breaker = await registry.get("input-processor")
    assert breaker.state in {CircuitState.OPEN, CircuitState.HALF_OPEN}


@pytest.mark.asyncio
async def test_forwarder_raises_when_circuit_open():
    registry = CircuitBreakerRegistry(get_gateway_settings())
    breaker = await registry.get("input-processor")
    # Force OPEN
    for _ in range(10):
        await breaker.record_failure()

    fwd = UpstreamForwarder(breakers=registry)
    fwd._client = _mock_client(lambda r: httpx.Response(200))

    with pytest.raises(CircuitOpenError):
        await fwd.forward("GET", "/whatever")
