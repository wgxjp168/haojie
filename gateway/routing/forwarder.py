"""Forwards requests to upstream services with retry / circuit-breaker.

All calls go through a single httpx.AsyncClient; each logical upstream
(addressed by service name) is fronted by a circuit breaker so a flapping
backend doesn't take the gateway down.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import httpx

from gateway.auth.user_types import GatewayPrincipal
from gateway.circuit_breaker.breaker import (
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
)
from gateway.config import GatewaySettings, get_gateway_settings
from gateway.core.exceptions import (
    BadGatewayError,
    CircuitOpenError,
    UpstreamTimeoutError,
)
from gateway.core.logging import get_logger
from gateway.core.metrics import GW_UPSTREAM_DURATION, GW_UPSTREAM_REQUESTS_TOTAL

logger = get_logger(__name__)


class UpstreamForwarder:
    def __init__(
        self,
        settings: Optional[GatewaySettings] = None,
        breakers: Optional[CircuitBreakerRegistry] = None,
    ) -> None:
        self._settings = settings or get_gateway_settings()
        self._breakers = breakers or get_circuit_breaker_registry()
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self._client is None:
            timeout = httpx.Timeout(
                timeout=self._settings.upstream_timeout_seconds,
                connect=self._settings.upstream_connect_timeout_seconds,
            )
            self._client = httpx.AsyncClient(
                base_url=self._settings.upstream_base_url,
                timeout=timeout,
                follow_redirects=False,
                http2=False,
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def forward(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        json: Any = None,
        service_name: str = "input-processor",
        principal: Optional[GatewayPrincipal] = None,
    ) -> httpx.Response:
        """Send the request to the upstream service.

        Raises :class:`GatewayError` subclasses for all failure modes so the
        FastAPI exception handler can serialise them uniformly.
        """
        if self._client is None:
            await self.start()
        assert self._client is not None

        breaker = await self._breakers.get(service_name)
        if not await breaker.allow():
            raise CircuitOpenError(service_name)

        forward_headers = self._build_forward_headers(headers or {}, principal)

        last_exc: Optional[Exception] = None
        attempts = self._settings.upstream_max_retries + 1
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                response = await self._client.request(
                    method=method,
                    url=path,
                    headers=forward_headers,
                    params=params,
                    content=content,
                    json=json,
                )
                duration = time.perf_counter() - start
                GW_UPSTREAM_DURATION.labels(method=method).observe(duration)
                GW_UPSTREAM_REQUESTS_TOTAL.labels(
                    status=str(response.status_code), method=method
                ).inc()
                # 5xx is a failure signal for the breaker; 4xx is not.
                if response.status_code >= 500:
                    await breaker.record_failure()
                    if attempt < attempts - 1:
                        continue
                else:
                    await breaker.record_success()
                return response
            except httpx.TimeoutException as e:
                last_exc = e
                await breaker.record_failure()
                GW_UPSTREAM_REQUESTS_TOTAL.labels(
                    status="timeout", method=method
                ).inc()
                if attempt >= attempts - 1:
                    raise UpstreamTimeoutError(
                        self._settings.upstream_timeout_seconds
                    ) from e
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.HTTPError) as e:
                last_exc = e
                await breaker.record_failure()
                GW_UPSTREAM_REQUESTS_TOTAL.labels(
                    status="error", method=method
                ).inc()
                if attempt >= attempts - 1:
                    raise BadGatewayError(
                        "Upstream connection failed", error=str(e)
                    ) from e

        # Should be unreachable — defensive.
        raise BadGatewayError(
            "Upstream retries exhausted", error=str(last_exc) if last_exc else ""
        )

    # --- helpers ---

    @staticmethod
    def _build_forward_headers(
        headers: Dict[str, str], principal: Optional[GatewayPrincipal]
    ) -> Dict[str, str]:
        """Filter hop-by-hop headers and inject gateway trust headers."""
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
            "host",
            "content-length",  # httpx will set this correctly itself
        }
        forward = {
            k: v for k, v in headers.items() if k.lower() not in hop_by_hop
        }
        forward["X-Forwarded-By"] = "ilbuyai-gateway"
        if principal and not principal.is_anonymous():
            forward["X-User-Id"] = principal.user_id
            forward["X-User-Type"] = principal.user_type.value
            if principal.roles:
                forward["X-User-Roles"] = ",".join(r.value for r in principal.roles)
            if principal.username:
                forward["X-User-Name"] = principal.username
        return forward


_forwarder: Optional[UpstreamForwarder] = None


def get_forwarder() -> UpstreamForwarder:
    global _forwarder
    if _forwarder is None:
        _forwarder = UpstreamForwarder()
    return _forwarder


def reset_forwarder() -> None:
    global _forwarder
    _forwarder = None
