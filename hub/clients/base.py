"""Abstract base for dual-mode stage clients.

A *stage client* knows how to talk to one downstream Part 3 service —
intent / decision / llm / report — in one of two modes:

* ``embedded`` — instantiate the downstream ``Service`` class in this
  Python process and invoke it directly. Zero-network, deterministic,
  and perfect for tests and monolithic deployments.
* ``http`` — hit the remote FastAPI instance via ``httpx.AsyncClient``
  with retries + exponential backoff + per-stage circuit breaker.

Every client exposes the *same async Python API* in both modes; the
pipeline orchestrator never has to care which is active.
"""
from __future__ import annotations

import abc
import asyncio
import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dep for http mode only
    httpx = None  # type: ignore

from hub.config import ClientMode
from hub.core.exceptions import (
    UpstreamFailedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from hub.core.logging import get_logger
from hub.core.metrics import (
    HUB_CIRCUIT_STATE,
    HUB_CLIENT_MODE,
    HUB_STAGE_DURATION,
    HUB_STAGE_RETRIES,
    HUB_STAGE_TOTAL,
)

log = get_logger(__name__)


class _CircuitState(enum.Enum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


_LABEL = {
    _CircuitState.CLOSED: "closed",
    _CircuitState.HALF_OPEN: "half_open",
    _CircuitState.OPEN: "open",
}


class _StageCircuitBreaker:
    """Thread-safe breaker shared by one stage's HTTP mode."""

    def __init__(
        self, *, stage: str, failure_threshold: int, recovery_seconds: float
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        self.stage = stage
        self.failure_threshold = failure_threshold
        self.recovery_seconds = float(recovery_seconds)
        self._state = _CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()
        HUB_CIRCUIT_STATE.labels(stage=stage).set(self._state.value)

    @property
    def state_label(self) -> str:
        with self._lock:
            return _LABEL[self._state]

    def allow(self) -> bool:
        with self._lock:
            if self._state == _CircuitState.CLOSED:
                return True
            if self._state == _CircuitState.OPEN:
                assert self._opened_at is not None
                if time.monotonic() - self._opened_at >= self.recovery_seconds:
                    self._transition(_CircuitState.HALF_OPEN)
                    return True
                return False
            return True  # HALF_OPEN — allow one probe

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state != _CircuitState.CLOSED:
                self._transition(_CircuitState.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == _CircuitState.HALF_OPEN:
                self._transition(_CircuitState.OPEN)
                self._opened_at = time.monotonic()
            elif self._failures >= self.failure_threshold:
                self._transition(_CircuitState.OPEN)
                self._opened_at = time.monotonic()

    def _transition(self, new_state: _CircuitState) -> None:
        self._state = new_state
        HUB_CIRCUIT_STATE.labels(stage=self.stage).set(new_state.value)


@dataclass
class StageCallResult:
    """Outcome of one stage call — success or failure."""

    success: bool
    latency_ms: float
    mode: str
    result: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStageClient(abc.ABC):
    """Base class every concrete stage client inherits from."""

    #: Stage identifier used by metrics and logs (``intent``/``decision``/...).
    stage: str = "base"

    def __init__(
        self,
        *,
        mode: ClientMode,
        base_url: str,
        timeout_seconds: float,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.2,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.api_key = api_key
        self._breaker = _StageCircuitBreaker(
            stage=self.stage,
            failure_threshold=failure_threshold,
            recovery_seconds=recovery_seconds,
        )
        #: Optional httpx ``Transport`` — when set, ``_send_http`` uses it
        #: instead of opening a real network connection. Used by tests to
        #: inject ``httpx.MockTransport``.
        self._transport = None  # type: ignore[assignment]
        HUB_CLIENT_MODE.labels(stage=self.stage).set(
            1 if mode == ClientMode.EMBEDDED else 2
        )

    # ------------------------------------------------------------------
    # Public API — the orchestrator only calls ``call``.

    async def call(self, payload: Dict[str, Any]) -> StageCallResult:
        start = time.perf_counter()
        try:
            if self.mode == ClientMode.EMBEDDED:
                result = await self._call_embedded(payload)
            else:
                result = await self._call_http_with_retries(payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            HUB_STAGE_TOTAL.labels(stage=self.stage, status="success").inc()
            HUB_STAGE_DURATION.labels(
                stage=self.stage, mode=self.mode.value
            ).observe(latency_ms / 1000.0)
            return StageCallResult(
                success=True,
                latency_ms=round(latency_ms, 3),
                mode=self.mode.value,
                result=result,
            )
        except (
            UpstreamTimeoutError,
            UpstreamFailedError,
            UpstreamUnavailableError,
        ) as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            HUB_STAGE_TOTAL.labels(stage=self.stage, status="failed").inc()
            HUB_STAGE_DURATION.labels(
                stage=self.stage, mode=self.mode.value
            ).observe(latency_ms / 1000.0)
            return StageCallResult(
                success=False,
                latency_ms=round(latency_ms, 3),
                mode=self.mode.value,
                error_code=exc.code.value,
                error_message=exc.message,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            HUB_STAGE_TOTAL.labels(stage=self.stage, status="failed").inc()
            HUB_STAGE_DURATION.labels(
                stage=self.stage, mode=self.mode.value
            ).observe(latency_ms / 1000.0)
            log.error(
                "hub.client.unhandled",
                stage=self.stage,
                error=str(exc),
            )
            return StageCallResult(
                success=False,
                latency_ms=round(latency_ms, 3),
                mode=self.mode.value,
                error_code="INTERNAL_ERROR",
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Abstract methods.

    @abc.abstractmethod
    async def _call_embedded(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the downstream service in-process. Raise ``HubError`` on failure."""

    @property
    @abc.abstractmethod
    def _http_path(self) -> str:
        """Path appended to ``base_url`` in http mode (e.g. ``/api/v1/...``)."""

    # ------------------------------------------------------------------
    # HTTP mode with retry + circuit breaker.

    async def _call_http_with_retries(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        if httpx is None:  # pragma: no cover - requires httpx missing
            raise UpstreamUnavailableError(
                f"{self.stage} http mode requires httpx",
                details={"stage": self.stage},
            )
        if not self._breaker.allow():
            raise UpstreamUnavailableError(
                f"{self.stage} circuit open",
                details={"stage": self.stage, "state": self._breaker.state_label},
            )

        last_exc: Optional[Exception] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            if attempt > 0:
                await asyncio.sleep(
                    self.retry_backoff_seconds * (2 ** (attempt - 1))
                )
                HUB_STAGE_RETRIES.labels(stage=self.stage, outcome="success").inc()
            try:
                result = await self._send_http(payload)
                self._breaker.record_success()
                return result
            except (UpstreamTimeoutError, UpstreamFailedError) as exc:
                last_exc = exc
                self._breaker.record_failure()
                # Retry on timeouts and 5xx; stop immediately on 4xx.
                if isinstance(exc, UpstreamFailedError):
                    status = exc.details.get("status_code")
                    if isinstance(status, int) and 400 <= status < 500:
                        raise
                continue
            except UpstreamUnavailableError:
                raise

        HUB_STAGE_RETRIES.labels(stage=self.stage, outcome="exhausted").inc()
        assert last_exc is not None
        raise last_exc

    async def _send_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert httpx is not None
        url = f"{self.base_url}{self._http_path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        client_kwargs = {"timeout": self.timeout_seconds}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        async with httpx.AsyncClient(**client_kwargs) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                raise UpstreamTimeoutError(
                    f"{self.stage} timed out after {self.timeout_seconds}s",
                    details={"stage": self.stage, "url": url},
                ) from exc
            except httpx.HTTPError as exc:  # pragma: no cover - network errors
                raise UpstreamFailedError(
                    f"{self.stage} http error: {exc}",
                    details={"stage": self.stage, "url": url},
                ) from exc

        if resp.status_code >= 500:
            raise UpstreamFailedError(
                f"{self.stage} returned HTTP {resp.status_code}",
                details={
                    "stage": self.stage,
                    "status_code": resp.status_code,
                    "body": resp.text[:500],
                },
            )
        if resp.status_code >= 400:
            raise UpstreamFailedError(
                f"{self.stage} rejected request: HTTP {resp.status_code}",
                details={
                    "stage": self.stage,
                    "status_code": resp.status_code,
                    "body": resp.text[:500],
                },
            )

        try:
            return resp.json()
        except ValueError as exc:  # pragma: no cover - provider bug
            raise UpstreamFailedError(
                f"{self.stage} returned invalid JSON",
                details={"stage": self.stage},
            ) from exc

    # ------------------------------------------------------------------

    async def aclose(self) -> None:  # pragma: no cover - placeholder
        """Release any per-client resources. No-op by default."""
        return None

    @property
    def circuit_state(self) -> str:
        return self._breaker.state_label

    @property
    def available(self) -> bool:
        return self._breaker.state_label != "open"
