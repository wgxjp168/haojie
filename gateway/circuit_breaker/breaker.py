"""Circuit breaker with exponential-backoff auto recovery.

The breaker transitions between three states:

* ``CLOSED``   — normal operation.
* ``OPEN``     — failures exceeded threshold; reject fast.
* ``HALF_OPEN``— after a cooldown, allow a small number of probe calls.

Each time a breaker opens, the recovery timeout is doubled (capped) so that
flapping services back off.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Dict, Optional

from gateway.config import GatewaySettings, get_gateway_settings
from gateway.core.logging import get_logger
from gateway.core.metrics import GW_CIRCUIT_STATE, GW_CIRCUIT_TRIPS_TOTAL

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


_STATE_VALUE = {
    CircuitState.CLOSED: 0.0,
    CircuitState.OPEN: 1.0,
    CircuitState.HALF_OPEN: 2.0,
}


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_seconds: int = 30,
        half_open_max_calls: int = 3,
        max_recovery_seconds: int = 300,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._base_recovery = recovery_seconds
        self._recovery = recovery_seconds
        self._max_recovery = max_recovery_seconds
        self._half_open_max = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._publish_state()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def allow(self) -> bool:
        """Returns True if a call is allowed to proceed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._transition(CircuitState.HALF_OPEN)
                    self._half_open_calls = 1
                    return True
                return False
            # HALF_OPEN
            if self._half_open_calls < self._half_open_max:
                self._half_open_calls += 1
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Require the full quota of probe successes before closing.
                if self._half_open_calls >= self._half_open_max:
                    self._transition(CircuitState.CLOSED)
                    self._recovery = self._base_recovery  # reset backoff
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def record_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._trip()
                return
            if (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._failure_threshold
            ):
                self._trip()

    async def reset(self) -> None:
        async with self._lock:
            self._transition(CircuitState.CLOSED)
            self._failure_count = 0
            self._recovery = self._base_recovery

    # --- internals ---

    def _trip(self) -> None:
        self._transition(CircuitState.OPEN)
        self._opened_at = time.monotonic()
        # Exponential backoff on repeated trips.
        self._recovery = min(self._max_recovery, self._recovery * 2)
        GW_CIRCUIT_TRIPS_TOTAL.labels(service=self.name).inc()
        logger.warning(
            "circuit.open",
            service=self.name,
            failures=self._failure_count,
            recovery_seconds=self._recovery,
        )

    def _transition(self, new_state: CircuitState) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        self._half_open_calls = 0
        if new_state != CircuitState.OPEN:
            self._opened_at = None
        self._publish_state()
        logger.info(
            "circuit.transition",
            service=self.name,
            from_state=old.value,
            to_state=new_state.value,
        )

    def _publish_state(self) -> None:
        GW_CIRCUIT_STATE.labels(service=self.name).set(_STATE_VALUE[self._state])

    def snapshot(self) -> dict:
        return {
            "service": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "recovery_seconds": self._recovery,
            "opened_at": self._opened_at,
        }


class CircuitBreakerRegistry:
    """Lazy map from service name -> circuit breaker."""

    def __init__(self, settings: Optional[GatewaySettings] = None):
        self._settings = settings or get_gateway_settings()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(self, service: str) -> CircuitBreaker:
        async with self._lock:
            if service not in self._breakers:
                self._breakers[service] = CircuitBreaker(
                    name=service,
                    failure_threshold=self._settings.cb_failure_threshold,
                    recovery_seconds=self._settings.cb_recovery_seconds,
                    half_open_max_calls=self._settings.cb_half_open_max_calls,
                )
            return self._breakers[service]

    def snapshot(self) -> list[dict]:
        return [b.snapshot() for b in self._breakers.values()]


_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


def reset_circuit_breaker_registry() -> None:
    global _registry
    _registry = None
