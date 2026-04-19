"""Per-backend circuit breaker for storage writes.

Mirrors ``llm.circuit_breaker`` so operators see the same shape across
services. One breaker per backend (``memory``, ``local``, ``s3``...) so a
flaky S3 endpoint does not pull down the local/memory fan-out.
"""
from __future__ import annotations

import enum
import threading
import time
from typing import Dict, Optional

from report.core.exceptions import CircuitOpenError
from report.core.metrics import REPORT_CIRCUIT_STATE


class CircuitState(enum.Enum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


_STATE_LABEL = {
    CircuitState.CLOSED: "closed",
    CircuitState.HALF_OPEN: "half_open",
    CircuitState.OPEN: "open",
}


class StorageCircuitBreaker:
    def __init__(
        self,
        *,
        backend: str,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        self.backend = backend
        self.failure_threshold = failure_threshold
        self.recovery_seconds = float(recovery_seconds)
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()
        REPORT_CIRCUIT_STATE.labels(backend=backend).set(self._state.value)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def state_label(self) -> str:
        return _STATE_LABEL[self.state]

    def allow(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None
                if time.monotonic() - self._opened_at >= self.recovery_seconds:
                    self._transition(CircuitState.HALF_OPEN)
                    return True
                return False
            return True  # HALF_OPEN

    def ensure_allowed(self) -> None:
        if not self.allow():
            raise CircuitOpenError(
                f"circuit for backend {self.backend!r} is open",
                details={"backend": self.backend, "state": self.state_label},
            )

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state != CircuitState.CLOSED:
                self._transition(CircuitState.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
                self._opened_at = time.monotonic()
            elif self._failures >= self.failure_threshold:
                self._transition(CircuitState.OPEN)
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._transition(CircuitState.CLOSED)

    def _transition(self, new_state: CircuitState) -> None:
        self._state = new_state
        REPORT_CIRCUIT_STATE.labels(backend=self.backend).set(new_state.value)


class StorageCircuitRegistry:
    """Thread-safe registry of per-backend breakers."""

    def __init__(
        self, *, failure_threshold: int = 5, recovery_seconds: float = 30.0
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._breakers: Dict[str, StorageCircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, backend: str) -> StorageCircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(backend)
            if breaker is None:
                breaker = StorageCircuitBreaker(
                    backend=backend,
                    failure_threshold=self._failure_threshold,
                    recovery_seconds=self._recovery_seconds,
                )
                self._breakers[backend] = breaker
            return breaker

    def all_states(self) -> Dict[str, str]:
        with self._lock:
            return {b: br.state_label for b, br in self._breakers.items()}

    def reset_all(self) -> None:
        with self._lock:
            for b in self._breakers.values():
                b.reset()
