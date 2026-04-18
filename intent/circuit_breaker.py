"""Minimal synchronous circuit breaker around the transformer backend.

We don't re-use the gateway circuit breaker because that one is async and
also tracks per-service state; here we only need a single-instance, simple
state machine.
"""
from __future__ import annotations

import enum
import threading
import time
from typing import Optional

from intent.core.exceptions import CircuitOpenError
from intent.core.metrics import INTENT_CIRCUIT_STATE


class CircuitState(enum.Enum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


_STATE_LABEL = {
    CircuitState.CLOSED: "closed",
    CircuitState.HALF_OPEN: "half_open",
    CircuitState.OPEN: "open",
}


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        name: str = "intent_transformer",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = float(recovery_seconds)
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()
        INTENT_CIRCUIT_STATE.set(self._state.value)

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
            # HALF_OPEN: allow one probe.
            return True

    def ensure_allowed(self) -> None:
        if not self.allow():
            raise CircuitOpenError(
                f"circuit {self.name!r} is open",
                details={"state": self.state_label},
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
        INTENT_CIRCUIT_STATE.set(new_state.value)
