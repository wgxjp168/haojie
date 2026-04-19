"""Per-provider circuit breakers for the LLM service.

The LLM service maintains one breaker *per provider* (``openai``,
``anthropic``, ...) rather than a single global one, so a broken
OpenAI deployment does not take down Anthropic traffic. State is pushed
to ``LLM_CIRCUIT_STATE{provider=...}`` on every transition.
"""
from __future__ import annotations

import enum
import threading
import time
from typing import Dict, Optional

from llm.core.exceptions import CircuitOpenError
from llm.core.metrics import LLM_CIRCUIT_STATE


class CircuitState(enum.Enum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


_STATE_LABEL = {
    CircuitState.CLOSED: "closed",
    CircuitState.HALF_OPEN: "half_open",
    CircuitState.OPEN: "open",
}


class ProviderCircuitBreaker:
    """Single-provider circuit breaker."""

    def __init__(
        self,
        *,
        provider: str,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be > 0")
        self.provider = provider
        self.failure_threshold = failure_threshold
        self.recovery_seconds = float(recovery_seconds)
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()
        LLM_CIRCUIT_STATE.labels(provider=provider).set(self._state.value)

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
                f"circuit for provider {self.provider!r} is open",
                details={"provider": self.provider, "state": self.state_label},
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
        LLM_CIRCUIT_STATE.labels(provider=self.provider).set(new_state.value)


class CircuitBreakerRegistry:
    """Thread-safe registry of per-provider breakers."""

    def __init__(
        self, *, failure_threshold: int = 5, recovery_seconds: float = 30.0
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._breakers: Dict[str, ProviderCircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, provider: str) -> ProviderCircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(provider)
            if breaker is None:
                breaker = ProviderCircuitBreaker(
                    provider=provider,
                    failure_threshold=self._failure_threshold,
                    recovery_seconds=self._recovery_seconds,
                )
                self._breakers[provider] = breaker
            return breaker

    def all_states(self) -> Dict[str, str]:
        with self._lock:
            return {p: b.state_label for p, b in self._breakers.items()}

    def reset_all(self) -> None:
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
