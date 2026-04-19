import time

import pytest

from llm.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
    ProviderCircuitBreaker,
)
from llm.core.exceptions import CircuitOpenError


def test_initial_state_closed():
    cb = ProviderCircuitBreaker(
        provider="x", failure_threshold=2, recovery_seconds=10
    )
    assert cb.state == CircuitState.CLOSED
    assert cb.allow() is True


def test_opens_after_threshold():
    cb = ProviderCircuitBreaker(
        provider="x", failure_threshold=2, recovery_seconds=10
    )
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_blocks_until_recovery():
    cb = ProviderCircuitBreaker(
        provider="x", failure_threshold=1, recovery_seconds=0.01
    )
    cb.record_failure()
    assert cb.allow() is False
    time.sleep(0.02)
    assert cb.allow() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes():
    cb = ProviderCircuitBreaker(
        provider="x", failure_threshold=1, recovery_seconds=0.01
    )
    cb.record_failure()
    time.sleep(0.02)
    cb.allow()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens():
    cb = ProviderCircuitBreaker(
        provider="x", failure_threshold=1, recovery_seconds=0.01
    )
    cb.record_failure()
    time.sleep(0.02)
    cb.allow()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_ensure_allowed_raises_when_open():
    cb = ProviderCircuitBreaker(
        provider="x", failure_threshold=1, recovery_seconds=10
    )
    cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.ensure_allowed()


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        ProviderCircuitBreaker(provider="x", failure_threshold=0)
    with pytest.raises(ValueError):
        ProviderCircuitBreaker(provider="x", recovery_seconds=0)


def test_registry_returns_same_instance():
    reg = CircuitBreakerRegistry(failure_threshold=3, recovery_seconds=10)
    a = reg.get("openai")
    b = reg.get("openai")
    assert a is b


def test_registry_tracks_per_provider():
    reg = CircuitBreakerRegistry(failure_threshold=1, recovery_seconds=10)
    reg.get("openai").record_failure()
    states = reg.all_states()
    assert states.get("openai") == "open"
    assert reg.get("anthropic").state_label == "closed"


def test_registry_reset_all():
    reg = CircuitBreakerRegistry(failure_threshold=1, recovery_seconds=10)
    reg.get("openai").record_failure()
    reg.reset_all()
    assert reg.get("openai").state_label == "closed"
