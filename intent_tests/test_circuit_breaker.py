import time

import pytest

from intent.circuit_breaker import CircuitBreaker, CircuitState
from intent.core.exceptions import CircuitOpenError


def test_circuit_closed_allows_by_default():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=1)
    assert cb.allow() is True
    assert cb.state == CircuitState.CLOSED


def test_circuit_opens_after_threshold_failures():
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=10)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow() is False
    with pytest.raises(CircuitOpenError):
        cb.ensure_allowed()


def test_circuit_transitions_to_half_open_after_recovery():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    time.sleep(0.07)
    assert cb.allow() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes_circuit():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
    cb.record_failure()
    time.sleep(0.02)
    assert cb.allow() is True  # moves to half-open
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens_circuit():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
    cb.record_failure()
    time.sleep(0.02)
    assert cb.allow() is True  # half-open
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_reset_clears_state():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=10)
    cb.record_failure()
    cb.record_failure()
    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow() is True


def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0, recovery_seconds=1)
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=1, recovery_seconds=0)
