import pytest

from decision.circuit_breaker import CircuitBreaker, CircuitState
from decision.core.exceptions import CircuitOpenError


def test_initial_state_closed():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=10)
    assert cb.state == CircuitState.CLOSED
    assert cb.allow() is True


def test_opens_after_threshold_failures():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=10)
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_open_blocks_calls_until_recovery():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
    cb.record_failure()
    assert cb.allow() is False
    import time
    time.sleep(0.02)
    assert cb.allow() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
    cb.record_failure()
    import time
    time.sleep(0.02)
    cb.allow()  # transition to HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
    cb.record_failure()
    import time
    time.sleep(0.02)
    cb.allow()  # transition to HALF_OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_ensure_allowed_raises_when_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=10)
    cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.ensure_allowed()


def test_reset_returns_to_closed():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=10)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    cb.reset()
    assert cb.state == CircuitState.CLOSED


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_seconds=0)


def test_state_label_strings():
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=10)
    assert cb.state_label == "closed"
    cb.record_failure()
    assert cb.state_label == "open"
