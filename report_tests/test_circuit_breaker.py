import time

import pytest

from report.circuit_breaker import CircuitState, StorageCircuitBreaker, StorageCircuitRegistry
from report.core.exceptions import CircuitOpenError


def test_initial_state_closed():
    cb = StorageCircuitBreaker(backend="s3", failure_threshold=2, recovery_seconds=10)
    assert cb.state == CircuitState.CLOSED
    assert cb.allow() is True


def test_opens_after_failures():
    cb = StorageCircuitBreaker(backend="s3", failure_threshold=2, recovery_seconds=10)
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_blocks_until_recovery():
    cb = StorageCircuitBreaker(backend="s3", failure_threshold=1, recovery_seconds=0.01)
    cb.record_failure()
    assert cb.allow() is False
    time.sleep(0.02)
    assert cb.allow() is True


def test_ensure_allowed_raises():
    cb = StorageCircuitBreaker(backend="s3", failure_threshold=1, recovery_seconds=10)
    cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.ensure_allowed()


def test_registry_returns_same_instance():
    reg = StorageCircuitRegistry()
    a = reg.get("local")
    b = reg.get("local")
    assert a is b


def test_registry_tracks_per_backend():
    reg = StorageCircuitRegistry(failure_threshold=1, recovery_seconds=10)
    reg.get("local").record_failure()
    assert reg.all_states()["local"] == "open"
    assert reg.get("s3").state_label == "closed"


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        StorageCircuitBreaker(backend="x", failure_threshold=0)
    with pytest.raises(ValueError):
        StorageCircuitBreaker(backend="x", recovery_seconds=0)
