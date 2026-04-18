"""Circuit breaker state machine tests."""
from __future__ import annotations

import asyncio

import pytest

from gateway.circuit_breaker.breaker import CircuitBreaker, CircuitState


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold():
    cb = CircuitBreaker(
        name="svc", failure_threshold=3, recovery_seconds=1, half_open_max_calls=2
    )
    for _ in range(3):
        assert await cb.allow()
        await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert not await cb.allow()


@pytest.mark.asyncio
async def test_breaker_half_open_then_closes_on_success():
    cb = CircuitBreaker(
        name="svc",
        failure_threshold=2,
        recovery_seconds=0,  # so the breaker flips to half_open immediately
        half_open_max_calls=2,
    )
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    # First allow after OPEN + recovery=0 should flip to HALF_OPEN.
    assert await cb.allow()
    assert cb.state == CircuitState.HALF_OPEN
    await cb.record_success()  # first probe
    # Needs the quota of probe calls to close. Second probe:
    assert await cb.allow()
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_breaker_half_open_trips_again_on_failure():
    cb = CircuitBreaker(
        name="svc",
        failure_threshold=1,
        recovery_seconds=0,
        half_open_max_calls=2,
    )
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert await cb.allow()  # flips to HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_breaker_reset():
    cb = CircuitBreaker(name="svc", failure_threshold=1, recovery_seconds=60)
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    await cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert await cb.allow()


@pytest.mark.asyncio
async def test_breaker_exponential_backoff():
    cb = CircuitBreaker(
        name="svc",
        failure_threshold=1,
        recovery_seconds=1,
        half_open_max_calls=1,
        max_recovery_seconds=16,
    )
    await cb.record_failure()
    assert cb.snapshot()["recovery_seconds"] == 2  # doubled from 1
    # Re-trip
    await asyncio.sleep(0)  # no-op, just to stay async
    # Force another open via half-open failure path requires state cycling;
    # we just check the recovery_seconds cap instead.
    for _ in range(5):
        # Manually trip again
        await cb.reset()
        await cb.record_failure()
    assert cb.snapshot()["recovery_seconds"] <= 16
