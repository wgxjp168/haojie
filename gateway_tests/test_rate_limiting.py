"""Token-bucket and user-type strategy tests."""
from __future__ import annotations

import pytest

from gateway.auth.user_types import GatewayPrincipal, GatewayUserType
from gateway.rate_limiting.strategies import (
    DefaultRateLimitStrategy,
    RateLimiter,
)
from gateway.rate_limiting.token_bucket import InMemoryTokenBucketStore, TokenBucket


@pytest.mark.asyncio
async def test_token_bucket_allows_up_to_capacity():
    bucket = TokenBucket(
        InMemoryTokenBucketStore(), capacity=5, refill_per_second=0.0001
    )
    for _ in range(5):
        assert (await bucket.try_consume("k")).allowed is True
    assert (await bucket.try_consume("k")).allowed is False


@pytest.mark.asyncio
async def test_token_bucket_refills_over_time():
    # 10/sec => 1 token every 100ms. capacity=1 means we alternate allow/deny.
    bucket = TokenBucket(
        InMemoryTokenBucketStore(), capacity=1, refill_per_second=10.0
    )
    assert (await bucket.try_consume("k")).allowed is True
    assert (await bucket.try_consume("k")).allowed is False
    # Wait enough for a refill.
    import asyncio

    await asyncio.sleep(0.15)
    assert (await bucket.try_consume("k")).allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_differentiates_by_user_type():
    limiter = RateLimiter(
        strategy=DefaultRateLimitStrategy(),
        store=InMemoryTokenBucketStore(),
    )

    b2c = GatewayPrincipal(user_id="c1", user_type=GatewayUserType.B2C_CONSUMER)
    b2b = GatewayPrincipal(user_id="b1", user_type=GatewayUserType.B2B_PURCHASER)

    b2c_dec = await limiter.check(b2c, "/api/v1/inputs/text")
    b2b_dec = await limiter.check(b2b, "/api/v1/inputs/text")

    assert b2c_dec.allowed
    assert b2b_dec.allowed
    # B2B bucket should be larger than B2C per defaults.
    assert b2b_dec.limit > b2c_dec.limit


@pytest.mark.asyncio
async def test_rate_limiter_isolates_per_user():
    limiter = RateLimiter(
        strategy=DefaultRateLimitStrategy(),
        store=InMemoryTokenBucketStore(),
    )
    a = GatewayPrincipal(user_id="a", user_type=GatewayUserType.ANONYMOUS)
    b = GatewayPrincipal(user_id="b", user_type=GatewayUserType.ANONYMOUS)

    # Burn anon quota for user a
    dec = await limiter.check(a, "/any")
    cap = dec.limit
    for _ in range(cap + 10):
        await limiter.check(a, "/any")

    # User b is unaffected.
    b_dec = await limiter.check(b, "/any")
    assert b_dec.allowed
