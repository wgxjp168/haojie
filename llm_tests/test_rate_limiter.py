import pytest

from llm.core.exceptions import RateLimitedError
from llm.rate_limiter import RateLimiterRegistry, TokenBucket


def test_token_bucket_starts_full():
    tb = TokenBucket(rate_per_minute=60, burst=5)
    for _ in range(5):
        assert tb.try_acquire() is True
    # 6th should be throttled because refill hasn't happened yet
    assert tb.try_acquire() is False


def test_rate_zero_disables_limiting():
    tb = TokenBucket(rate_per_minute=0, burst=0)
    for _ in range(100):
        assert tb.try_acquire() is True


def test_registry_separates_providers():
    reg = RateLimiterRegistry(rate_per_minute=60, burst=1)
    assert reg.get("openai").try_acquire() is True
    # second openai call hits the cap
    assert reg.get("openai").try_acquire() is False
    # anthropic has its own bucket
    assert reg.get("anthropic").try_acquire() is True


def test_ensure_allowed_raises_when_throttled():
    reg = RateLimiterRegistry(rate_per_minute=60, burst=1)
    reg.ensure_allowed("openai")  # consume the single token
    with pytest.raises(RateLimitedError):
        reg.ensure_allowed("openai")
