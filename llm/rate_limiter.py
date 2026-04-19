"""Per-provider token-bucket rate limiter.

Protects both our local CPU budget and the *provider's* own quota. Uses
a simple in-process token bucket per provider; for multi-worker setups
this is a best-effort local cap that complements the provider's own
server-side enforcement.
"""
from __future__ import annotations

import threading
import time
from typing import Dict

from llm.core.exceptions import RateLimitedError
from llm.core.metrics import LLM_RATE_LIMIT_EVENTS


class TokenBucket:
    """Thread-safe token bucket."""

    def __init__(self, *, rate_per_minute: int, burst: int) -> None:
        self.rate_per_minute = max(0, int(rate_per_minute))
        self.capacity = max(0, int(burst))
        # Tokens per second for internal math.
        self._refill_per_second = self.rate_per_minute / 60.0 if self.rate_per_minute else 0.0
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self, tokens: int = 1) -> bool:
        if self.rate_per_minute == 0:
            # rate_per_minute == 0 disables rate limiting entirely.
            return True

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                float(self.capacity),
                self._tokens + elapsed * self._refill_per_second,
            )
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


class RateLimiterRegistry:
    """Registry of per-provider token buckets."""

    def __init__(self, *, rate_per_minute: int, burst: int) -> None:
        self._rate_per_minute = rate_per_minute
        self._burst = burst
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def get(self, provider: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(provider)
            if bucket is None:
                bucket = TokenBucket(
                    rate_per_minute=self._rate_per_minute, burst=self._burst
                )
                self._buckets[provider] = bucket
            return bucket

    def ensure_allowed(self, provider: str) -> None:
        bucket = self.get(provider)
        if bucket.try_acquire():
            LLM_RATE_LIMIT_EVENTS.labels(provider=provider, event="allowed").inc()
            return
        LLM_RATE_LIMIT_EVENTS.labels(provider=provider, event="throttled").inc()
        raise RateLimitedError(
            f"rate limit exceeded for provider {provider!r}",
            details={"provider": provider},
        )
