"""Token-bucket rate limiter.

Two back-ends are provided:

* ``InMemoryTokenBucketStore`` — lock-per-key, process-local (fine for dev
  and for single-process deployments).
* ``RedisTokenBucketStore`` — uses a small Lua script to atomically
  refill-and-consume so it works correctly across multiple replicas.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RateLimitDecision:
    """Return value of a rate-limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_in_seconds: float


@dataclass
class _BucketState:
    tokens: float
    updated_at: float


class TokenBucketStore(ABC):
    @abstractmethod
    async def try_consume(
        self, key: str, capacity: int, refill_per_second: float, cost: int = 1
    ) -> RateLimitDecision: ...


class InMemoryTokenBucketStore(TokenBucketStore):
    def __init__(self) -> None:
        self._buckets: Dict[str, _BucketState] = {}
        self._lock = asyncio.Lock()

    async def try_consume(
        self, key: str, capacity: int, refill_per_second: float, cost: int = 1
    ) -> RateLimitDecision:
        now = time.monotonic()
        async with self._lock:
            state = self._buckets.get(key)
            if state is None:
                state = _BucketState(tokens=capacity, updated_at=now)
            else:
                elapsed = max(0.0, now - state.updated_at)
                state.tokens = min(
                    capacity, state.tokens + elapsed * refill_per_second
                )
                state.updated_at = now
            if state.tokens >= cost:
                state.tokens -= cost
                self._buckets[key] = state
                return RateLimitDecision(
                    allowed=True,
                    remaining=int(state.tokens),
                    limit=capacity,
                    reset_in_seconds=(capacity - state.tokens) / refill_per_second
                    if refill_per_second > 0
                    else 0.0,
                )
            # Not enough tokens
            self._buckets[key] = state
            missing = cost - state.tokens
            retry_after = missing / refill_per_second if refill_per_second > 0 else 0.0
            return RateLimitDecision(
                allowed=False,
                remaining=int(state.tokens),
                limit=capacity,
                reset_in_seconds=retry_after,
            )


class RedisTokenBucketStore(TokenBucketStore):
    """Redis-backed atomic token bucket.

    Key layout::

        tb:{key} -> hash { tokens, updated_at }

    The Lua script is short and idempotent.
    """

    _SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill = tonumber(ARGV[2])
    local cost = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])

    local bucket = redis.call('HMGET', key, 'tokens', 'updated_at')
    local tokens = tonumber(bucket[1])
    local updated_at = tonumber(bucket[2])

    if tokens == nil then
      tokens = capacity
      updated_at = now
    else
      local delta = math.max(0, now - updated_at)
      tokens = math.min(capacity, tokens + delta * refill)
      updated_at = now
    end

    local allowed = 0
    if tokens >= cost then
      tokens = tokens - cost
      allowed = 1
    end

    redis.call('HSET', key, 'tokens', tokens, 'updated_at', updated_at)
    -- expire the key so idle buckets don't leak
    redis.call('EXPIRE', key, 3600)

    return { allowed, tokens, capacity }
    """

    def __init__(self, redis_client):
        self._redis = redis_client
        self._script_sha: Optional[str] = None

    async def _ensure_script(self) -> str:
        if self._script_sha is None:
            self._script_sha = await self._redis.script_load(self._SCRIPT)
        return self._script_sha

    async def try_consume(
        self, key: str, capacity: int, refill_per_second: float, cost: int = 1
    ) -> RateLimitDecision:
        sha = await self._ensure_script()
        now = time.time()
        result = await self._redis.evalsha(
            sha,
            1,
            f"tb:{key}",
            capacity,
            refill_per_second,
            cost,
            now,
        )
        allowed_i, tokens, cap = result
        tokens = float(tokens)
        allowed = bool(int(allowed_i))
        retry_after = 0.0
        if not allowed and refill_per_second > 0:
            retry_after = max(0.0, (cost - tokens) / refill_per_second)
        return RateLimitDecision(
            allowed=allowed,
            remaining=max(0, int(tokens)),
            limit=int(cap),
            reset_in_seconds=retry_after,
        )


class TokenBucket:
    """Thin convenience wrapper binding a store + parameters."""

    def __init__(
        self,
        store: TokenBucketStore,
        capacity: int,
        refill_per_second: float,
    ):
        self._store = store
        self._capacity = capacity
        self._refill = refill_per_second

    async def try_consume(self, key: str, cost: int = 1) -> RateLimitDecision:
        return await self._store.try_consume(
            key, self._capacity, self._refill, cost
        )
