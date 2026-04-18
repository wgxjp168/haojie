"""Differentiated rate-limiting strategies per user type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from gateway.auth.user_types import GatewayPrincipal, GatewayUserType
from gateway.config import GatewaySettings, get_gateway_settings
from gateway.core.logging import get_logger
from gateway.core.metrics import GW_RATE_LIMIT_HITS
from gateway.rate_limiting.token_bucket import (
    InMemoryTokenBucketStore,
    RateLimitDecision,
    RedisTokenBucketStore,
    TokenBucketStore,
)

logger = get_logger(__name__)


class RateLimitStrategy(ABC):
    """Produces (capacity, refill_per_second) for a principal."""

    @abstractmethod
    def get_parameters(
        self, principal: GatewayPrincipal
    ) -> tuple[int, float]: ...

    @abstractmethod
    def get_key(self, principal: GatewayPrincipal, path: str) -> str: ...


class DefaultRateLimitStrategy(RateLimitStrategy):
    """Per-user-type limits, falling back to anonymous IP-keyed limits."""

    def __init__(self, settings: Optional[GatewaySettings] = None) -> None:
        self._settings = settings or get_gateway_settings()
        self._map = {
            GatewayUserType.B2B_PURCHASER: self._settings.rl_b2b_purchaser_per_minute,
            GatewayUserType.B2C_CONSUMER: self._settings.rl_b2c_consumer_per_minute,
            GatewayUserType.PARTNER: self._settings.rl_partner_per_minute,
            GatewayUserType.INTERNAL: self._settings.rl_internal_per_minute,
            GatewayUserType.ADMIN: self._settings.rl_internal_per_minute,
            GatewayUserType.ANONYMOUS: self._settings.rl_anonymous_per_minute,
        }

    def get_parameters(self, principal: GatewayPrincipal) -> tuple[int, float]:
        per_minute = self._map.get(
            principal.user_type, self._settings.rl_default_per_minute
        )
        refill = per_minute / 60.0
        # Allow bursts up to the per-minute quota.
        capacity = max(1, per_minute)
        return capacity, refill

    def get_key(self, principal: GatewayPrincipal, path: str) -> str:
        # Key by user; anonymous buckets are keyed by IP (embedded in user_id).
        return f"{principal.user_type.value}:{principal.user_id}"


class RateLimiter:
    """High-level rate limiter combining a strategy and a store."""

    def __init__(
        self,
        strategy: Optional[RateLimitStrategy] = None,
        store: Optional[TokenBucketStore] = None,
    ):
        self._strategy = strategy or DefaultRateLimitStrategy()
        self._store = store or InMemoryTokenBucketStore()

    async def check(
        self, principal: GatewayPrincipal, path: str = "/"
    ) -> RateLimitDecision:
        capacity, refill = self._strategy.get_parameters(principal)
        key = self._strategy.get_key(principal, path)
        decision = await self._store.try_consume(key, capacity, refill)
        if not decision.allowed:
            GW_RATE_LIMIT_HITS.labels(user_type=principal.user_type.value).inc()
        return decision


_rate_limiter: Optional[RateLimiter] = None


async def _maybe_redis_store(settings: GatewaySettings) -> Optional[TokenBucketStore]:
    if not settings.redis_url:
        return None
    try:
        import redis.asyncio as redis

        client = redis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        return RedisTokenBucketStore(client)
    except Exception as e:
        logger.warning("rate_limiter.redis_unavailable", error=str(e))
        return None


async def build_rate_limiter(
    settings: Optional[GatewaySettings] = None,
) -> RateLimiter:
    cfg = settings or get_gateway_settings()
    store = await _maybe_redis_store(cfg) or InMemoryTokenBucketStore()
    return RateLimiter(
        strategy=DefaultRateLimitStrategy(cfg),
        store=store,
    )


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    global _rate_limiter
    _rate_limiter = limiter


def reset_rate_limiter() -> None:
    global _rate_limiter
    _rate_limiter = None
