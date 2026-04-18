"""Token blacklist (revocation) with Redis + in-memory fallback.

Tokens are stored by their ``jti`` claim with a TTL set to the remaining
lifetime of the token, so we never need to clean expired entries.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

from gateway.config import GatewaySettings, get_gateway_settings
from gateway.core.logging import get_logger

logger = get_logger(__name__)


class TokenBlacklist:
    """Stores revoked ``jti`` values with per-token TTL.

    Redis-backed when a Redis URL is configured; falls back to an in-memory
    dict otherwise (suitable for single-process dev/tests).
    """

    def __init__(self, settings: Optional[GatewaySettings] = None):
        self._settings = settings or get_gateway_settings()
        self._memory: Dict[str, float] = {}
        self._redis = None  # lazily created
        self._lock = asyncio.Lock()

    async def _get_redis(self):
        if not self._settings.redis_url:
            return None
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self._settings.redis_url, decode_responses=True
            )
            await self._redis.ping()
            return self._redis
        except Exception as e:
            logger.warning("blacklist.redis_unavailable", error=str(e))
            self._redis = None
            return None

    async def add(self, jti: str, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        r = await self._get_redis()
        key = f"blacklist:{jti}"
        if r is not None:
            try:
                await r.setex(key, ttl_seconds, "1")
                return
            except Exception as e:
                logger.warning("blacklist.redis_set_failed", error=str(e))
        async with self._lock:
            self._memory[jti] = time.time() + ttl_seconds

    async def contains(self, jti: str) -> bool:
        r = await self._get_redis()
        if r is not None:
            try:
                return bool(await r.exists(f"blacklist:{jti}"))
            except Exception as e:
                logger.warning("blacklist.redis_get_failed", error=str(e))
        async with self._lock:
            exp = self._memory.get(jti)
            if exp is None:
                return False
            if exp < time.time():
                self._memory.pop(jti, None)
                return False
            return True

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except AttributeError:
                await self._redis.close()
            self._redis = None


_blacklist: Optional[TokenBlacklist] = None


def get_token_blacklist() -> TokenBlacklist:
    global _blacklist
    if _blacklist is None:
        _blacklist = TokenBlacklist()
    return _blacklist


def reset_token_blacklist() -> None:
    """Test helper."""
    global _blacklist
    _blacklist = None
