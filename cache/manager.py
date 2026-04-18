"""Async cache manager with Redis and in-memory back-ends.

The in-memory backend is process-local and is only suitable for
development / tests. Production deployments should use Redis.

Values are serialised with pickle for the Redis backend; callers should
therefore only cache picklable objects.
"""
from __future__ import annotations

import asyncio
import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from config.settings import Settings, get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class AbstractCacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str) -> Any: ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear_prefix(self, prefix: str) -> int: ...

    @abstractmethod
    async def ping(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...


class InMemoryCache(AbstractCacheBackend):
    """Process-local cache with TTL and a simple LRU-ish eviction policy."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any:
        async with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expires_at = item
            if expires_at and expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int) -> None:
        async with self._lock:
            if len(self._store) >= self._max_size:
                # Drop the oldest ~10% of entries.
                drop = max(1, self._max_size // 10)
                for old_key in list(self._store.keys())[:drop]:
                    self._store.pop(old_key, None)
            expires_at = time.time() + ttl if ttl > 0 else 0
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear_prefix(self, prefix: str) -> int:
        async with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                self._store.pop(k, None)
            return len(keys)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        async with self._lock:
            self._store.clear()


class RedisCache(AbstractCacheBackend):
    """Redis-backed cache using the async redis client."""

    def __init__(self, url: str) -> None:
        # Import lazily so tests that don't use Redis don't need the lib.
        import redis.asyncio as redis

        self._client = redis.from_url(url, encoding=None, decode_responses=False)

    async def get(self, key: str) -> Any:
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception as e:  # corrupted cache entry — drop it
            logger.warning("cache.deserialize_failed", key=key, error=str(e))
            await self._client.delete(key)
            return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        data = pickle.dumps(value)
        if ttl > 0:
            await self._client.set(key, data, ex=ttl)
        else:
            await self._client.set(key, data)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def clear_prefix(self, prefix: str) -> int:
        deleted = 0
        async for k in self._client.scan_iter(match=f"{prefix}*"):
            await self._client.delete(k)
            deleted += 1
        return deleted

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except AttributeError:  # older redis-py
            await self._client.close()


class CacheManager:
    """Front-door cache used by the rest of the app.

    If ``cache_enabled`` is false, all operations are no-ops.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._backend: Optional[AbstractCacheBackend] = None

    async def initialize(self) -> None:
        if not self._settings.cache_enabled:
            logger.info("cache.disabled")
            return
        if self._settings.redis_url:
            try:
                self._backend = RedisCache(self._settings.redis_url)
                healthy = await self._backend.ping()
                if healthy:
                    logger.info("cache.redis_ready", url=self._redacted_url())
                    return
                logger.warning("cache.redis_unhealthy_falling_back_to_memory")
            except Exception as e:
                logger.warning("cache.redis_init_failed", error=str(e))
        self._backend = InMemoryCache()
        logger.info("cache.in_memory_ready")

    def _redacted_url(self) -> str:
        url = self._settings.redis_url or ""
        # Strip password if embedded.
        if "@" in url:
            scheme, rest = url.split("://", 1)
            _, host = rest.split("@", 1)
            return f"{scheme}://***@{host}"
        return url

    def _full_key(self, namespace: str, key: str) -> str:
        return f"{self._settings.service_name}:{namespace}:{key}"

    async def get(self, key: str, namespace: str = "default") -> Any:
        if not self._backend:
            return None
        try:
            return await self._backend.get(self._full_key(namespace, key))
        except Exception as e:
            logger.warning("cache.get_failed", key=key, error=str(e))
            return None

    async def set(
        self, key: str, value: Any, namespace: str = "default", ttl: Optional[int] = None
    ) -> None:
        if not self._backend:
            return
        try:
            await self._backend.set(
                self._full_key(namespace, key),
                value,
                ttl if ttl is not None else self._settings.cache_ttl,
            )
        except Exception as e:
            logger.warning("cache.set_failed", key=key, error=str(e))

    async def delete(self, key: str, namespace: str = "default") -> None:
        if not self._backend:
            return
        try:
            await self._backend.delete(self._full_key(namespace, key))
        except Exception as e:
            logger.warning("cache.delete_failed", key=key, error=str(e))

    async def clear_namespace(self, namespace: str) -> int:
        if not self._backend:
            return 0
        prefix = f"{self._settings.service_name}:{namespace}:"
        try:
            return await self._backend.clear_prefix(prefix)
        except Exception as e:
            logger.warning("cache.clear_failed", namespace=namespace, error=str(e))
            return 0

    async def ping(self) -> bool:
        if not self._backend:
            return True
        return await self._backend.ping()

    async def close(self) -> None:
        if self._backend:
            await self._backend.close()
            self._backend = None


# --- singleton accessor ---

_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# --- decorator ---

def cached(namespace: str = "default", ttl: Optional[int] = None) -> Callable:
    """Decorator that caches the return value of an async function.

    The cache key is derived from the function name and the stringified
    positional/keyword arguments. Unhashable argument types will still
    work but are less deduplicated-friendly.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            key_material = f"{func.__module__}.{func.__name__}:{args!r}:{kwargs!r}"
            key = hashlib.md5(key_material.encode("utf-8")).hexdigest()
            cached_value = await cache.get(key, namespace)
            if cached_value is not None:
                return cached_value
            result = await func(*args, **kwargs)
            await cache.set(key, result, namespace, ttl)
            return result

        return wrapper

    return decorator
