"""Two-tier cache for rendered reports (memory LRU + optional Redis DB 5).

Keys are the request fingerprint (sha256 over the normalized input). We
cache the full ``ReportResponse`` so a repeat request returns identical
content/checksum without re-running the renderer or re-uploading to the
storage fan-out.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, Optional

try:
    import redis  # type: ignore
    from redis.exceptions import RedisError  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore

from report.core.logging import get_logger
from report.core.metrics import REPORT_CACHE_EVENTS

log = get_logger(__name__)


class _LRU:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.capacity == 0:
            return None
        val = self._store.get(key)
        if val is None:
            return None
        self._store.move_to_end(key)
        return val

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if self.capacity == 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class ReportCache:
    """Two-tier cache. Safe to construct when Redis is unavailable."""

    def __init__(
        self,
        *,
        memory_capacity: int = 2048,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 600,
        namespace: str = "report",
    ) -> None:
        self._mem = _LRU(memory_capacity)
        self._ttl = max(0, int(ttl_seconds))
        self._namespace = namespace
        self._redis = None
        if redis_url and redis is not None:
            try:
                self._redis = redis.Redis.from_url(
                    redis_url,
                    socket_timeout=0.5,
                    socket_connect_timeout=0.5,
                )
                self._redis.ping()
                log.info("report.cache.redis_connected", url=redis_url)
            except RedisError as exc:  # pragma: no cover - depends on env
                log.warning("report.cache.redis_unavailable", error=str(exc))
                self._redis = None

    @property
    def ready(self) -> bool:
        return self._mem.capacity > 0 or self._redis is not None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        hit = self._mem.get(key)
        if hit is not None:
            REPORT_CACHE_EVENTS.labels(tier="memory", event="hit").inc()
            return hit
        REPORT_CACHE_EVENTS.labels(tier="memory", event="miss").inc()

        if self._redis is not None:
            try:
                raw = self._redis.get(self._key(key))
                if raw is not None:
                    REPORT_CACHE_EVENTS.labels(tier="redis", event="hit").inc()
                    payload = json.loads(raw)
                    self._mem.set(key, payload)
                    return payload
                REPORT_CACHE_EVENTS.labels(tier="redis", event="miss").inc()
            except (RedisError, ValueError) as exc:
                REPORT_CACHE_EVENTS.labels(tier="redis", event="error").inc()
                log.warning("report.cache.redis_get_failed", error=str(exc))
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._mem.set(key, value)
        if self._redis is not None and self._ttl > 0:
            try:
                self._redis.setex(
                    self._key(key), self._ttl, json.dumps(value, default=str)
                )
            except RedisError as exc:
                REPORT_CACHE_EVENTS.labels(tier="redis", event="error").inc()
                log.warning("report.cache.redis_set_failed", error=str(exc))

    def clear(self) -> None:
        self._mem.clear()
        if self._redis is not None:
            try:
                self._redis.flushdb()
            except RedisError as exc:
                log.warning("report.cache.redis_flush_failed", error=str(exc))

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}"
