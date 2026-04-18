"""In-memory session store with Redis persistence when available.

Sessions live in-process for low-latency access, and are mirrored to the
cache backend so they survive a process restart (as long as the cache
backend is Redis).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from cache.manager import CacheManager, get_cache_manager
from config.settings import Settings, get_settings
from core.logging import get_logger
from core.metrics import ACTIVE_SESSIONS
from entities.enums import DeviceType, SessionStatus
from entities.session import InputItem, Session

logger = get_logger(__name__)

_SESSION_NAMESPACE = "session"


class SessionManager:
    def __init__(
        self,
        cache: Optional[CacheManager] = None,
        settings: Optional[Settings] = None,
    ):
        self._cache = cache or get_cache_manager()
        self._settings = settings or get_settings()
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    def start_cleanup_task(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def create_session(
        self,
        user_id: str,
        device_type: DeviceType = DeviceType.UNKNOWN,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Session:
        session = Session(
            session_id="",
            user_id=user_id,
            device_type=device_type,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        async with self._lock:
            self._sessions[session.session_id] = session
            ACTIVE_SESSIONS.set(len(self._sessions))
        await self._persist(session)
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            if session.is_expired():
                await self.end_session(session_id, reason="timeout")
                return None
            return session
        # Not in memory — try cache.
        cached = await self._cache.get(session_id, _SESSION_NAMESPACE)
        if cached is None:
            return None
        if not isinstance(cached, Session):
            return None
        async with self._lock:
            self._sessions[session_id] = cached
            ACTIVE_SESSIONS.set(len(self._sessions))
        return cached

    async def add_input(self, session_id: str, item: InputItem) -> bool:
        session = await self.get_session(session_id)
        if session is None:
            return False
        session.add_input(item)
        await self._persist(session)
        return True

    async def end_session(self, session_id: str, reason: str = "user_end") -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            ACTIVE_SESSIONS.set(len(self._sessions))
        if session is None:
            # maybe in cache only
            session = await self._cache.get(session_id, _SESSION_NAMESPACE)
            if session is None:
                return False
        session.end(reason=reason)
        # Keep ended sessions in cache for retrieval, shorter TTL.
        await self._cache.set(
            session_id, session, _SESSION_NAMESPACE, ttl=3_600
        )
        return True

    async def list_user_sessions(self, user_id: str) -> List[Session]:
        async with self._lock:
            return [s for s in self._sessions.values() if s.user_id == user_id]

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._sessions)

    async def _persist(self, session: Session) -> None:
        await self._cache.set(
            session.session_id,
            session,
            _SESSION_NAMESPACE,
            ttl=self._settings.session_max_age_hours * 3_600,
        )

    async def _cleanup_loop(self) -> None:
        interval = self._settings.session_cleanup_interval_minutes * 60
        while True:
            try:
                await asyncio.sleep(interval)
                expired = await self._cleanup_once()
                if expired:
                    logger.info("session.cleanup", expired=expired)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("session.cleanup_error", error=str(e))

    async def _cleanup_once(self) -> int:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            hours=self._settings.session_max_age_hours
        )
        async with self._lock:
            expired_ids = [
                sid
                for sid, s in self._sessions.items()
                if s.is_expired() or s.last_activity_at < cutoff
            ]
        for sid in expired_ids:
            await self.end_session(sid, reason="timeout")
        return len(expired_ids)


_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
