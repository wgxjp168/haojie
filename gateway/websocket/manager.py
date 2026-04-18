"""WebSocket connection manager with rooms/broadcasts.

Kept intentionally small:

* Connections are tracked in-process (one gateway replica = one hub).
* For multi-replica fan-out, extend ``broadcast_to_room`` to publish over
  Redis pubsub; the public API stays the same.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from gateway.auth.user_types import GatewayPrincipal
from gateway.core.logging import get_logger
from gateway.core.metrics import GW_WS_CONNECTIONS, GW_WS_MESSAGES_TOTAL

logger = get_logger(__name__)


@dataclass
class WebSocketConnection:
    connection_id: str
    websocket: WebSocket
    principal: GatewayPrincipal
    rooms: Set[str] = field(default_factory=set)
    connected_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    last_seen_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_open(self) -> bool:
        return self.websocket.application_state == WebSocketState.CONNECTED

    def touch(self) -> None:
        self.last_seen_at = datetime.now(tz=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "user_id": self.principal.user_id,
            "user_type": self.principal.user_type.value,
            "rooms": sorted(self.rooms),
            "connected_at": self.connected_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": self.metadata,
        }


class WebSocketManager:
    def __init__(self, max_connections: int = 10_000):
        self._max = max_connections
        self._connections: Dict[str, WebSocketConnection] = {}
        self._rooms: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        principal: GatewayPrincipal,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WebSocketConnection:
        async with self._lock:
            if len(self._connections) >= self._max:
                await websocket.close(code=1013, reason="Server at capacity")
                raise RuntimeError("Max connections reached")
            connection_id = str(uuid.uuid4())
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                principal=principal,
                metadata=metadata or {},
            )
            self._connections[connection_id] = connection
            GW_WS_CONNECTIONS.set(len(self._connections))
        logger.info(
            "ws.connect",
            connection_id=connection_id,
            user_id=principal.user_id,
            user_type=principal.user_type.value,
        )
        return connection

    async def disconnect(self, connection_id: str) -> None:
        async with self._lock:
            conn = self._connections.pop(connection_id, None)
            if conn is None:
                return
            for room in list(conn.rooms):
                self._rooms.get(room, set()).discard(connection_id)
                if not self._rooms.get(room):
                    self._rooms.pop(room, None)
            GW_WS_CONNECTIONS.set(len(self._connections))
        if conn.is_open():
            try:
                await conn.websocket.close()
            except Exception:
                pass
        logger.info(
            "ws.disconnect",
            connection_id=connection_id,
            user_id=conn.principal.user_id,
        )

    async def join_room(self, connection_id: str, room: str) -> bool:
        async with self._lock:
            conn = self._connections.get(connection_id)
            if conn is None:
                return False
            conn.rooms.add(room)
            self._rooms.setdefault(room, set()).add(connection_id)
            return True

    async def leave_room(self, connection_id: str, room: str) -> bool:
        async with self._lock:
            conn = self._connections.get(connection_id)
            if conn is None:
                return False
            conn.rooms.discard(room)
            self._rooms.get(room, set()).discard(connection_id)
            if not self._rooms.get(room):
                self._rooms.pop(room, None)
            return True

    async def send_to_connection(
        self, connection_id: str, message: Dict[str, Any]
    ) -> bool:
        async with self._lock:
            conn = self._connections.get(connection_id)
        if conn is None or not conn.is_open():
            return False
        try:
            await conn.websocket.send_json(message)
            conn.touch()
            GW_WS_MESSAGES_TOTAL.labels(
                direction="out", type=str(message.get("type", "unknown"))
            ).inc()
            return True
        except Exception as e:
            logger.warning("ws.send_failed", connection_id=connection_id, error=str(e))
            await self.disconnect(connection_id)
            return False

    async def broadcast_to_room(
        self,
        room: str,
        message: Dict[str, Any],
        exclude: Optional[Set[str]] = None,
    ) -> int:
        exclude = exclude or set()
        async with self._lock:
            targets = list(self._rooms.get(room, set()))
        sent = 0
        for cid in targets:
            if cid in exclude:
                continue
            if await self.send_to_connection(cid, message):
                sent += 1
        return sent

    async def broadcast_to_user(
        self, user_id: str, message: Dict[str, Any]
    ) -> int:
        async with self._lock:
            targets = [
                c.connection_id
                for c in self._connections.values()
                if c.principal.user_id == user_id
            ]
        sent = 0
        for cid in targets:
            if await self.send_to_connection(cid, message):
                sent += 1
        return sent

    async def close_all(self) -> None:
        async with self._lock:
            cids = list(self._connections.keys())
        for cid in cids:
            await self.disconnect(cid)

    def stats(self) -> Dict[str, Any]:
        return {
            "connections": len(self._connections),
            "rooms": len(self._rooms),
            "max_connections": self._max,
        }

    def list_connections(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._connections.values()]


_manager: Optional[WebSocketManager] = None


def get_ws_manager() -> WebSocketManager:
    global _manager
    if _manager is None:
        from gateway.config import get_gateway_settings

        settings = get_gateway_settings()
        _manager = WebSocketManager(max_connections=settings.ws_max_connections)
    return _manager


def reset_ws_manager() -> None:
    global _manager
    _manager = None
