"""WebSocket endpoints.

Protocol:

Client → gateway::

    { "type": "join",  "room": "<name>" }
    { "type": "leave", "room": "<name>" }
    { "type": "broadcast", "room": "<name>", "payload": { ... } }
    { "type": "direct", "user_id": "...", "payload": { ... } }
    { "type": "ping" }

Gateway → client::

    { "type": "welcome", ... }
    { "type": "pong" }
    { "type": "ack", "message_id": "..." }
    { "type": "error", "message": "..." }
    { "type": "broadcast", "room": "...", "from": "...", "payload": {...} }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from gateway.auth.jwt_manager import get_jwt_manager
from gateway.auth.user_types import GatewayPrincipal
from gateway.core.exceptions import GatewayError
from gateway.core.logging import get_logger
from gateway.core.metrics import GW_WS_MESSAGES_TOTAL
from gateway.websocket.manager import WebSocketManager, get_ws_manager

logger = get_logger(__name__)
router = APIRouter(tags=["ws"])


async def _authenticate_ws(token: str) -> GatewayPrincipal:
    manager = get_jwt_manager()
    return await manager.verify_access(token)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="Access token (JWT)"),
):
    try:
        principal = await _authenticate_ws(token)
    except GatewayError as e:
        await websocket.close(code=4401, reason=e.message)
        return

    if not principal.has_permission("ws:connect") and not principal.has_permission(
        "inputs:*"
    ):
        await websocket.close(code=4403, reason="ws:connect permission required")
        return

    await websocket.accept()
    ws_manager: WebSocketManager = get_ws_manager()
    try:
        connection = await ws_manager.connect(websocket, principal)
    except RuntimeError:
        return

    await websocket.send_json(
        {
            "type": "welcome",
            "connection_id": connection.connection_id,
            "user_id": principal.user_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    )

    try:
        while True:
            message: Dict[str, Any] = await websocket.receive_json()
            connection.touch()
            GW_WS_MESSAGES_TOTAL.labels(
                direction="in", type=str(message.get("type", "unknown"))
            ).inc()
            await _dispatch(ws_manager, connection.connection_id, principal, message)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(
            "ws.loop_error",
            connection_id=connection.connection_id,
            error=str(e),
        )
    finally:
        await ws_manager.disconnect(connection.connection_id)


async def _dispatch(
    manager: WebSocketManager,
    connection_id: str,
    principal: GatewayPrincipal,
    message: Dict[str, Any],
) -> None:
    mtype = message.get("type")
    if mtype == "ping":
        await manager.send_to_connection(connection_id, {"type": "pong"})
        return
    if mtype == "join":
        room = message.get("room")
        if not room:
            await _error(manager, connection_id, "room is required")
            return
        await manager.join_room(connection_id, str(room))
        await manager.send_to_connection(
            connection_id, {"type": "joined", "room": room}
        )
        return
    if mtype == "leave":
        room = message.get("room")
        if not room:
            await _error(manager, connection_id, "room is required")
            return
        await manager.leave_room(connection_id, str(room))
        await manager.send_to_connection(
            connection_id, {"type": "left", "room": room}
        )
        return
    if mtype == "broadcast":
        room = message.get("room")
        payload = message.get("payload")
        if not room:
            await _error(manager, connection_id, "room is required")
            return
        count = await manager.broadcast_to_room(
            str(room),
            {
                "type": "broadcast",
                "room": room,
                "from": principal.user_id,
                "payload": payload,
            },
            exclude={connection_id},
        )
        await manager.send_to_connection(
            connection_id, {"type": "ack", "delivered_to": count}
        )
        return
    if mtype == "direct":
        target = message.get("user_id")
        payload = message.get("payload")
        if not target:
            await _error(manager, connection_id, "user_id is required")
            return
        count = await manager.broadcast_to_user(
            str(target),
            {
                "type": "direct",
                "from": principal.user_id,
                "payload": payload,
            },
        )
        await manager.send_to_connection(
            connection_id, {"type": "ack", "delivered_to": count}
        )
        return
    await _error(manager, connection_id, f"Unknown message type: {mtype}")


async def _error(manager: WebSocketManager, connection_id: str, msg: str) -> None:
    await manager.send_to_connection(connection_id, {"type": "error", "message": msg})
