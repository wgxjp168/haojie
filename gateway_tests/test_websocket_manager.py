"""Unit tests for the WebSocketManager (no real WS needed)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.websockets import WebSocketState

from gateway.auth.user_types import GatewayPrincipal, GatewayUserType
from gateway.websocket.manager import WebSocketManager


def _fake_ws() -> MagicMock:
    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _principal(user_id: str = "u1") -> GatewayPrincipal:
    return GatewayPrincipal(
        user_id=user_id, user_type=GatewayUserType.B2C_CONSUMER
    )


@pytest.mark.asyncio
async def test_connect_and_send():
    mgr = WebSocketManager(max_connections=10)
    ws = _fake_ws()
    conn = await mgr.connect(ws, _principal())
    assert mgr.stats()["connections"] == 1

    ok = await mgr.send_to_connection(conn.connection_id, {"type": "hello"})
    assert ok
    ws.send_json.assert_awaited_with({"type": "hello"})


@pytest.mark.asyncio
async def test_rooms_and_broadcast():
    mgr = WebSocketManager(max_connections=10)
    ws1, ws2 = _fake_ws(), _fake_ws()
    c1 = await mgr.connect(ws1, _principal("a"))
    c2 = await mgr.connect(ws2, _principal("b"))

    await mgr.join_room(c1.connection_id, "lobby")
    await mgr.join_room(c2.connection_id, "lobby")

    sent = await mgr.broadcast_to_room(
        "lobby", {"type": "chat", "text": "hi"}, exclude={c1.connection_id}
    )
    assert sent == 1
    ws2.send_json.assert_awaited_with({"type": "chat", "text": "hi"})
    ws1.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_to_user():
    mgr = WebSocketManager(max_connections=10)
    ws_a1 = _fake_ws()
    ws_a2 = _fake_ws()
    ws_b = _fake_ws()
    await mgr.connect(ws_a1, _principal("alice"))
    await mgr.connect(ws_a2, _principal("alice"))
    await mgr.connect(ws_b, _principal("bob"))

    sent = await mgr.broadcast_to_user("alice", {"type": "ping"})
    assert sent == 2
    ws_a1.send_json.assert_awaited()
    ws_a2.send_json.assert_awaited()
    ws_b.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_removes_from_rooms_and_closes():
    mgr = WebSocketManager(max_connections=10)
    ws = _fake_ws()
    conn = await mgr.connect(ws, _principal())
    await mgr.join_room(conn.connection_id, "r1")
    await mgr.disconnect(conn.connection_id)
    assert mgr.stats()["connections"] == 0
    assert mgr.stats()["rooms"] == 0


@pytest.mark.asyncio
async def test_max_connections_enforced():
    mgr = WebSocketManager(max_connections=1)
    ws1 = _fake_ws()
    await mgr.connect(ws1, _principal("a"))
    ws2 = _fake_ws()
    with pytest.raises(RuntimeError):
        await mgr.connect(ws2, _principal("b"))
