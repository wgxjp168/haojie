"""Admin endpoints: inspect connections, reset circuit breakers, stats."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from gateway.api.dependencies import require_roles
from gateway.auth.user_types import GatewayPrincipal, GatewayRole
from gateway.circuit_breaker.breaker import (
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
)
from gateway.websocket.manager import WebSocketManager, get_ws_manager

router = APIRouter(prefix="/admin", tags=["admin"])

_admin_dep = require_roles(GatewayRole.ADMIN, GatewayRole.SUPER_ADMIN)


@router.get("/ws/connections", summary="List active WebSocket connections")
async def list_ws_connections(
    _principal: GatewayPrincipal = Depends(_admin_dep),
    manager: WebSocketManager = Depends(get_ws_manager),
):
    return {"stats": manager.stats(), "connections": manager.list_connections()}


@router.get("/circuits", summary="Current circuit-breaker state")
async def list_circuits(
    _principal: GatewayPrincipal = Depends(_admin_dep),
    registry: CircuitBreakerRegistry = Depends(get_circuit_breaker_registry),
):
    return {"circuits": registry.snapshot()}


@router.post("/circuits/{service}/reset", summary="Force-close a circuit")
async def reset_circuit(
    service: str,
    _principal: GatewayPrincipal = Depends(_admin_dep),
    registry: CircuitBreakerRegistry = Depends(get_circuit_breaker_registry),
):
    breaker = await registry.get(service)
    await breaker.reset()
    return {"service": service, "state": breaker.state.value}
