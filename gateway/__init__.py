"""Gateway (Access Layer) for ILBuyAI - Part 2/10.

Sits in front of the Part 1 input processor. Provides:
    * JWT + API key authentication, token refresh, revocation
    * RBAC with multi-user-type (B2B / B2C / Partner / Internal)
    * Token-bucket rate limiting, differentiated by user type
    * Circuit breaker with exponential auto-recovery
    * Upstream request forwarder with retry/timeout
    * WebSocket hub (connection manager, rooms, broadcasts)
    * Prometheus metrics + health checks
"""
from gateway.config import GatewaySettings, get_gateway_settings

__all__ = ["GatewaySettings", "get_gateway_settings"]
