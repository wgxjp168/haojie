"""FastAPI dependencies used by gateway routes."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gateway.auth.jwt_manager import JWTManager, get_jwt_manager
from gateway.auth.user_types import GatewayPrincipal, GatewayRole, GatewayUserType
from gateway.config import GatewaySettings, get_gateway_settings
from gateway.core.exceptions import ForbiddenError, UnauthorizedError
from gateway.core.security import verify_api_key
from gateway.rate_limiting.strategies import RateLimiter, get_rate_limiter

_bearer = HTTPBearer(auto_error=False)


async def get_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: GatewaySettings = Depends(get_gateway_settings),
) -> GatewayPrincipal:
    """Resolve the caller identity. Sets ``request.state.principal``."""
    # 1) JWT Bearer
    if credentials is not None and credentials.scheme.lower() == "bearer":
        manager: JWTManager = get_jwt_manager()
        principal = await manager.verify_access(credentials.credentials)
        request.state.principal = principal
        return principal

    # 2) API key
    if x_api_key is not None:
        if not verify_api_key(x_api_key, settings.api_keys):
            raise UnauthorizedError("Invalid API key")
        principal = GatewayPrincipal(
            user_id=f"apikey:{x_api_key[-6:]}",
            user_type=GatewayUserType.PARTNER,
            roles=[GatewayRole.PARTNER],
            permissions=["inputs:*", "sessions:*"],
            auth_method="api_key",
        )
        request.state.principal = principal
        return principal

    # 3) Anonymous (if allowed)
    if settings.api_key_required:
        raise UnauthorizedError("Authentication required")
    ip = request.client.host if request.client else "unknown"
    principal = GatewayPrincipal.anonymous(ip_address=ip)
    request.state.principal = principal
    return principal


async def get_authenticated_principal(
    principal: GatewayPrincipal = Depends(get_principal),
) -> GatewayPrincipal:
    if principal.is_anonymous():
        raise UnauthorizedError("Authentication required")
    return principal


def require_roles(*roles: GatewayRole):
    async def _dep(
        principal: GatewayPrincipal = Depends(get_authenticated_principal),
    ) -> GatewayPrincipal:
        if not principal.has_any_role(list(roles)):
            raise ForbiddenError(
                "Missing required role", required=[r.value for r in roles]
            )
        return principal

    return _dep


async def rate_limit_dependency(
    request: Request,
    principal: GatewayPrincipal = Depends(get_principal),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> GatewayPrincipal:
    from gateway.core.exceptions import RateLimitedError

    decision = await limiter.check(principal, request.url.path)
    # Surface rate-limit headers even on success.
    request.state.rate_limit = decision
    if not decision.allowed:
        raise RateLimitedError(
            limit=decision.limit,
            window_seconds=60,
            retry_after=max(1, int(decision.reset_in_seconds)),
        )
    return principal
