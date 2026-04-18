"""JWT issue/verify/refresh + revocation.

The manager produces short-lived access tokens and longer-lived refresh
tokens. Refresh tokens are rotated on use: the old refresh token's ``jti`` is
added to the blacklist as soon as a new pair is minted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt

from gateway.auth.token_blacklist import TokenBlacklist, get_token_blacklist
from gateway.auth.user_types import GatewayPrincipal, GatewayRole, GatewayUserType
from gateway.config import GatewaySettings, get_gateway_settings
from gateway.core.exceptions import (
    InvalidTokenError,
    TokenExpiredError,
    TokenRevokedError,
)
from gateway.core.logging import get_logger
from gateway.core.metrics import GW_AUTH_EVENTS_TOTAL

logger = get_logger(__name__)

_ACCESS = "access"
_REFRESH = "refresh"


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    access_expires_in: int = 0  # seconds
    refresh_expires_in: int = 0  # seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.access_expires_in,
            "refresh_expires_in": self.refresh_expires_in,
        }


class JWTManager:
    def __init__(
        self,
        settings: Optional[GatewaySettings] = None,
        blacklist: Optional[TokenBlacklist] = None,
    ):
        self._settings = settings or get_gateway_settings()
        self._blacklist = blacklist or get_token_blacklist()

    # --- issue ---

    def issue(
        self,
        user_id: str,
        user_type: GatewayUserType,
        roles: Optional[List[GatewayRole]] = None,
        permissions: Optional[List[str]] = None,
        scopes: Optional[List[str]] = None,
        username: Optional[str] = None,
        company_id: Optional[str] = None,
        extra_claims: Optional[Dict[str, Any]] = None,
    ) -> TokenPair:
        import uuid

        now = datetime.now(tz=timezone.utc)
        access_exp = now + timedelta(minutes=self._settings.jwt_access_minutes)
        refresh_exp = now + timedelta(days=self._settings.jwt_refresh_days)

        base: Dict[str, Any] = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "iss": self._settings.service_name,
            "user_type": user_type.value,
            "roles": [r.value for r in (roles or [])],
            "permissions": list(permissions or []),
            "scopes": list(scopes or []),
        }
        if username:
            base["username"] = username
        if company_id:
            base["company_id"] = company_id
        if extra_claims:
            base.update(extra_claims)

        access_payload = {
            **base,
            "typ": _ACCESS,
            "jti": str(uuid.uuid4()),
            "exp": int(access_exp.timestamp()),
        }
        refresh_payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "iss": self._settings.service_name,
            "user_type": user_type.value,
            "typ": _REFRESH,
            "jti": str(uuid.uuid4()),
            "exp": int(refresh_exp.timestamp()),
        }

        access = jwt.encode(
            access_payload,
            self._settings.jwt_secret_key,
            algorithm=self._settings.jwt_algorithm,
        )
        refresh = jwt.encode(
            refresh_payload,
            self._settings.jwt_secret_key,
            algorithm=self._settings.jwt_algorithm,
        )
        GW_AUTH_EVENTS_TOTAL.labels(event="issue", result="success").inc()
        return TokenPair(
            access_token=access,
            refresh_token=refresh,
            access_expires_in=self._settings.jwt_access_minutes * 60,
            refresh_expires_in=self._settings.jwt_refresh_days * 86_400,
        )

    # --- verify ---

    def _decode(self, token: str) -> Dict[str, Any]:
        try:
            return jwt.decode(
                token,
                self._settings.jwt_secret_key,
                algorithms=[self._settings.jwt_algorithm],
                options={"require": ["exp", "iat", "sub", "typ", "jti"]},
            )
        except jwt.ExpiredSignatureError:
            GW_AUTH_EVENTS_TOTAL.labels(event="verify", result="expired").inc()
            raise TokenExpiredError()
        except jwt.InvalidTokenError as e:
            GW_AUTH_EVENTS_TOTAL.labels(event="verify", result="invalid").inc()
            raise InvalidTokenError(str(e))

    async def verify_access(self, token: str) -> GatewayPrincipal:
        claims = self._decode(token)
        if claims.get("typ") != _ACCESS:
            raise InvalidTokenError("Not an access token")
        if await self._blacklist.contains(claims["jti"]):
            GW_AUTH_EVENTS_TOTAL.labels(event="verify", result="revoked").inc()
            raise TokenRevokedError()
        principal = self._principal_from_claims(claims)
        GW_AUTH_EVENTS_TOTAL.labels(event="verify", result="success").inc()
        return principal

    async def verify_refresh(self, token: str) -> Dict[str, Any]:
        claims = self._decode(token)
        if claims.get("typ") != _REFRESH:
            raise InvalidTokenError("Not a refresh token")
        if await self._blacklist.contains(claims["jti"]):
            raise TokenRevokedError()
        return claims

    # --- refresh (rotation) ---

    async def refresh(
        self,
        refresh_token: str,
        roles: Optional[List[GatewayRole]] = None,
        permissions: Optional[List[str]] = None,
    ) -> TokenPair:
        claims = await self.verify_refresh(refresh_token)
        # revoke the old refresh token immediately
        ttl = max(0, int(claims["exp"] - datetime.now(tz=timezone.utc).timestamp()))
        await self._blacklist.add(claims["jti"], ttl)
        pair = self.issue(
            user_id=claims["sub"],
            user_type=GatewayUserType(claims.get("user_type", "B2C_CONSUMER")),
            roles=roles,
            permissions=permissions,
            username=claims.get("username"),
            company_id=claims.get("company_id"),
        )
        GW_AUTH_EVENTS_TOTAL.labels(event="refresh", result="success").inc()
        return pair

    async def revoke(self, token: str) -> bool:
        """Blacklist the token so future verification fails."""
        try:
            claims = self._decode(token)
        except TokenExpiredError:
            # Already dead — no-op.
            GW_AUTH_EVENTS_TOTAL.labels(event="revoke", result="already_expired").inc()
            return False
        except InvalidTokenError:
            GW_AUTH_EVENTS_TOTAL.labels(event="revoke", result="invalid").inc()
            return False
        ttl = max(0, int(claims["exp"] - datetime.now(tz=timezone.utc).timestamp()))
        await self._blacklist.add(claims["jti"], ttl)
        GW_AUTH_EVENTS_TOTAL.labels(event="revoke", result="success").inc()
        return True

    # --- helpers ---

    @staticmethod
    def _principal_from_claims(claims: Dict[str, Any]) -> GatewayPrincipal:
        user_type_value = claims.get("user_type", "B2C_CONSUMER")
        try:
            user_type = GatewayUserType(user_type_value)
        except ValueError:
            user_type = GatewayUserType.B2C_CONSUMER

        roles = []
        for r in claims.get("roles", []):
            try:
                roles.append(GatewayRole(r))
            except ValueError:
                continue

        return GatewayPrincipal(
            user_id=claims["sub"],
            user_type=user_type,
            roles=roles,
            permissions=list(claims.get("permissions", [])),
            scopes=list(claims.get("scopes", [])),
            username=claims.get("username"),
            company_id=claims.get("company_id"),
            auth_method="jwt",
            token_id=claims.get("jti"),
            issued_at=datetime.fromtimestamp(claims["iat"], tz=timezone.utc)
            if "iat" in claims
            else None,
            expires_at=datetime.fromtimestamp(claims["exp"], tz=timezone.utc)
            if "exp" in claims
            else None,
        )


_manager: Optional[JWTManager] = None


def get_jwt_manager() -> JWTManager:
    global _manager
    if _manager is None:
        _manager = JWTManager()
    return _manager


def reset_jwt_manager() -> None:
    global _manager
    _manager = None
