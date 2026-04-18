"""Auth endpoints.

For brevity, the gateway accepts a trusted internal credential check that
returns a :class:`GatewayPrincipal`. In a real deployment the credential
check talks to a user service — swap ``authenticate_credentials`` for your
org's IdP.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from gateway.auth.jwt_manager import JWTManager, TokenPair, get_jwt_manager
from gateway.auth.user_types import GatewayPrincipal, GatewayRole, GatewayUserType
from gateway.core.exceptions import UnauthorizedError
from gateway.core.logging import get_logger
from gateway.core.security import verify_password

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# --- schemas ---


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    refresh_expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class WhoAmIResponse(BaseModel):
    user_id: str
    user_type: str
    roles: List[str]
    permissions: List[str]
    username: Optional[str] = None
    auth_method: str


# --- in-memory demo user store (replace with real IdP lookup) ---

# bcrypt hashes for "s3cret!"
_DEMO_HASH_PLACEHOLDER = None


def _demo_users() -> dict:
    """Lazily hash demo passwords so tests don't pay the bcrypt cost at import."""
    from gateway.core.security import hash_password

    global _DEMO_HASH_PLACEHOLDER
    if _DEMO_HASH_PLACEHOLDER is None:
        _DEMO_HASH_PLACEHOLDER = hash_password("s3cret!")
    return {
        "b2b": {
            "user_id": "u-b2b-1",
            "password_hash": _DEMO_HASH_PLACEHOLDER,
            "user_type": GatewayUserType.B2B_PURCHASER,
            "roles": [GatewayRole.B2B_PURCHASER],
            "permissions": ["inputs:*", "sessions:*", "ws:connect"],
            "company_id": "acme",
        },
        "b2c": {
            "user_id": "u-b2c-1",
            "password_hash": _DEMO_HASH_PLACEHOLDER,
            "user_type": GatewayUserType.B2C_CONSUMER,
            "roles": [GatewayRole.B2C_CONSUMER],
            "permissions": ["inputs:write", "sessions:read", "ws:connect"],
        },
        "admin": {
            "user_id": "u-admin-1",
            "password_hash": _DEMO_HASH_PLACEHOLDER,
            "user_type": GatewayUserType.ADMIN,
            "roles": [GatewayRole.ADMIN, GatewayRole.SUPER_ADMIN],
            "permissions": ["*"],
        },
    }


def authenticate_credentials(username: str, password: str) -> Optional[dict]:
    user = _demo_users().get(username)
    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


# --- routes ---


@router.post("/token", response_model=TokenResponse, summary="Issue a token pair")
async def login(
    body: LoginRequest,
    jwt_manager: JWTManager = Depends(get_jwt_manager),
):
    user = authenticate_credentials(body.username, body.password)
    if user is None:
        raise UnauthorizedError("Invalid username or password")
    pair: TokenPair = jwt_manager.issue(
        user_id=user["user_id"],
        user_type=user["user_type"],
        roles=user["roles"],
        permissions=user["permissions"],
        username=body.username,
        company_id=user.get("company_id"),
    )
    return pair.to_dict()


@router.post("/refresh", response_model=TokenResponse, summary="Refresh tokens")
async def refresh(
    body: RefreshRequest,
    jwt_manager: JWTManager = Depends(get_jwt_manager),
):
    # Look up permissions again so changes propagate on refresh.
    # For the demo we leave permissions as-is in the token claims.
    pair = await jwt_manager.refresh(body.refresh_token)
    return pair.to_dict()


@router.post("/logout", summary="Revoke the caller's access token")
async def logout(
    authorization: str = Header(..., alias="Authorization"),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
):
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=400, detail="Bearer token required")
    revoked = await jwt_manager.revoke(token)
    return {"revoked": revoked}


@router.get("/whoami", response_model=WhoAmIResponse, summary="Describe caller")
async def whoami(
    # Imported here to avoid circular import at module load.
    principal: GatewayPrincipal = Depends(
        __import__(
            "gateway.api.dependencies", fromlist=["get_authenticated_principal"]
        ).get_authenticated_principal
    ),
):
    return WhoAmIResponse(
        user_id=principal.user_id,
        user_type=principal.user_type.value,
        roles=[r.value for r in principal.roles],
        permissions=principal.permissions,
        username=principal.username,
        auth_method=principal.auth_method,
    )
