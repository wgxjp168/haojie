"""FastAPI dependencies for authentication and shared services."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import Settings, get_settings
from core.exceptions import InputProcessorError
from core.security import decode_token, verify_api_key
from entities.enums import UserRole, UserType
from entities.user import User, UserContact
from managers.input_manager import MultiModalInputManager, get_input_manager
from managers.session_manager import SessionManager, get_session_manager

# Auto-error=False so we can support both JWT and API-key paths.
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> User:
    """Resolve the caller identity from Bearer JWT or API-Key header.

    When `api_key_required` is false and no credentials are supplied,
    an anonymous system user is returned. This is convenient for local
    development and for the healthcheck endpoints.
    """
    if credentials is not None and credentials.scheme.lower() == "bearer":
        try:
            claims = decode_token(credentials.credentials, settings)
        except InputProcessorError as e:
            raise HTTPException(status_code=e.status_code, detail=e.to_dict())
        subject = claims.get("sub")
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "UNAUTHORIZED", "message": "missing sub"},
            )
        role_values = claims.get("roles", [])
        roles = [UserRole(r) for r in role_values if r in UserRole.__members__.values()]
        user_type = UserType.B2C_CONSUMER
        if UserRole.B2B_PURCHASER in roles:
            user_type = UserType.B2B_PURCHASER
        elif UserRole.ADMIN in roles or UserRole.SUPER_ADMIN in roles:
            user_type = UserType.ADMIN
        return User(
            user_id=subject,
            username=claims.get("username", f"user_{subject[:8]}"),
            display_name=claims.get("name", f"user_{subject[:8]}"),
            user_type=user_type,
            roles=roles,
            contact=UserContact(
                email=claims.get("email", f"{subject}@unknown"),
                phone=claims.get("phone", "0000000000"),
            ),
        )

    if x_api_key is not None:
        if not verify_api_key(x_api_key, settings.api_keys):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "UNAUTHORIZED", "message": "invalid api key"},
            )
        return User(
            user_id=f"apikey-{x_api_key[-6:]}",
            username="api_user",
            display_name="API User",
            user_type=UserType.B2B_PURCHASER,
            roles=[UserRole.B2B_PURCHASER],
            contact=UserContact(email="api@internal", phone="0000000000"),
        )

    if settings.api_key_required:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "authentication required"},
        )

    return User(
        user_id="anonymous",
        username="anonymous",
        display_name="Anonymous",
        user_type=UserType.B2C_CONSUMER,
        roles=[UserRole.GUEST],
        contact=UserContact(email="anon@local", phone="0000000000"),
    )


def require_roles(*roles: UserRole):
    """Factory that returns a dependency enforcing the given roles."""

    async def _dep(user: User = Depends(get_current_user)) -> User:
        if not user.has_any_role(list(roles)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "FORBIDDEN",
                    "message": f"requires one of {[r.value for r in roles]}",
                },
            )
        return user

    return _dep


def get_manager() -> MultiModalInputManager:
    return get_input_manager()


def get_sessions() -> SessionManager:
    return get_session_manager()
