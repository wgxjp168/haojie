"""User type / role enums and the authenticated-principal dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class GatewayUserType(str, Enum):
    B2B_PURCHASER = "B2B_PURCHASER"
    B2C_CONSUMER = "B2C_CONSUMER"
    PARTNER = "PARTNER"
    INTERNAL = "INTERNAL"
    ADMIN = "ADMIN"
    ANONYMOUS = "ANONYMOUS"


class GatewayRole(str, Enum):
    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    B2B_PURCHASER = "B2B_PURCHASER"
    B2C_CONSUMER = "B2C_CONSUMER"
    PARTNER = "PARTNER"
    INTERNAL_USER = "INTERNAL_USER"
    ANALYST = "ANALYST"
    GUEST = "GUEST"


@dataclass
class GatewayPrincipal:
    """A resolved caller identity — the result of authentication."""

    user_id: str
    user_type: GatewayUserType
    roles: List[GatewayRole] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    username: Optional[str] = None
    company_id: Optional[str] = None
    auth_method: str = "anonymous"  # jwt | api_key | anonymous
    token_id: Optional[str] = None
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def is_anonymous(self) -> bool:
        return self.user_type == GatewayUserType.ANONYMOUS

    def has_role(self, role: GatewayRole) -> bool:
        return role in self.roles

    def has_any_role(self, roles: List[GatewayRole]) -> bool:
        return any(r in self.roles for r in roles)

    def has_permission(self, permission: str) -> bool:
        # Wildcard support: "inputs:*" grants "inputs:read" etc.
        if permission in self.permissions:
            return True
        resource = permission.split(":", 1)[0]
        return f"{resource}:*" in self.permissions or "*" in self.permissions

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "user_type": self.user_type.value,
            "roles": [r.value for r in self.roles],
            "permissions": self.permissions,
            "scopes": self.scopes,
            "username": self.username,
            "company_id": self.company_id,
            "auth_method": self.auth_method,
            "token_id": self.token_id,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def anonymous(cls, ip_address: Optional[str] = None) -> "GatewayPrincipal":
        return cls(
            user_id=f"anon:{ip_address or 'unknown'}",
            user_type=GatewayUserType.ANONYMOUS,
            roles=[GatewayRole.GUEST],
            auth_method="anonymous",
        )
