"""User domain entity."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from entities.enums import UserRole, UserType


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?\d{7,15}$")


@dataclass
class UserContact:
    email: str
    phone: str
    address: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not _EMAIL_RE.match(self.email or ""):
            errors.append("invalid email")
        if not _PHONE_RE.match(self.phone or ""):
            errors.append("invalid phone")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {"email": self.email, "phone": self.phone, "address": self.address}


@dataclass
class User:
    user_id: str
    username: str
    display_name: str
    user_type: UserType
    contact: UserContact
    roles: List[UserRole] = field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    company_name: Optional[str] = None  # B2B users
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_login_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_id:
            self.user_id = str(uuid.uuid4())
        if not self.roles:
            default_role = {
                UserType.B2B_PURCHASER: UserRole.B2B_PURCHASER,
                UserType.B2C_CONSUMER: UserRole.B2C_CONSUMER,
                UserType.ADMIN: UserRole.ADMIN,
                UserType.SYSTEM: UserRole.ADMIN,
            }.get(self.user_type)
            if default_role:
                self.roles = [default_role]

    def has_role(self, role: UserRole) -> bool:
        return role in self.roles

    def has_any_role(self, roles: List[UserRole]) -> bool:
        return any(r in self.roles for r in roles)

    def validate(self) -> List[str]:
        errors = self.contact.validate()
        if not self.username or len(self.username) < 3:
            errors.append("username must be at least 3 chars")
        if not self.display_name:
            errors.append("display_name required")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "user_type": self.user_type.value,
            "roles": [r.value for r in self.roles],
            "contact": self.contact.to_dict(),
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "company_name": self.company_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def create_b2c(
        cls,
        username: str,
        display_name: str,
        email: str,
        phone: str,
    ) -> "User":
        return cls(
            user_id=str(uuid.uuid4()),
            username=username,
            display_name=display_name,
            user_type=UserType.B2C_CONSUMER,
            contact=UserContact(email=email, phone=phone),
        )

    @classmethod
    def create_b2b(
        cls,
        username: str,
        display_name: str,
        email: str,
        phone: str,
        company_name: str,
    ) -> "User":
        return cls(
            user_id=str(uuid.uuid4()),
            username=username,
            display_name=display_name,
            user_type=UserType.B2B_PURCHASER,
            contact=UserContact(email=email, phone=phone),
            company_name=company_name,
        )
