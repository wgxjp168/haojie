"""Role-based access control (RBAC).

We keep this declarative: a policy maps (method, path-prefix) to the set of
permissions any one of which satisfies the check. The policy is evaluated at
gateway-edge, so downstream services can trust the propagated principal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from fastapi import Depends

from gateway.auth.user_types import GatewayPrincipal, GatewayRole, GatewayUserType
from gateway.core.exceptions import ForbiddenError, UnauthorizedError


@dataclass
class PolicyRule:
    method: str  # "GET" / "POST" / "*"
    path_prefix: str  # e.g. "/api/v1/inputs"
    required_permissions: Tuple[str, ...] = ()  # any of these
    required_roles: Tuple[GatewayRole, ...] = ()
    allow_anonymous: bool = False

    def matches(self, method: str, path: str) -> bool:
        if self.method != "*" and self.method.upper() != method.upper():
            return False
        return path.startswith(self.path_prefix)


@dataclass
class RBACPolicy:
    """Ordered list of rules — first match wins."""

    rules: List[PolicyRule] = field(default_factory=list)

    def add(self, rule: PolicyRule) -> "RBACPolicy":
        self.rules.append(rule)
        return self

    def authorize(
        self, principal: GatewayPrincipal, method: str, path: str
    ) -> None:
        for rule in self.rules:
            if not rule.matches(method, path):
                continue
            if rule.allow_anonymous:
                return
            if principal.is_anonymous():
                raise UnauthorizedError("Authentication required")
            if rule.required_roles and not principal.has_any_role(
                list(rule.required_roles)
            ):
                raise ForbiddenError(
                    "Missing required role",
                    required=[r.value for r in rule.required_roles],
                )
            if rule.required_permissions and not any(
                principal.has_permission(p) for p in rule.required_permissions
            ):
                raise ForbiddenError(
                    "Missing required permission",
                    required=list(rule.required_permissions),
                )
            return
        # Default deny: if no rule matched, allow only non-anonymous callers.
        if principal.is_anonymous():
            raise UnauthorizedError("Authentication required")


# --- default policy suitable for Part 1 ---


def _default_policy() -> RBACPolicy:
    p = RBACPolicy()

    # Public
    p.add(PolicyRule("*", "/health", allow_anonymous=True))
    p.add(PolicyRule("*", "/metrics", allow_anonymous=True))
    p.add(PolicyRule("*", "/docs", allow_anonymous=True))
    p.add(PolicyRule("*", "/openapi.json", allow_anonymous=True))
    p.add(PolicyRule("POST", "/auth/token", allow_anonymous=True))
    p.add(PolicyRule("POST", "/auth/refresh", allow_anonymous=True))

    # Core APIs — any authenticated user may process their own inputs.
    p.add(PolicyRule("*", "/api/v1/inputs", required_permissions=("inputs:write", "inputs:*")))
    p.add(PolicyRule("*", "/api/v1/sessions", required_permissions=("sessions:read", "sessions:*")))
    p.add(PolicyRule("*", "/api/v1/system", required_roles=(GatewayRole.ADMIN, GatewayRole.SUPER_ADMIN, GatewayRole.ANALYST)))

    # WebSocket
    p.add(PolicyRule("*", "/ws", required_permissions=("ws:connect", "inputs:*")))

    # Admin
    p.add(PolicyRule("*", "/admin", required_roles=(GatewayRole.ADMIN, GatewayRole.SUPER_ADMIN)))

    return p


_DEFAULT_POLICY: Optional[RBACPolicy] = None


def default_policy() -> RBACPolicy:
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = _default_policy()
    return _DEFAULT_POLICY


def require_permission(permission: str) -> Callable:
    """Create a FastAPI dependency that checks a single permission."""

    async def _dep(principal: GatewayPrincipal) -> GatewayPrincipal:
        if not principal.has_permission(permission):
            raise ForbiddenError(
                "Missing required permission", required=[permission]
            )
        return principal

    return _dep
