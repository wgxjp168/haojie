"""Tests for JWT issuance, verification, refresh, revocation, RBAC."""
from __future__ import annotations

import pytest

from gateway.auth.jwt_manager import JWTManager, get_jwt_manager
from gateway.auth.rbac import PolicyRule, RBACPolicy, default_policy
from gateway.auth.user_types import GatewayPrincipal, GatewayRole, GatewayUserType
from gateway.core.exceptions import (
    ForbiddenError,
    InvalidTokenError,
    TokenRevokedError,
    UnauthorizedError,
)


@pytest.mark.asyncio
async def test_jwt_issue_and_verify():
    mgr: JWTManager = get_jwt_manager()
    pair = mgr.issue(
        user_id="u1",
        user_type=GatewayUserType.B2C_CONSUMER,
        roles=[GatewayRole.B2C_CONSUMER],
        permissions=["inputs:write"],
    )
    principal = await mgr.verify_access(pair.access_token)
    assert principal.user_id == "u1"
    assert principal.user_type == GatewayUserType.B2C_CONSUMER
    assert GatewayRole.B2C_CONSUMER in principal.roles
    assert principal.has_permission("inputs:write")


@pytest.mark.asyncio
async def test_jwt_refresh_rotates_token():
    mgr = get_jwt_manager()
    pair = mgr.issue(
        user_id="u2",
        user_type=GatewayUserType.B2B_PURCHASER,
        roles=[GatewayRole.B2B_PURCHASER],
    )
    new_pair = await mgr.refresh(pair.refresh_token)
    assert new_pair.access_token != pair.access_token
    assert new_pair.refresh_token != pair.refresh_token
    # Old refresh token must now be rejected.
    with pytest.raises(TokenRevokedError):
        await mgr.refresh(pair.refresh_token)


@pytest.mark.asyncio
async def test_jwt_revoke_rejects_future_use():
    mgr = get_jwt_manager()
    pair = mgr.issue(
        user_id="u3",
        user_type=GatewayUserType.ADMIN,
        roles=[GatewayRole.ADMIN],
    )
    assert await mgr.revoke(pair.access_token) is True
    with pytest.raises(TokenRevokedError):
        await mgr.verify_access(pair.access_token)


@pytest.mark.asyncio
async def test_jwt_invalid_token():
    mgr = get_jwt_manager()
    with pytest.raises(InvalidTokenError):
        await mgr.verify_access("not-a-token")


@pytest.mark.asyncio
async def test_jwt_wrong_typ_rejected():
    mgr = get_jwt_manager()
    pair = mgr.issue(
        user_id="u4",
        user_type=GatewayUserType.INTERNAL,
    )
    # Using refresh as access, or vice versa, must raise InvalidTokenError.
    with pytest.raises(InvalidTokenError):
        await mgr.verify_access(pair.refresh_token)
    with pytest.raises(InvalidTokenError):
        await mgr.verify_refresh(pair.access_token)


def test_principal_permission_wildcard():
    principal = GatewayPrincipal(
        user_id="u",
        user_type=GatewayUserType.ADMIN,
        permissions=["inputs:*"],
    )
    assert principal.has_permission("inputs:write")
    assert principal.has_permission("inputs:read")
    assert not principal.has_permission("sessions:read")

    super_principal = GatewayPrincipal(
        user_id="s",
        user_type=GatewayUserType.ADMIN,
        permissions=["*"],
    )
    assert super_principal.has_permission("anything:goes")


def test_rbac_policy_allows_anonymous_public_paths():
    p = default_policy()
    anon = GatewayPrincipal.anonymous("1.2.3.4")
    # /health is allow-anonymous
    p.authorize(anon, "GET", "/health")


def test_rbac_policy_rejects_anonymous_on_protected_paths():
    p = default_policy()
    anon = GatewayPrincipal.anonymous()
    with pytest.raises(UnauthorizedError):
        p.authorize(anon, "POST", "/api/v1/inputs/text")


def test_rbac_policy_requires_role_for_system():
    p = default_policy()
    user = GatewayPrincipal(
        user_id="u",
        user_type=GatewayUserType.B2C_CONSUMER,
        roles=[GatewayRole.B2C_CONSUMER],
        permissions=["inputs:write"],
    )
    with pytest.raises(ForbiddenError):
        p.authorize(user, "GET", "/api/v1/system/stats")

    admin = GatewayPrincipal(
        user_id="a",
        user_type=GatewayUserType.ADMIN,
        roles=[GatewayRole.ADMIN],
    )
    p.authorize(admin, "GET", "/api/v1/system/stats")


def test_rbac_policy_custom():
    policy = RBACPolicy()
    policy.add(PolicyRule("GET", "/public", allow_anonymous=True))
    policy.add(
        PolicyRule(
            "*",
            "/secret",
            required_permissions=("secret:read",),
        )
    )
    anon = GatewayPrincipal.anonymous()
    policy.authorize(anon, "GET", "/public")
    with pytest.raises(UnauthorizedError):
        policy.authorize(anon, "GET", "/secret")

    viewer = GatewayPrincipal(
        user_id="u",
        user_type=GatewayUserType.INTERNAL,
        permissions=["secret:read"],
    )
    policy.authorize(viewer, "GET", "/secret")
