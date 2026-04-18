"""Catch-all proxy that forwards ``/api/v1/*`` to the upstream input processor.

Flow per request:

    dependencies → RBAC policy → rate limiter → circuit breaker (via forwarder)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request, Response

from gateway.api.dependencies import get_principal, rate_limit_dependency
from gateway.auth.rbac import RBACPolicy, default_policy
from gateway.auth.user_types import GatewayPrincipal
from gateway.routing.forwarder import UpstreamForwarder, get_forwarder

router = APIRouter(prefix="/api/v1", tags=["proxy"])


def _get_policy() -> RBACPolicy:
    return default_policy()


async def _proxy(
    request: Request,
    path_suffix: str,
    principal: GatewayPrincipal,
    forwarder: UpstreamForwarder,
    policy: RBACPolicy,
) -> Response:
    full_path = request.url.path
    policy.authorize(principal, request.method, full_path)

    body = await request.body()
    upstream_response = await forwarder.forward(
        method=request.method,
        path=full_path,
        headers=dict(request.headers),
        params=dict(request.query_params),
        content=body if body else None,
    )
    headers = {
        k: v
        for k, v in upstream_response.headers.items()
        if k.lower() not in {"content-encoding", "transfer-encoding", "connection"}
    }
    # Attach rate-limit telemetry headers when present.
    decision = getattr(request.state, "rate_limit", None)
    if decision is not None:
        headers["X-RateLimit-Limit"] = str(decision.limit)
        headers["X-RateLimit-Remaining"] = str(decision.remaining)
        headers["X-RateLimit-Reset"] = f"{decision.reset_in_seconds:.2f}"
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=headers,
        media_type=upstream_response.headers.get("content-type"),
    )


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def proxy_any(
    request: Request,
    path: str,
    principal: GatewayPrincipal = Depends(rate_limit_dependency),
    forwarder: UpstreamForwarder = Depends(get_forwarder),
    policy: RBACPolicy = Depends(_get_policy),
):
    return await _proxy(request, path, principal, forwarder, policy)
