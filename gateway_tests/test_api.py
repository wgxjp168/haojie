"""End-to-end gateway HTTP tests using FastAPI TestClient.

We stub out the upstream httpx client on the singleton forwarder so tests
don't depend on a live Part 1 service.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.main import app
from gateway.routing.forwarder import get_forwarder


def _mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/inputs/text"):
            return httpx.Response(200, json={"echo": "ok", "path": request.url.path})
        if request.url.path.startswith("/api/v1/system"):
            return httpx.Response(200, json={"service": "input-processor"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.fixture
def client():
    # Function-scoped so the autouse reset in conftest runs first each test,
    # then we patch the fresh forwarder singleton.
    with TestClient(app) as c:
        forwarder = get_forwarder()
        forwarder._client = httpx.AsyncClient(
            transport=_mock_transport(), base_url="http://upstream.invalid"
        )
        yield c


def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"]
    assert "/auth" in body["endpoints"]["auth"]

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text exposition format starts with comments.
    assert r.text.startswith("# HELP") or "gateway_requests_total" in r.text


def test_login_and_whoami_and_logout(client):
    r = client.post(
        "/auth/token", json={"username": "b2c", "password": "s3cret!"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    access = body["access_token"]
    refresh = body["refresh_token"]

    # Who am I?
    r = client.get("/auth/whoami", headers={"Authorization": f"Bearer {access}"})
    assert r.status_code == 200, r.text
    whoami = r.json()
    assert whoami["user_id"] == "u-b2c-1"
    assert whoami["user_type"] == "B2C_CONSUMER"

    # Refresh
    r = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200, r.text
    new_access = r.json()["access_token"]
    assert new_access != access

    # Old refresh now revoked
    r = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 401

    # Logout revokes the new access token
    r = client.post(
        "/auth/logout", headers={"Authorization": f"Bearer {new_access}"}
    )
    assert r.status_code == 200

    # Subsequent use rejected
    r = client.get("/auth/whoami", headers={"Authorization": f"Bearer {new_access}"})
    assert r.status_code == 401


def test_login_wrong_password(client):
    r = client.post("/auth/token", json={"username": "b2c", "password": "nope"})
    assert r.status_code == 401


def test_proxy_requires_auth_and_forwards(client):
    # Unauthenticated
    r = client.post("/api/v1/inputs/text", json={"text": "hi"})
    assert r.status_code == 401

    # Authenticated B2C user
    login = client.post(
        "/auth/token", json={"username": "b2c", "password": "s3cret!"}
    ).json()
    token = login["access_token"]
    r = client.post(
        "/api/v1/inputs/text",
        json={"text": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["echo"] == "ok"
    # Rate-limit headers surfaced
    assert "x-ratelimit-limit" in {k.lower() for k in r.headers.keys()}


def test_proxy_admin_path_forbidden_for_b2c(client):
    login = client.post(
        "/auth/token", json={"username": "b2c", "password": "s3cret!"}
    ).json()
    token = login["access_token"]
    r = client.get(
        "/api/v1/system/stats", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403


def test_proxy_admin_path_allowed_for_admin(client):
    login = client.post(
        "/auth/token", json={"username": "admin", "password": "s3cret!"}
    ).json()
    token = login["access_token"]
    r = client.get(
        "/api/v1/system/stats", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["service"] == "input-processor"


def test_admin_circuit_endpoints(client):
    login = client.post(
        "/auth/token", json={"username": "admin", "password": "s3cret!"}
    ).json()
    token = login["access_token"]
    r = client.get("/admin/circuits", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "circuits" in r.json()

    r = client.post(
        "/admin/circuits/some-service/reset",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["state"] == "closed"


def test_admin_ws_connections_requires_admin(client):
    login = client.post(
        "/auth/token", json={"username": "b2c", "password": "s3cret!"}
    ).json()
    token = login["access_token"]
    r = client.get(
        "/admin/ws/connections", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403
