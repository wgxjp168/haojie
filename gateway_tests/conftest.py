"""Gateway test configuration."""
from __future__ import annotations

import os

import pytest

# Set before any gateway module imports Settings.
os.environ.setdefault("GW_ENV", "testing")
os.environ.setdefault("GW_JWT_SECRET_KEY", "test-secret-key-unit-tests")
os.environ.setdefault("GW_REDIS_URL", "")  # in-memory fallback
os.environ.setdefault("GW_UPSTREAM_BASE_URL", "http://upstream.invalid")
os.environ.setdefault("GW_API_KEY_REQUIRED", "false")

from gateway.config import reload_gateway_settings  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def settings():
    return reload_gateway_settings()


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-level singletons between tests."""
    from gateway.auth.jwt_manager import reset_jwt_manager
    from gateway.auth.token_blacklist import reset_token_blacklist
    from gateway.circuit_breaker.breaker import reset_circuit_breaker_registry
    from gateway.rate_limiting.strategies import reset_rate_limiter
    from gateway.routing.forwarder import reset_forwarder
    from gateway.websocket.manager import reset_ws_manager

    yield
    reset_jwt_manager()
    reset_token_blacklist()
    reset_circuit_breaker_registry()
    reset_rate_limiter()
    reset_forwarder()
    reset_ws_manager()
