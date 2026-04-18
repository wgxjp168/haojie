"""Shared fixtures + env setup for intent tests."""
from __future__ import annotations

import os

# Force deterministic settings *before* any intent module is imported.
os.environ.setdefault("INTENT_ENV", "testing")
os.environ.setdefault("INTENT_BACKEND", "rules")   # transformer deps not installed
os.environ.setdefault("INTENT_CACHE_ENABLED", "true")
os.environ.setdefault("INTENT_REDIS_URL", "")      # force memory-only cache
os.environ.setdefault("INTENT_API_KEY_REQUIRED", "false")
os.environ.setdefault("INTENT_PRELOAD_MODEL", "false")
os.environ.setdefault("INTENT_CONFIDENT_THRESHOLD", "0.5")
os.environ.setdefault("INTENT_LOW_CONFIDENCE_THRESHOLD", "0.4")
os.environ.setdefault("INTENT_METRICS_ENABLED", "true")

import pytest  # noqa: E402

from intent.config import reset_intent_settings  # noqa: E402
from intent.service import reset_intent_service  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_intent_singletons():
    yield
    reset_intent_settings()
    reset_intent_service()


@pytest.fixture
def fresh_service():
    """Fresh service instance for every test."""
    reset_intent_settings()
    reset_intent_service()
    from intent.service import IntentService
    return IntentService()
