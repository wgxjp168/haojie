"""Shared fixtures + env setup for LLM-service tests."""
from __future__ import annotations

import os

# Force deterministic settings *before* any llm module is imported.
os.environ.setdefault("LLM_ENV", "testing")
os.environ.setdefault("LLM_DEFAULT_MODEL_ID", "stub-small")
os.environ.setdefault("LLM_DEFAULT_STRATEGY", "fallback")
os.environ.setdefault("LLM_FALLBACK_CHAIN", "stub-small,stub-fast")
os.environ.setdefault("LLM_CACHE_ENABLED", "true")
os.environ.setdefault("LLM_REDIS_URL", "")
os.environ.setdefault("LLM_API_KEY_REQUIRED", "false")
os.environ.setdefault("LLM_METRICS_ENABLED", "true")
os.environ.setdefault("LLM_CONFIDENT_THRESHOLD", "0.5")
os.environ.setdefault("LLM_LOW_CONFIDENCE_THRESHOLD", "0.3")
os.environ.setdefault("LLM_RATE_LIMIT_PER_MINUTE", "6000")
os.environ.setdefault("LLM_RATE_LIMIT_BURST", "600")
os.environ.setdefault("LLM_MAX_REQUEST_COST_USD", "1.0")

import pytest  # noqa: E402

from llm.config import reset_llm_settings  # noqa: E402
from llm.service import reset_llm_service  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_llm_singletons():
    yield
    reset_llm_settings()
    reset_llm_service()


@pytest.fixture
def fresh_service():
    reset_llm_settings()
    reset_llm_service()
    from llm.service import LLMService

    return LLMService()
