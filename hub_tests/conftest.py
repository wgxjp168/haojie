"""Shared fixtures + env setup for hub-service tests.

The hub's integration tests exercise the four downstream services
embedded in-process, so every ``REDIS_URL`` is cleared and every
downstream service is pushed into its fastest, least-dependent mode.
"""
from __future__ import annotations

import os

# --- Hub config ---
os.environ.setdefault("HUB_ENV", "testing")
os.environ.setdefault("HUB_INTENT_MODE", "embedded")
os.environ.setdefault("HUB_DECISION_MODE", "embedded")
os.environ.setdefault("HUB_LLM_MODE", "embedded")
os.environ.setdefault("HUB_REPORT_MODE", "embedded")
os.environ.setdefault("HUB_API_KEY_REQUIRED", "false")
os.environ.setdefault("HUB_METRICS_ENABLED", "true")
os.environ.setdefault("HUB_MAX_BATCH_SIZE", "16")
os.environ.setdefault("HUB_PIPELINE_TIMEOUT_SECONDS", "30")
os.environ.setdefault("HUB_HTTP_MAX_RETRIES", "1")
os.environ.setdefault("HUB_HTTP_RETRY_BACKOFF_SECONDS", "0.05")
os.environ.setdefault("HUB_CB_FAILURE_THRESHOLD", "3")
os.environ.setdefault("HUB_CB_RECOVERY_SECONDS", "1")

# --- Downstream service config (embedded mode) ---
os.environ.setdefault("INTENT_ENV", "testing")
os.environ.setdefault("INTENT_BACKEND", "rules")
os.environ.setdefault("INTENT_REDIS_URL", "")
os.environ.setdefault("INTENT_API_KEY_REQUIRED", "false")
os.environ.setdefault("INTENT_PRELOAD_MODEL", "false")
os.environ.setdefault("INTENT_CONFIDENT_THRESHOLD", "0.5")

os.environ.setdefault("DECISION_ENV", "testing")
os.environ.setdefault("DECISION_STRATEGY", "rules")
os.environ.setdefault("DECISION_REDIS_URL", "")
os.environ.setdefault("DECISION_API_KEY_REQUIRED", "false")
os.environ.setdefault("DECISION_PRELOAD_MODEL", "false")

os.environ.setdefault("LLM_ENV", "testing")
os.environ.setdefault("LLM_DEFAULT_MODEL_ID", "stub-small")
os.environ.setdefault("LLM_DEFAULT_STRATEGY", "single")
os.environ.setdefault("LLM_FALLBACK_CHAIN", "stub-small,stub-fast")
os.environ.setdefault("LLM_REDIS_URL", "")
os.environ.setdefault("LLM_API_KEY_REQUIRED", "false")
os.environ.setdefault("LLM_RATE_LIMIT_PER_MINUTE", "0")

os.environ.setdefault("REPORT_ENV", "testing")
os.environ.setdefault("REPORT_STORAGE_BACKENDS", "memory")
os.environ.setdefault("REPORT_REDIS_URL", "")
os.environ.setdefault("REPORT_API_KEY_REQUIRED", "false")

import pytest  # noqa: E402

from hub.config import reset_hub_settings  # noqa: E402
from hub.service import reset_hub_service  # noqa: E402
from intent.config import reset_intent_settings  # noqa: E402
from intent.service import reset_intent_service  # noqa: E402
from decision.config import reset_decision_settings  # noqa: E402
from decision.service import reset_decision_service  # noqa: E402
from llm.config import reset_llm_settings  # noqa: E402
from llm.service import reset_llm_service  # noqa: E402
from report.config import reset_report_settings  # noqa: E402
from report.service import reset_report_service  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_all_singletons():
    yield
    reset_hub_settings()
    reset_hub_service()
    reset_intent_settings()
    reset_intent_service()
    reset_decision_settings()
    reset_decision_service()
    reset_llm_settings()
    reset_llm_service()
    reset_report_settings()
    reset_report_service()


@pytest.fixture
def fresh_service():
    reset_hub_settings()
    reset_hub_service()
    from hub.service import HubService

    return HubService()


@pytest.fixture
def sample_b2b_request():
    """A typical B2B procurement request."""
    from hub.schemas import (
        PipelineBuyer,
        PipelineOptions,
        PipelineProduct,
        PipelineRequest,
    )

    return PipelineRequest(
        request_id="hub-test-001",
        session_id="sess-001",
        text="这个办公椅多少钱？我想批量采购 50 把。",
        language="zh-CN",
        user_type="b2b",
        buyer=PipelineBuyer(user_id="u-b2b-1", user_type="b2b", budget=100_000.0),
        product=PipelineProduct(
            product_id="P001",
            name="办公椅",
            unit_price=500.0,
            quantity=50,
            in_stock=True,
            supplier_rating=4.5,
        ),
        options=PipelineOptions(
            include_llm=False,
            include_report=True,
            report_format="markdown",
            report_audience="technical",
        ),
    )
