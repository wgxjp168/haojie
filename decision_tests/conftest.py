"""Shared fixtures + env setup for decision-engine tests."""
from __future__ import annotations

import os

# Force deterministic settings *before* any decision module is imported.
os.environ.setdefault("DECISION_ENV", "testing")
os.environ.setdefault("DECISION_STRATEGY", "rules")        # ML deps may be absent
os.environ.setdefault("DECISION_ML_MODEL_KIND", "dummy")
os.environ.setdefault("DECISION_CACHE_ENABLED", "true")
os.environ.setdefault("DECISION_REDIS_URL", "")            # memory-only cache
os.environ.setdefault("DECISION_API_KEY_REQUIRED", "false")
os.environ.setdefault("DECISION_PRELOAD_MODEL", "false")
os.environ.setdefault("DECISION_CONFIDENT_THRESHOLD", "0.55")
os.environ.setdefault("DECISION_LOW_CONFIDENCE_THRESHOLD", "0.4")
os.environ.setdefault("DECISION_RISK_REVIEW_THRESHOLD", "0.7")
os.environ.setdefault("DECISION_METRICS_ENABLED", "true")
os.environ.setdefault("DECISION_MAX_AUTO_APPROVE_AMOUNT", "50000")
os.environ.setdefault("DECISION_REQUIRE_HUMAN_REVIEW_ABOVE", "100000")

import pytest  # noqa: E402

from decision.config import reset_decision_settings  # noqa: E402
from decision.service import reset_decision_service  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_decision_singletons():
    yield
    reset_decision_settings()
    reset_decision_service()


@pytest.fixture
def fresh_service():
    """Fresh service instance for every test."""
    reset_decision_settings()
    reset_decision_service()
    from decision.service import DecisionService

    return DecisionService()


@pytest.fixture
def make_request():
    """Factory for building DecisionRequest objects in tests."""
    from decision.schemas import (
        BuyerContext,
        DecisionRequest,
        IntentSignal,
        ProductContext,
    )

    def _make(
        *,
        intent: str = "search_product",
        intent_confidence: float = 0.85,
        user_type: str = "b2c",
        budget: float | None = None,
        unit_price: float | None = None,
        quantity: int | None = None,
        in_stock: bool | None = None,
        supplier_rating: float | None = None,
        urgency: str = "medium",
        language: str = "en",
        **extra,
    ):
        return DecisionRequest(
            intent=IntentSignal(intent=intent, confidence=intent_confidence),
            buyer=BuyerContext(
                user_type=user_type,
                budget=budget,
                **{k: v for k, v in extra.items() if k in {"history_orders", "history_returns", "risk_score", "loyalty_tier", "region"}},
            ),
            product=ProductContext(
                unit_price=unit_price,
                quantity=quantity,
                in_stock=in_stock,
                supplier_rating=supplier_rating,
                **{k: v for k, v in extra.items() if k in {"product_id", "promotion_active", "discount_pct", "has_alternatives", "lead_time_days"}},
            ),
            urgency=urgency,
            language=language,
        )

    return _make
