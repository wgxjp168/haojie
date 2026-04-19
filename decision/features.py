"""Feature extraction from a ``DecisionRequest``.

Strategies (rules / ML / ensemble) consume the same ``FeatureVector`` so
that swapping one for the other does not change the contract.

The extractor is intentionally pure / dependency-free so it can run
inside the test suite without optional ML dependencies.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from decision.schemas import DecisionRequest


@dataclass(frozen=True)
class FeatureVector:
    """Numerical + categorical features ready for any strategy."""

    # ---- categorical / identity ----
    intent: str
    user_type: str          # b2b | b2c
    urgency: str            # low | medium | high
    language: str

    # ---- numerical ----
    intent_confidence: float
    unit_price: float
    quantity: float
    total_amount: float
    budget: float
    price_to_budget_ratio: float
    discount_pct: float
    supplier_rating: float
    history_orders: float
    history_returns: float
    return_ratio: float
    buyer_risk_score: float
    lead_time_days: float

    # ---- boolean (encoded as 0.0 / 1.0) ----
    in_stock: float
    promotion_active: float
    has_alternatives: float
    has_budget: float
    has_product: float

    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    # ----- helpers used by rule strategy -----

    @property
    def affordability(self) -> float:
        """1.0 = affordable, 0.0 = unaffordable, 0.5 = unknown."""
        if self.budget <= 0 or self.total_amount <= 0:
            return 0.5
        if self.total_amount <= self.budget:
            return 1.0 - min(1.0, self.total_amount / self.budget) * 0.5
        return max(0.0, 1.0 - (self.total_amount / self.budget - 1.0))

    @property
    def is_high_value(self) -> bool:
        return self.total_amount > 0 and self.total_amount >= 10_000.0

    def cache_key(self) -> str:
        """Stable key for caching identical decision requests."""
        # Round floats to 4 dp so tiny floating-point jitter (e.g. from
        # a Pydantic round-trip) doesn't blow the cache.
        items = (
            ("intent", self.intent),
            ("user_type", self.user_type),
            ("urgency", self.urgency),
            ("language", self.language),
            ("intent_confidence", round(self.intent_confidence, 4)),
            ("unit_price", round(self.unit_price, 4)),
            ("quantity", int(self.quantity)),
            ("total_amount", round(self.total_amount, 4)),
            ("budget", round(self.budget, 4)),
            ("discount_pct", round(self.discount_pct, 4)),
            ("supplier_rating", round(self.supplier_rating, 4)),
            ("history_orders", int(self.history_orders)),
            ("history_returns", int(self.history_returns)),
            ("buyer_risk_score", round(self.buyer_risk_score, 4)),
            ("lead_time_days", int(self.lead_time_days)),
            ("in_stock", int(self.in_stock)),
            ("promotion_active", int(self.promotion_active)),
            ("has_alternatives", int(self.has_alternatives)),
            ("has_budget", int(self.has_budget)),
            ("has_product", int(self.has_product)),
        )
        blob = "|".join(f"{k}={v}" for k, v in items).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class FeatureExtractor:
    """Build a ``FeatureVector`` from a ``DecisionRequest``."""

    def __init__(self, *, default_language: str = "zh-CN") -> None:
        self.default_language = default_language

    def extract(self, request: DecisionRequest) -> FeatureVector:
        product = request.product
        buyer = request.buyer

        unit_price = float(product.unit_price or 0.0)
        quantity = float(product.quantity or 0)
        total_amount = float(product.total_amount or unit_price * max(quantity, 1.0))
        budget = float(buyer.budget or 0.0)
        if budget > 0 and total_amount > 0:
            ratio = total_amount / budget
        elif budget > 0:
            ratio = 0.0
        else:
            ratio = 0.0

        history_orders = float(buyer.history_orders)
        history_returns = float(buyer.history_returns)
        return_ratio = (
            history_returns / history_orders if history_orders > 0 else 0.0
        )

        return FeatureVector(
            intent=request.intent.intent,
            user_type=buyer.user_type,
            urgency=request.urgency,
            language=request.language or buyer.preferred_language or self.default_language,
            intent_confidence=float(request.intent.confidence),
            unit_price=unit_price,
            quantity=quantity,
            total_amount=total_amount,
            budget=budget,
            price_to_budget_ratio=_clip(ratio, 0.0, 10.0),
            discount_pct=float(product.discount_pct or 0.0),
            supplier_rating=float(product.supplier_rating or 0.0),
            history_orders=history_orders,
            history_returns=history_returns,
            return_ratio=_clip(return_ratio, 0.0, 1.0),
            buyer_risk_score=float(buyer.risk_score or 0.0),
            lead_time_days=float(product.lead_time_days or 0),
            in_stock=_bool_to_float(product.in_stock),
            promotion_active=_bool_to_float(product.promotion_active),
            has_alternatives=_bool_to_float(product.has_alternatives),
            has_budget=1.0 if budget > 0 else 0.0,
            has_product=1.0 if (product.product_id or product.sku) else 0.0,
            raw_metadata=dict(request.metadata),
        )


def _bool_to_float(v: Optional[bool]) -> float:
    if v is None:
        return 0.5  # unknown
    return 1.0 if v else 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return lo
    return max(lo, min(hi, value))
