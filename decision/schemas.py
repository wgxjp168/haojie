"""Pydantic request/response schemas for the decision engine.

The wire format is intentionally explicit: ``DecisionRequest`` carries the
*intent* (typically forwarded from Part 3.1), the buyer profile, the
product/order context, and any free-form metadata. ``DecisionResponse``
emits a ranked list of candidate actions plus an explainable rationale,
risk score, and concrete next steps.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


UserType = Literal["b2b", "b2c"]
Urgency = Literal["low", "medium", "high"]


# ---------------------------------------------------------------- request


class IntentSignal(BaseModel):
    """Recognised intent (usually produced by Part 3.1)."""

    intent: str = Field(..., min_length=1, description="Canonical intent id.")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    label: Optional[str] = Field(None, description="Human-readable label.")

    @field_validator("intent")
    @classmethod
    def _strip_intent(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("intent must not be empty")
        return v


class ProductContext(BaseModel):
    """Optional product / order context the engine reasons over."""

    product_id: Optional[str] = None
    sku: Optional[str] = None
    category: Optional[str] = None
    unit_price: Optional[float] = Field(None, ge=0.0)
    quantity: Optional[int] = Field(None, ge=1)
    in_stock: Optional[bool] = None
    lead_time_days: Optional[int] = Field(None, ge=0)
    supplier_id: Optional[str] = None
    supplier_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    promotion_active: Optional[bool] = None
    discount_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    has_alternatives: Optional[bool] = None

    @property
    def total_amount(self) -> Optional[float]:
        if self.unit_price is None or self.quantity is None:
            return None
        return float(self.unit_price) * float(self.quantity)


class BuyerContext(BaseModel):
    """Buyer profile features."""

    user_id: Optional[str] = None
    user_type: UserType = "b2c"
    budget: Optional[float] = Field(None, ge=0.0)
    preferred_language: Optional[str] = None
    history_orders: int = Field(0, ge=0)
    history_returns: int = Field(0, ge=0)
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    loyalty_tier: Optional[str] = None
    region: Optional[str] = None


class DecisionRequest(BaseModel):
    """A single decision request."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None
    intent: IntentSignal
    buyer: BuyerContext = Field(default_factory=BuyerContext)
    product: ProductContext = Field(default_factory=ProductContext)
    urgency: Urgency = "medium"
    language: Optional[str] = None
    strategy_override: Optional[str] = Field(
        None,
        description="Force a specific strategy (rules|ml|ensemble) for this request.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("strategy_override")
    @classmethod
    def _norm_strategy(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in {"rules", "ml", "ensemble"}:
            raise ValueError("strategy_override must be one of: rules, ml, ensemble")
        return v


class BatchDecisionRequest(BaseModel):
    items: List[DecisionRequest]

    @field_validator("items")
    @classmethod
    def _non_empty(cls, v: List[DecisionRequest]) -> List[DecisionRequest]:
        if not v:
            raise ValueError("items must contain at least one decision request")
        return v

    @model_validator(mode="after")
    def _check_size(self) -> "BatchDecisionRequest":
        # Hard upper bound regardless of settings — settings may further
        # restrict this at the service layer.
        if len(self.items) > 256:
            raise ValueError("batch size must be <= 256")
        return self


# ---------------------------------------------------------------- response


class DecisionCandidate(BaseModel):
    """One ranked candidate action."""

    action: str = Field(..., description="Canonical action id.")
    label: str = Field(..., description="Localised display label.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: Optional[str] = Field(
        None, description="One-sentence explanation of why this action ranked here."
    )


class DecisionResponse(BaseModel):
    """Result of evaluating a single decision request."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None

    action: str = Field(..., description="Top-ranked action id.")
    label: str = Field(..., description="Localised label of the top action.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    confident: bool = Field(
        ..., description="True iff confidence >= configured confident_threshold."
    )
    risk_score: float = Field(..., ge=0.0, le=1.0)
    requires_review: bool = Field(
        ..., description="True iff a human reviewer should sign-off."
    )

    rationale: str = Field(..., description="Top-line, human-readable reasoning.")
    next_steps: List[str] = Field(default_factory=list)
    candidates: List[DecisionCandidate] = Field(default_factory=list)

    strategy: str = Field(..., description="Which strategy produced this result.")
    language: str
    latency_ms: float = Field(..., ge=0.0)
    cached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchDecisionResponse(BaseModel):
    results: List[DecisionResponse]
    total: int
    latency_ms: float = Field(..., ge=0.0)


# ---------------------------------------------------------------- meta


class HealthResponse(BaseModel):
    status: str
    version: str
    strategy: str
    model_loaded: bool
    cache_ready: bool
    circuit_state: str
    uptime_seconds: float


class ActionInfo(BaseModel):
    """Describes one action to clients (used by /actions)."""

    id: str
    label_zh: str
    label_en: str
    description: str
    audiences: List[str]
    severity: str
    terminal: bool
