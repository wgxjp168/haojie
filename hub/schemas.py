"""Pydantic request/response schemas for the hub pipeline.

The hub accepts a single high-level ``PipelineRequest`` and returns a
``PipelineResponse`` containing whatever stage outcomes were produced.
Each stage is optional; the request controls which ones run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


UserType = Literal["b2b", "b2c"]
PipelineStatus = Literal["success", "partial", "failed"]
StageStatus = Literal["success", "skipped", "failed"]


# ---------------------------------------------------------------- request


class PipelineBuyer(BaseModel):
    """Compact buyer profile the hub forwards to decision + report."""

    user_id: Optional[str] = None
    user_type: Optional[UserType] = None
    region: Optional[str] = None
    loyalty_tier: Optional[str] = None
    budget: Optional[float] = Field(None, ge=0.0)
    history_orders: int = Field(0, ge=0)
    history_returns: int = Field(0, ge=0)
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class PipelineProduct(BaseModel):
    """Compact product/order profile."""

    product_id: Optional[str] = None
    sku: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    unit_price: Optional[float] = Field(None, ge=0.0)
    quantity: Optional[int] = Field(None, ge=1)
    in_stock: Optional[bool] = None
    lead_time_days: Optional[int] = Field(None, ge=0)
    supplier_id: Optional[str] = None
    supplier_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    has_alternatives: Optional[bool] = None


class PipelineOptions(BaseModel):
    """Per-stage toggles + overrides."""

    # Intent overrides.
    intent_top_k: Optional[int] = Field(None, ge=1, le=20)

    # Decision overrides.
    decision_strategy: Optional[str] = Field(
        None,
        description="Force decision-engine strategy: rules|ml|ensemble.",
    )
    urgency: Optional[str] = Field(None, description="low|medium|high")

    # LLM stage (opt-in).
    include_llm: bool = Field(
        False,
        description=(
            "When true, generate an LLM narrative explaining the decision."
        ),
    )
    llm_model_id: Optional[str] = None
    llm_strategy: Optional[str] = None
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

    # Report stage (opt-in).
    include_report: bool = Field(
        True,
        description="When true, render a report of the intent+decision[+llm].",
    )
    report_format: Optional[str] = Field(
        None, description="markdown|html|json|text"
    )
    report_audience: Optional[str] = Field(
        None, description="executive|technical|customer"
    )
    report_store: bool = True


class PipelineRequest(BaseModel):
    """The hub's top-level request."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None

    text: str = Field(..., min_length=1, description="Normalised user query.")
    language: Optional[str] = None
    user_type: Optional[UserType] = None

    buyer: Optional[PipelineBuyer] = None
    product: Optional[PipelineProduct] = None

    options: PipelineOptions = Field(default_factory=PipelineOptions)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class BatchPipelineRequest(BaseModel):
    items: List[PipelineRequest]

    @field_validator("items")
    @classmethod
    def _non_empty(cls, v: List[PipelineRequest]) -> List[PipelineRequest]:
        if not v:
            raise ValueError("items must contain at least one pipeline request")
        if len(v) > 256:
            raise ValueError("batch size must be <= 256")
        return v


# ---------------------------------------------------------------- stages


class StageOutcome(BaseModel):
    """Common envelope for every stage outcome."""

    stage: Literal["intent", "decision", "llm", "report"]
    status: StageStatus
    latency_ms: float = Field(..., ge=0.0)
    mode: Optional[str] = Field(None, description="embedded | http")
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------- response


class PipelineResponse(BaseModel):
    """The hub's top-level response."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None

    overall_status: PipelineStatus
    total_latency_ms: float = Field(..., ge=0.0)

    intent: Optional[StageOutcome] = None
    decision: Optional[StageOutcome] = None
    llm: Optional[StageOutcome] = None
    report: Optional[StageOutcome] = None

    stage_sequence: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchPipelineResponse(BaseModel):
    results: List[PipelineResponse]
    total: int
    latency_ms: float = Field(..., ge=0.0)


# ---------------------------------------------------------------- meta


class StageHealth(BaseModel):
    stage: str
    mode: str
    available: bool
    circuit_state: str
    endpoint: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    stages: List[StageHealth]
