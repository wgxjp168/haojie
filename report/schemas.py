"""Pydantic request/response schemas for the report service.

The input is designed to accept the outputs of Parts 3.1 (intent),
3.2 (decision), and 3.3 (LLM) directly, with all three pieces being
optional — so callers can render a partial report from whichever
upstream stage they have.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------- inputs


class BuyerSummary(BaseModel):
    """Compact buyer profile echoed into the report."""

    user_id: Optional[str] = None
    user_type: Literal["b2b", "b2c"] = "b2c"
    region: Optional[str] = None
    loyalty_tier: Optional[str] = None
    budget: Optional[float] = Field(None, ge=0.0)


class ProductSummary(BaseModel):
    """Compact product/order reference used in the report body."""

    product_id: Optional[str] = None
    sku: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    unit_price: Optional[float] = Field(None, ge=0.0)
    quantity: Optional[int] = Field(None, ge=1)
    supplier_id: Optional[str] = None

    @property
    def total_amount(self) -> Optional[float]:
        if self.unit_price is None or self.quantity is None:
            return None
        return float(self.unit_price) * float(self.quantity)


class IntentSlice(BaseModel):
    """Slice of the intent-service output relevant to the report."""

    intent: str = Field(..., min_length=1)
    label: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    backend: Optional[str] = None


class DecisionCandidateSlice(BaseModel):
    action: str = Field(..., min_length=1)
    label: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class DecisionSlice(BaseModel):
    """Slice of the decision-engine output consumed by the report."""

    action: str = Field(..., min_length=1)
    label: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    confident: bool = False
    risk_score: float = Field(0.0, ge=0.0, le=1.0)
    requires_review: bool = False
    rationale: str = ""
    next_steps: List[str] = Field(default_factory=list)
    candidates: List[DecisionCandidateSlice] = Field(default_factory=list)
    strategy: Optional[str] = None
    matched_rules: List[Dict[str, Any]] = Field(default_factory=list)


class LLMSlice(BaseModel):
    """Optional free-form narrative produced by the LLM service."""

    content: str = Field(..., min_length=1)
    model_id: Optional[str] = None
    provider: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    cached: bool = False


class ReportRequest(BaseModel):
    """A single report generation request."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None
    subject_id: Optional[str] = None

    # What to render.
    format: Optional[str] = Field(None, description="markdown | html | json | text")
    audience: Optional[str] = Field(
        None, description="executive | technical | customer"
    )
    template_id: Optional[str] = Field(
        None,
        description="Override template id; defaults to the audience-specific template.",
    )
    language: Optional[str] = Field(
        None, description="IETF tag (zh-CN | en). Defaults to service default."
    )

    # Input payload — all optional; at least one must be present.
    buyer: Optional[BuyerSummary] = None
    product: Optional[ProductSummary] = None
    intent: Optional[IntentSlice] = None
    decision: Optional[DecisionSlice] = None
    llm: Optional[LLMSlice] = None

    # Output control.
    store: bool = Field(
        True,
        description="When true, persist the rendered report via configured backends.",
    )
    include_trace: bool = Field(
        True, description="Include matched_rules / decision trace when available."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_something_to_render(self) -> "ReportRequest":
        if not any(
            [self.intent, self.decision, self.llm, self.product, self.buyer]
        ):
            raise ValueError(
                "at least one of intent/decision/llm/product/buyer must be provided"
            )
        return self

    @field_validator("format")
    @classmethod
    def _norm_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in {"markdown", "html", "json", "text"}:
            raise ValueError(
                "format must be one of: markdown, html, json, text"
            )
        return v

    @field_validator("audience")
    @classmethod
    def _norm_audience(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in {"executive", "technical", "customer"}:
            raise ValueError(
                "audience must be one of: executive, technical, customer"
            )
        return v

    def fingerprint(self) -> str:
        """Deterministic hash over the semantic content — used as cache key."""
        payload: Dict[str, Any] = {
            "format": self.format,
            "audience": self.audience,
            "template_id": self.template_id,
            "language": self.language,
            "buyer": self.buyer.model_dump() if self.buyer else None,
            "product": self.product.model_dump() if self.product else None,
            "intent": self.intent.model_dump() if self.intent else None,
            "decision": self.decision.model_dump() if self.decision else None,
            "llm": self.llm.model_dump() if self.llm else None,
            "include_trace": self.include_trace,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class BatchReportRequest(BaseModel):
    items: List[ReportRequest]

    @field_validator("items")
    @classmethod
    def _non_empty(cls, v: List[ReportRequest]) -> List[ReportRequest]:
        if not v:
            raise ValueError("items must contain at least one report request")
        if len(v) > 256:
            raise ValueError("batch size must be <= 256")
        return v


# --------------------------------------------------------------- outputs


class StorageResult(BaseModel):
    """Per-backend storage outcome."""

    backend: str
    success: bool
    location: Optional[str] = Field(None, description="URI / key / path.")
    bytes_written: int = Field(0, ge=0)
    latency_ms: float = Field(..., ge=0.0)
    error: Optional[str] = None


class ReportResponse(BaseModel):
    """Result of rendering a single report."""

    report_id: str
    request_id: Optional[str] = None
    session_id: Optional[str] = None

    format: str
    audience: str
    template_id: str
    language: str

    content: str = Field(..., description="Rendered body (may be base64 for binary).")
    content_type: str
    size_bytes: int = Field(..., ge=0)
    checksum: str = Field(..., description="sha256 of the rendered content.")

    render_latency_ms: float = Field(..., ge=0.0)
    storage_results: List[StorageResult] = Field(default_factory=list)
    cached: bool = False

    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchReportResponse(BaseModel):
    results: List[ReportResponse]
    total: int
    latency_ms: float = Field(..., ge=0.0)


# --------------------------------------------------------------- meta


class HealthResponse(BaseModel):
    status: str
    version: str
    default_format: str
    default_audience: str
    storage_backends: List[str]
    cache_ready: bool
    storage_circuit_states: Dict[str, str]
    uptime_seconds: float


class TemplateInfo(BaseModel):
    template_id: str
    audience: str
    formats: List[str]
    description: str


class StoredReport(BaseModel):
    """Envelope returned when retrieving a previously-persisted report."""

    report_id: str
    format: str
    language: str
    content: str
    content_type: str
    checksum: str
    stored_at: datetime
    backend: str
