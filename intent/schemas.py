"""Pydantic request/response schemas for the intent service."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class IntentRequest(BaseModel):
    """A single classification request."""

    text: str = Field(..., description="Normalized user text to classify.", min_length=1)
    user_type: Optional[str] = Field(
        None,
        description="Hint for audience-biased rule voting: b2b | b2c | any.",
    )
    language: Optional[str] = Field(
        None, description="IETF language tag (e.g. zh-CN, en). Auto-detected if omitted."
    )
    session_id: Optional[str] = Field(None, description="Upstream session id (for tracing).")
    request_id: Optional[str] = Field(None, description="Client-supplied id (for tracing).")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override top-k.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v

    @field_validator("user_type")
    @classmethod
    def _normalise_audience(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v in {"b2b", "purchaser", "b2b_purchaser"}:
            return "b2b"
        if v in {"b2c", "consumer", "b2c_consumer"}:
            return "b2c"
        return "any"


class IntentCandidate(BaseModel):
    """A single ranked intent candidate in the response."""

    intent: str = Field(..., description="Canonical intent id.")
    label: str = Field(..., description="Human-readable label in the requested language.")
    confidence: float = Field(..., ge=0.0, le=1.0)


class IntentResponse(BaseModel):
    """Result of classifying one text."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None
    top_intent: str
    top_label: str
    top_confidence: float = Field(..., ge=0.0, le=1.0)
    confident: bool = Field(
        ..., description="Whether top_confidence >= configured confident_threshold."
    )
    candidates: List[IntentCandidate]
    backend: str = Field(..., description="Which backend produced the result.")
    language: str
    latency_ms: float = Field(..., ge=0.0)
    cached: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchIntentRequest(BaseModel):
    """Batch version — all items share language / top-k / user_type by default."""

    items: List[IntentRequest]

    @field_validator("items")
    @classmethod
    def _non_empty(cls, v: List[IntentRequest]) -> List[IntentRequest]:
        if not v:
            raise ValueError("items must contain at least one request")
        if len(v) > 256:
            raise ValueError("batch size must be <= 256")
        return v


class BatchIntentResponse(BaseModel):
    results: List[IntentResponse]
    total: int
    latency_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str
    model_loaded: bool
    cache_ready: bool
    circuit_state: str
    uptime_seconds: float


class IntentInfo(BaseModel):
    """Describes one intent to clients (used by /intents)."""

    id: str
    label_zh: str
    label_en: str
    description: str
    audiences: List[str]
