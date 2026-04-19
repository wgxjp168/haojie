"""Pydantic request/response schemas for the LLM service."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


MessageRole = Literal["system", "user", "assistant", "tool"]
Priority = Literal["low", "normal", "high", "critical"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1)
    name: Optional[str] = Field(
        None,
        description="Optional speaker name (used by tools/assistant).",
    )

    @field_validator("content")
    @classmethod
    def _strip_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message content must not be empty")
        return v


# ----------------------------------------------------------- request


class CompletionRequest(BaseModel):
    """A single chat completion request."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    priority: Priority = "normal"

    # Either ``messages`` or ``prompt`` must be supplied. ``prompt`` is
    # sugar for [{"role": "user", "content": prompt}].
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None

    # Targeting.
    model_id: Optional[str] = Field(
        None,
        description="Preferred model id. Defaults to the service default.",
    )
    strategy: Optional[str] = Field(
        None,
        description=(
            "Routing strategy override: single | fallback | parallel_any | "
            "parallel_vote | confidence."
        ),
    )
    target_model_ids: Optional[List[str]] = Field(
        None,
        description="Explicit fallback / parallel target list; overrides fallback_chain.",
    )

    # Sampling.
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32_768)
    stop: Optional[List[str]] = None

    # Timeouts + metadata.
    timeout_seconds: Optional[float] = Field(None, ge=0.1, le=600.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_prompt_or_messages(self) -> "CompletionRequest":
        if not self.prompt and not self.messages:
            raise ValueError("either 'prompt' or 'messages' must be provided")
        return self

    @field_validator("strategy")
    @classmethod
    def _norm_strategy(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in {
            "single",
            "fallback",
            "parallel_any",
            "parallel_vote",
            "confidence",
        }:
            raise ValueError(
                "strategy must be one of: single, fallback, parallel_any, "
                "parallel_vote, confidence"
            )
        return v

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("stop")
    @classmethod
    def _cap_stop(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and len(v) > 8:
            raise ValueError("stop sequences limited to 8")
        return v

    def normalized_messages(self) -> List[ChatMessage]:
        """Always return a messages list for downstream use."""
        if self.messages:
            return list(self.messages)
        return [ChatMessage(role="user", content=self.prompt or "")]


class BatchCompletionRequest(BaseModel):
    items: List[CompletionRequest]

    @field_validator("items")
    @classmethod
    def _non_empty(cls, v: List[CompletionRequest]) -> List[CompletionRequest]:
        if not v:
            raise ValueError("items must contain at least one completion request")
        if len(v) > 256:
            raise ValueError("batch size must be <= 256")
        return v


# ---------------------------------------------------------- response


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _check_total(self) -> "TokenUsage":
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class CompletionAttempt(BaseModel):
    """One provider attempt — may have succeeded or failed."""

    provider: str
    model_id: str
    remote_name: Optional[str] = None
    status: Literal["success", "failure", "skipped"]
    latency_ms: float = Field(..., ge=0.0)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    tokens: Optional[TokenUsage] = None
    cost_usd: float = Field(0.0, ge=0.0)


class CompletionResponse(BaseModel):
    """Result of a completion request."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None

    content: str
    provider: str
    model_id: str
    strategy: str
    finish_reason: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    confident: bool = Field(
        ..., description="True iff confidence >= configured confident_threshold."
    )
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(0.0, ge=0.0)
    latency_ms: float = Field(..., ge=0.0)
    attempts: List[CompletionAttempt] = Field(default_factory=list)
    cached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchCompletionResponse(BaseModel):
    results: List[CompletionResponse]
    total: int
    latency_ms: float = Field(..., ge=0.0)


# ---------------------------------------------------------- meta


class HealthResponse(BaseModel):
    status: str
    version: str
    default_strategy: str
    default_model_id: str
    providers_available: List[str]
    cache_ready: bool
    circuit_states: Dict[str, str]
    uptime_seconds: float


class ModelInfoPublic(BaseModel):
    model_id: str
    provider: str
    context_window: int
    max_output_tokens: int
    price_per_1k_prompt_usd: float
    price_per_1k_completion_usd: float
    available: bool
    description: str
