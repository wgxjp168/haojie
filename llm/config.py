"""LLM-service settings (env-prefix ``LLM_``)."""
from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class LLMEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMStrategy(str, Enum):
    """Routing strategies for completion requests.

    * ``single``   — call exactly one model (the best-ranked target).
    * ``fallback`` — sequential; on failure fall through to the next.
    * ``parallel_any`` — race all targets, return the first success.
    * ``parallel_vote`` — call all, majority-vote on the textual answer.
    * ``confidence`` — call all, pick the highest-confidence reply.
    """

    SINGLE = "single"
    FALLBACK = "fallback"
    PARALLEL_ANY = "parallel_any"
    PARALLEL_VOTE = "parallel_vote"
    CONFIDENCE = "confidence"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- server ---
    env: LLMEnvironment = LLMEnvironment.DEVELOPMENT
    service_name: str = "ilbuyai-llm"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(8004, ge=1, le=65535)
    workers: int = Field(2, ge=1, le=64)
    debug: bool = False
    log_level: str = "INFO"

    # --- routing ---
    default_strategy: LLMStrategy = LLMStrategy.FALLBACK
    default_model_id: str = "stub-small"
    # ``NoDecode`` keeps pydantic-settings from trying to JSON-decode the
    # env var before our CSV-friendly validator runs.
    fallback_chain: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["stub-small"],
        description="Ordered model_ids tried on fallback strategy.",
    )
    parallel_fanout: int = Field(3, ge=1, le=16)
    confident_threshold: float = Field(0.65, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(0.45, ge=0.0, le=1.0)
    request_timeout_seconds: float = Field(30.0, ge=0.1, le=600.0)
    max_batch_size: int = Field(32, ge=1, le=256)

    # --- prompt handling ---
    max_prompt_chars: int = Field(32_000, ge=1, le=2_000_000)
    max_completion_tokens: int = Field(1024, ge=1, le=32_768)
    default_temperature: float = Field(0.7, ge=0.0, le=2.0)
    default_top_p: float = Field(1.0, ge=0.0, le=1.0)

    # --- provider credentials (all optional; stub provider is always
    # available so the service runs without any real provider configured)
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_organization: Optional[str] = None

    anthropic_api_key: Optional[str] = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"

    google_api_key: Optional[str] = None
    google_base_url: str = "https://generativelanguage.googleapis.com/v1"

    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-02-01"

    # --- cache ---
    cache_enabled: bool = True
    redis_url: Optional[str] = "redis://localhost:6379/4"
    cache_ttl_seconds: int = Field(600, ge=0)
    memory_cache_size: int = Field(2048, ge=0, le=1_000_000)

    # --- circuit breaker (per provider) ---
    cb_failure_threshold: int = Field(5, ge=1)
    cb_recovery_seconds: int = Field(30, ge=1)

    # --- rate limiting (per provider) ---
    # Applied per provider; "anonymous" buckets to the caller's client_id
    # when present in the request header.
    rate_limit_per_minute: int = Field(300, ge=0)
    rate_limit_burst: int = Field(60, ge=0)

    # --- budget / cost guard ---
    # Cap on per-request estimated spend in USD; requests whose projected
    # cost (prompt tokens * unit-price) exceeds this are refused with 402.
    max_request_cost_usd: float = Field(1.00, ge=0.0)

    # --- content safety (hooks; actual checking left to upstream) ---
    content_safety_enabled: bool = False

    # --- security ---
    api_key_required: bool = False
    api_keys: Annotated[List[str], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["*"]
    )

    # --- monitoring ---
    metrics_enabled: bool = True

    @field_validator(
        "api_keys",
        "cors_origins",
        "fallback_chain",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("fallback_chain")
    @classmethod
    def _fallback_non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("fallback_chain must contain at least one model id")
        return v

    @field_validator("low_confidence_threshold")
    @classmethod
    def _thresholds_ordered(cls, v, info):
        confident = info.data.get("confident_threshold")
        if confident is not None and v > confident:
            raise ValueError(
                "low_confidence_threshold must be <= confident_threshold"
            )
        return v

    @property
    def is_production(self) -> bool:
        return self.env == LLMEnvironment.PRODUCTION

    def provider_credentials(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Collect non-empty provider credentials."""
        return {
            "openai": {
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "organization": self.openai_organization,
            },
            "anthropic": {
                "api_key": self.anthropic_api_key,
                "base_url": self.anthropic_base_url,
            },
            "google": {
                "api_key": self.google_api_key,
                "base_url": self.google_base_url,
            },
            "azure": {
                "api_key": self.azure_api_key,
                "endpoint": self.azure_endpoint,
                "api_version": self.azure_api_version,
            },
        }


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    return LLMSettings()


def reset_llm_settings() -> None:
    get_llm_settings.cache_clear()
