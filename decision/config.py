"""Decision-engine settings (env-prefix ``DECISION_``)."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DecisionEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DecisionStrategyKind(str, Enum):
    """Which strategy drives the engine.

    * ``rules`` — deterministic policy tree (always available, zero-dep).
    * ``ml`` — learned model (requires ``scikit-learn`` or ``torch``).
    * ``ensemble`` — weighted blend of ``rules`` + ``ml`` with rules
      acting as fallback when ML is unavailable or low-confidence.
    """

    RULES = "rules"
    ML = "ml"
    ENSEMBLE = "ensemble"


class DecisionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DECISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- server ---
    env: DecisionEnvironment = DecisionEnvironment.DEVELOPMENT
    service_name: str = "ilbuyai-decision"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(8003, ge=1, le=65535)
    workers: int = Field(2, ge=1, le=64)
    debug: bool = False
    log_level: str = "INFO"

    # --- strategy ---
    strategy: DecisionStrategyKind = DecisionStrategyKind.ENSEMBLE
    # Ensemble weights — rules vs. ML. Must sum to > 0.
    ensemble_rules_weight: float = Field(0.5, ge=0.0, le=1.0)
    ensemble_ml_weight: float = Field(0.5, ge=0.0, le=1.0)
    # If ML top-1 confidence is below this threshold, the rules vote.
    low_confidence_threshold: float = Field(0.55, ge=0.0, le=1.0)
    # Final decision must clear this bar to be marked ``confident``.
    confident_threshold: float = Field(0.65, ge=0.0, le=1.0)
    # Decisions with risk_score >= this are flagged ``requires_review``.
    risk_review_threshold: float = Field(0.7, ge=0.0, le=1.0)
    # Maximum number of alternative actions returned alongside the top choice.
    top_k_alternatives: int = Field(3, ge=0, le=10)
    # Hard cap on batch size (matches intent service for symmetry).
    max_batch_size: int = Field(64, ge=1, le=512)

    # --- ML strategy (optional) ---
    ml_model_path: Optional[str] = None
    ml_model_kind: str = "sklearn"  # sklearn | torch | dummy
    ml_device: str = "cpu"          # cpu | cuda | auto
    ml_batch_size: int = Field(16, ge=1, le=256)
    # If ML load fails we *continue* with rules only; when false we refuse
    # to start unless ML loads (useful for prod hard-dependence on ML).
    allow_rules_fallback: bool = True
    preload_model: bool = False

    # --- cache ---
    cache_enabled: bool = True
    redis_url: Optional[str] = "redis://localhost:6379/3"
    cache_ttl_seconds: int = Field(300, ge=0)
    memory_cache_size: int = Field(2048, ge=0, le=1_000_000)

    # --- circuit breaker (around the ML strategy) ---
    cb_failure_threshold: int = Field(5, ge=1)
    cb_recovery_seconds: int = Field(30, ge=1)

    # --- policy guardrails ---
    # Hard ceilings used by the rule engine to clamp dangerous decisions.
    max_auto_approve_amount: float = Field(50_000.0, ge=0.0)
    require_human_review_above: float = Field(100_000.0, ge=0.0)
    # Bilingual-friendly default language for human-readable rationale.
    default_language: str = "zh-CN"
    supported_languages: List[str] = Field(default_factory=lambda: ["zh-CN", "en"])

    # --- security ---
    api_key_required: bool = False
    api_keys: List[str] = Field(default_factory=list)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # --- monitoring ---
    metrics_enabled: bool = True

    @field_validator(
        "api_keys", "supported_languages", "cors_origins", mode="before"
    )
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @field_validator("default_language")
    @classmethod
    def _validate_default_lang(cls, v, info):
        supported = info.data.get("supported_languages") or ["zh-CN", "en"]
        if v not in supported:
            raise ValueError(f"default_language {v!r} must be one of {supported}")
        return v

    @field_validator("ensemble_ml_weight")
    @classmethod
    def _validate_weights(cls, v, info):
        rules_w = info.data.get("ensemble_rules_weight", 0.0)
        if rules_w + v <= 0:
            raise ValueError(
                "ensemble weights must sum to > 0 "
                f"(got rules={rules_w}, ml={v})"
            )
        return v

    @field_validator("require_human_review_above")
    @classmethod
    def _ceiling_above_floor(cls, v, info):
        floor = info.data.get("max_auto_approve_amount", 0.0)
        if v < floor:
            raise ValueError(
                "require_human_review_above must be >= max_auto_approve_amount"
            )
        return v

    @property
    def is_production(self) -> bool:
        return self.env == DecisionEnvironment.PRODUCTION


@lru_cache(maxsize=1)
def get_decision_settings() -> DecisionSettings:
    return DecisionSettings()


def reset_decision_settings() -> None:
    get_decision_settings.cache_clear()
