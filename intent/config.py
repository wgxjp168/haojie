"""Intent-service settings (env-prefix ``INTENT_``)."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IntentEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class IntentBackend(str, Enum):
    """Which backend drives the classifier.

    * ``rules`` — keyword/regex rules only (always available, zero-dep).
    * ``transformer`` — HuggingFace transformer model (requires torch).
    * ``ensemble`` — transformer with rule-based fallback on low confidence.
    """

    RULES = "rules"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class IntentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- server ---
    env: IntentEnvironment = IntentEnvironment.DEVELOPMENT
    service_name: str = "ilbuyai-intent"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(8002, ge=1, le=65535)
    workers: int = Field(2, ge=1, le=64)
    debug: bool = False
    log_level: str = "INFO"

    # --- classifier ---
    backend: IntentBackend = IntentBackend.ENSEMBLE
    default_language: str = "zh-CN"
    supported_languages: List[str] = Field(default_factory=lambda: ["zh-CN", "en"])
    # When ensemble is used and the transformer's top-1 confidence is below
    # this threshold, we ask the rules to vote; if they disagree with the
    # transformer we keep the transformer pick but demote confidence.
    low_confidence_threshold: float = Field(0.55, ge=0.0, le=1.0)
    # Final answer must clear this bar to be marked ``confident``.
    confident_threshold: float = Field(0.70, ge=0.0, le=1.0)
    top_k: int = Field(3, ge=1, le=20)
    max_text_length: int = Field(2000, ge=1, le=100_000)
    min_text_length: int = Field(1, ge=1)

    # --- transformer model (optional) ---
    model_name_or_path: str = "bert-base-multilingual-cased"
    model_cache_dir: Optional[str] = None
    model_max_seq_length: int = Field(128, ge=8, le=1024)
    model_device: str = "auto"  # auto|cpu|cuda|cuda:0...
    model_batch_size: int = Field(16, ge=1, le=256)
    # If transformer load fails we *continue* with rules only; when false we
    # refuse to start unless the transformer loads (useful for prod).
    allow_rules_fallback: bool = True
    # Download / initialise transformer on boot (otherwise lazy).
    preload_model: bool = False

    # --- cache ---
    cache_enabled: bool = True
    redis_url: Optional[str] = "redis://localhost:6379/2"
    cache_ttl_seconds: int = Field(600, ge=0)
    # Size of the in-memory LRU that sits in front of Redis.
    memory_cache_size: int = Field(4096, ge=0, le=1_000_000)

    # --- circuit breaker (around the transformer) ---
    cb_failure_threshold: int = Field(5, ge=1)
    cb_recovery_seconds: int = Field(30, ge=1)

    # --- security ---
    api_key_required: bool = False
    api_keys: List[str] = Field(default_factory=list)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # --- monitoring ---
    metrics_enabled: bool = True

    @field_validator("api_keys", "supported_languages", "cors_origins", mode="before")
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

    @property
    def is_production(self) -> bool:
        return self.env == IntentEnvironment.PRODUCTION


@lru_cache(maxsize=1)
def get_intent_settings() -> IntentSettings:
    return IntentSettings()


def reset_intent_settings() -> None:
    get_intent_settings.cache_clear()
