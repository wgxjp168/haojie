"""Application settings loaded from environment variables and optional YAML.

All settings are validated at startup. Secrets must NEVER be hardcoded —
use environment variables or a secret manager.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"


class SpeechProvider(str, Enum):
    AZURE = "azure"
    GOOGLE = "google"
    LOCAL = "local"  # mock / on-prem


class ImageRecognitionProvider(str, Enum):
    AZURE = "azure"
    GOOGLE = "google"
    LOCAL = "local"


class Settings(BaseSettings):
    """Root settings object. Field names use snake_case and are read from
    environment variables with the prefix ``APP_`` (case insensitive)."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- environment ---
    env: Environment = Environment.DEVELOPMENT
    service_name: str = "multi-modal-input-processor"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- server ---
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    workers: int = Field(4, ge=1, le=128)

    # --- persistence ---
    redis_url: Optional[str] = "redis://localhost:6379/0"

    # --- cache ---
    cache_enabled: bool = True
    cache_ttl: int = Field(300, ge=1)

    # --- storage ---
    storage_backend: StorageBackend = StorageBackend.LOCAL
    storage_local_path: str = "./storage"
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_bucket: str = "input-processor"
    s3_region: str = "us-east-1"

    # --- security ---
    jwt_secret_key: str = "change-me-to-a-strong-random-value"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    api_key_required: bool = False
    api_keys: List[str] = Field(default_factory=list)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 100
    max_request_size_mb: int = 50

    # --- text processor ---
    text_max_length: int = Field(10_000, ge=1, le=1_000_000)

    # --- image processor ---
    image_max_size_mb: int = Field(20, ge=1, le=200)
    image_max_dimension: int = 4096
    image_recognition_provider: ImageRecognitionProvider = ImageRecognitionProvider.LOCAL
    azure_cv_endpoint: Optional[str] = None
    azure_cv_key: Optional[str] = None

    # --- link processor ---
    link_timeout: int = Field(10, ge=1, le=120)
    link_user_agent: str = "ShoppingAI-InputProcessor/1.0"

    # --- voice processor ---
    voice_max_size_mb: int = Field(20, ge=1, le=200)
    voice_max_duration_seconds: int = Field(300, ge=1, le=3_600)
    speech_provider: SpeechProvider = SpeechProvider.LOCAL
    speech_default_language: str = "zh-CN"
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None

    # --- session ---
    session_max_age_hours: int = Field(24, ge=1)
    session_cleanup_interval_minutes: int = Field(60, ge=1)

    # --- monitoring ---
    metrics_enabled: bool = True
    tracing_enabled: bool = False

    # --- validators ---
    @field_validator("cors_origins", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    # --- computed / helpers ---
    @property
    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        return self.env == Environment.TESTING

    def production_safety_check(self) -> List[str]:
        """Return a list of warnings for dangerous production config."""
        warnings: List[str] = []
        if not self.is_production:
            return warnings
        if self.debug:
            warnings.append("debug=True in production")
        if self.jwt_secret_key == "change-me-to-a-strong-random-value":
            warnings.append("default JWT secret in production")
        if self.cors_origins == ["*"]:
            warnings.append("wildcard CORS origin in production")
        return warnings


def _maybe_load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Loads a YAML file if ``APP_CONFIG_FILE`` is set, merges with env vars
    (env vars win), and returns a validated Settings instance.
    """
    yaml_path = os.environ.get("APP_CONFIG_FILE")
    overrides = _maybe_load_yaml(yaml_path) if yaml_path else {}
    return Settings(**overrides)


def reload_settings() -> Settings:
    """Bust the cache and reload settings (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()
