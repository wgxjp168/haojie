"""Report-service settings (env-prefix ``REPORT_``)."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class ReportEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ReportFormat(str, Enum):
    """Supported output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


class ReportAudience(str, Enum):
    """Target audience — controls template selection and tone."""

    EXECUTIVE = "executive"   # concise, high-level
    TECHNICAL = "technical"   # full trace / rule matches
    CUSTOMER = "customer"     # plain-language, buyer-facing


class StorageBackend(str, Enum):
    """Pluggable storage backend selection."""

    MEMORY = "memory"         # in-process (default; always available)
    LOCAL = "local"           # local filesystem
    S3 = "s3"                 # AWS S3 (optional; requires boto3)


class ReportSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REPORT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- server ---
    env: ReportEnvironment = ReportEnvironment.DEVELOPMENT
    service_name: str = "ilbuyai-report"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(8005, ge=1, le=65535)
    workers: int = Field(2, ge=1, le=64)
    debug: bool = False
    log_level: str = "INFO"

    # --- rendering defaults ---
    default_format: ReportFormat = ReportFormat.MARKDOWN
    default_audience: ReportAudience = ReportAudience.TECHNICAL
    default_language: str = "zh-CN"
    supported_languages: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["zh-CN", "en"]
    )
    max_payload_bytes: int = Field(1_048_576, ge=1024, le=100_000_000)  # 1 MiB default
    max_batch_size: int = Field(32, ge=1, le=256)
    render_timeout_seconds: float = Field(15.0, ge=0.1, le=300.0)

    # --- storage ---
    storage_backend: StorageBackend = StorageBackend.MEMORY
    storage_local_path: str = "./storage/reports"
    storage_filename_pattern: str = "{report_id}.{ext}"
    # Multi-backend fan-out (e.g. ``memory,local`` to keep a hot cache +
    # durable copy). Comma-separated list; first entry is primary.
    storage_backends: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["memory"]
    )

    # --- S3 (optional) ---
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    s3_prefix: str = "reports/"
    s3_endpoint_url: Optional[str] = None  # for MinIO / localstack
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None

    # --- cache ---
    cache_enabled: bool = True
    redis_url: Optional[str] = "redis://localhost:6379/5"
    cache_ttl_seconds: int = Field(600, ge=0)
    memory_cache_size: int = Field(2048, ge=0, le=1_000_000)

    # --- circuit breaker (per storage backend) ---
    cb_failure_threshold: int = Field(5, ge=1)
    cb_recovery_seconds: int = Field(30, ge=1)

    # --- security ---
    api_key_required: bool = False
    api_keys: Annotated[List[str], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["*"]
    )

    # --- monitoring ---
    metrics_enabled: bool = True

    @field_validator("api_keys", "supported_languages", "cors_origins", "storage_backends", mode="before")
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
            raise ValueError(
                f"default_language {v!r} must be one of {supported}"
            )
        return v

    @field_validator("storage_backends")
    @classmethod
    def _validate_backends(cls, v: List[str]) -> List[str]:
        valid = {b.value for b in StorageBackend}
        unknown = [x for x in v if x not in valid]
        if unknown:
            raise ValueError(
                f"storage_backends contains unknown backend(s): {unknown}. "
                f"Supported: {sorted(valid)}"
            )
        if not v:
            raise ValueError("storage_backends must contain at least one backend")
        return v

    @property
    def is_production(self) -> bool:
        return self.env == ReportEnvironment.PRODUCTION


@lru_cache(maxsize=1)
def get_report_settings() -> ReportSettings:
    return ReportSettings()


def reset_report_settings() -> None:
    get_report_settings.cache_clear()
