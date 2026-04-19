"""Hub-service settings (env-prefix ``HUB_``)."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class HubEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ClientMode(str, Enum):
    """How the hub talks to each downstream service.

    * ``embedded`` — instantiates the downstream ``Service`` class in
      the same Python process. Ideal for monolithic deployments and
      integration tests; skips the network entirely.
    * ``http`` — talks to a remote service over HTTP via httpx. Used
      when the hub runs as its own microservice with upstreams reached
      over the network.
    """

    EMBEDDED = "embedded"
    HTTP = "http"


class HubSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- server ---
    env: HubEnvironment = HubEnvironment.DEVELOPMENT
    service_name: str = "ilbuyai-hub"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(8006, ge=1, le=65535)
    workers: int = Field(2, ge=1, le=64)
    debug: bool = False
    log_level: str = "INFO"

    # --- pipeline control ---
    # Defaults the hub uses when a pipeline request does not override.
    default_language: str = "zh-CN"
    supported_languages: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["zh-CN", "en"]
    )
    default_user_type: str = "b2c"
    default_report_format: str = "markdown"
    default_report_audience: str = "technical"
    pipeline_timeout_seconds: float = Field(60.0, ge=1.0, le=600.0)
    max_batch_size: int = Field(32, ge=1, le=256)

    # --- per-stage client modes ---
    intent_mode: ClientMode = ClientMode.EMBEDDED
    decision_mode: ClientMode = ClientMode.EMBEDDED
    llm_mode: ClientMode = ClientMode.EMBEDDED
    report_mode: ClientMode = ClientMode.EMBEDDED

    # --- per-stage HTTP base URLs (used when mode=http) ---
    intent_base_url: str = "http://localhost:8002"
    decision_base_url: str = "http://localhost:8003"
    llm_base_url: str = "http://localhost:8004"
    report_base_url: str = "http://localhost:8005"

    # --- per-stage HTTP timeouts (seconds) ---
    intent_timeout_seconds: float = Field(10.0, ge=0.1, le=120.0)
    decision_timeout_seconds: float = Field(10.0, ge=0.1, le=120.0)
    llm_timeout_seconds: float = Field(60.0, ge=0.1, le=600.0)
    report_timeout_seconds: float = Field(20.0, ge=0.1, le=120.0)

    # --- per-stage HTTP retries ---
    http_max_retries: int = Field(2, ge=0, le=10)
    http_retry_backoff_seconds: float = Field(0.2, ge=0.0, le=10.0)

    # --- circuit breaker (per stage) ---
    cb_failure_threshold: int = Field(5, ge=1)
    cb_recovery_seconds: int = Field(30, ge=1)

    # --- downstream api key (forwarded as X-API-Key when set) ---
    downstream_api_key: Optional[str] = None

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
        "supported_languages",
        mode="before",
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
            raise ValueError(
                f"default_language {v!r} must be one of {supported}"
            )
        return v

    @field_validator("default_user_type")
    @classmethod
    def _validate_user_type(cls, v: str) -> str:
        if v not in {"b2b", "b2c"}:
            raise ValueError("default_user_type must be b2b or b2c")
        return v

    @property
    def is_production(self) -> bool:
        return self.env == HubEnvironment.PRODUCTION


@lru_cache(maxsize=1)
def get_hub_settings() -> HubSettings:
    return HubSettings()


def reset_hub_settings() -> None:
    get_hub_settings.cache_clear()
