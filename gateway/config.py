"""Gateway settings (env-driven, pydantic-settings based).

The gateway is deployed as a separate process/container; its settings are
independent from the Part 1 input-processor settings, though they may share
Redis.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewayEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GW_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- server ---
    env: GatewayEnvironment = GatewayEnvironment.DEVELOPMENT
    service_name: str = "ilbuyai-gateway"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(8001, ge=1, le=65535)
    workers: int = Field(4, ge=1, le=128)
    debug: bool = False
    log_level: str = "INFO"

    # --- upstream (Part 1) ---
    upstream_base_url: str = "http://localhost:8000"
    upstream_timeout_seconds: float = 30.0
    upstream_connect_timeout_seconds: float = 5.0
    upstream_max_retries: int = 2
    upstream_retry_backoff: float = 0.2

    # --- redis (for rate-limit / token blacklist / ws pubsub) ---
    redis_url: Optional[str] = "redis://localhost:6379/1"

    # --- security ---
    jwt_secret_key: str = "change-me-to-a-strong-random-value"
    jwt_algorithm: str = "HS256"
    jwt_access_minutes: int = 30
    jwt_refresh_days: int = 7
    api_keys: List[str] = Field(default_factory=list)
    api_key_required: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    max_request_size_mb: int = 50

    # --- rate limiting (per user-type, requests per minute) ---
    rl_b2b_purchaser_per_minute: int = 1000
    rl_b2c_consumer_per_minute: int = 100
    rl_partner_per_minute: int = 5000
    rl_internal_per_minute: int = 10000
    rl_anonymous_per_minute: int = 20
    rl_default_per_minute: int = 100

    # --- circuit breaker ---
    cb_failure_threshold: int = 5
    cb_recovery_seconds: int = 30
    cb_half_open_max_calls: int = 3

    # --- websocket ---
    ws_max_connections: int = 10_000
    ws_max_message_size_kb: int = 256
    ws_heartbeat_seconds: int = 30
    ws_idle_timeout_seconds: int = 300

    # --- monitoring ---
    metrics_enabled: bool = True

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

    @property
    def is_production(self) -> bool:
        return self.env == GatewayEnvironment.PRODUCTION


@lru_cache(maxsize=1)
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


def reload_gateway_settings() -> GatewaySettings:
    get_gateway_settings.cache_clear()
    return get_gateway_settings()
