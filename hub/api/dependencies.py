"""FastAPI dependencies for the hub service."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header

from hub.config import HubSettings, get_hub_settings
from hub.core.exceptions import UnauthorizedError
from hub.service import HubService, get_hub_service


def settings_dep() -> HubSettings:
    return get_hub_settings()


def service_dep() -> HubService:
    return get_hub_service()


def require_api_key(
    settings: HubSettings = Depends(settings_dep),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.api_key_required:
        return
    if not x_api_key or x_api_key not in settings.api_keys:
        raise UnauthorizedError("missing or invalid api key")
