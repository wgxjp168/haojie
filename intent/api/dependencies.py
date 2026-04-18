"""FastAPI dependencies for the intent service."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header

from intent.config import IntentSettings, get_intent_settings
from intent.core.exceptions import UnauthorizedError
from intent.service import IntentService, get_intent_service


def settings_dep() -> IntentSettings:
    return get_intent_settings()


def service_dep() -> IntentService:
    return get_intent_service()


def require_api_key(
    settings: IntentSettings = Depends(settings_dep),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.api_key_required:
        return
    if not x_api_key or x_api_key not in settings.api_keys:
        raise UnauthorizedError("missing or invalid api key")
