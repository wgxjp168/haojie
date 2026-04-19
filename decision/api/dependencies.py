"""FastAPI dependencies for the decision service."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header

from decision.config import DecisionSettings, get_decision_settings
from decision.core.exceptions import UnauthorizedError
from decision.service import DecisionService, get_decision_service


def settings_dep() -> DecisionSettings:
    return get_decision_settings()


def service_dep() -> DecisionService:
    return get_decision_service()


def require_api_key(
    settings: DecisionSettings = Depends(settings_dep),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.api_key_required:
        return
    if not x_api_key or x_api_key not in settings.api_keys:
        raise UnauthorizedError("missing or invalid api key")
