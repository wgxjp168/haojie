"""FastAPI dependencies for the LLM service."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header

from llm.config import LLMSettings, get_llm_settings
from llm.core.exceptions import UnauthorizedError
from llm.service import LLMService, get_llm_service


def settings_dep() -> LLMSettings:
    return get_llm_settings()


def service_dep() -> LLMService:
    return get_llm_service()


def require_api_key(
    settings: LLMSettings = Depends(settings_dep),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.api_key_required:
        return
    if not x_api_key or x_api_key not in settings.api_keys:
        raise UnauthorizedError("missing or invalid api key")
