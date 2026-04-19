"""FastAPI dependencies for the report service."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header

from report.config import ReportSettings, get_report_settings
from report.core.exceptions import UnauthorizedError
from report.service import ReportService, get_report_service


def settings_dep() -> ReportSettings:
    return get_report_settings()


def service_dep() -> ReportService:
    return get_report_service()


def require_api_key(
    settings: ReportSettings = Depends(settings_dep),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.api_key_required:
        return
    if not x_api_key or x_api_key not in settings.api_keys:
        raise UnauthorizedError("missing or invalid api key")
