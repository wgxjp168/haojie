"""Report-service exception hierarchy.

Mirrors ``intent.core.exceptions``, ``decision.core.exceptions`` and
``llm.core.exceptions`` so operators see the same error envelope shape
across the AI Decision Hub.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ReportErrorCode(str, Enum):
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_FORMAT = "UNKNOWN_FORMAT"
    UNKNOWN_TEMPLATE = "UNKNOWN_TEMPLATE"
    UNKNOWN_AUDIENCE = "UNKNOWN_AUDIENCE"
    RENDER_FAILED = "RENDER_FAILED"
    STORAGE_FAILED = "STORAGE_FAILED"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"


_STATUS: Dict[ReportErrorCode, int] = {
    ReportErrorCode.INTERNAL_ERROR: 500,
    ReportErrorCode.VALIDATION_ERROR: 422,
    ReportErrorCode.UNKNOWN_FORMAT: 400,
    ReportErrorCode.UNKNOWN_TEMPLATE: 400,
    ReportErrorCode.UNKNOWN_AUDIENCE: 400,
    ReportErrorCode.RENDER_FAILED: 500,
    ReportErrorCode.STORAGE_FAILED: 502,
    ReportErrorCode.STORAGE_UNAVAILABLE: 503,
    ReportErrorCode.CIRCUIT_OPEN: 503,
    ReportErrorCode.NOT_FOUND: 404,
    ReportErrorCode.UNAUTHORIZED: 401,
    ReportErrorCode.PAYLOAD_TOO_LARGE: 413,
}


class ReportError(Exception):
    """Base class for report-service errors."""

    code: ReportErrorCode = ReportErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[ReportErrorCode] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}

    @property
    def http_status(self) -> int:
        return _STATUS.get(self.code, 500)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "details": self.details}


class ValidationReportError(ReportError):
    code = ReportErrorCode.VALIDATION_ERROR


class UnknownFormatError(ReportError):
    code = ReportErrorCode.UNKNOWN_FORMAT


class UnknownTemplateError(ReportError):
    code = ReportErrorCode.UNKNOWN_TEMPLATE


class UnknownAudienceError(ReportError):
    code = ReportErrorCode.UNKNOWN_AUDIENCE


class RenderFailedError(ReportError):
    code = ReportErrorCode.RENDER_FAILED


class StorageFailedError(ReportError):
    code = ReportErrorCode.STORAGE_FAILED


class StorageUnavailableError(ReportError):
    code = ReportErrorCode.STORAGE_UNAVAILABLE


class CircuitOpenError(ReportError):
    code = ReportErrorCode.CIRCUIT_OPEN


class ReportNotFoundError(ReportError):
    code = ReportErrorCode.NOT_FOUND


class UnauthorizedError(ReportError):
    code = ReportErrorCode.UNAUTHORIZED


class PayloadTooLargeError(ReportError):
    code = ReportErrorCode.PAYLOAD_TOO_LARGE
