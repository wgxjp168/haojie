"""Domain-specific exceptions.

Every exception carries a stable ``code`` (for API clients), an HTTP
``status_code``, and a human-readable ``message``. Optional ``details``
are attached for structured client handling.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCode(str, Enum):
    # generic
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    FORBIDDEN = "FORBIDDEN"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    CONFLICT = "CONFLICT"
    # input
    INVALID_INPUT = "INVALID_INPUT"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    # external
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    EXTERNAL_SERVICE_TIMEOUT = "EXTERNAL_SERVICE_TIMEOUT"


_STATUS_MAP: Dict[ErrorCode, int] = {
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.CONFLICT: 409,
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.INPUT_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_FORMAT: 415,
    ErrorCode.PROCESSING_TIMEOUT: 408,
    ErrorCode.PROCESSING_FAILED: 500,
    ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
    ErrorCode.EXTERNAL_SERVICE_TIMEOUT: 504,
}


class InputProcessorError(Exception):
    """Base for all domain errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code or _STATUS_MAP.get(code, 500)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(InputProcessorError):
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details: Dict[str, Any] = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)[:200]
        super().__init__(ErrorCode.VALIDATION_ERROR, message, details)


class UnsupportedFormatError(InputProcessorError):
    def __init__(self, fmt: str, supported: Optional[List[str]] = None):
        super().__init__(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"Unsupported format: {fmt}",
            {"format": fmt, "supported": supported or []},
        )


class InputTooLargeError(InputProcessorError):
    def __init__(self, size: int, limit: int):
        super().__init__(
            ErrorCode.INPUT_TOO_LARGE,
            f"Input exceeds limit: {size} > {limit} bytes",
            {"size": size, "limit": limit},
        )


class RateLimitExceededError(InputProcessorError):
    def __init__(self, limit: int, window_seconds: int, retry_after: Optional[int] = None):
        details: Dict[str, Any] = {"limit": limit, "window_seconds": window_seconds}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(
            ErrorCode.RATE_LIMIT_EXCEEDED,
            f"Rate limit exceeded: {limit} per {window_seconds}s",
            details,
        )


class ExternalServiceError(InputProcessorError):
    def __init__(self, service: str, error: str, status: Optional[int] = None):
        details: Dict[str, Any] = {"service": service, "upstream_error": error}
        if status is not None:
            details["upstream_status"] = status
        super().__init__(
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            f"External service {service} failed: {error}",
            details,
        )


class ProcessingFailedError(InputProcessorError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.PROCESSING_FAILED, message, details)


class NotFoundError(InputProcessorError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            ErrorCode.NOT_FOUND,
            f"{resource} not found: {identifier}",
            {"resource": resource, "id": identifier},
        )
