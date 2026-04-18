"""Intent-service exceptions."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class IntentErrorCode(str, Enum):
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    MODEL_FAILED = "MODEL_FAILED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    CACHE_ERROR = "CACHE_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"


_STATUS: Dict[IntentErrorCode, int] = {
    IntentErrorCode.INTERNAL_ERROR: 500,
    IntentErrorCode.VALIDATION_ERROR: 422,
    IntentErrorCode.UNSUPPORTED_LANGUAGE: 400,
    IntentErrorCode.MODEL_NOT_READY: 503,
    IntentErrorCode.MODEL_FAILED: 500,
    IntentErrorCode.CIRCUIT_OPEN: 503,
    IntentErrorCode.CACHE_ERROR: 500,
    IntentErrorCode.UNAUTHORIZED: 401,
}


class IntentError(Exception):
    """Base class for intent-service errors."""

    code: IntentErrorCode = IntentErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[IntentErrorCode] = None,
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


class ValidationIntentError(IntentError):
    code = IntentErrorCode.VALIDATION_ERROR


class UnsupportedLanguageError(IntentError):
    code = IntentErrorCode.UNSUPPORTED_LANGUAGE


class ModelNotReadyError(IntentError):
    code = IntentErrorCode.MODEL_NOT_READY


class ModelFailedError(IntentError):
    code = IntentErrorCode.MODEL_FAILED


class CircuitOpenError(IntentError):
    code = IntentErrorCode.CIRCUIT_OPEN


class UnauthorizedError(IntentError):
    code = IntentErrorCode.UNAUTHORIZED
