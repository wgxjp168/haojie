"""LLM-service exception hierarchy.

Each error carries a stable code, an HTTP-status hint, and a structured
``details`` dict so the FastAPI layer can render a deterministic JSON
envelope. Mirrors ``intent.core.exceptions`` and
``decision.core.exceptions`` to keep operator muscle-memory consistent.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class LLMErrorCode(str, Enum):
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_PROVIDER = "UNKNOWN_PROVIDER"
    UNKNOWN_MODEL = "UNKNOWN_MODEL"
    NO_PROVIDER_AVAILABLE = "NO_PROVIDER_AVAILABLE"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    RATE_LIMITED = "RATE_LIMITED"
    CACHE_ERROR = "CACHE_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    CONTENT_FILTERED = "CONTENT_FILTERED"


_STATUS: Dict[LLMErrorCode, int] = {
    LLMErrorCode.INTERNAL_ERROR: 500,
    LLMErrorCode.VALIDATION_ERROR: 422,
    LLMErrorCode.UNKNOWN_PROVIDER: 400,
    LLMErrorCode.UNKNOWN_MODEL: 400,
    LLMErrorCode.NO_PROVIDER_AVAILABLE: 503,
    LLMErrorCode.PROVIDER_FAILED: 502,
    LLMErrorCode.PROVIDER_TIMEOUT: 504,
    LLMErrorCode.CIRCUIT_OPEN: 503,
    LLMErrorCode.RATE_LIMITED: 429,
    LLMErrorCode.CACHE_ERROR: 500,
    LLMErrorCode.UNAUTHORIZED: 401,
    LLMErrorCode.BUDGET_EXCEEDED: 402,
    LLMErrorCode.CONTENT_FILTERED: 451,
}


class LLMError(Exception):
    """Base class for LLM-service errors."""

    code: LLMErrorCode = LLMErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[LLMErrorCode] = None,
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


class ValidationLLMError(LLMError):
    code = LLMErrorCode.VALIDATION_ERROR


class UnknownProviderError(LLMError):
    code = LLMErrorCode.UNKNOWN_PROVIDER


class UnknownModelError(LLMError):
    code = LLMErrorCode.UNKNOWN_MODEL


class NoProviderAvailableError(LLMError):
    code = LLMErrorCode.NO_PROVIDER_AVAILABLE


class ProviderFailedError(LLMError):
    code = LLMErrorCode.PROVIDER_FAILED


class ProviderTimeoutError(LLMError):
    code = LLMErrorCode.PROVIDER_TIMEOUT


class CircuitOpenError(LLMError):
    code = LLMErrorCode.CIRCUIT_OPEN


class RateLimitedError(LLMError):
    code = LLMErrorCode.RATE_LIMITED


class CacheBackendError(LLMError):
    code = LLMErrorCode.CACHE_ERROR


class UnauthorizedError(LLMError):
    code = LLMErrorCode.UNAUTHORIZED


class BudgetExceededError(LLMError):
    code = LLMErrorCode.BUDGET_EXCEEDED


class ContentFilteredError(LLMError):
    code = LLMErrorCode.CONTENT_FILTERED
