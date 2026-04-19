"""Decision-engine exception hierarchy.

Each error carries a stable code, an HTTP status hint, and a structured
``details`` dict so the FastAPI layer can render a deterministic JSON
envelope. The hierarchy intentionally mirrors ``intent.core.exceptions``
to keep operator muscle-memory consistent across services.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class DecisionErrorCode(str, Enum):
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_STRATEGY = "UNSUPPORTED_STRATEGY"
    UNKNOWN_ACTION = "UNKNOWN_ACTION"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    MODEL_FAILED = "MODEL_FAILED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    CACHE_ERROR = "CACHE_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    POLICY_VIOLATION = "POLICY_VIOLATION"


_STATUS: Dict[DecisionErrorCode, int] = {
    DecisionErrorCode.INTERNAL_ERROR: 500,
    DecisionErrorCode.VALIDATION_ERROR: 422,
    DecisionErrorCode.UNSUPPORTED_STRATEGY: 400,
    DecisionErrorCode.UNKNOWN_ACTION: 500,
    DecisionErrorCode.MODEL_NOT_READY: 503,
    DecisionErrorCode.MODEL_FAILED: 500,
    DecisionErrorCode.CIRCUIT_OPEN: 503,
    DecisionErrorCode.CACHE_ERROR: 500,
    DecisionErrorCode.UNAUTHORIZED: 401,
    DecisionErrorCode.POLICY_VIOLATION: 409,
}


class DecisionError(Exception):
    """Base class for decision-engine errors."""

    code: DecisionErrorCode = DecisionErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[DecisionErrorCode] = None,
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


class ValidationDecisionError(DecisionError):
    code = DecisionErrorCode.VALIDATION_ERROR


class UnsupportedStrategyError(DecisionError):
    code = DecisionErrorCode.UNSUPPORTED_STRATEGY


class UnknownActionError(DecisionError):
    code = DecisionErrorCode.UNKNOWN_ACTION


class ModelNotReadyError(DecisionError):
    code = DecisionErrorCode.MODEL_NOT_READY


class ModelFailedError(DecisionError):
    code = DecisionErrorCode.MODEL_FAILED


class CircuitOpenError(DecisionError):
    code = DecisionErrorCode.CIRCUIT_OPEN


class CacheBackendError(DecisionError):
    code = DecisionErrorCode.CACHE_ERROR


class UnauthorizedError(DecisionError):
    code = DecisionErrorCode.UNAUTHORIZED


class PolicyViolationError(DecisionError):
    code = DecisionErrorCode.POLICY_VIOLATION
