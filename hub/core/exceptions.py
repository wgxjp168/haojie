"""Hub-service exception hierarchy.

Mirrors the error envelopes used by intent/decision/llm/report so the
gateway and end clients see the same JSON shape regardless of which
stage failed.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class HubErrorCode(str, Enum):
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    UNKNOWN_STAGE = "UNKNOWN_STAGE"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UPSTREAM_FAILED = "UPSTREAM_FAILED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    PIPELINE_FAILED = "PIPELINE_FAILED"


_STATUS: Dict[HubErrorCode, int] = {
    HubErrorCode.INTERNAL_ERROR: 500,
    HubErrorCode.VALIDATION_ERROR: 422,
    HubErrorCode.UNAUTHORIZED: 401,
    HubErrorCode.UNKNOWN_STAGE: 400,
    HubErrorCode.UPSTREAM_TIMEOUT: 504,
    HubErrorCode.UPSTREAM_FAILED: 502,
    HubErrorCode.UPSTREAM_UNAVAILABLE: 503,
    HubErrorCode.PIPELINE_FAILED: 500,
}


class HubError(Exception):
    """Base class for hub-service errors."""

    code: HubErrorCode = HubErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[HubErrorCode] = None,
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


class ValidationHubError(HubError):
    code = HubErrorCode.VALIDATION_ERROR


class UnauthorizedError(HubError):
    code = HubErrorCode.UNAUTHORIZED


class UnknownStageError(HubError):
    code = HubErrorCode.UNKNOWN_STAGE


class UpstreamTimeoutError(HubError):
    code = HubErrorCode.UPSTREAM_TIMEOUT


class UpstreamFailedError(HubError):
    code = HubErrorCode.UPSTREAM_FAILED


class UpstreamUnavailableError(HubError):
    code = HubErrorCode.UPSTREAM_UNAVAILABLE


class PipelineFailedError(HubError):
    code = HubErrorCode.PIPELINE_FAILED
