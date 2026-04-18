"""Gateway-specific exceptions."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class GatewayErrorCode(str, Enum):
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    BAD_GATEWAY = "BAD_GATEWAY"
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_REVOKED = "TOKEN_REVOKED"
    TOKEN_INVALID = "TOKEN_INVALID"


_STATUS = {
    GatewayErrorCode.INTERNAL_ERROR: 500,
    GatewayErrorCode.UNAUTHORIZED: 401,
    GatewayErrorCode.FORBIDDEN: 403,
    GatewayErrorCode.NOT_FOUND: 404,
    GatewayErrorCode.BAD_REQUEST: 400,
    GatewayErrorCode.VALIDATION_ERROR: 422,
    GatewayErrorCode.RATE_LIMITED: 429,
    GatewayErrorCode.CIRCUIT_OPEN: 503,
    GatewayErrorCode.BAD_GATEWAY: 502,
    GatewayErrorCode.UPSTREAM_TIMEOUT: 504,
    GatewayErrorCode.TOKEN_EXPIRED: 401,
    GatewayErrorCode.TOKEN_REVOKED: 401,
    GatewayErrorCode.TOKEN_INVALID: 401,
}


class GatewayError(Exception):
    def __init__(
        self,
        code: GatewayErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code or _STATUS.get(code, 500)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class UnauthorizedError(GatewayError):
    def __init__(self, message: str = "Unauthorized", **details):
        super().__init__(GatewayErrorCode.UNAUTHORIZED, message, details)


class TokenExpiredError(GatewayError):
    def __init__(self):
        super().__init__(GatewayErrorCode.TOKEN_EXPIRED, "Token has expired")


class TokenRevokedError(GatewayError):
    def __init__(self):
        super().__init__(GatewayErrorCode.TOKEN_REVOKED, "Token has been revoked")


class InvalidTokenError(GatewayError):
    def __init__(self, message: str = "Invalid token"):
        super().__init__(GatewayErrorCode.TOKEN_INVALID, message)


class ForbiddenError(GatewayError):
    def __init__(self, message: str = "Forbidden", **details):
        super().__init__(GatewayErrorCode.FORBIDDEN, message, details)


class RateLimitedError(GatewayError):
    def __init__(self, limit: int, window_seconds: int, retry_after: Optional[int] = None):
        details: Dict[str, Any] = {"limit": limit, "window_seconds": window_seconds}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(
            GatewayErrorCode.RATE_LIMITED,
            f"Rate limit exceeded: {limit}/{window_seconds}s",
            details,
        )


class CircuitOpenError(GatewayError):
    def __init__(self, service: str):
        super().__init__(
            GatewayErrorCode.CIRCUIT_OPEN,
            f"Circuit open for {service}",
            {"service": service},
        )


class BadGatewayError(GatewayError):
    def __init__(self, message: str, **details):
        super().__init__(GatewayErrorCode.BAD_GATEWAY, message, details)


class UpstreamTimeoutError(GatewayError):
    def __init__(self, timeout_seconds: float):
        super().__init__(
            GatewayErrorCode.UPSTREAM_TIMEOUT,
            f"Upstream timed out after {timeout_seconds}s",
            {"timeout_seconds": timeout_seconds},
        )


class ValidationGatewayError(GatewayError):
    def __init__(self, message: str, **details):
        super().__init__(GatewayErrorCode.VALIDATION_ERROR, message, details)
