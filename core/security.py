"""Security helpers: password hashing, JWT issuing/verification, API keys.

Passwords are hashed with bcrypt via passlib. JWTs use PyJWT with
configurable algorithm. All time values are in UTC.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
from passlib.context import CryptContext

from config.settings import Settings, get_settings
from core.exceptions import InputProcessorError, ErrorCode

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --- passwords ---

def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd_context.verify(plain, hashed)
    except Exception:
        return False


# --- JWT ---

def create_access_token(
    subject: str,
    roles: Optional[List[str]] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
    expires_minutes: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> str:
    cfg = settings or get_settings()
    now = datetime.now(tz=timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(
            (now + timedelta(minutes=expires_minutes or cfg.jwt_access_token_expire_minutes))
            .timestamp()
        ),
        "type": "access",
        "roles": roles or [],
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def decode_token(token: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    cfg = settings or get_settings()
    try:
        return jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise InputProcessorError(ErrorCode.UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError as e:
        raise InputProcessorError(ErrorCode.UNAUTHORIZED, f"Invalid token: {e}")


# --- API keys ---

def hash_api_key(api_key: str) -> str:
    """API keys are compared by hash to avoid timing attacks on equality."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(api_key: str, valid_keys: List[str]) -> bool:
    if not api_key or not valid_keys:
        return False
    candidate = hash_api_key(api_key)
    for key in valid_keys:
        if secrets.compare_digest(candidate, hash_api_key(key)):
            return True
    return False


def generate_api_key(prefix: str = "sk") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"
