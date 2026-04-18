"""Security primitives for the gateway: password hashing + API-key compare.

JWT handling is in ``gateway.auth.jwt_manager`` since it needs more state
(blacklist, refresh, rotation).
"""
from __future__ import annotations

import hashlib
import secrets

from passlib.context import CryptContext

# Primary scheme: bcrypt. Fall back to pbkdf2_sha256 in environments where
# bcrypt native bindings are mis-configured (e.g. some CI images).
_pwd_context = CryptContext(
    schemes=["pbkdf2_sha256", "bcrypt"],
    default="pbkdf2_sha256",
    deprecated="auto",
)


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd_context.verify(plain, hashed)
    except Exception:
        return False


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_api_key(provided: str, valid_keys: list[str]) -> bool:
    if not provided or not valid_keys:
        return False
    candidate = _hash_key(provided)
    return any(secrets.compare_digest(candidate, _hash_key(k)) for k in valid_keys)


def generate_api_key(prefix: str = "sk") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"
