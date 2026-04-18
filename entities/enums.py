"""Domain enums shared across the input processor."""
from __future__ import annotations

from enum import Enum


class UserType(str, Enum):
    B2B_PURCHASER = "B2B_PURCHASER"
    B2C_CONSUMER = "B2C_CONSUMER"
    ADMIN = "ADMIN"
    SYSTEM = "SYSTEM"


class UserRole(str, Enum):
    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    B2B_PURCHASER = "B2B_PURCHASER"
    B2C_CONSUMER = "B2C_CONSUMER"
    ANALYST = "ANALYST"
    GUEST = "GUEST"


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    LINK = "link"
    VOICE = "voice"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class DeviceType(str, Enum):
    WEB = "web"
    MOBILE = "mobile"
    IOS = "ios"
    ANDROID = "android"
    API = "api"
    UNKNOWN = "unknown"


class ProcessingMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"


class PriorityLevel(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
