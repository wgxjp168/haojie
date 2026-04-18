"""Session and input-item entities."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from entities.enums import DeviceType, InputType, ProcessingStatus, SessionStatus


@dataclass
class SessionContext:
    """Free-form per-session context accumulated during the conversation."""

    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    state: str = "initial"
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key == "entities" and isinstance(value, dict):
                self.entities.update(value)
            elif key == "parameters" and isinstance(value, dict):
                self.parameters.update(value)
            elif key == "metadata" and isinstance(value, dict):
                self.metadata.update(value)
            elif hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "parameters": self.parameters,
            "state": self.state,
            "step": self.step,
            "metadata": self.metadata,
        }


@dataclass
class InputItem:
    """A single user input ingested during a session."""

    input_id: str
    session_id: str
    user_id: str
    input_type: InputType
    input_summary: str  # short description / URL / storage key; never raw blob
    status: ProcessingStatus = ProcessingStatus.PENDING
    processed_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    processed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_processing(self) -> None:
        self.status = ProcessingStatus.PROCESSING

    def mark_success(self, data: Dict[str, Any], duration_ms: float) -> None:
        self.status = ProcessingStatus.SUCCESS
        self.processed_data = data
        self.processing_time_ms = duration_ms
        self.processed_at = datetime.now(tz=timezone.utc)

    def mark_failed(self, error: str, duration_ms: Optional[float] = None) -> None:
        self.status = ProcessingStatus.FAILED
        self.error_message = error
        self.processing_time_ms = duration_ms
        self.processed_at = datetime.now(tz=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_id": self.input_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "input_type": self.input_type.value,
            "input_summary": self.input_summary,
            "status": self.status.value,
            "processed_data": self.processed_data,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class Session:
    """Conversation / interaction session belonging to one user."""

    session_id: str
    user_id: str
    device_type: DeviceType = DeviceType.UNKNOWN
    status: SessionStatus = SessionStatus.ACTIVE
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    context: SessionContext = field(default_factory=SessionContext)
    inputs: List[InputItem] = field(default_factory=list)
    max_inactivity_minutes: int = 30
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

    def touch(self) -> None:
        now = datetime.now(tz=timezone.utc)
        self.last_activity_at = now
        self.updated_at = now

    def is_expired(self) -> bool:
        if self.status != SessionStatus.ACTIVE:
            return True
        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            minutes=self.max_inactivity_minutes
        )
        return self.last_activity_at < cutoff

    def is_active(self) -> bool:
        return self.status == SessionStatus.ACTIVE and not self.is_expired()

    def add_input(self, item: InputItem) -> None:
        self.inputs.append(item)
        # Keep only the most recent 500 inputs to bound memory.
        if len(self.inputs) > 500:
            self.inputs = self.inputs[-500:]
        self.touch()

    def duration_seconds(self) -> float:
        end = self.ended_at or datetime.now(tz=timezone.utc)
        return (end - self.created_at).total_seconds()

    def end(self, reason: str = "user_end") -> None:
        self.status = SessionStatus.COMPLETED
        self.ended_at = datetime.now(tz=timezone.utc)
        self.metadata["end_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "device_type": self.device_type.value,
            "status": self.status.value,
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "context": self.context.to_dict(),
            "input_count": len(self.inputs),
            "recent_inputs": [i.to_dict() for i in self.inputs[-10:]],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds(),
            "metadata": self.metadata,
        }
