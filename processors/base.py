"""Base classes for input processors.

An ``BaseInputProcessor`` implementation takes raw input (of a specific
modality) plus an ``InputContext`` and returns a uniform
``ProcessorResult`` dict. The manager orchestrates the call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from entities.enums import InputType


@dataclass
class InputContext:
    """Per-request context shared with the processor."""

    user_id: str
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    language: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# A processor's ``process`` returns a plain dict — the manager wraps it
# into a ProcessingResult with timing/metadata. The convention for the
# returned dict is:
#   {
#     "data": {...},      # processed fields specific to the modality
#     "warnings": [...],  # optional non-fatal warnings
#     "summary": "..."    # short human-readable summary
#   }
ProcessorResult = Dict[str, Any]


class BaseInputProcessor(ABC):
    """Contract every input processor implements."""

    #: The modality this processor handles.
    input_type: InputType

    @abstractmethod
    def validate(self, input_data: Any) -> None:
        """Raise a ``ValidationError`` (or subclass) if input is invalid.

        Implementations should keep this cheap — deeper validation can
        happen inside ``process``.
        """

    @abstractmethod
    async def process(
        self, input_data: Any, context: InputContext
    ) -> ProcessorResult: ...

    async def health_check(self) -> bool:  # pragma: no cover — trivial default
        return True

    async def close(self) -> None:  # pragma: no cover — trivial default
        return None

    @property
    def supported_formats(self) -> List[str]:
        return []
