"""Abstract storage backend for persisted reports.

Concrete backends implement ``put`` / ``get`` / ``exists`` / ``delete``
so upstream callers (``ReportEngine``) can persist reports without
knowing whether they land in memory, on disk, or in S3.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass(frozen=True)
class StoredObject:
    """Envelope returned by ``get`` — payload + minimal metadata."""

    report_id: str
    content: bytes
    content_type: str
    checksum: str
    stored_at: datetime
    backend: str
    metadata: Dict[str, str] = field(default_factory=dict)


class BaseStorageBackend(abc.ABC):
    """Abstract storage backend.

    Each backend has a stable ``name`` label surfaced in metrics and
    ``StorageResult.backend``; make sure subclasses set it.
    """

    name: str = "base"

    @property
    def available(self) -> bool:
        """Whether the backend can currently serve writes/reads."""
        return True

    @abc.abstractmethod
    def put(
        self,
        report_id: str,
        content: bytes,
        *,
        content_type: str,
        checksum: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Persist ``content`` under ``report_id``. Return a location hint."""

    @abc.abstractmethod
    def get(self, report_id: str) -> StoredObject:
        """Retrieve a stored report. Raises on miss."""

    @abc.abstractmethod
    def exists(self, report_id: str) -> bool:
        """Check whether a report is persisted."""

    @abc.abstractmethod
    def delete(self, report_id: str) -> bool:
        """Remove a stored report; return ``True`` if something was removed."""

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)
