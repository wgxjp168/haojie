"""In-process memory storage backend.

Always available, zero-dep. Used as the default in tests and as a hot
cache layer in production when fanning out to a durable backend.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, Optional

from report.core.exceptions import ReportNotFoundError
from report.storage.base import BaseStorageBackend, StoredObject


class MemoryStorageBackend(BaseStorageBackend):
    """Thread-safe LRU-backed memory store."""

    name = "memory"

    def __init__(self, *, capacity: int = 1024) -> None:
        self.capacity = max(1, int(capacity))
        self._store: "OrderedDict[str, StoredObject]" = OrderedDict()
        self._lock = threading.Lock()

    def put(
        self,
        report_id: str,
        content: bytes,
        *,
        content_type: str,
        checksum: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        obj = StoredObject(
            report_id=report_id,
            content=bytes(content),
            content_type=content_type,
            checksum=checksum,
            stored_at=self._utcnow(),
            backend=self.name,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._store[report_id] = obj
            self._store.move_to_end(report_id)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)
        return f"memory://{report_id}"

    def get(self, report_id: str) -> StoredObject:
        with self._lock:
            obj = self._store.get(report_id)
            if obj is None:
                raise ReportNotFoundError(
                    f"report {report_id!r} not found in memory backend",
                    details={"report_id": report_id, "backend": self.name},
                )
            self._store.move_to_end(report_id)
            return obj

    def exists(self, report_id: str) -> bool:
        with self._lock:
            return report_id in self._store

    def delete(self, report_id: str) -> bool:
        with self._lock:
            return self._store.pop(report_id, None) is not None

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._store)
