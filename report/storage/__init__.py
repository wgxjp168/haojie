"""Storage backends for persisted reports."""
from __future__ import annotations

from report.storage.base import BaseStorageBackend, StoredObject  # noqa: F401
from report.storage.local import LocalFilesystemBackend  # noqa: F401
from report.storage.memory import MemoryStorageBackend  # noqa: F401
from report.storage.s3 import S3StorageBackend  # noqa: F401
