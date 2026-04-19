"""Local filesystem storage backend.

Writes reports into ``<base_dir>/<report_id>.<ext>`` and stores sidecar
``*.meta.json`` files so ``get`` can restore content-type / checksum on
read. Safe to use in containerised deployments via a mounted volume.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from report.core.exceptions import (
    ReportNotFoundError,
    StorageFailedError,
    StorageUnavailableError,
)
from report.core.logging import get_logger
from report.storage.base import BaseStorageBackend, StoredObject

log = get_logger(__name__)


_EXT_BY_CONTENT_TYPE = {
    "text/markdown; charset=utf-8": "md",
    "text/html; charset=utf-8": "html",
    "application/json; charset=utf-8": "json",
    "text/plain; charset=utf-8": "txt",
}


class LocalFilesystemBackend(BaseStorageBackend):
    """Local filesystem storage (one file per report + sidecar JSON)."""

    name = "local"

    def __init__(self, *, base_dir: str = "./storage/reports") -> None:
        self.base_dir = Path(base_dir).resolve()
        self._lock = threading.Lock()
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depends on env
            log.warning(
                "report.storage.local.unavailable",
                base_dir=str(self.base_dir),
                error=str(exc),
            )
            self._available = False
        else:
            self._available = True

    @property
    def available(self) -> bool:
        return self._available and self.base_dir.exists() and os.access(self.base_dir, os.W_OK)

    def put(
        self,
        report_id: str,
        content: bytes,
        *,
        content_type: str,
        checksum: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        if not self.available:
            raise StorageUnavailableError(
                "local filesystem backend is not writable",
                details={"base_dir": str(self.base_dir)},
            )
        safe_id = self._safe_id(report_id)
        ext = _EXT_BY_CONTENT_TYPE.get(content_type, "bin")
        target = self.base_dir / f"{safe_id}.{ext}"
        sidecar = self.base_dir / f"{safe_id}.meta.json"
        stored_at = self._utcnow()
        with self._lock:
            try:
                target.write_bytes(bytes(content))
                sidecar.write_text(
                    json.dumps(
                        {
                            "report_id": report_id,
                            "content_type": content_type,
                            "checksum": checksum,
                            "stored_at": stored_at.isoformat(),
                            "metadata": metadata or {},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                raise StorageFailedError(
                    f"failed to write report {report_id!r} to local backend",
                    details={"path": str(target), "error": str(exc)},
                ) from exc
        return f"file://{target}"

    def get(self, report_id: str) -> StoredObject:
        if not self.available:
            raise StorageUnavailableError("local filesystem backend is not readable")
        safe_id = self._safe_id(report_id)
        sidecar = self.base_dir / f"{safe_id}.meta.json"
        if not sidecar.exists():
            raise ReportNotFoundError(
                f"report {report_id!r} not found on disk",
                details={"sidecar": str(sidecar)},
            )
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageFailedError(
                f"corrupt sidecar for {report_id!r}",
                details={"error": str(exc)},
            ) from exc

        content_type = meta.get("content_type", "application/octet-stream")
        ext = _EXT_BY_CONTENT_TYPE.get(content_type, "bin")
        target = self.base_dir / f"{safe_id}.{ext}"
        if not target.exists():
            raise ReportNotFoundError(
                f"report body for {report_id!r} missing",
                details={"path": str(target)},
            )
        try:
            content = target.read_bytes()
        except OSError as exc:
            raise StorageFailedError(
                f"failed to read report {report_id!r}",
                details={"error": str(exc)},
            ) from exc

        try:
            stored_at = datetime.fromisoformat(meta["stored_at"])
        except (KeyError, ValueError):
            stored_at = self._utcnow()

        return StoredObject(
            report_id=report_id,
            content=content,
            content_type=content_type,
            checksum=meta.get("checksum", ""),
            stored_at=stored_at,
            backend=self.name,
            metadata=meta.get("metadata", {}),
        )

    def exists(self, report_id: str) -> bool:
        safe_id = self._safe_id(report_id)
        return (self.base_dir / f"{safe_id}.meta.json").exists()

    def delete(self, report_id: str) -> bool:
        safe_id = self._safe_id(report_id)
        sidecar = self.base_dir / f"{safe_id}.meta.json"
        if not sidecar.exists():
            return False
        removed = False
        with self._lock:
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                ext = _EXT_BY_CONTENT_TYPE.get(
                    meta.get("content_type", ""), "bin"
                )
                body = self.base_dir / f"{safe_id}.{ext}"
                if body.exists():
                    body.unlink()
                    removed = True
                sidecar.unlink()
                removed = True
            except OSError as exc:  # pragma: no cover - depends on env
                log.warning(
                    "report.storage.local.delete_failed",
                    report_id=report_id,
                    error=str(exc),
                )
        return removed

    @staticmethod
    def _safe_id(report_id: str) -> str:
        # Never trust caller-supplied ids to be fs-safe.
        return "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in report_id
        )[:128]
