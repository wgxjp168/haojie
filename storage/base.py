"""Object storage abstraction with local-disk and S3 back-ends.

The public surface (``StorageManager``) is identical regardless of
backend. All file operations run on a worker thread so they don't block
the event loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import Settings
from config.settings import StorageBackend as StorageBackendType
from config.settings import get_settings
from core.exceptions import ErrorCode, InputProcessorError, NotFoundError
from core.logging import get_logger

logger = get_logger(__name__)


class StorageBackend(ABC):
    @abstractmethod
    async def save(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    @abstractmethod
    async def get(self, key: str) -> bytes: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def delete(self, key: str) -> bool: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


class LocalStorageBackend(StorageBackend):
    """Write files to a directory on the local filesystem."""

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Prevent path traversal — resolve and assert containment.
        target = (self._base / key).resolve()
        if not str(target).startswith(str(self._base)):
            raise InputProcessorError(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Invalid storage key: {key}",
            )
        return target

    async def save(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _write():
            with path.open("wb") as f:
                f.write(data)

        await asyncio.to_thread(_write)
        return str(path)

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise NotFoundError("storage_object", key)

        def _read() -> bytes:
            with path.open("rb") as f:
                return f.read()

        return await asyncio.to_thread(_read)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)

    async def delete(self, key: str) -> bool:
        path = self._path(key)

        def _delete() -> bool:
            if path.exists():
                path.unlink()
                return True
            return False

        return await asyncio.to_thread(_delete)

    async def health_check(self) -> bool:
        try:
            return self._base.is_dir()
        except Exception:
            return False


class S3StorageBackend(StorageBackend):
    """S3 / MinIO compatible backend using boto3 in a worker thread."""

    def __init__(self, settings: Settings) -> None:
        import boto3
        from botocore.client import Config

        self._bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    async def save(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        def _put():
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )

        await asyncio.to_thread(_put)
        return key

    async def get(self, key: str) -> bytes:
        def _get() -> bytes:
            try:
                response = self._client.get_object(Bucket=self._bucket, Key=key)
                return response["Body"].read()
            except self._client.exceptions.NoSuchKey:
                raise NotFoundError("storage_object", key)

        return await asyncio.to_thread(_get)

    async def exists(self, key: str) -> bool:
        def _head() -> bool:
            try:
                self._client.head_object(Bucket=self._bucket, Key=key)
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_head)

    async def delete(self, key: str) -> bool:
        def _delete() -> bool:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return True

        return await asyncio.to_thread(_delete)

    async def health_check(self) -> bool:
        def _head_bucket() -> bool:
            try:
                self._client.head_bucket(Bucket=self._bucket)
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_head_bucket)


class StorageManager:
    """Main storage facade."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._backend = self._build_backend()

    def _build_backend(self) -> StorageBackend:
        if self._settings.storage_backend == StorageBackendType.S3:
            return S3StorageBackend(self._settings)
        return LocalStorageBackend(self._settings.storage_local_path)

    @staticmethod
    def _make_key(input_type: str, user_id: str, extension: str = "") -> str:
        day = datetime.now(tz=timezone.utc).strftime("%Y/%m/%d")
        unique = uuid.uuid4().hex[:12]
        # Sanitise user_id (no path separators)
        safe_user = user_id.replace("/", "_").replace("\\", "_")[:64]
        suffix = f".{extension.lstrip('.')}" if extension else ""
        return f"inputs/{input_type}/{day}/{safe_user}/{unique}{suffix}"

    async def save_input(
        self,
        data: bytes,
        input_type: str,
        user_id: str,
        extension: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        key = self._make_key(input_type, user_id, extension)
        await self._backend.save(key, data)
        # Side-car metadata file so we can retrieve context later.
        meta = {
            "user_id": user_id,
            "input_type": input_type,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "stored_at": datetime.now(tz=timezone.utc).isoformat(),
            "custom": metadata or {},
        }
        await self._backend.save(
            f"{key}.meta.json",
            json.dumps(meta, ensure_ascii=False).encode("utf-8"),
            "application/json",
        )
        logger.debug("storage.saved", key=key, size_bytes=len(data))
        return key

    async def get_input(self, key: str) -> bytes:
        return await self._backend.get(key)

    async def get_input_metadata(self, key: str) -> Dict[str, Any]:
        try:
            raw = await self._backend.get(f"{key}.meta.json")
            return json.loads(raw.decode("utf-8"))
        except NotFoundError:
            return {}
        except Exception as e:
            logger.warning("storage.metadata_read_failed", key=key, error=str(e))
            return {}

    async def delete_input(self, key: str) -> bool:
        ok = await self._backend.delete(key)
        await self._backend.delete(f"{key}.meta.json")
        return ok

    async def health_check(self) -> bool:
        return await self._backend.health_check()


_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager
