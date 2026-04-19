"""AWS S3 storage backend (optional: requires ``boto3``).

When ``boto3`` is not installed or credentials are missing, the backend
reports ``available == False`` and is skipped by the engine's fan-out.
Supports MinIO / localstack via ``endpoint_url``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from report.core.exceptions import (
    ReportNotFoundError,
    StorageFailedError,
    StorageUnavailableError,
)
from report.core.logging import get_logger
from report.storage.base import BaseStorageBackend, StoredObject

log = get_logger(__name__)

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover - optional dep
    boto3 = None  # type: ignore
    BotoCoreError = Exception  # type: ignore
    ClientError = Exception  # type: ignore


_EXT_BY_CONTENT_TYPE = {
    "text/markdown; charset=utf-8": "md",
    "text/html; charset=utf-8": "html",
    "application/json; charset=utf-8": "json",
    "text/plain; charset=utf-8": "txt",
}


class S3StorageBackend(BaseStorageBackend):
    """S3 / S3-compatible storage."""

    name = "s3"

    def __init__(
        self,
        *,
        bucket: Optional[str],
        region: Optional[str] = None,
        prefix: str = "reports/",
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.endpoint_url = endpoint_url
        self._client: Optional[Any] = None

        if boto3 is None or not bucket:
            log.info(
                "report.storage.s3.disabled",
                reason="boto3 missing" if boto3 is None else "no bucket",
            )
            return

        try:
            self._client = boto3.client(
                "s3",
                region_name=region,
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            # Best-effort probe.
            self._client.head_bucket(Bucket=bucket)
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - needs aws
            log.warning(
                "report.storage.s3.unavailable",
                bucket=bucket,
                error=str(exc),
            )
            self._client = None

    @property
    def available(self) -> bool:
        return bool(self._client is not None and self.bucket)

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
                "S3 backend is unavailable (boto3 or credentials missing)",
                details={"bucket": self.bucket},
            )
        key = self._key_for(report_id, content_type)
        extra_meta = {"checksum": checksum, "report_id": report_id}
        if metadata:
            extra_meta.update({k: str(v) for k, v in metadata.items()})

        try:
            self._client.put_object(  # type: ignore[union-attr]
                Bucket=self.bucket,
                Key=key,
                Body=bytes(content),
                ContentType=content_type,
                Metadata=extra_meta,
            )
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - needs aws
            raise StorageFailedError(
                f"S3 put failed for {report_id!r}",
                details={"bucket": self.bucket, "key": key, "error": str(exc)},
            ) from exc

        return f"s3://{self.bucket}/{key}"

    def get(self, report_id: str) -> StoredObject:
        if not self.available:
            raise StorageUnavailableError("S3 backend is unavailable")
        # We don't know the extension without the metadata; try common ones.
        for content_type in _EXT_BY_CONTENT_TYPE.keys():
            key = self._key_for(report_id, content_type)
            try:
                resp = self._client.get_object(  # type: ignore[union-attr]
                    Bucket=self.bucket, Key=key
                )
            except ClientError as exc:  # pragma: no cover - needs aws
                code = getattr(exc, "response", {}).get("Error", {}).get("Code")
                if code in {"NoSuchKey", "404"}:
                    continue
                raise StorageFailedError(
                    f"S3 get failed for {report_id!r}",
                    details={"error": str(exc)},
                ) from exc
            body = resp["Body"].read()
            meta = resp.get("Metadata", {})
            return StoredObject(
                report_id=report_id,
                content=body,
                content_type=resp.get("ContentType", content_type),
                checksum=meta.get("checksum", ""),
                stored_at=resp.get("LastModified", self._utcnow()),
                backend=self.name,
                metadata=meta,
            )
        raise ReportNotFoundError(
            f"report {report_id!r} not found in S3",
            details={"bucket": self.bucket},
        )

    def exists(self, report_id: str) -> bool:
        if not self.available:
            return False
        for content_type in _EXT_BY_CONTENT_TYPE.keys():
            key = self._key_for(report_id, content_type)
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)  # type: ignore[union-attr]
                return True
            except ClientError:  # pragma: no cover
                continue
        return False

    def delete(self, report_id: str) -> bool:
        if not self.available:
            return False
        removed = False
        for content_type in _EXT_BY_CONTENT_TYPE.keys():
            key = self._key_for(report_id, content_type)
            try:
                self._client.delete_object(Bucket=self.bucket, Key=key)  # type: ignore[union-attr]
                removed = True
            except ClientError:  # pragma: no cover
                continue
        return removed

    def _key_for(self, report_id: str, content_type: str) -> str:
        ext = _EXT_BY_CONTENT_TYPE.get(content_type, "bin")
        return f"{self.prefix}{report_id}.{ext}"
