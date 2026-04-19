"""Public report service — composes engine + cache + storage registry.

The service owns caching, request validation, singletons, and the
retrieval-by-id API (``get_report``). The engine stays I/O-focused.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from report.cache import ReportCache
from report.config import ReportSettings, StorageBackend, get_report_settings
from report.core.exceptions import (
    PayloadTooLargeError,
    ReportError,
    ReportNotFoundError,
    ValidationReportError,
)
from report.core.logging import get_logger
from report.core.metrics import REPORT_BATCH_SIZE, REPORT_REQUESTS_TOTAL
from report.engine import ReportEngine
from report.schemas import (
    BatchReportRequest,
    BatchReportResponse,
    ReportRequest,
    ReportResponse,
    StoredReport,
)
from report.storage import (
    BaseStorageBackend,
    LocalFilesystemBackend,
    MemoryStorageBackend,
    S3StorageBackend,
)

log = get_logger(__name__)


def _build_default_storages(settings: ReportSettings) -> List[BaseStorageBackend]:
    """Materialise configured storage backends in the requested order."""
    backends: List[BaseStorageBackend] = []
    seen: set = set()
    for name in settings.storage_backends:
        if name in seen:
            continue
        seen.add(name)
        try:
            kind = StorageBackend(name)
        except ValueError:
            log.warning("report.service.unknown_backend", backend=name)
            continue

        if kind == StorageBackend.MEMORY:
            backends.append(MemoryStorageBackend(capacity=settings.memory_cache_size))
        elif kind == StorageBackend.LOCAL:
            backends.append(
                LocalFilesystemBackend(base_dir=settings.storage_local_path)
            )
        elif kind == StorageBackend.S3:
            backends.append(
                S3StorageBackend(
                    bucket=settings.s3_bucket,
                    region=settings.s3_region,
                    prefix=settings.s3_prefix,
                    endpoint_url=settings.s3_endpoint_url,
                    access_key=settings.s3_access_key,
                    secret_key=settings.s3_secret_key,
                )
            )
    return backends


class ReportService:
    """High-level facade used by FastAPI routes and in-process callers."""

    def __init__(
        self,
        *,
        settings: Optional[ReportSettings] = None,
        engine: Optional[ReportEngine] = None,
        storages: Optional[List[BaseStorageBackend]] = None,
        cache: Optional[ReportCache] = None,
    ) -> None:
        self.settings = settings or get_report_settings()

        if storages is not None:
            self.storages = storages
        else:
            self.storages = _build_default_storages(self.settings)

        if engine is not None:
            self.engine = engine
        else:
            self.engine = ReportEngine(
                settings=self.settings, storages=self.storages
            )

        if cache is not None:
            self.cache = cache
        elif self.settings.cache_enabled:
            self.cache = ReportCache(
                memory_capacity=self.settings.memory_cache_size,
                redis_url=self.settings.redis_url,
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
        else:
            self.cache = ReportCache(
                memory_capacity=0, redis_url=None, ttl_seconds=0
            )

        self._storages_by_name: Dict[str, BaseStorageBackend] = {
            s.name: s for s in self.storages
        }
        self._started_at = time.time()

    # ---- public api ----

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    @property
    def cache_ready(self) -> bool:
        return bool(getattr(self.cache, "ready", False))

    @property
    def available_backends(self) -> List[str]:
        return self.engine.available_backends()

    @property
    def circuit_states(self) -> Dict[str, str]:
        return self.engine.circuit_states()

    def generate(self, request: ReportRequest) -> ReportResponse:
        start = time.perf_counter()
        self._validate_request(request)

        cache_key = request.fingerprint() if self.settings.cache_enabled else None
        cached = self.cache.get(cache_key) if cache_key else None
        if cached is not None:
            response = ReportResponse.model_validate(cached)
            response.cached = True
            response.request_id = request.request_id or response.request_id
            response.session_id = request.session_id or response.session_id
            response.metadata.setdefault("cache", "hit")
            return response

        try:
            response = self.engine.render(request)
        except ReportError:
            # Metrics already recorded by the engine path; rethrow for the API.
            raise

        if cache_key:
            self.cache.set(cache_key, response.model_dump(mode="json"))

        # Total latency incl. cache lookup is useful for dashboards.
        response.metadata["total_latency_ms"] = round(
            (time.perf_counter() - start) * 1000.0, 3
        )
        return response

    def generate_batch(self, batch: BatchReportRequest) -> BatchReportResponse:
        start = time.perf_counter()
        if len(batch.items) > self.settings.max_batch_size:
            raise ValidationReportError(
                f"batch size {len(batch.items)} exceeds max_batch_size "
                f"{self.settings.max_batch_size}",
                details={"max": self.settings.max_batch_size},
            )
        REPORT_BATCH_SIZE.observe(len(batch.items))

        results: List[ReportResponse] = []
        for item in batch.items:
            try:
                results.append(self.generate(item))
            except ReportError as exc:
                log.warning(
                    "report.service.batch_item_failed",
                    request_id=item.request_id,
                    code=exc.code.value,
                    error=exc.message,
                )
                REPORT_REQUESTS_TOTAL.labels(
                    format=(item.format or self.settings.default_format.value),
                    audience=(item.audience or self.settings.default_audience.value),
                    status="error",
                ).inc()
                results.append(self._error_response(item, exc))

        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return BatchReportResponse(
            results=results, total=len(results), latency_ms=latency_ms
        )

    def get_report(
        self, report_id: str, backend: Optional[str] = None
    ) -> StoredReport:
        """Retrieve a persisted report by id, optionally pinned to a backend."""
        order: List[BaseStorageBackend]
        if backend:
            if backend not in self._storages_by_name:
                raise ValidationReportError(
                    f"unknown storage backend {backend!r}",
                    details={"available": list(self._storages_by_name.keys())},
                )
            order = [self._storages_by_name[backend]]
        else:
            order = list(self.storages)

        for store in order:
            if not store.available:
                continue
            if not store.exists(report_id):
                continue
            obj = store.get(report_id)
            return StoredReport(
                report_id=obj.report_id,
                format=self._infer_format(obj.content_type),
                language=obj.metadata.get("language", self.settings.default_language),
                content=obj.content.decode("utf-8", errors="replace"),
                content_type=obj.content_type,
                checksum=obj.checksum,
                stored_at=obj.stored_at,
                backend=obj.backend,
            )

        raise ReportNotFoundError(
            f"report {report_id!r} not found in any configured backend",
            details={"report_id": report_id, "backends_checked": [s.name for s in order]},
        )

    def delete_report(self, report_id: str) -> Dict[str, bool]:
        outcomes: Dict[str, bool] = {}
        for store in self.storages:
            if not store.available:
                outcomes[store.name] = False
                continue
            try:
                outcomes[store.name] = store.delete(report_id)
            except ReportError as exc:
                log.warning(
                    "report.service.delete_failed",
                    backend=store.name,
                    error=exc.message,
                )
                outcomes[store.name] = False
        return outcomes

    # ---- internals ----

    def _validate_request(self, request: ReportRequest) -> None:
        # Rough payload-size check (serialized bytes).
        try:
            payload = request.model_dump_json()
        except Exception:
            payload = ""
        if len(payload) > self.settings.max_payload_bytes:
            raise PayloadTooLargeError(
                f"request body {len(payload)}B exceeds max_payload_bytes "
                f"{self.settings.max_payload_bytes}",
                details={"max_bytes": self.settings.max_payload_bytes},
            )

    def _error_response(
        self, request: ReportRequest, exc: ReportError
    ) -> ReportResponse:
        fmt = request.format or self.settings.default_format.value
        audience = request.audience or self.settings.default_audience.value
        template = request.template_id or "unknown"
        language = request.language or self.settings.default_language
        return ReportResponse(
            report_id=f"err_{int(time.time() * 1000)}",
            request_id=request.request_id,
            session_id=request.session_id,
            format=fmt,
            audience=audience,
            template_id=template,
            language=language,
            content="",
            content_type="text/plain",
            size_bytes=0,
            checksum="",
            render_latency_ms=0.0,
            storage_results=[],
            cached=False,
            metadata={"error_code": exc.code.value, "error_message": exc.message},
        )

    @staticmethod
    def _infer_format(content_type: str) -> str:
        if "markdown" in content_type:
            return "markdown"
        if "html" in content_type:
            return "html"
        if "json" in content_type:
            return "json"
        return "text"


# -------- singleton helpers (used by FastAPI lifespan) --------

_service_lock = threading.Lock()
_service_instance: Optional[ReportService] = None


def get_report_service() -> ReportService:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ReportService()
    return _service_instance


def reset_report_service() -> None:
    global _service_instance
    with _service_lock:
        _service_instance = None
