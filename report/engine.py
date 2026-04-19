"""Report engine — coordinates renderers, storage, and metrics.

The engine is the I/O-agnostic orchestrator. ``ReportService`` owns the
HTTP/cache boundary; the engine only knows how to turn a validated
``ReportRequest`` into a rendered body and fan it out to one or more
storage backends.
"""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Tuple

from report.catalog import (
    TemplateMetadata,
    copy_for,
    default_template_for,
    resolve_template,
    sections_for_template,
)
from report.circuit_breaker import StorageCircuitRegistry
from report.config import ReportAudience, ReportFormat, ReportSettings
from report.core.exceptions import (
    CircuitOpenError,
    RenderFailedError,
    ReportError,
    StorageFailedError,
    StorageUnavailableError,
    UnknownAudienceError,
    UnknownFormatError,
    UnknownTemplateError,
)
from report.core.logging import get_logger
from report.core.metrics import (
    REPORT_RENDER_DURATION,
    REPORT_RENDER_SIZE,
    REPORT_REQUESTS_TOTAL,
    REPORT_STORAGE_DURATION,
    REPORT_STORAGE_EVENTS,
    REPORT_TEMPLATE_ERRORS,
)
from report.renderers import (
    BaseRenderer,
    HtmlRenderer,
    JsonRenderer,
    MarkdownRenderer,
    TextRenderer,
)
from report.renderers.base import RenderContext, RenderOutput
from report.schemas import ReportRequest, ReportResponse, StorageResult
from report.storage.base import BaseStorageBackend


log = get_logger(__name__)


def _default_renderers() -> Dict[str, BaseRenderer]:
    return {
        ReportFormat.MARKDOWN.value: MarkdownRenderer(),
        ReportFormat.HTML.value: HtmlRenderer(),
        ReportFormat.JSON.value: JsonRenderer(),
        ReportFormat.TEXT.value: TextRenderer(),
    }


class ReportEngine:
    """Stateful orchestrator — safe to share across requests."""

    def __init__(
        self,
        *,
        settings: ReportSettings,
        renderers: Optional[Dict[str, BaseRenderer]] = None,
        storages: Optional[List[BaseStorageBackend]] = None,
        breakers: Optional[StorageCircuitRegistry] = None,
    ) -> None:
        self.settings = settings
        self.renderers = renderers or _default_renderers()
        self.storages = storages or []
        self.breakers = breakers or StorageCircuitRegistry(
            failure_threshold=settings.cb_failure_threshold,
            recovery_seconds=settings.cb_recovery_seconds,
        )

    # ---- public api ----

    def render(self, request: ReportRequest) -> ReportResponse:
        start = time.perf_counter()

        fmt, audience, template, language = self._resolve_targets(request)
        renderer = self.renderers.get(fmt.value)
        if renderer is None:
            raise UnknownFormatError(
                f"no renderer registered for format {fmt.value!r}",
                details={"format": fmt.value},
            )

        report_id = request.request_id or f"rpt_{uuid.uuid4().hex[:12]}"
        context = RenderContext(
            report_id=report_id,
            request=request,
            template=template,
            audience=audience.value,
            language=language,
            copy=copy_for(language),
            sections=sections_for_template(template.template_id),
        )

        try:
            render_start = time.perf_counter()
            output: RenderOutput = renderer.render(context)
            render_latency_ms = (time.perf_counter() - render_start) * 1000.0
        except ReportError:
            REPORT_TEMPLATE_ERRORS.labels(
                template=template.template_id, reason="format"
            ).inc()
            raise
        except Exception as exc:  # noqa: BLE001
            REPORT_TEMPLATE_ERRORS.labels(
                template=template.template_id, reason="unknown"
            ).inc()
            log.error(
                "report.engine.render_failed",
                template=template.template_id,
                format=fmt.value,
                error=str(exc),
            )
            raise RenderFailedError(
                f"renderer {fmt.value!r} failed: {exc}",
                details={"template": template.template_id},
            ) from exc

        # Record metrics.
        REPORT_RENDER_DURATION.labels(
            format=fmt.value, template=template.template_id
        ).observe(render_latency_ms / 1000.0)
        REPORT_RENDER_SIZE.labels(format=fmt.value).observe(output.size_bytes)

        storage_results: List[StorageResult] = []
        if request.store and self.storages:
            storage_results = self._persist(report_id, output)

        REPORT_REQUESTS_TOTAL.labels(
            format=fmt.value, audience=audience.value, status="ok"
        ).inc()

        total_latency_ms = (time.perf_counter() - start) * 1000.0
        return ReportResponse(
            report_id=report_id,
            request_id=request.request_id,
            session_id=request.session_id,
            format=fmt.value,
            audience=audience.value,
            template_id=template.template_id,
            language=language,
            content=output.content,
            content_type=output.content_type,
            size_bytes=output.size_bytes,
            checksum=output.checksum,
            render_latency_ms=round(render_latency_ms, 3),
            storage_results=storage_results,
            cached=False,
            metadata={
                "total_latency_ms": round(total_latency_ms, 3),
                "sections": list(context.sections),
            },
        )

    # ---- target resolution ----

    def _resolve_targets(
        self, request: ReportRequest
    ) -> Tuple[ReportFormat, ReportAudience, TemplateMetadata, str]:
        fmt_raw = request.format or self.settings.default_format.value
        try:
            fmt = ReportFormat(fmt_raw)
        except ValueError as exc:
            raise UnknownFormatError(
                f"unknown format {fmt_raw!r}",
                details={"supported": [f.value for f in ReportFormat]},
            ) from exc

        audience_raw = request.audience or self.settings.default_audience.value
        try:
            audience = ReportAudience(audience_raw)
        except ValueError as exc:
            raise UnknownAudienceError(
                f"unknown audience {audience_raw!r}",
                details={"supported": [a.value for a in ReportAudience]},
            ) from exc

        template_id = request.template_id or default_template_for(audience)
        try:
            template = resolve_template(template_id, audience)
        except KeyError as exc:
            raise UnknownTemplateError(
                f"unknown template {template_id!r}",
                details={"template_id": template_id},
            ) from exc

        language = request.language or self.settings.default_language
        if language not in self.settings.supported_languages:
            language = self.settings.default_language

        return fmt, audience, template, language

    # ---- storage fan-out ----

    def _persist(
        self, report_id: str, output: RenderOutput
    ) -> List[StorageResult]:
        results: List[StorageResult] = []
        for backend in self.storages:
            breaker = self.breakers.get(backend.name)
            start = time.perf_counter()
            try:
                breaker.ensure_allowed()
                if not backend.available:
                    raise StorageUnavailableError(
                        f"backend {backend.name!r} reports unavailable"
                    )
                location = backend.put(
                    report_id,
                    output.content.encode("utf-8")
                    if isinstance(output.content, str)
                    else output.content,
                    content_type=output.content_type,
                    checksum=output.checksum,
                )
                breaker.record_success()
                latency_ms = (time.perf_counter() - start) * 1000.0
                REPORT_STORAGE_EVENTS.labels(
                    backend=backend.name, event="success"
                ).inc()
                REPORT_STORAGE_DURATION.labels(backend=backend.name).observe(
                    latency_ms / 1000.0
                )
                results.append(
                    StorageResult(
                        backend=backend.name,
                        success=True,
                        location=location,
                        bytes_written=output.size_bytes,
                        latency_ms=round(latency_ms, 3),
                    )
                )
            except CircuitOpenError as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                REPORT_STORAGE_EVENTS.labels(
                    backend=backend.name, event="skipped"
                ).inc()
                results.append(
                    StorageResult(
                        backend=backend.name,
                        success=False,
                        latency_ms=round(latency_ms, 3),
                        error=exc.message,
                    )
                )
            except (StorageFailedError, StorageUnavailableError) as exc:
                breaker.record_failure()
                latency_ms = (time.perf_counter() - start) * 1000.0
                REPORT_STORAGE_EVENTS.labels(
                    backend=backend.name, event="failure"
                ).inc()
                results.append(
                    StorageResult(
                        backend=backend.name,
                        success=False,
                        latency_ms=round(latency_ms, 3),
                        error=exc.message,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                breaker.record_failure()
                latency_ms = (time.perf_counter() - start) * 1000.0
                REPORT_STORAGE_EVENTS.labels(
                    backend=backend.name, event="failure"
                ).inc()
                log.error(
                    "report.engine.storage_unhandled",
                    backend=backend.name,
                    error=str(exc),
                )
                results.append(
                    StorageResult(
                        backend=backend.name,
                        success=False,
                        latency_ms=round(latency_ms, 3),
                        error=str(exc),
                    )
                )
        return results

    def circuit_states(self) -> Dict[str, str]:
        return self.breakers.all_states()

    def available_backends(self) -> List[str]:
        return [b.name for b in self.storages if b.available]
