"""Prometheus metrics for the report service.

Uses a dedicated ``CollectorRegistry`` so the report service can run in
the same Python process as Parts 1 / 2 / 3.1 / 3.2 / 3.3 without
metric-name collisions.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REPORT_REGISTRY = CollectorRegistry()

REPORT_REQUESTS_TOTAL = Counter(
    "report_requests_total",
    "Total report generation requests handled.",
    ["format", "audience", "status"],
    registry=REPORT_REGISTRY,
)

REPORT_RENDER_DURATION = Histogram(
    "report_render_duration_seconds",
    "Latency of report rendering in seconds.",
    ["format", "template"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REPORT_REGISTRY,
)

REPORT_RENDER_SIZE = Histogram(
    "report_render_size_bytes",
    "Rendered report size in bytes.",
    ["format"],
    buckets=(256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576),
    registry=REPORT_REGISTRY,
)

REPORT_STORAGE_EVENTS = Counter(
    "report_storage_events_total",
    "Storage backend events (per backend).",
    ["backend", "event"],  # event: success|failure|skipped
    registry=REPORT_REGISTRY,
)

REPORT_STORAGE_DURATION = Histogram(
    "report_storage_duration_seconds",
    "Storage backend write latency.",
    ["backend"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REPORT_REGISTRY,
)

REPORT_CACHE_EVENTS = Counter(
    "report_cache_events_total",
    "Cache hits/misses for the report service.",
    ["tier", "event"],  # tier: memory|redis; event: hit|miss|error
    registry=REPORT_REGISTRY,
)

REPORT_CIRCUIT_STATE = Gauge(
    "report_circuit_state",
    "Per-storage-backend circuit state (0=closed,1=half_open,2=open).",
    ["backend"],
    registry=REPORT_REGISTRY,
)

REPORT_BATCH_SIZE = Histogram(
    "report_batch_size",
    "Batch size of report generation requests.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=REPORT_REGISTRY,
)

REPORT_TEMPLATE_ERRORS = Counter(
    "report_template_errors_total",
    "Template rendering errors grouped by reason.",
    ["template", "reason"],  # reason: missing_field|format|unknown
    registry=REPORT_REGISTRY,
)
