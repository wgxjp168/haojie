"""Prometheus metrics for the service.

Counters, histograms and gauges exposed on ``/metrics``. Import the
pre-built instruments from this module rather than constructing new ones,
so the Prometheus registry stays consistent across reloads.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INPUT_REQUESTS_TOTAL = Counter(
    "input_requests_total",
    "Total number of input requests processed",
    ["input_type", "status"],
)

INPUT_PROCESSING_SECONDS = Histogram(
    "input_processing_seconds",
    "Time spent processing an input",
    ["input_type"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

INPUT_SIZE_BYTES = Histogram(
    "input_size_bytes",
    "Size of the input in bytes",
    ["input_type"],
    buckets=(1024, 10_240, 102_400, 1_048_576, 10_485_760, 52_428_800),
)

ACTIVE_SESSIONS = Gauge(
    "active_sessions",
    "Number of currently active sessions",
)

EXTERNAL_API_CALLS_TOTAL = Counter(
    "external_api_calls_total",
    "Calls to external services",
    ["service", "status"],
)

EXTERNAL_API_LATENCY_SECONDS = Histogram(
    "external_api_latency_seconds",
    "External API call latency",
    ["service"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
)
