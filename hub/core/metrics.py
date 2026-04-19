"""Prometheus metrics for the hub service.

Uses a dedicated registry so the hub can live in the same process as
Parts 1/2/3.1/3.2/3.3/3.4 without metric-name collisions.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

HUB_REGISTRY = CollectorRegistry()

HUB_PIPELINE_TOTAL = Counter(
    "hub_pipeline_total",
    "Total pipeline requests handled.",
    ["status"],  # success | partial | failed
    registry=HUB_REGISTRY,
)

HUB_PIPELINE_DURATION = Histogram(
    "hub_pipeline_duration_seconds",
    "End-to-end pipeline latency.",
    ["status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=HUB_REGISTRY,
)

HUB_STAGE_TOTAL = Counter(
    "hub_stage_total",
    "Per-stage outcomes.",
    ["stage", "status"],  # stage: intent|decision|llm|report ; status: success|skipped|failed
    registry=HUB_REGISTRY,
)

HUB_STAGE_DURATION = Histogram(
    "hub_stage_duration_seconds",
    "Per-stage latency.",
    ["stage", "mode"],  # mode: embedded|http
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=HUB_REGISTRY,
)

HUB_STAGE_RETRIES = Counter(
    "hub_stage_retries_total",
    "Per-stage retry events (HTTP mode only).",
    ["stage", "outcome"],  # outcome: success|exhausted
    registry=HUB_REGISTRY,
)

HUB_CIRCUIT_STATE = Gauge(
    "hub_circuit_state",
    "Per-stage circuit-breaker state (0=closed,1=half_open,2=open).",
    ["stage"],
    registry=HUB_REGISTRY,
)

HUB_CLIENT_MODE = Gauge(
    "hub_client_mode",
    "Active client mode per stage (1=embedded, 2=http).",
    ["stage"],
    registry=HUB_REGISTRY,
)
