"""Prometheus metrics for the intent service.

Uses a dedicated ``CollectorRegistry`` so the intent service can run in the
same Python process as Part 1 / Part 2 without metric-name collisions.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

INTENT_REGISTRY = CollectorRegistry()

INTENT_REQUESTS_TOTAL = Counter(
    "intent_requests_total",
    "Total intent predictions handled.",
    ["backend", "status"],
    registry=INTENT_REGISTRY,
)

INTENT_PREDICTION_DURATION = Histogram(
    "intent_prediction_duration_seconds",
    "Latency of intent predictions in seconds.",
    ["backend"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=INTENT_REGISTRY,
)

INTENT_CONFIDENCE = Histogram(
    "intent_prediction_confidence",
    "Top-1 confidence distribution.",
    ["backend"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=INTENT_REGISTRY,
)

INTENT_LABEL_TOTAL = Counter(
    "intent_label_total",
    "Count of predictions per intent label.",
    ["intent"],
    registry=INTENT_REGISTRY,
)

INTENT_CACHE_EVENTS = Counter(
    "intent_cache_events_total",
    "Cache hits/misses for the intent service.",
    ["tier", "event"],  # tier: memory|redis; event: hit|miss|error
    registry=INTENT_REGISTRY,
)

INTENT_CIRCUIT_STATE = Gauge(
    "intent_circuit_state",
    "Transformer circuit-breaker state (0=closed,1=half_open,2=open).",
    registry=INTENT_REGISTRY,
)

INTENT_MODEL_LOAD_EVENTS = Counter(
    "intent_model_load_events_total",
    "Events related to transformer model loading.",
    ["event"],  # attempt|success|failure
    registry=INTENT_REGISTRY,
)

INTENT_BATCH_SIZE = Histogram(
    "intent_batch_size",
    "Size of intent batches.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    registry=INTENT_REGISTRY,
)
