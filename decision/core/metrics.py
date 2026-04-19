"""Prometheus metrics for the decision engine.

Uses a dedicated ``CollectorRegistry`` so the decision service can run in
the same Python process as Parts 1 / 2 / 3.1 without metric-name
collisions.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

DECISION_REGISTRY = CollectorRegistry()

DECISION_REQUESTS_TOTAL = Counter(
    "decision_requests_total",
    "Total decision requests handled.",
    ["strategy", "status"],
    registry=DECISION_REGISTRY,
)

DECISION_DURATION = Histogram(
    "decision_duration_seconds",
    "Latency of decision evaluation in seconds.",
    ["strategy"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=DECISION_REGISTRY,
)

DECISION_CONFIDENCE = Histogram(
    "decision_confidence",
    "Confidence distribution of the chosen action.",
    ["strategy"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=DECISION_REGISTRY,
)

DECISION_RISK = Histogram(
    "decision_risk_score",
    "Risk score of recommended decisions.",
    buckets=(0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0),
    registry=DECISION_REGISTRY,
)

DECISION_ACTION_TOTAL = Counter(
    "decision_action_total",
    "Count of decisions per action id.",
    ["action"],
    registry=DECISION_REGISTRY,
)

DECISION_REVIEW_TOTAL = Counter(
    "decision_review_total",
    "Decisions flagged for human review.",
    ["reason"],
    registry=DECISION_REGISTRY,
)

DECISION_CACHE_EVENTS = Counter(
    "decision_cache_events_total",
    "Cache hits/misses for the decision engine.",
    ["tier", "event"],  # tier: memory|redis; event: hit|miss|error
    registry=DECISION_REGISTRY,
)

DECISION_CIRCUIT_STATE = Gauge(
    "decision_circuit_state",
    "ML strategy circuit-breaker state (0=closed,1=half_open,2=open).",
    registry=DECISION_REGISTRY,
)

DECISION_MODEL_LOAD_EVENTS = Counter(
    "decision_model_load_events_total",
    "Events related to ML strategy model loading.",
    ["event"],  # attempt|success|failure
    registry=DECISION_REGISTRY,
)

DECISION_BATCH_SIZE = Histogram(
    "decision_batch_size",
    "Size of decision batches.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    registry=DECISION_REGISTRY,
)

DECISION_FALLBACK_TOTAL = Counter(
    "decision_fallback_total",
    "Number of times the engine fell back from one strategy to another.",
    ["from_strategy", "to_strategy", "reason"],
    registry=DECISION_REGISTRY,
)
