"""Prometheus metrics for the LLM service.

Uses a dedicated ``CollectorRegistry`` so the LLM service can run in the
same Python process as Parts 1 / 2 / 3.1 / 3.2 without metric-name
collisions.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LLM_REGISTRY = CollectorRegistry()

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM completion requests handled.",
    ["provider", "model", "status"],
    registry=LLM_REGISTRY,
)

LLM_DURATION = Histogram(
    "llm_request_duration_seconds",
    "End-to-end latency of LLM completion requests.",
    ["provider", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=LLM_REGISTRY,
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed across completion requests.",
    ["provider", "model", "kind"],  # kind: prompt|completion|total
    registry=LLM_REGISTRY,
)

LLM_COST_TOTAL = Counter(
    "llm_cost_usd_total",
    "Estimated USD cost accrued across completion requests.",
    ["provider", "model"],
    registry=LLM_REGISTRY,
)

LLM_CACHE_EVENTS = Counter(
    "llm_cache_events_total",
    "Cache hits/misses for the LLM service.",
    ["tier", "event"],  # tier: memory|redis; event: hit|miss|error
    registry=LLM_REGISTRY,
)

LLM_CIRCUIT_STATE = Gauge(
    "llm_circuit_state",
    "Per-provider circuit-breaker state (0=closed,1=half_open,2=open).",
    ["provider"],
    registry=LLM_REGISTRY,
)

LLM_RATE_LIMIT_EVENTS = Counter(
    "llm_rate_limit_events_total",
    "Per-provider rate-limit outcomes.",
    ["provider", "event"],  # event: allowed|throttled
    registry=LLM_REGISTRY,
)

LLM_PROVIDER_FAILURES = Counter(
    "llm_provider_failures_total",
    "Provider-level failures broken down by reason.",
    ["provider", "reason"],  # reason: timeout|connection|http_error|parse|unknown
    registry=LLM_REGISTRY,
)

LLM_FALLBACK_TOTAL = Counter(
    "llm_fallback_total",
    "Number of times the router fell back from one provider to another.",
    ["from_provider", "to_provider", "reason"],
    registry=LLM_REGISTRY,
)

LLM_BATCH_SIZE = Histogram(
    "llm_batch_size",
    "Size of LLM batches.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=LLM_REGISTRY,
)
