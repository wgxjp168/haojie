"""Prometheus metrics for the gateway.

We use a dedicated registry so gateway metrics can be scraped independently
of any Part 1 metrics co-located in the same Python process (e.g. during
tests).
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

GATEWAY_REGISTRY = CollectorRegistry()

GW_REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total gateway requests",
    ["method", "path", "status", "user_type"],
    registry=GATEWAY_REGISTRY,
)

GW_REQUEST_DURATION = Histogram(
    "gateway_request_duration_seconds",
    "Gateway request duration",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=GATEWAY_REGISTRY,
)

GW_UPSTREAM_REQUESTS_TOTAL = Counter(
    "gateway_upstream_requests_total",
    "Total upstream forwarded requests",
    ["status", "method"],
    registry=GATEWAY_REGISTRY,
)

GW_UPSTREAM_DURATION = Histogram(
    "gateway_upstream_duration_seconds",
    "Upstream request duration",
    ["method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=GATEWAY_REGISTRY,
)

GW_RATE_LIMIT_HITS = Counter(
    "gateway_rate_limit_hits_total",
    "Rate-limit rejections",
    ["user_type"],
    registry=GATEWAY_REGISTRY,
)

GW_CIRCUIT_STATE = Gauge(
    "gateway_circuit_state",
    "Circuit state (0=closed, 1=open, 2=half_open)",
    ["service"],
    registry=GATEWAY_REGISTRY,
)

GW_CIRCUIT_TRIPS_TOTAL = Counter(
    "gateway_circuit_trips_total",
    "Total times a circuit opened",
    ["service"],
    registry=GATEWAY_REGISTRY,
)

GW_AUTH_EVENTS_TOTAL = Counter(
    "gateway_auth_events_total",
    "Auth events (login/refresh/verify/revoke)",
    ["event", "result"],
    registry=GATEWAY_REGISTRY,
)

GW_WS_CONNECTIONS = Gauge(
    "gateway_ws_connections",
    "Current live WebSocket connections",
    registry=GATEWAY_REGISTRY,
)

GW_WS_MESSAGES_TOTAL = Counter(
    "gateway_ws_messages_total",
    "WebSocket messages processed",
    ["direction", "type"],
    registry=GATEWAY_REGISTRY,
)
