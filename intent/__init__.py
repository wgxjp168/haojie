"""ILBuyAI Part 3.1 — Intent Recognition Service.

First stage of the AI Decision Hub. Classifies normalized B2B/B2C procurement
queries into a fixed taxonomy of intents (search, compare, inquire-price,
check-stock, get-recommendation, place-order, track-order, complaint, ...).

Design goals:
  * Graceful degradation: rule-based classifier always works; transformer
    backend is an optional upgrade.
  * Low-latency: in-memory + Redis cache keyed on normalized text.
  * Observable: Prometheus metrics + structured logs.
  * Composable: exposed as both an importable Python API and a REST service.
"""
from intent.config import IntentSettings, get_intent_settings
from intent.schemas import (
    IntentCandidate,
    IntentRequest,
    IntentResponse,
    BatchIntentRequest,
    BatchIntentResponse,
)
from intent.service import IntentService, get_intent_service, reset_intent_service

__all__ = [
    "IntentSettings",
    "get_intent_settings",
    "IntentCandidate",
    "IntentRequest",
    "IntentResponse",
    "BatchIntentRequest",
    "BatchIntentResponse",
    "IntentService",
    "get_intent_service",
    "reset_intent_service",
]
