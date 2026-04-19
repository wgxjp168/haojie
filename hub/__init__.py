"""ILBuyAI AI Decision Hub — Part 3 orchestrator.

Thin edge service that fans a single user query across the four Part 3
stages:

    intent (3.1) → decision (3.2) → [llm (3.3)] → report (3.4)

Each downstream call goes through a *dual-mode client* (``embedded`` =
in-process Service object; ``http`` = remote FastAPI instance via
httpx), so the same code can back a monolith and a microservice
topology. Ships with an independent Prometheus registry and runs on
port 8006 alongside Parts 1/2/3.1/3.2/3.3/3.4.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
