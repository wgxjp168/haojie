"""ILBuyAI Decision Engine — Part 3.2.

Consumes a recognised intent (Part 3.1) plus contextual features and emits a
ranked, audited procurement decision (e.g. ``approve_purchase``,
``request_quote``, ``defer_decision``). The service exposes a REST API on
port 8003 and is designed to live in the same process as Parts 1/2/3.1
without metric-name collisions.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
