"""ILBuyAI Report Generation Service — Part 3.4.

Final stage of the AI Decision Hub: consumes intent (Part 3.1), decision
(Part 3.2), and optional LLM narrative (Part 3.3) outputs, rendering a
bilingual audit-ready report in one of several formats
(markdown / html / json / text) for one of three audiences
(executive / technical / customer).

Runs as a standalone FastAPI service on port 8005 with an independent
Prometheus registry. Ships with zero-dependency template rendering and a
pluggable storage layer (memory + local filesystem; S3 is opt-in).
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
