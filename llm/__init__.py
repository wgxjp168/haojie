"""ILBuyAI LLM Service — Part 3.3.

Unified multi-provider LLM access layer for the AI Decision Hub. Wraps
OpenAI / Anthropic / Google / Azure and any custom HTTP-compatible
providers behind one async REST surface, with provider-level rate
limiting, per-provider circuit breakers, two-tier caching, and routing
strategies (single / fallback / parallel-vote / confidence-weighted).

Providers are *optional*: without ``openai`` / ``anthropic`` / ``httpx``
installed the service still runs on a zero-dep stub so deployments can
light up providers incrementally.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
