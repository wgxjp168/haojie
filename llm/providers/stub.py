"""Deterministic stub provider.

Zero third-party dependencies, always available. Used as:
* the default provider in unit / integration tests,
* the zero-config fallback when no real provider credentials are set,
* a reference implementation for the ``BaseProvider`` contract.

The stub produces a human-readable echo of the last user message with a
short decorative preamble so downstream voting / confidence strategies
have something to discriminate on.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from llm.catalog import ModelInfo
from llm.providers.base import BaseProvider, ProviderResult
from llm.schemas import ChatMessage, TokenUsage


class StubProvider(BaseProvider):
    """Always-available deterministic provider."""

    name = "stub"

    @property
    def available(self) -> bool:  # pragma: no cover - trivial
        return True

    async def complete(
        self,
        *,
        model: ModelInfo,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        timeout_seconds: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ProviderResult:
        start = time.perf_counter()

        user_text = ""
        system_text = ""
        for m in messages:
            if m.role == "system" and not system_text:
                system_text = m.content
            if m.role == "user":
                user_text = m.content  # keep last

        prompt_tokens = sum(self.estimate_tokens(m.content) for m in messages)

        # Deterministic content: prefixed echo + short hash so two
        # identical prompts across providers produce identical content,
        # but different model ids produce distinguishable output.
        digest = hashlib.sha256(
            f"{model.model_id}|{user_text}".encode("utf-8")
        ).hexdigest()[:8]
        reply = (
            f"[{model.model_id}] {user_text.strip()[:500]} "
            f"(stub:{digest})"
        ).strip()

        # Confidence decreases with temperature (deterministic but
        # dependent on sampling). ``stub-fast`` is marked slightly less
        # confident so ranking strategies see variance.
        base_conf = 0.9 if model.model_id == "stub-small" else 0.75
        confidence = max(0.0, min(1.0, base_conf - temperature * 0.1))

        # Respect max_tokens by truncating.
        cap = max(1, min(max_tokens, model.max_output_tokens))
        # approximate char cap; 4 chars per token
        char_cap = cap * 4
        if len(reply) > char_cap:
            reply = reply[:char_cap]

        completion_tokens = self.estimate_tokens(reply)
        total = prompt_tokens + completion_tokens

        latency_ms = (time.perf_counter() - start) * 1000.0

        return ProviderResult(
            content=reply,
            tokens=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total,
            ),
            finish_reason="stop",
            confidence=round(confidence, 4),
            latency_ms=round(latency_ms, 3),
            raw={"stub": True, "digest": digest},
        )
