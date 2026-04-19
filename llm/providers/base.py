"""Provider abstract base class.

Providers are the I/O boundary of the LLM service: they turn a
normalized ``CompletionRequest`` into a ``ProviderResult`` by talking to
a remote API (or, for ``StubProvider``, by running a deterministic
local stub).

Providers must be:
* **async** — every public entrypoint is a coroutine.
* **dependency-safe** — optional dependencies (``httpx``,
  ``openai``-SDK, etc.) are imported lazily inside ``_call_remote`` so
  the service boots without them.
* **side-effect-free in ``__init__``** — no network calls at construction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm.catalog import ModelInfo
from llm.schemas import ChatMessage, TokenUsage


@dataclass
class ProviderResult:
    """Envelope returned by every provider call."""

    content: str
    tokens: TokenUsage
    finish_reason: str = "stop"
    confidence: float = 0.9
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract async provider adapter."""

    #: Canonical provider key (``openai``, ``anthropic``, ``google``,
    #: ``azure``, ``stub``, ``custom``...). Must match ``ModelInfo.provider``.
    name: str = "base"

    def __init__(self, *, credentials: Optional[Dict[str, Any]] = None) -> None:
        self.credentials = credentials or {}

    # ---------- public API ----------

    @property
    def available(self) -> bool:
        """Whether the provider can actually serve requests.

        Real providers should override this and return ``False`` when
        required credentials or SDKs are missing. ``StubProvider``
        always returns ``True``.
        """
        return True

    @abstractmethod
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
        """Execute one chat completion."""

    # ---------- helpers ----------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Very rough token estimate.

        Production providers should trust server-reported usage and only
        use this helper as a fallback when the remote endpoint does not
        report token counts.
        """
        if not text:
            return 0
        # ~4 chars per token for English, ~1.5 for Chinese. Use a blend.
        ascii_chars = sum(1 for c in text if c.isascii())
        cjk_chars = len(text) - ascii_chars
        return max(1, ascii_chars // 4 + int(cjk_chars / 1.5))
