"""LLM model catalog.

The catalog is a registry of known ``(provider, model_id)`` pairs with
pricing, context-window, and capability hints. It is *descriptive* — the
router trusts what the catalog says about a model to make decisions
(e.g. price caps, token caps) but does not enforce provider-side
capabilities. The ``stub-small`` model is always present so the service
can run without any real provider configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelCapabilities:
    chat: bool = True
    completion: bool = True
    embedding: bool = False
    vision: bool = False
    tools: bool = False
    streaming: bool = False  # Not wired in v1.


@dataclass(frozen=True)
class ModelInfo:
    """One entry in the model catalog."""

    model_id: str               # canonical id — stable across providers
    provider: str               # provider key (openai|anthropic|google|azure|stub)
    remote_name: str            # the model name to send in the API call
    context_window: int
    max_output_tokens: int
    # USD per 1K tokens (prompt, completion).
    price_per_1k_prompt_usd: float
    price_per_1k_completion_usd: float
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    audiences: Tuple[str, ...] = ("any",)
    description: str = ""

    def estimated_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        p = (prompt_tokens / 1000.0) * self.price_per_1k_prompt_usd
        c = (completion_tokens / 1000.0) * self.price_per_1k_completion_usd
        return round(p + c, 6)


_RAW: List[ModelInfo] = [
    # ---------- Stub (zero-dep; always available) ----------
    ModelInfo(
        model_id="stub-small",
        provider="stub",
        remote_name="stub-small",
        context_window=8192,
        max_output_tokens=1024,
        price_per_1k_prompt_usd=0.0,
        price_per_1k_completion_usd=0.0,
        description="Zero-dependency deterministic stub used for tests & dev.",
    ),
    ModelInfo(
        model_id="stub-fast",
        provider="stub",
        remote_name="stub-fast",
        context_window=8192,
        max_output_tokens=512,
        price_per_1k_prompt_usd=0.0,
        price_per_1k_completion_usd=0.0,
        description="Fast stub variant used to exercise ranking paths.",
    ),
    # ---------- OpenAI ----------
    ModelInfo(
        model_id="gpt-4o",
        provider="openai",
        remote_name="gpt-4o",
        context_window=128_000,
        max_output_tokens=4_096,
        price_per_1k_prompt_usd=0.005,
        price_per_1k_completion_usd=0.015,
        capabilities=ModelCapabilities(chat=True, vision=True, tools=True),
        description="OpenAI GPT-4o flagship multi-modal model.",
    ),
    ModelInfo(
        model_id="gpt-4o-mini",
        provider="openai",
        remote_name="gpt-4o-mini",
        context_window=128_000,
        max_output_tokens=16_384,
        price_per_1k_prompt_usd=0.00015,
        price_per_1k_completion_usd=0.0006,
        capabilities=ModelCapabilities(chat=True, tools=True),
        description="Cost-effective OpenAI model for high volume.",
    ),
    # ---------- Anthropic ----------
    ModelInfo(
        model_id="claude-3-5-sonnet",
        provider="anthropic",
        remote_name="claude-3-5-sonnet-20241022",
        context_window=200_000,
        max_output_tokens=8_192,
        price_per_1k_prompt_usd=0.003,
        price_per_1k_completion_usd=0.015,
        capabilities=ModelCapabilities(chat=True, vision=True, tools=True),
        description="Anthropic Claude 3.5 Sonnet, balanced quality/price.",
    ),
    ModelInfo(
        model_id="claude-3-haiku",
        provider="anthropic",
        remote_name="claude-3-haiku-20240307",
        context_window=200_000,
        max_output_tokens=4_096,
        price_per_1k_prompt_usd=0.00025,
        price_per_1k_completion_usd=0.00125,
        capabilities=ModelCapabilities(chat=True),
        description="Cheapest Anthropic model for high throughput.",
    ),
    # ---------- Google ----------
    ModelInfo(
        model_id="gemini-1.5-pro",
        provider="google",
        remote_name="gemini-1.5-pro",
        context_window=1_000_000,
        max_output_tokens=8_192,
        price_per_1k_prompt_usd=0.00125,
        price_per_1k_completion_usd=0.005,
        capabilities=ModelCapabilities(chat=True, vision=True, tools=True),
        description="Google Gemini 1.5 Pro, very long context.",
    ),
    # ---------- Azure-hosted (same model-id namespace as OpenAI) ----------
    ModelInfo(
        model_id="azure-gpt-4o",
        provider="azure",
        remote_name="gpt-4o",
        context_window=128_000,
        max_output_tokens=4_096,
        price_per_1k_prompt_usd=0.005,
        price_per_1k_completion_usd=0.015,
        capabilities=ModelCapabilities(chat=True, vision=True, tools=True),
        description="Azure-hosted GPT-4o deployment.",
    ),
]

MODEL_CATALOG: Dict[str, ModelInfo] = {m.model_id: m for m in _RAW}
MODEL_IDS: Tuple[str, ...] = tuple(MODEL_CATALOG.keys())
STUB_MODEL_ID = "stub-small"


def get_model(model_id: str) -> ModelInfo:
    try:
        return MODEL_CATALOG[model_id]
    except KeyError as exc:
        raise KeyError(f"unknown model id: {model_id!r}") from exc


def models_for_provider(provider: str) -> List[ModelInfo]:
    return [m for m in MODEL_CATALOG.values() if m.provider == provider]


def models_for_audience(audience: str) -> List[ModelInfo]:
    audience = (audience or "any").lower()
    if audience == "any":
        return list(MODEL_CATALOG.values())
    return [
        m for m in MODEL_CATALOG.values()
        if audience in m.audiences or "any" in m.audiences
    ]


def cheapest_model() -> ModelInfo:
    """Return the cheapest model (by prompt price) — useful as a floor."""
    return min(MODEL_CATALOG.values(), key=lambda m: m.price_per_1k_prompt_usd)
