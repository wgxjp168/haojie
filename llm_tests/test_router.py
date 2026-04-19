import asyncio

import pytest

from llm.catalog import get_model
from llm.circuit_breaker import CircuitBreakerRegistry
from llm.config import LLMSettings, LLMStrategy
from llm.core.exceptions import (
    NoProviderAvailableError,
    ProviderFailedError,
    UnknownModelError,
)
from llm.providers.base import BaseProvider, ProviderResult
from llm.providers.stub import StubProvider
from llm.rate_limiter import RateLimiterRegistry
from llm.router import LLMRouter
from llm.schemas import ChatMessage, CompletionRequest, TokenUsage


class _BoomProvider(BaseProvider):
    name = "openai"

    def __init__(self):
        super().__init__(credentials={"api_key": "x"})

    @property
    def available(self):
        return True

    async def complete(self, **kwargs):
        raise ProviderFailedError("kaboom")


class _FixedProvider(BaseProvider):
    """Fixed-content provider used to exercise vote/confidence logic."""

    name = "anthropic"

    def __init__(self, content: str, confidence: float):
        super().__init__(credentials={"api_key": "x"})
        self._content = content
        self._confidence = confidence

    @property
    def available(self):
        return True

    async def complete(self, **kwargs):
        return ProviderResult(
            content=self._content,
            tokens=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="stop",
            confidence=self._confidence,
            latency_ms=1.0,
        )


def _make_router(providers, **settings_overrides) -> LLMRouter:
    settings_overrides.setdefault("fallback_chain", ["stub-small"])
    settings_overrides.setdefault("default_model_id", "stub-small")
    settings_overrides.setdefault("rate_limit_per_minute", 0)
    s = LLMSettings(**settings_overrides)
    return LLMRouter(
        settings=s,
        providers=providers,
        breakers=CircuitBreakerRegistry(
            failure_threshold=s.cb_failure_threshold,
            recovery_seconds=s.cb_recovery_seconds,
        ),
        rate_limiters=RateLimiterRegistry(rate_per_minute=0, burst=0),
    )


@pytest.mark.asyncio
async def test_single_strategy_success():
    r = _make_router({"stub": StubProvider()})
    req = CompletionRequest(prompt="hi", strategy="single")
    resp = await r.run(req)
    assert resp.provider == "stub"
    assert resp.model_id == "stub-small"
    assert len(resp.attempts) == 1


@pytest.mark.asyncio
async def test_unknown_model_raises():
    r = _make_router({"stub": StubProvider()})
    req = CompletionRequest(prompt="hi", model_id="does-not-exist")
    # service validates this, but router-level catch also triggers when
    # a target list is explicit.
    with pytest.raises(UnknownModelError):
        await r.run(CompletionRequest(
            prompt="hi", target_model_ids=["bogus-id"]
        ))


@pytest.mark.asyncio
async def test_fallback_skips_unavailable_providers():
    # openai provider is missing entirely, stub should pick up.
    r = _make_router(
        {"stub": StubProvider()},
        fallback_chain=["gpt-4o", "stub-small"],
    )
    req = CompletionRequest(prompt="hi", strategy="fallback")
    resp = await r.run(req)
    assert resp.model_id == "stub-small"
    # first attempt was skipped with provider_unavailable
    assert resp.attempts[0].status == "skipped"
    assert resp.attempts[0].error_code == "provider_unavailable"
    assert resp.attempts[-1].status == "success"


@pytest.mark.asyncio
async def test_fallback_on_provider_failure():
    r = _make_router(
        {"stub": StubProvider(), "openai": _BoomProvider()},
        fallback_chain=["gpt-4o", "stub-small"],
    )
    req = CompletionRequest(prompt="hi", strategy="fallback")
    resp = await r.run(req)
    assert resp.model_id == "stub-small"
    failure = [a for a in resp.attempts if a.status == "failure"]
    assert failure and failure[0].provider == "openai"


@pytest.mark.asyncio
async def test_fallback_raises_when_all_fail():
    providers = {"openai": _BoomProvider()}
    r = _make_router(
        providers,
        fallback_chain=["gpt-4o"],  # only boom
    )
    req = CompletionRequest(prompt="hi", strategy="fallback")
    with pytest.raises(ProviderFailedError):
        await r.run(req)


@pytest.mark.asyncio
async def test_parallel_any_returns_first_success():
    r = _make_router(
        {"stub": StubProvider()},
        fallback_chain=["stub-small", "stub-fast"],
    )
    req = CompletionRequest(prompt="hi", strategy="parallel_any")
    resp = await r.run(req)
    assert resp.provider == "stub"
    # Should have at least one success attempt.
    assert any(a.status == "success" for a in resp.attempts)


@pytest.mark.asyncio
async def test_parallel_vote_picks_majority():
    # Provide two providers returning the same content + a stub that
    # returns a distinct string. Majority wins.
    r = _make_router(
        {
            "stub": StubProvider(),
            "anthropic": _FixedProvider("SAME", 0.7),
        },
        fallback_chain=["stub-small", "claude-3-5-sonnet", "claude-3-haiku"],
    )
    req = CompletionRequest(prompt="hi", strategy="parallel_vote")
    resp = await r.run(req)
    # SAME wins 2-1 over the stub reply
    assert resp.content == "SAME"


@pytest.mark.asyncio
async def test_confidence_picks_highest():
    r = _make_router(
        {
            "stub": StubProvider(),
            "anthropic": _FixedProvider("high-confidence-reply", 0.98),
        },
        fallback_chain=["stub-small", "claude-3-5-sonnet"],
    )
    req = CompletionRequest(prompt="hi", strategy="confidence")
    resp = await r.run(req)
    assert resp.content == "high-confidence-reply"
    assert resp.confidence >= 0.9


@pytest.mark.asyncio
async def test_no_provider_available_raises():
    # No providers configured at all => every attempt skipped.
    r = _make_router({}, fallback_chain=["stub-small"])
    req = CompletionRequest(prompt="hi", strategy="fallback")
    with pytest.raises(NoProviderAvailableError):
        await r.run(req)


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_failures():
    r = _make_router(
        {"openai": _BoomProvider()},
        fallback_chain=["gpt-4o"],
        cb_failure_threshold=2,
        cb_recovery_seconds=10,
    )
    req = CompletionRequest(prompt="hi", strategy="fallback")
    with pytest.raises(ProviderFailedError):
        await r.run(req)
    with pytest.raises(ProviderFailedError):
        await r.run(req)
    # Third call should not even attempt openai -> circuit open.
    with pytest.raises(NoProviderAvailableError):
        await r.run(req)
    assert r.breakers.get("openai").state_label == "open"


@pytest.mark.asyncio
async def test_rate_limiting_blocks_excess_calls():
    s = LLMSettings(
        fallback_chain=["stub-small"],
        rate_limit_per_minute=60,  # tiny refill
    )
    r = LLMRouter(
        settings=s,
        providers={"stub": StubProvider()},
        breakers=CircuitBreakerRegistry(
            failure_threshold=s.cb_failure_threshold,
            recovery_seconds=s.cb_recovery_seconds,
        ),
        rate_limiters=RateLimiterRegistry(rate_per_minute=60, burst=1),
    )
    req = CompletionRequest(prompt="hi", strategy="fallback")
    # First call consumes the single available token.
    resp = await r.run(req)
    assert resp.provider == "stub"
    # Second call is throttled => attempts include a "skipped" entry and
    # with no fallback target the router raises NoProviderAvailable.
    with pytest.raises(NoProviderAvailableError):
        await r.run(req)


@pytest.mark.asyncio
async def test_prompt_too_large_refused():
    r = _make_router(
        {"stub": StubProvider()}, max_prompt_chars=10
    )
    req = CompletionRequest(prompt="x" * 50)
    from llm.core.exceptions import BudgetExceededError
    with pytest.raises(BudgetExceededError):
        await r.run(req)


@pytest.mark.asyncio
async def test_request_explicit_targets_override_defaults():
    r = _make_router({"stub": StubProvider()})
    req = CompletionRequest(
        prompt="hi",
        strategy="fallback",
        target_model_ids=["stub-fast", "stub-small"],
    )
    resp = await r.run(req)
    # The first target (stub-fast) should appear first in attempts.
    assert resp.attempts[0].model_id == "stub-fast"
