import pytest

from llm.cache import LLMCache
from llm.config import LLMSettings, LLMStrategy
from llm.core.exceptions import ValidationLLMError
from llm.schemas import (
    BatchCompletionRequest,
    CompletionRequest,
)
from llm.service import LLMService


def _service(**over) -> LLMService:
    # redis_url=None so cache is pure memory
    settings = LLMSettings(
        fallback_chain=over.pop("fallback_chain", ["stub-small"]),
        default_strategy=over.pop("default_strategy", LLMStrategy.FALLBACK),
        cache_enabled=over.pop("cache_enabled", True),
        memory_cache_size=8,
        redis_url=None,
        rate_limit_per_minute=0,  # disable for tests
        **over,
    )
    return LLMService(settings=settings)


@pytest.mark.asyncio
async def test_basic_complete():
    svc = _service()
    resp = await svc.complete(CompletionRequest(prompt="Hello"))
    assert resp.provider == "stub"
    assert resp.model_id == "stub-small"
    assert resp.content
    assert resp.cached is False


@pytest.mark.asyncio
async def test_cache_hit_on_repeated_prompt():
    svc = _service()
    r1 = await svc.complete(CompletionRequest(prompt="same question"))
    r2 = await svc.complete(CompletionRequest(prompt="same question"))
    assert r2.cached is True
    assert r1.content == r2.content


@pytest.mark.asyncio
async def test_different_prompts_not_cached():
    svc = _service()
    r1 = await svc.complete(CompletionRequest(prompt="a"))
    r2 = await svc.complete(CompletionRequest(prompt="b"))
    assert r1.cached is False
    assert r2.cached is False


@pytest.mark.asyncio
async def test_cache_disabled_never_caches():
    svc = _service(cache_enabled=False)
    await svc.complete(CompletionRequest(prompt="x"))
    r2 = await svc.complete(CompletionRequest(prompt="x"))
    assert r2.cached is False


@pytest.mark.asyncio
async def test_strategy_override_per_request():
    svc = _service()
    resp = await svc.complete(
        CompletionRequest(prompt="hi", strategy="single")
    )
    assert resp.strategy == "single"


@pytest.mark.asyncio
async def test_unknown_model_rejected_by_service():
    svc = _service()
    with pytest.raises(ValidationLLMError):
        await svc.complete(CompletionRequest(prompt="hi", model_id="no-such-model"))


@pytest.mark.asyncio
async def test_unknown_target_list_rejected():
    svc = _service()
    with pytest.raises(ValidationLLMError):
        await svc.complete(
            CompletionRequest(prompt="hi", target_model_ids=["bogus-a", "bogus-b"])
        )


@pytest.mark.asyncio
async def test_batch_completion():
    svc = _service()
    batch = BatchCompletionRequest(
        items=[CompletionRequest(prompt="a"), CompletionRequest(prompt="b")]
    )
    resp = await svc.complete_batch(batch)
    assert resp.total == 2
    assert all(r.provider == "stub" for r in resp.results)


@pytest.mark.asyncio
async def test_batch_respects_max_batch_size():
    svc = _service(max_batch_size=1)
    with pytest.raises(ValidationLLMError):
        await svc.complete_batch(
            BatchCompletionRequest(items=[
                CompletionRequest(prompt="a"),
                CompletionRequest(prompt="b"),
            ])
        )


@pytest.mark.asyncio
async def test_batch_tolerates_per_item_errors():
    svc = _service()
    batch = BatchCompletionRequest(items=[
        CompletionRequest(prompt="ok"),
        CompletionRequest(prompt="ok", model_id="bogus"),
    ])
    resp = await svc.complete_batch(batch)
    # first succeeds, second returns error_response
    assert resp.total == 2
    ok, err = resp.results
    assert ok.content
    assert err.content == ""
    assert err.metadata.get("error_code") == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_providers_available_contains_stub():
    svc = _service()
    assert "stub" in svc.providers_available


@pytest.mark.asyncio
async def test_confident_flag_below_threshold():
    svc = _service(confident_threshold=0.99, low_confidence_threshold=0.5)
    resp = await svc.complete(CompletionRequest(prompt="hi"))
    # stub-small at temperature 0.7 gives confidence ~0.83 < 0.99
    assert resp.confident is False


@pytest.mark.asyncio
async def test_response_carries_request_and_session_ids():
    svc = _service()
    resp = await svc.complete(CompletionRequest(
        prompt="hi", request_id="r1", session_id="s1"
    ))
    assert resp.request_id == "r1"
    assert resp.session_id == "s1"


@pytest.mark.asyncio
async def test_cached_response_refreshes_request_and_session_ids():
    svc = _service()
    await svc.complete(CompletionRequest(prompt="same", request_id="r1"))
    r2 = await svc.complete(CompletionRequest(prompt="same", request_id="r2"))
    assert r2.cached is True
    assert r2.request_id == "r2"


@pytest.mark.asyncio
async def test_parallel_vote_returns_some_content():
    svc = _service(default_strategy=LLMStrategy.PARALLEL_VOTE,
                   fallback_chain=["stub-small", "stub-fast"])
    resp = await svc.complete(CompletionRequest(prompt="vote please"))
    assert resp.content


@pytest.mark.asyncio
async def test_confidence_strategy_picks_higher():
    svc = _service(default_strategy=LLMStrategy.CONFIDENCE,
                   fallback_chain=["stub-small", "stub-fast"])
    resp = await svc.complete(CompletionRequest(prompt="hello"))
    # stub-small has higher baseline confidence than stub-fast.
    assert resp.model_id == "stub-small"
