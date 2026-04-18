"""End-to-end tests for the MultiModalInputManager (in-process)."""
from __future__ import annotations

import pytest

from core.exceptions import RateLimitExceededError, ValidationError
from entities.enums import InputType
from managers.input_manager import (
    InputRequest,
    MultiModalInputManager,
    ProcessingConfig,
)


@pytest.mark.asyncio
async def test_text_input_round_trip():
    manager = MultiModalInputManager()
    await manager._cache.initialize()

    req = InputRequest(
        input_data="我想买一台笔记本电脑，预算5000元",
        input_type=InputType.TEXT,
        user_id="u-test-1",
    )
    result = await manager.process(req)
    assert result.success
    assert result.input_type == InputType.TEXT
    assert result.data["extracted"]["prices"] == [5000.0]
    assert result.session_id  # auto-created
    assert result.processing_time_ms >= 0


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_flag():
    manager = MultiModalInputManager()
    # enable in-memory cache for this test
    manager._settings.cache_enabled = True
    await manager._cache.initialize()

    req = InputRequest(
        input_data="repeated query",
        input_type=InputType.TEXT,
        user_id="u-test-2",
        config=ProcessingConfig(cache_result=True),
    )
    first = await manager.process(req)
    second = await manager.process(
        InputRequest(
            input_data="repeated query",
            input_type=InputType.TEXT,
            user_id="u-test-2",
            config=ProcessingConfig(cache_result=True),
        )
    )
    assert first.cached is False
    assert second.cached is True
    # Same data regardless of cache hit/miss
    assert first.data["cleaned_text"] == second.data["cleaned_text"]


@pytest.mark.asyncio
async def test_validation_error_bubbles_up():
    manager = MultiModalInputManager()
    await manager._cache.initialize()
    req = InputRequest(
        input_data="",
        input_type=InputType.TEXT,
        user_id="u-test-3",
    )
    with pytest.raises(ValidationError):
        await manager.process(req)


@pytest.mark.asyncio
async def test_rate_limiter_triggers():
    manager = MultiModalInputManager()
    await manager._cache.initialize()
    # Force a tiny limit
    from managers.input_manager import AsyncRateLimiter

    manager._rate_limiter = AsyncRateLimiter(max_requests=2, window_seconds=60)

    async def _do():
        return await manager.process(
            InputRequest(
                input_data="x",
                input_type=InputType.TEXT,
                user_id="rate-user",
            )
        )

    await _do()
    await _do()
    with pytest.raises(RateLimitExceededError):
        await _do()


@pytest.mark.asyncio
async def test_batch_processing_reports_mixed_results():
    manager = MultiModalInputManager()
    await manager._cache.initialize()
    requests = [
        InputRequest(
            input_data="I want to buy X",
            input_type=InputType.TEXT,
            user_id="batch-user",
        ),
        InputRequest(
            input_data="",  # invalid
            input_type=InputType.TEXT,
            user_id="batch-user",
        ),
    ]
    results = await manager.process_batch(requests)
    assert len(results) == 2
    assert results[0].success
    assert not results[1].success
    assert results[1].error is not None
