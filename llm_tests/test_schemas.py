import pytest
from pydantic import ValidationError

from llm.schemas import (
    BatchCompletionRequest,
    ChatMessage,
    CompletionAttempt,
    CompletionRequest,
    TokenUsage,
)


def test_message_requires_non_empty():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")


def test_request_prompt_or_messages_required():
    with pytest.raises(ValidationError):
        CompletionRequest()


def test_request_with_prompt():
    req = CompletionRequest(prompt="hi there")
    assert req.prompt == "hi there"
    msgs = req.normalized_messages()
    assert len(msgs) == 1
    assert msgs[0].role == "user"


def test_request_with_messages():
    req = CompletionRequest(
        messages=[
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="hi"),
        ]
    )
    msgs = req.normalized_messages()
    assert len(msgs) == 2


def test_strategy_validated():
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", strategy="bogus")


def test_strategy_normalised():
    req = CompletionRequest(prompt="x", strategy="FALLBACK")
    assert req.strategy == "fallback"


def test_temperature_bounds():
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", temperature=3.0)


def test_stop_size_capped():
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", stop=[f"s{i}" for i in range(20)])


def test_batch_must_be_non_empty():
    with pytest.raises(ValidationError):
        BatchCompletionRequest(items=[])


def test_batch_size_capped():
    with pytest.raises(ValidationError):
        BatchCompletionRequest(
            items=[CompletionRequest(prompt=str(i)) for i in range(257)]
        )


def test_token_usage_autocomputes_total():
    u = TokenUsage(prompt_tokens=10, completion_tokens=5)
    assert u.total_tokens == 15


def test_token_usage_respects_explicit_total():
    u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=20)
    assert u.total_tokens == 20


def test_attempt_fields():
    a = CompletionAttempt(
        provider="stub",
        model_id="stub-small",
        status="success",
        latency_ms=1.0,
    )
    assert a.cost_usd == 0.0
    assert a.tokens is None
