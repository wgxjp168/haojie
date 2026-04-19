import pytest

from llm.catalog import get_model
from llm.core.exceptions import ProviderFailedError
from llm.providers.anthropic import AnthropicProvider
from llm.providers.google import GoogleProvider
from llm.providers.openai import AzureOpenAIProvider, OpenAIProvider
from llm.providers.stub import StubProvider
from llm.schemas import ChatMessage


@pytest.mark.asyncio
async def test_stub_provider_deterministic():
    p = StubProvider()
    assert p.available is True
    m = get_model("stub-small")
    result = await p.complete(
        model=m,
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.2,
        top_p=1.0,
        max_tokens=256,
        stop=None,
        timeout_seconds=5.0,
    )
    assert result.content
    assert result.tokens.prompt_tokens > 0
    assert result.finish_reason == "stop"
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_stub_provider_different_models_produce_distinct_output():
    p = StubProvider()
    msg = [ChatMessage(role="user", content="same prompt")]
    r_small = await p.complete(
        model=get_model("stub-small"),
        messages=msg,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        stop=None,
        timeout_seconds=5.0,
    )
    r_fast = await p.complete(
        model=get_model("stub-fast"),
        messages=msg,
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        stop=None,
        timeout_seconds=5.0,
    )
    assert r_small.content != r_fast.content
    # stub-small declares higher baseline confidence than stub-fast.
    assert r_small.confidence > r_fast.confidence


@pytest.mark.asyncio
async def test_stub_provider_respects_temperature_confidence():
    p = StubProvider()
    msg = [ChatMessage(role="user", content="x")]
    hot = await p.complete(
        model=get_model("stub-small"),
        messages=msg,
        temperature=1.5,
        top_p=1.0,
        max_tokens=64,
        stop=None,
        timeout_seconds=5.0,
    )
    cold = await p.complete(
        model=get_model("stub-small"),
        messages=msg,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        stop=None,
        timeout_seconds=5.0,
    )
    assert cold.confidence > hot.confidence


def test_openai_unavailable_without_api_key():
    p = OpenAIProvider(credentials={"api_key": None})
    assert p.available is False


def test_anthropic_unavailable_without_api_key():
    p = AnthropicProvider(credentials={"api_key": None})
    assert p.available is False


def test_google_unavailable_without_api_key():
    p = GoogleProvider(credentials={"api_key": None})
    assert p.available is False


def test_azure_unavailable_without_endpoint():
    p = AzureOpenAIProvider(credentials={"api_key": "x"})
    assert p.available is False


@pytest.mark.asyncio
async def test_openai_rejects_call_when_unavailable():
    p = OpenAIProvider(credentials={"api_key": None})
    with pytest.raises(ProviderFailedError):
        await p.complete(
            model=get_model("gpt-4o"),
            messages=[ChatMessage(role="user", content="x")],
            temperature=0.7,
            top_p=1.0,
            max_tokens=32,
            stop=None,
            timeout_seconds=5.0,
        )


def test_base_estimator():
    p = StubProvider()
    assert p.estimate_tokens("") == 0
    # "hello world" is 11 ASCII chars -> 11//4 = 2 tokens
    assert p.estimate_tokens("hello world") >= 1
    # Mixed CJK should roughly count too.
    assert p.estimate_tokens("你好 world") >= 1
