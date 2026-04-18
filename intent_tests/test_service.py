import pytest

from intent.cache import IntentCache
from intent.classifier import IntentClassifier
from intent.config import IntentBackend, IntentSettings
from intent.schemas import BatchIntentRequest, IntentRequest
from intent.service import IntentService


def _build_service(backend: IntentBackend = IntentBackend.RULES) -> IntentService:
    s = IntentSettings()
    s.backend = backend
    s.cache_enabled = True
    s.redis_url = None
    s.memory_cache_size = 32
    s.preload_model = False
    s.confident_threshold = 0.5
    svc = IntentService(
        settings=s,
        classifier=IntentClassifier(settings=s),
        cache=IntentCache(memory_capacity=32, redis_url=None, ttl_seconds=60),
    )
    return svc


def test_predict_returns_ranked_candidates():
    svc = _build_service()
    resp = svc.predict(IntentRequest(text="这款手机多少钱?", language="zh-CN"))
    assert resp.top_intent == "inquire_price"
    assert resp.top_label == "价格咨询"
    assert resp.backend == "rules"
    assert resp.candidates
    assert resp.cached is False


def test_predict_returns_cached_on_second_call():
    svc = _build_service()
    req = IntentRequest(text="请问有没有现货", language="zh-CN")
    first = svc.predict(req)
    second = svc.predict(req)
    assert first.cached is False
    assert second.cached is True
    assert first.top_intent == second.top_intent


def test_predict_respects_top_k_override():
    svc = _build_service()
    resp = svc.predict(
        IntentRequest(text="价格 报价 多少钱", language="zh-CN", top_k=1)
    )
    assert len(resp.candidates) == 1


def test_predict_marks_low_confidence_as_not_confident():
    svc = _build_service()
    resp = svc.predict(IntentRequest(text="xxxxx yyyyy", language="en"))
    assert resp.top_intent == "other"
    assert resp.confident is False


def test_predict_rejects_empty_text_at_schema_layer():
    # Empty/whitespace-only text is rejected by the IntentRequest schema
    # before it ever reaches the service.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        IntentRequest(text="   ", language="en")


def test_predict_rejects_unsupported_language():
    svc = _build_service()
    from intent.core.exceptions import UnsupportedLanguageError
    with pytest.raises(UnsupportedLanguageError):
        svc.predict(IntentRequest(text="bonjour", language="fr-FR"))


def test_batch_predict_returns_per_item_results():
    svc = _build_service()
    batch = BatchIntentRequest(
        items=[
            IntentRequest(text="这款多少钱", language="zh-CN"),
            IntentRequest(text="有货吗", language="zh-CN"),
            IntentRequest(text="这两个哪个好", language="zh-CN"),
        ]
    )
    resp = svc.predict_batch(batch)
    assert resp.total == 3
    assert len(resp.results) == 3
    intents = {r.top_intent for r in resp.results}
    assert "inquire_price" in intents
    assert "check_stock" in intents


def test_batch_predict_caches_across_calls():
    svc = _build_service()
    batch = BatchIntentRequest(
        items=[IntentRequest(text="现货吗", language="zh-CN")]
    )
    first = svc.predict_batch(batch)
    second = svc.predict_batch(batch)
    assert first.results[0].cached is False
    assert second.results[0].cached is True


def test_english_audience_biased_classification():
    svc = _build_service()
    resp = svc.predict(
        IntentRequest(
            text="We need an RFQ for a bulk order of 1000 units",
            language="en",
            user_type="b2b",
        )
    )
    assert resp.top_intent in {"request_quote", "bulk_order"}
