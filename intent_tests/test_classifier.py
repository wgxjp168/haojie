from unittest.mock import MagicMock

from intent.classifier import ClassificationOutcome, IntentClassifier
from intent.config import IntentBackend, IntentSettings
from intent.model import TransformerPrediction
from intent.preprocessor import TextNormalizer


def _normalize(text: str, language: str = "zh-CN"):
    return TextNormalizer().normalize(text, language=language)


def _settings(**overrides):
    s = IntentSettings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_classifier_uses_rules_when_backend_rules():
    s = _settings(backend=IntentBackend.RULES)
    clf = IntentClassifier(settings=s, transformer=None)
    outcome = clf.classify(_normalize("这个多少钱?"))
    assert outcome.backend == "rules"
    assert outcome.top_intent == "inquire_price"


def test_ensemble_keeps_transformer_when_confident():
    s = _settings(backend=IntentBackend.ENSEMBLE, low_confidence_threshold=0.4)
    fake_tf = MagicMock()
    fake_tf.loaded = True
    fake_tf.predict.return_value = [
        TransformerPrediction(scores=[("compare_products", 0.9), ("other", 0.1)])
    ]
    clf = IntentClassifier(settings=s, transformer=fake_tf)
    outcome = clf.classify(_normalize("iPhone 15 vs Galaxy S24"))
    assert outcome.top_intent == "compare_products"
    assert outcome.backend == "ensemble:transformer"


def test_ensemble_falls_back_to_rules_when_transformer_low_confidence():
    s = _settings(backend=IntentBackend.ENSEMBLE, low_confidence_threshold=0.8)
    fake_tf = MagicMock()
    fake_tf.loaded = True
    # Transformer unsure between two intents.
    fake_tf.predict.return_value = [
        TransformerPrediction(scores=[("other", 0.3), ("greeting", 0.25)])
    ]
    clf = IntentClassifier(settings=s, transformer=fake_tf)
    # Rules should rescue this with a strong price match.
    outcome = clf.classify(_normalize("请问这个报价是多少 折扣 价格"))
    assert outcome.top_intent == "inquire_price"
    assert outcome.backend == "ensemble:rules"


def test_transformer_failure_trips_breaker_and_falls_back():
    from intent.core.exceptions import ModelFailedError

    s = _settings(backend=IntentBackend.ENSEMBLE, cb_failure_threshold=1)
    fake_tf = MagicMock()
    fake_tf.loaded = True
    fake_tf.predict.side_effect = ModelFailedError("boom")
    clf = IntentClassifier(settings=s, transformer=fake_tf)

    # First call: transformer raises, fallback kicks in, breaker opens.
    outcome = clf.classify(_normalize("多少钱 报价"))
    assert outcome.backend == "rules"
    assert clf.breaker.state_label == "open"

    # Second call: breaker short-circuits straight to rules.
    outcome2 = clf.classify(_normalize("到货没有"))
    assert outcome2.backend == "rules"
    # Predict should only have been called once — the second time the
    # breaker was open so we went straight to rules.
    assert fake_tf.predict.call_count == 1


def test_classify_batch_delegates_to_transformer_batch_when_possible():
    s = _settings(backend=IntentBackend.ENSEMBLE, low_confidence_threshold=0.0)
    fake_tf = MagicMock()
    fake_tf.loaded = True
    fake_tf.predict.return_value = [
        TransformerPrediction(scores=[("search_product", 0.9), ("other", 0.1)]),
        TransformerPrediction(scores=[("inquire_price", 0.8), ("other", 0.2)]),
    ]
    clf = IntentClassifier(settings=s, transformer=fake_tf)

    items = [_normalize("搜索手机"), _normalize("多少钱")]
    outcomes = clf.classify_batch(items)
    assert [o.top_intent for o in outcomes] == ["search_product", "inquire_price"]
    # Transformer called exactly once for the whole batch.
    assert fake_tf.predict.call_count == 1


def test_outcome_candidates_respect_top_k():
    s = _settings(backend=IntentBackend.TRANSFORMER, top_k=2)
    fake_tf = MagicMock()
    fake_tf.loaded = True
    fake_tf.predict.return_value = [
        TransformerPrediction(
            scores=[
                ("compare_products", 0.5),
                ("inquire_price", 0.3),
                ("other", 0.2),
            ]
        )
    ]
    clf = IntentClassifier(settings=s, transformer=fake_tf)
    outcome = clf.classify(_normalize("which is cheaper A or B"), top_k=2)
    assert len(outcome.candidates) == 2
    assert outcome.backend == "transformer"
