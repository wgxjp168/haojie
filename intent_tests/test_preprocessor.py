import pytest

from intent.core.exceptions import UnsupportedLanguageError, ValidationIntentError
from intent.preprocessor import TextNormalizer


def test_normalize_strips_whitespace_and_urls():
    n = TextNormalizer()
    out = n.normalize("  请问 https://example.com  有货吗? \n\n ")
    assert "https" not in out.clean
    assert out.clean.endswith("有货吗?")
    assert out.language == "zh-CN"


def test_language_detection_english():
    n = TextNormalizer()
    out = n.normalize("Do you have this in stock?")
    assert out.language == "en"


def test_language_override_takes_priority():
    n = TextNormalizer()
    out = n.normalize("Hello 你好", language="zh-CN")
    assert out.language == "zh-CN"


def test_unsupported_language_override_falls_back_to_default():
    n = TextNormalizer(default_language="zh-CN", supported_languages=["zh-CN", "en"])
    out = n.normalize("hola mundo", language=None)
    # Latin-only -> detected as en.
    assert out.language == "en"


def test_empty_text_raises():
    n = TextNormalizer()
    with pytest.raises(ValidationIntentError):
        n.normalize("   \n  ")


def test_very_long_text_is_truncated():
    n = TextNormalizer(max_length=50)
    out = n.normalize("a" * 1000)
    assert len(out.clean) == 50


def test_ensure_language_supported_raises():
    n = TextNormalizer(supported_languages=["zh-CN", "en"])
    with pytest.raises(UnsupportedLanguageError):
        n.ensure_language_supported("fr-FR")


def test_cache_key_is_deterministic_for_same_input():
    n = TextNormalizer()
    a = n.normalize("我想买手机")
    b = n.normalize("我想买手机")
    assert a.cache_key == b.cache_key
    c = n.normalize("我想买平板")
    assert c.cache_key != a.cache_key
