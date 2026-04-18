"""Light-weight text normalisation & language detection.

This module must stay dependency-free so that it can run anywhere, including
the unit-test environment (no torch/transformers). Anything heavier belongs in
``intent.model``.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from intent.core.exceptions import UnsupportedLanguageError, ValidationIntentError

# Matches common full-width punctuation / emoji / control characters that we
# want to collapse into plain whitespace or drop.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    clean: str
    language: str
    length: int

    @property
    def cache_key(self) -> str:
        blob = f"{self.language}|{self.clean}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class TextNormalizer:
    """Normalises text and detects the dominant language.

    Language detection here is intentionally trivial — it only distinguishes
    CJK (``zh-CN``) from Latin (``en``). A real deployment can plug a proper
    detector by subclassing and overriding ``detect_language``.
    """

    def __init__(
        self,
        *,
        max_length: int = 2000,
        min_length: int = 1,
        default_language: str = "zh-CN",
        supported_languages: Optional[list[str]] = None,
    ) -> None:
        if min_length < 1:
            raise ValueError("min_length must be >= 1")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        self.max_length = max_length
        self.min_length = min_length
        self.default_language = default_language
        self.supported = set(supported_languages or ["zh-CN", "en"])
        if default_language not in self.supported:
            raise ValueError(
                f"default_language {default_language!r} not in supported {self.supported}"
            )

    # ----- public api -----

    def normalize(self, text: str, *, language: Optional[str] = None) -> NormalizedText:
        if text is None:
            raise ValidationIntentError("text must not be null")
        raw = text
        clean = self._clean(text)
        if len(clean) < self.min_length:
            raise ValidationIntentError(
                "text is too short after normalisation",
                details={"min_length": self.min_length, "actual": len(clean)},
            )
        if len(clean) > self.max_length:
            # Truncate instead of rejecting — intent queries are usually short
            # and long prefixes are dominated by the beginning anyway.
            clean = clean[: self.max_length]
        lang = (language or self.detect_language(clean) or self.default_language).strip()
        if lang not in self.supported:
            # Fall back instead of failing hard: the transformer is
            # multilingual so we can still classify, we just log the hint.
            lang = self.default_language
        return NormalizedText(raw=raw, clean=clean, language=lang, length=len(clean))

    def ensure_language_supported(self, language: str) -> None:
        if language not in self.supported:
            raise UnsupportedLanguageError(
                f"language {language!r} is not supported",
                details={"supported": sorted(self.supported)},
            )

    # ----- internals -----

    def _clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = _URL_RE.sub(" ", text)
        text = _EMAIL_RE.sub(" ", text)
        text = _CONTROL_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text)
        return text.strip()

    def detect_language(self, text: str) -> Optional[str]:
        if not text:
            return None
        cjk = len(_CJK_RE.findall(text))
        latin = len(_LATIN_RE.findall(text))
        if cjk == 0 and latin == 0:
            return None
        return "zh-CN" if cjk >= latin else "en"
