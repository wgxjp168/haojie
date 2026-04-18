"""Rule-based intent classifier.

Uses keyword matching over the taxonomy's ``keywords_*`` fields plus a few
hand-written heuristics. This classifier:

* Always returns a result (falls back to ``other`` with low confidence).
* Has predictable latency (< 1 ms on typical queries).
* Serves as fallback when the transformer is unavailable or low-confidence.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from intent.taxonomy import (
    FALLBACK_INTENT,
    INTENT_REGISTRY,
    IntentDefinition,
    intents_for_audience,
)


@dataclass(frozen=True)
class RuleMatch:
    intent: str
    score: float
    matched_terms: Tuple[str, ...]


class RuleBasedClassifier:
    """Keyword-voting classifier with audience biasing.

    Scoring:
        score = 1 - exp(-k * matched_terms) * audience_penalty
    This gives quickly-saturating confidence as more keywords match,
    capped at ``1 - epsilon`` so the transformer can still override.
    """

    def __init__(self, *, saturation: float = 0.9, k: float = 0.55) -> None:
        self._saturation = saturation
        self._k = k
        # Precompile keyword indices for both languages.
        self._zh_index = self._build_index("keywords_zh")
        self._en_index = self._build_index("keywords_en")

    # ---- public api ----

    def classify(
        self,
        text: str,
        *,
        language: str = "zh-CN",
        user_type: Optional[str] = None,
        top_k: int = 3,
    ) -> List[RuleMatch]:
        if not text:
            return [RuleMatch(intent=FALLBACK_INTENT, score=0.0, matched_terms=())]

        lowered = text.lower()
        index = self._zh_index if language.lower().startswith("zh") else self._en_index

        hits: Dict[str, List[str]] = {}
        for term, intent_ids in index.items():
            if term in lowered:
                for intent_id in intent_ids:
                    hits.setdefault(intent_id, []).append(term)

        if not hits:
            return [RuleMatch(intent=FALLBACK_INTENT, score=0.1, matched_terms=())]

        audience = (user_type or "any").lower()
        audience_allowed = {d.id for d in intents_for_audience(audience)}

        scored: List[RuleMatch] = []
        for intent_id, terms in hits.items():
            penalty = 1.0 if intent_id in audience_allowed else 0.6
            raw = self._saturation * (1.0 - math.exp(-self._k * len(terms)))
            scored.append(
                RuleMatch(
                    intent=intent_id,
                    score=round(raw * penalty, 4),
                    matched_terms=tuple(sorted(set(terms))),
                )
            )

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def classify_best(
        self,
        text: str,
        *,
        language: str = "zh-CN",
        user_type: Optional[str] = None,
    ) -> RuleMatch:
        return self.classify(text, language=language, user_type=user_type, top_k=1)[0]

    # ---- internals ----

    def _build_index(self, field_name: str) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        for intent_id, definition in INTENT_REGISTRY.items():
            for term in getattr(definition, field_name, ()) or ():
                term = term.lower()
                if not term:
                    continue
                index.setdefault(term, []).append(intent_id)
        return index
