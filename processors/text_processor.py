"""Text input processor.

Normalises Chinese/English procurement-style text, extracts lightweight
signals (keywords, intent hints, price/quantity mentions) that are cheap
to compute. Heavyweight NLP (NER, sentiment) is intentionally out of
scope for the input stage — the AI decision hub handles those.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from core.exceptions import ValidationError
from entities.enums import InputType
from processors.base import BaseInputProcessor, InputContext, ProcessorResult

_WHITESPACE_RE = re.compile(r"\s+")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-zA-Z]")

# Money patterns: ¥1,234.56 / $99 / RMB 200 / 5000元
_PRICE_PATTERNS = [
    re.compile(r"(?:¥|￥|RMB\s*)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)"),
    re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:元|块)"),
]

# Budget keywords
_BUDGET_WORDS = ("预算", "价位", "多少钱", "budget", "price range")

# Quantity: 10件 / 5 台 / 需要20个 / qty 100
_QUANTITY_PATTERNS = [
    re.compile(r"([0-9]+)\s*(?:个|件|台|套|箱|pcs|units?)", re.IGNORECASE),
    re.compile(r"(?:需要|买|采购)\s*([0-9]+)"),
]

# Intent heuristics — keep tiny and conservative. Downstream AI handles real NLU.
_INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "purchase": ("购买", "采购", "买", "下单", "order", "buy", "purchase"),
    "compare": ("比较", "对比", "哪个好", "vs", "versus", "compare"),
    "recommend": ("推荐", "建议", "帮我选", "recommend", "suggest"),
    "query": ("咨询", "问一下", "了解", "ask", "question", "info"),
}

# Spec keywords — useful to tell the decision hub the user has been specific.
_SPEC_KEYWORDS = (
    "规格", "尺寸", "型号", "颜色", "材质", "容量", "功率",
    "spec", "size", "model", "color", "capacity", "wattage",
)


class TextInputProcessor(BaseInputProcessor):
    input_type = InputType.TEXT

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    # --- validation ---

    def validate(self, input_data: Any) -> None:
        if not isinstance(input_data, str):
            raise ValidationError("input must be a string", field="text")
        if not input_data.strip():
            raise ValidationError("text is empty", field="text")
        if len(input_data) > self._settings.text_max_length:
            raise ValidationError(
                f"text exceeds max length of {self._settings.text_max_length}",
                field="text",
                value=len(input_data),
            )

    # --- processing ---

    async def process(self, input_data: str, context: InputContext) -> ProcessorResult:
        text = input_data
        cleaned = self._clean(text)
        language = self._detect_language(cleaned)
        keywords = self._extract_keywords(cleaned)
        intent = self._detect_intent(cleaned)
        prices = self._extract_prices(cleaned)
        quantity = self._extract_quantity(cleaned)
        has_specs = any(k in cleaned for k in _SPEC_KEYWORDS)
        has_budget = any(k in cleaned.lower() for k in _BUDGET_WORDS) or bool(prices)

        data: Dict[str, Any] = {
            "original_text": text,
            "cleaned_text": cleaned,
            "length": len(text),
            "language": language,
            "keywords": keywords,
            "intent": intent,
            "signals": {
                "has_specifications": has_specs,
                "has_budget_mention": has_budget,
                "has_quantity_mention": quantity is not None,
            },
            "extracted": {
                "prices": prices,
                "quantity": quantity,
            },
        }

        summary = self._summarise(data)
        return {"data": data, "warnings": [], "summary": summary}

    # --- helpers ---

    @staticmethod
    def _clean(text: str) -> str:
        collapsed = _WHITESPACE_RE.sub(" ", text).strip()
        return collapsed

    @staticmethod
    def _detect_language(text: str) -> str:
        has_cn = bool(_CHINESE_RE.search(text))
        has_en = bool(_LATIN_RE.search(text))
        if has_cn and has_en:
            return "mixed"
        if has_cn:
            return "zh"
        if has_en:
            return "en"
        return "unknown"

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        # Very lightweight tokenisation — split by whitespace & punctuation.
        # Real keyword extraction happens downstream in the AI decision hub.
        tokens = re.split(r"[\s,，.。;；:：!！?？/、]+", text)
        deduped: List[str] = []
        seen = set()
        for t in tokens:
            t = t.strip()
            if len(t) < 2 or t in seen:
                continue
            seen.add(t)
            deduped.append(t)
            if len(deduped) >= 20:
                break
        return deduped

    @staticmethod
    def _detect_intent(text: str) -> Dict[str, bool]:
        lower = text.lower()
        return {name: any(kw in lower for kw in kws) for name, kws in _INTENT_KEYWORDS.items()}

    @staticmethod
    def _extract_prices(text: str) -> List[float]:
        results: List[float] = []
        seen = set()
        for pattern in _PRICE_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(1).replace(",", "")
                try:
                    value = float(Decimal(raw))
                except (InvalidOperation, ValueError):
                    continue
                if value in seen:
                    continue
                seen.add(value)
                results.append(value)
        return results

    @staticmethod
    def _extract_quantity(text: str) -> Optional[int]:
        for pattern in _QUANTITY_PATTERNS:
            m = pattern.search(text)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _summarise(data: Dict[str, Any]) -> str:
        intent = ", ".join(k for k, v in data["intent"].items() if v) or "unspecified"
        prices = data["extracted"]["prices"]
        price_hint = f", prices={prices}" if prices else ""
        qty = data["extracted"]["quantity"]
        qty_hint = f", qty={qty}" if qty else ""
        return (
            f"text[{data['language']}, {data['length']} chars]: "
            f"intent={intent}{price_hint}{qty_hint}"
        )
