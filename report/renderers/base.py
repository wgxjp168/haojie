"""Abstract renderer + shared data classes.

The ``BaseRenderer`` is intentionally tiny: each concrete renderer
assembles the report body however makes sense for its format. A shared
``RenderContext`` carries the resolved template copy and sliced input so
subclasses do not have to re-validate anything.
"""
from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from report.catalog import TemplateCopy, TemplateMetadata, sections_for_template
from report.schemas import (
    BuyerSummary,
    DecisionSlice,
    IntentSlice,
    LLMSlice,
    ProductSummary,
    ReportRequest,
)


@dataclass(frozen=True)
class RenderContext:
    """Resolved context passed to every renderer.

    All input slices are normalized here so renderers do not need to
    repeat defensive ``None`` checks.
    """

    report_id: str
    request: ReportRequest
    template: TemplateMetadata
    audience: str
    language: str
    copy: TemplateCopy
    sections: Tuple[str, ...]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def buyer(self) -> Optional[BuyerSummary]:
        return self.request.buyer

    @property
    def product(self) -> Optional[ProductSummary]:
        return self.request.product

    @property
    def intent(self) -> Optional[IntentSlice]:
        return self.request.intent

    @property
    def decision(self) -> Optional[DecisionSlice]:
        return self.request.decision

    @property
    def llm(self) -> Optional[LLMSlice]:
        return self.request.llm

    def wants(self, section: str) -> bool:
        return section in self.sections


@dataclass
class RenderOutput:
    """The end product of a render call."""

    content: str
    content_type: str
    size_bytes: int
    checksum: str

    @classmethod
    def from_text(cls, content: str, content_type: str) -> "RenderOutput":
        data = content.encode("utf-8")
        return cls(
            content=content,
            content_type=content_type,
            size_bytes=len(data),
            checksum=hashlib.sha256(data).hexdigest(),
        )


class BaseRenderer(abc.ABC):
    """Every format-specific renderer implements this contract."""

    #: String id used by ``ReportSettings.default_format`` and request bodies.
    format: str = "base"

    #: MIME type used for Content-Type headers and storage metadata.
    content_type: str = "application/octet-stream"

    @abc.abstractmethod
    def render(self, ctx: RenderContext) -> RenderOutput:
        """Render the report body for ``ctx``."""

    # ------------------------------------------------------------------

    @staticmethod
    def format_amount(amount: Optional[float]) -> str:
        """Human-friendly money formatting, locale-agnostic."""
        if amount is None:
            return "-"
        try:
            return f"{float(amount):,.2f}"
        except (TypeError, ValueError):
            return str(amount)

    @staticmethod
    def format_percent(ratio: Optional[float]) -> str:
        if ratio is None:
            return "-"
        try:
            return f"{float(ratio) * 100:.1f}%"
        except (TypeError, ValueError):
            return str(ratio)

    @staticmethod
    def sections_for_template_id(template_id: str) -> Tuple[str, ...]:
        return sections_for_template(template_id)
