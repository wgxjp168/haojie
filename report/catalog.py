"""Template catalog — audience-specific, bilingual (zh-CN / en).

Templates are resolved by ``(template_id, audience, language)``. The
``template_id`` defaults to the audience name (``executive`` →
``executive_summary``), so most callers only need to pick the audience.

All template copy lives in Python (no external files, no Jinja2
dependency) so the service ships as a single wheel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from report.config import ReportAudience


# Key into ``TEMPLATE_COPY``: (template_id, audience, language).
_Key = Tuple[str, str, str]


@dataclass(frozen=True)
class TemplateMetadata:
    """Descriptor for clients (surfaced via ``/templates`` endpoint)."""

    template_id: str
    audience: ReportAudience
    description: str
    supported_languages: Tuple[str, ...] = ("zh-CN", "en")


TEMPLATE_REGISTRY: Dict[str, TemplateMetadata] = {
    "executive_summary": TemplateMetadata(
        template_id="executive_summary",
        audience=ReportAudience.EXECUTIVE,
        description="High-level one-page summary for leadership.",
    ),
    "technical_detail": TemplateMetadata(
        template_id="technical_detail",
        audience=ReportAudience.TECHNICAL,
        description="Full audit trail with matched rules and LLM rationale.",
    ),
    "customer_friendly": TemplateMetadata(
        template_id="customer_friendly",
        audience=ReportAudience.CUSTOMER,
        description="Buyer-facing plain-language explanation.",
    ),
}


def default_template_for(audience: ReportAudience) -> str:
    mapping = {
        ReportAudience.EXECUTIVE: "executive_summary",
        ReportAudience.TECHNICAL: "technical_detail",
        ReportAudience.CUSTOMER: "customer_friendly",
    }
    return mapping[audience]


# ---------------------------------------------------------------- copy


@dataclass(frozen=True)
class TemplateCopy:
    """Localized strings used by the renderer to assemble a report."""

    title: str
    section_subject: str
    section_intent: str
    section_decision: str
    section_rationale: str
    section_next_steps: str
    section_trace: str
    section_llm: str
    field_user: str
    field_user_type: str
    field_product: str
    field_amount: str
    field_budget: str
    field_action: str
    field_confidence: str
    field_risk: str
    field_requires_review: str
    field_strategy: str
    no_data: str
    generated_at: str
    report_id: str
    closing_customer: str = ""


def _base_copy_zh() -> TemplateCopy:
    return TemplateCopy(
        title="采购决策报告",
        section_subject="主体信息",
        section_intent="意图识别",
        section_decision="决策结论",
        section_rationale="决策理由",
        section_next_steps="后续步骤",
        section_trace="规则匹配轨迹",
        section_llm="AI 补充说明",
        field_user="用户",
        field_user_type="用户类型",
        field_product="商品",
        field_amount="金额",
        field_budget="预算",
        field_action="建议动作",
        field_confidence="置信度",
        field_risk="风险分",
        field_requires_review="需人工复核",
        field_strategy="策略",
        no_data="无",
        generated_at="生成时间",
        report_id="报告编号",
        closing_customer="如有疑问,请联系您的客户经理。",
    )


def _base_copy_en() -> TemplateCopy:
    return TemplateCopy(
        title="Procurement Decision Report",
        section_subject="Subject",
        section_intent="Intent Classification",
        section_decision="Decision",
        section_rationale="Rationale",
        section_next_steps="Next Steps",
        section_trace="Rule Match Trace",
        section_llm="AI Narrative",
        field_user="User",
        field_user_type="User type",
        field_product="Product",
        field_amount="Amount",
        field_budget="Budget",
        field_action="Recommended action",
        field_confidence="Confidence",
        field_risk="Risk score",
        field_requires_review="Requires review",
        field_strategy="Strategy",
        no_data="N/A",
        generated_at="Generated at",
        report_id="Report ID",
        closing_customer="If you have questions, please contact your account manager.",
    )


# For v1, all three audiences share the same localized copy but the
# renderer keys on the ``template_id`` to decide which sections to emit.
# Having a single copy dictionary per language keeps translations in one
# place; audience-specific overrides can be layered in as the product
# matures.
TEMPLATE_COPY: Dict[str, TemplateCopy] = {
    "zh-CN": _base_copy_zh(),
    "en": _base_copy_en(),
}


# ---------------------------------------------------------------- helpers


def resolve_template(
    template_id: str | None, audience: ReportAudience
) -> TemplateMetadata:
    """Resolve template metadata, falling back to the audience default."""
    chosen = template_id or default_template_for(audience)
    meta = TEMPLATE_REGISTRY.get(chosen)
    if meta is None:
        raise KeyError(f"unknown template id: {chosen!r}")
    return meta


def copy_for(language: str) -> TemplateCopy:
    """Return the localized copy, falling back to en."""
    return TEMPLATE_COPY.get(language) or TEMPLATE_COPY["en"]


def sections_for_template(template_id: str) -> Tuple[str, ...]:
    """Which logical sections a given template emits.

    Renderers key on this to decide which blocks to assemble. Keeping the
    wiring in one place makes it trivial to add a new audience without
    changing every renderer.
    """
    if template_id == "executive_summary":
        return ("subject", "decision", "rationale", "next_steps")
    if template_id == "technical_detail":
        return (
            "subject",
            "intent",
            "decision",
            "rationale",
            "trace",
            "llm",
            "next_steps",
        )
    if template_id == "customer_friendly":
        return ("subject", "decision", "rationale", "next_steps", "closing")
    return ("subject", "decision", "rationale", "next_steps")
