import pytest

from report.catalog import (
    TEMPLATE_COPY,
    TEMPLATE_REGISTRY,
    copy_for,
    default_template_for,
    resolve_template,
    sections_for_template,
)
from report.config import ReportAudience


def test_registry_has_three_templates():
    assert set(TEMPLATE_REGISTRY.keys()) == {
        "executive_summary",
        "technical_detail",
        "customer_friendly",
    }


def test_default_template_per_audience():
    assert default_template_for(ReportAudience.EXECUTIVE) == "executive_summary"
    assert default_template_for(ReportAudience.TECHNICAL) == "technical_detail"
    assert default_template_for(ReportAudience.CUSTOMER) == "customer_friendly"


def test_resolve_with_none_defaults_by_audience():
    meta = resolve_template(None, ReportAudience.EXECUTIVE)
    assert meta.template_id == "executive_summary"


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_template("ghost_template", ReportAudience.EXECUTIVE)


def test_copy_languages():
    assert "zh-CN" in TEMPLATE_COPY
    assert "en" in TEMPLATE_COPY
    assert copy_for("zh-CN").title != copy_for("en").title


def test_copy_unknown_language_falls_back_to_en():
    copy = copy_for("fr-FR")
    assert copy.title == TEMPLATE_COPY["en"].title


def test_sections_for_executive_are_minimal():
    sections = sections_for_template("executive_summary")
    assert "subject" in sections
    assert "trace" not in sections
    assert "llm" not in sections


def test_sections_for_technical_include_trace_and_llm():
    sections = sections_for_template("technical_detail")
    assert "trace" in sections
    assert "llm" in sections
    assert "intent" in sections


def test_sections_for_customer_include_closing():
    sections = sections_for_template("customer_friendly")
    assert "closing" in sections
    assert "trace" not in sections


def test_sections_for_unknown_template_has_safe_default():
    sections = sections_for_template("bogus_template")
    assert "decision" in sections
