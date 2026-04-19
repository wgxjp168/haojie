import json as jsonlib

import pytest

from report.catalog import copy_for, resolve_template, sections_for_template
from report.config import ReportAudience
from report.renderers import HtmlRenderer, JsonRenderer, MarkdownRenderer, TextRenderer
from report.renderers.base import RenderContext


def _ctx(sample_request, audience: ReportAudience = ReportAudience.TECHNICAL,
         language: str = "zh-CN") -> RenderContext:
    template = resolve_template(None, audience)
    return RenderContext(
        report_id="rpt-test-001",
        request=sample_request,
        template=template,
        audience=audience.value,
        language=language,
        copy=copy_for(language),
        sections=sections_for_template(template.template_id),
    )


# -------------------- markdown --------------------


def test_markdown_renderer_contains_expected_sections(sample_request):
    out = MarkdownRenderer().render(_ctx(sample_request))
    text = out.content
    assert "采购决策报告" in text
    assert "主体信息" in text
    assert "意图识别" in text
    assert "决策结论" in text
    assert "AI 补充说明" in text
    assert "后续步骤" in text
    assert out.content_type.startswith("text/markdown")
    assert out.size_bytes > 0
    assert out.checksum


def test_markdown_escapes_pipes(sample_request):
    # Inject a rationale with a pipe char to ensure table safety.
    sample_request.decision.rationale = "A | B | C"
    out = MarkdownRenderer().render(_ctx(sample_request))
    assert "A \\| B \\| C" in out.content


def test_markdown_english_when_language_en(sample_request):
    sample_request.language = "en"
    out = MarkdownRenderer().render(_ctx(sample_request, language="en"))
    assert "Procurement Decision Report" in out.content
    assert "Rationale" in out.content


def test_markdown_executive_omits_trace_and_llm(sample_request):
    out = MarkdownRenderer().render(_ctx(sample_request, ReportAudience.EXECUTIVE))
    assert "规则匹配轨迹" not in out.content
    assert "AI 补充说明" not in out.content


def test_markdown_customer_has_closing(sample_request):
    out = MarkdownRenderer().render(_ctx(sample_request, ReportAudience.CUSTOMER))
    assert "客户经理" in out.content or "account manager" in out.content


def test_markdown_respects_include_trace_false(sample_request):
    sample_request.include_trace = False
    out = MarkdownRenderer().render(_ctx(sample_request))
    assert "规则匹配轨迹" not in out.content


# -------------------- html --------------------


def test_html_renderer_produces_valid_document(sample_request):
    out = HtmlRenderer().render(_ctx(sample_request))
    assert out.content.startswith("<!DOCTYPE html>")
    assert "<style>" in out.content
    assert "<title>" in out.content
    assert out.content_type.startswith("text/html")


def test_html_escapes_dangerous_chars(sample_request):
    sample_request.decision.rationale = "<script>alert(1)</script>"
    out = HtmlRenderer().render(_ctx(sample_request))
    assert "<script>alert(1)</script>" not in out.content
    assert "&lt;script&gt;" in out.content


def test_html_executive_trims_sections(sample_request):
    out = HtmlRenderer().render(_ctx(sample_request, ReportAudience.EXECUTIVE))
    assert "规则匹配轨迹" not in out.content
    assert "AI 补充说明" not in out.content


# -------------------- json --------------------


def test_json_renderer_is_valid_json(sample_request):
    out = JsonRenderer().render(_ctx(sample_request))
    data = jsonlib.loads(out.content)
    assert data["schema_version"] == "v1"
    assert data["report_id"] == "rpt-test-001"
    assert "subject" in data
    assert "intent" in data
    assert "decision" in data
    assert "llm" in data


def test_json_excludes_trace_when_disabled(sample_request):
    sample_request.include_trace = False
    out = JsonRenderer().render(_ctx(sample_request))
    data = jsonlib.loads(out.content)
    assert "matched_rules" not in data["decision"]


def test_json_omits_missing_slices(sample_request):
    sample_request.llm = None
    out = JsonRenderer().render(_ctx(sample_request))
    data = jsonlib.loads(out.content)
    assert "llm" not in data


# -------------------- text --------------------


def test_text_renderer_produces_plain_text(sample_request):
    out = TextRenderer().render(_ctx(sample_request))
    assert "采购决策报告" in out.content
    assert "<" not in out.content  # no HTML tags
    assert "{" not in out.content[:200]  # no JSON
    assert out.content_type.startswith("text/plain")


def test_text_renderer_wraps_long_rationale(sample_request):
    sample_request.decision.rationale = "x" * 300
    out = TextRenderer().render(_ctx(sample_request))
    # Body should not contain a line wider than the column cap (+ a little slack).
    for line in out.content.splitlines():
        assert len(line) <= 120


# -------------------- base helpers --------------------


def test_format_amount_none():
    assert MarkdownRenderer.format_amount(None) == "-"


def test_format_amount_numeric():
    assert MarkdownRenderer.format_amount(12345.67) == "12,345.67"


def test_format_percent_none():
    assert MarkdownRenderer.format_percent(None) == "-"


def test_format_percent_value():
    assert MarkdownRenderer.format_percent(0.913) == "91.3%"
