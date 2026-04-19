import pytest
from pydantic import ValidationError

from report.schemas import (
    BatchReportRequest,
    BuyerSummary,
    DecisionCandidateSlice,
    DecisionSlice,
    IntentSlice,
    LLMSlice,
    ProductSummary,
    ReportRequest,
)


def test_request_needs_at_least_one_input():
    with pytest.raises(ValidationError):
        ReportRequest()


def test_request_accepts_buyer_only():
    req = ReportRequest(buyer=BuyerSummary(user_id="u1"))
    assert req.buyer is not None


def test_format_validated():
    with pytest.raises(ValidationError):
        ReportRequest(buyer=BuyerSummary(user_id="u1"), format="docx")


def test_format_lowercased():
    req = ReportRequest(buyer=BuyerSummary(user_id="u1"), format="HTML")
    assert req.format == "html"


def test_audience_validated():
    with pytest.raises(ValidationError):
        ReportRequest(buyer=BuyerSummary(user_id="u1"), audience="vip")


def test_product_total_amount():
    p = ProductSummary(unit_price=100.0, quantity=3)
    assert p.total_amount == 300.0


def test_product_total_amount_none_when_missing_qty():
    p = ProductSummary(unit_price=100.0)
    assert p.total_amount is None


def test_intent_slice_validated():
    with pytest.raises(ValidationError):
        IntentSlice(intent="", confidence=0.1)


def test_decision_candidate_bounds():
    with pytest.raises(ValidationError):
        DecisionCandidateSlice(action="x", confidence=1.5)


def test_decision_slice_defaults():
    d = DecisionSlice(action="approve_purchase", confidence=0.5)
    assert d.next_steps == []
    assert d.candidates == []


def test_llm_slice_requires_content():
    with pytest.raises(ValidationError):
        LLMSlice(content="")


def test_batch_must_be_non_empty():
    with pytest.raises(ValidationError):
        BatchReportRequest(items=[])


def test_batch_size_capped():
    with pytest.raises(ValidationError):
        BatchReportRequest(
            items=[
                ReportRequest(buyer=BuyerSummary(user_id=f"u{i}"))
                for i in range(300)
            ]
        )


def test_fingerprint_stable():
    req1 = ReportRequest(
        buyer=BuyerSummary(user_id="u1", user_type="b2b"),
        format="markdown",
        audience="technical",
    )
    req2 = ReportRequest(
        buyer=BuyerSummary(user_id="u1", user_type="b2b"),
        format="markdown",
        audience="technical",
    )
    assert req1.fingerprint() == req2.fingerprint()


def test_fingerprint_changes_with_content():
    req1 = ReportRequest(
        buyer=BuyerSummary(user_id="u1"), format="markdown"
    )
    req2 = ReportRequest(
        buyer=BuyerSummary(user_id="u2"), format="markdown"
    )
    assert req1.fingerprint() != req2.fingerprint()
