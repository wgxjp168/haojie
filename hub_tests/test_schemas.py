import pytest
from pydantic import ValidationError

from hub.schemas import (
    BatchPipelineRequest,
    PipelineBuyer,
    PipelineOptions,
    PipelineProduct,
    PipelineRequest,
    StageOutcome,
)


def test_request_requires_text():
    with pytest.raises(ValidationError):
        PipelineRequest(text="   ")


def test_request_trims_text():
    req = PipelineRequest(text="  hello  ")
    assert req.text == "hello"


def test_options_defaults():
    opts = PipelineOptions()
    assert opts.include_llm is False
    assert opts.include_report is True
    assert opts.report_store is True


def test_user_type_validated():
    with pytest.raises(ValidationError):
        PipelineRequest(text="hi", user_type="b2x")


def test_buyer_risk_score_bounded():
    with pytest.raises(ValidationError):
        PipelineBuyer(risk_score=2.0)


def test_product_quantity_positive():
    with pytest.raises(ValidationError):
        PipelineProduct(quantity=0)


def test_batch_must_be_non_empty():
    with pytest.raises(ValidationError):
        BatchPipelineRequest(items=[])


def test_batch_size_capped():
    with pytest.raises(ValidationError):
        BatchPipelineRequest(
            items=[PipelineRequest(text=f"hi {i}") for i in range(300)]
        )


def test_stage_outcome_shape():
    outcome = StageOutcome(
        stage="intent", status="success", latency_ms=12.5, mode="embedded"
    )
    assert outcome.stage == "intent"
    assert outcome.status == "success"
    assert outcome.latency_ms == 12.5
