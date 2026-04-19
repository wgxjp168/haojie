"""Pipeline orchestrator.

Turns one ``PipelineRequest`` into stage-by-stage calls across the four
Part 3 services. Responsible for:

* Sequencing the stages (``intent → decision → llm → report``).
* Passing each stage's output into the next stage's payload shape.
* Computing the overall pipeline status (success / partial / failed).

It is **I/O-agnostic** — the stage clients own retry / circuit-breaker /
HTTP concerns. This module is a pure state machine over stage outcomes.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from hub.clients.base import BaseStageClient, StageCallResult
from hub.config import HubSettings
from hub.core.logging import get_logger
from hub.core.metrics import HUB_PIPELINE_DURATION, HUB_PIPELINE_TOTAL
from hub.schemas import (
    PipelineRequest,
    PipelineResponse,
    StageOutcome,
)

log = get_logger(__name__)


class PipelineOrchestrator:
    """Stateless orchestrator — safe to share across requests."""

    def __init__(
        self,
        *,
        settings: HubSettings,
        intent_client: BaseStageClient,
        decision_client: BaseStageClient,
        llm_client: BaseStageClient,
        report_client: BaseStageClient,
    ) -> None:
        self.settings = settings
        self.intent_client = intent_client
        self.decision_client = decision_client
        self.llm_client = llm_client
        self.report_client = report_client

    # ------------------------------------------------------------------

    async def run(self, request: PipelineRequest) -> PipelineResponse:
        start = time.perf_counter()
        warnings: List[str] = []
        sequence: List[str] = []

        # ---------- 1. Intent ----------
        intent_payload = self._build_intent_payload(request)
        intent_result = await self.intent_client.call(intent_payload)
        intent_outcome = self._to_outcome("intent", intent_result)
        sequence.append("intent")

        if not intent_result.success:
            warnings.append(
                f"intent failed: {intent_result.error_message or 'unknown'}"
            )

        # ---------- 2. Decision (requires intent result) ----------
        decision_outcome: Optional[StageOutcome] = None
        decision_payload: Optional[Dict[str, Any]] = None
        if intent_result.success and intent_result.result:
            decision_payload = self._build_decision_payload(
                request, intent_result.result
            )
            decision_result = await self.decision_client.call(decision_payload)
            decision_outcome = self._to_outcome("decision", decision_result)
            sequence.append("decision")
            if not decision_result.success:
                warnings.append(
                    f"decision failed: {decision_result.error_message or 'unknown'}"
                )
        else:
            decision_outcome = self._skipped_outcome(
                "decision", reason="intent unavailable"
            )
            warnings.append("decision skipped because intent failed")

        # ---------- 3. LLM (opt-in) ----------
        llm_outcome: Optional[StageOutcome] = None
        if request.options.include_llm:
            llm_payload = self._build_llm_payload(
                request,
                intent_result.result if intent_result.success else None,
                decision_outcome.result if decision_outcome and decision_outcome.status == "success" else None,
            )
            llm_result = await self.llm_client.call(llm_payload)
            llm_outcome = self._to_outcome("llm", llm_result)
            sequence.append("llm")
            if not llm_result.success:
                warnings.append(
                    f"llm failed: {llm_result.error_message or 'unknown'}"
                )

        # ---------- 4. Report (opt-in, default on) ----------
        report_outcome: Optional[StageOutcome] = None
        if request.options.include_report:
            report_payload = self._build_report_payload(
                request,
                intent_result.result if intent_result.success else None,
                decision_outcome.result if decision_outcome and decision_outcome.status == "success" else None,
                llm_outcome.result if llm_outcome and llm_outcome.status == "success" else None,
            )
            report_result = await self.report_client.call(report_payload)
            report_outcome = self._to_outcome("report", report_result)
            sequence.append("report")
            if not report_result.success:
                warnings.append(
                    f"report failed: {report_result.error_message or 'unknown'}"
                )

        total_latency_ms = (time.perf_counter() - start) * 1000.0

        # ---------- overall status ----------
        overall = self._compute_overall_status(
            intent_outcome, decision_outcome, llm_outcome, report_outcome
        )

        HUB_PIPELINE_TOTAL.labels(status=overall).inc()
        HUB_PIPELINE_DURATION.labels(status=overall).observe(
            total_latency_ms / 1000.0
        )

        return PipelineResponse(
            request_id=request.request_id or f"hub_{uuid.uuid4().hex[:12]}",
            session_id=request.session_id,
            overall_status=overall,
            total_latency_ms=round(total_latency_ms, 3),
            intent=intent_outcome,
            decision=decision_outcome,
            llm=llm_outcome,
            report=report_outcome,
            stage_sequence=sequence,
            warnings=warnings,
            metadata=dict(request.metadata),
        )

    # ------------------------------------------------------------------
    # Payload builders.

    def _build_intent_payload(self, request: PipelineRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "text": request.text,
            "user_type": request.user_type or self.settings.default_user_type,
        }
        if request.language:
            payload["language"] = request.language
        if request.session_id:
            payload["session_id"] = request.session_id
        if request.request_id:
            payload["request_id"] = request.request_id
        if request.options.intent_top_k:
            payload["top_k"] = request.options.intent_top_k
        return payload

    def _build_decision_payload(
        self, request: PipelineRequest, intent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        intent_signal = {
            "intent": intent_result.get("top_intent") or intent_result.get("intent"),
            "confidence": intent_result.get("top_confidence", intent_result.get("confidence", 0.0)),
            "label": intent_result.get("top_label") or intent_result.get("label"),
        }

        buyer = self._buyer_to_decision(request)
        product = self._product_to_decision(request)

        payload: Dict[str, Any] = {
            "intent": intent_signal,
            "buyer": buyer,
            "product": product,
            "urgency": request.options.urgency or "medium",
        }
        if request.language:
            payload["language"] = request.language
        if request.session_id:
            payload["session_id"] = request.session_id
        if request.request_id:
            payload["request_id"] = request.request_id
        if request.options.decision_strategy:
            payload["strategy_override"] = request.options.decision_strategy
        return payload

    def _build_llm_payload(
        self,
        request: PipelineRequest,
        intent: Optional[Dict[str, Any]],
        decision: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_lines = [
            "You are a procurement decision assistant. Write a brief, "
            "audit-ready justification for the proposed action. Stay factual."
        ]
        user_lines = [request.text]
        if intent:
            user_lines.append(
                f"Intent: {intent.get('top_intent')} "
                f"(confidence={intent.get('top_confidence', 0):.2f})"
            )
        if decision:
            user_lines.append(
                f"Proposed action: {decision.get('action')} "
                f"(confidence={decision.get('confidence', 0):.2f}, "
                f"risk={decision.get('risk_score', 0):.2f})"
            )
            if decision.get("rationale"):
                user_lines.append(f"Rule rationale: {decision['rationale']}")

        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "\n".join(system_lines)},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
        }
        if request.options.llm_model_id:
            payload["model_id"] = request.options.llm_model_id
        if request.options.llm_strategy:
            payload["strategy"] = request.options.llm_strategy
        if request.options.llm_temperature is not None:
            payload["temperature"] = request.options.llm_temperature
        if request.request_id:
            payload["request_id"] = request.request_id
        if request.session_id:
            payload["session_id"] = request.session_id
        return payload

    def _build_report_payload(
        self,
        request: PipelineRequest,
        intent: Optional[Dict[str, Any]],
        decision: Optional[Dict[str, Any]],
        llm: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "format": (
                request.options.report_format or self.settings.default_report_format
            ),
            "audience": (
                request.options.report_audience or self.settings.default_report_audience
            ),
            "language": request.language or self.settings.default_language,
            "store": request.options.report_store,
        }
        if request.request_id:
            payload["request_id"] = request.request_id
        if request.session_id:
            payload["session_id"] = request.session_id

        # Buyer + product slices.
        if request.buyer is not None:
            payload["buyer"] = {
                "user_id": request.buyer.user_id,
                "user_type": request.buyer.user_type
                or request.user_type
                or self.settings.default_user_type,
                "region": request.buyer.region,
                "loyalty_tier": request.buyer.loyalty_tier,
                "budget": request.buyer.budget,
            }
        if request.product is not None:
            payload["product"] = {
                "product_id": request.product.product_id,
                "sku": request.product.sku,
                "name": request.product.name,
                "category": request.product.category,
                "unit_price": request.product.unit_price,
                "quantity": request.product.quantity,
                "supplier_id": request.product.supplier_id,
            }

        if intent:
            payload["intent"] = {
                "intent": intent.get("top_intent"),
                "label": intent.get("top_label"),
                "confidence": intent.get("top_confidence", 0.0),
                "backend": intent.get("backend"),
            }
        if decision:
            candidates = decision.get("candidates") or []
            payload["decision"] = {
                "action": decision.get("action"),
                "label": decision.get("label"),
                "confidence": decision.get("confidence", 0.0),
                "confident": decision.get("confident", False),
                "risk_score": decision.get("risk_score", 0.0),
                "requires_review": decision.get("requires_review", False),
                "rationale": decision.get("rationale", ""),
                "next_steps": decision.get("next_steps", []),
                "candidates": [
                    {
                        "action": c.get("action"),
                        "label": c.get("label"),
                        "confidence": c.get("confidence", 0.0),
                    }
                    for c in candidates
                ],
                "strategy": decision.get("strategy"),
                "matched_rules": (decision.get("metadata") or {}).get(
                    "matched_rules", []
                ),
            }
        if llm:
            payload["llm"] = {
                "content": llm.get("content", ""),
                "model_id": llm.get("model_id"),
                "provider": llm.get("provider"),
                "confidence": llm.get("confidence", 0.0),
                "cached": bool(llm.get("cached")),
            }
        return payload

    # ------------------------------------------------------------------
    # Mapping helpers.

    def _buyer_to_decision(self, request: PipelineRequest) -> Dict[str, Any]:
        buyer = request.buyer
        base: Dict[str, Any] = {
            "user_type": (
                (buyer.user_type if buyer else None)
                or request.user_type
                or self.settings.default_user_type
            ),
        }
        if buyer is None:
            return base
        if buyer.user_id:
            base["user_id"] = buyer.user_id
        if buyer.region is not None:
            base["region"] = buyer.region
        if buyer.loyalty_tier is not None:
            base["loyalty_tier"] = buyer.loyalty_tier
        if buyer.budget is not None:
            base["budget"] = buyer.budget
        if buyer.history_orders is not None:
            base["history_orders"] = buyer.history_orders
        if buyer.history_returns is not None:
            base["history_returns"] = buyer.history_returns
        if buyer.risk_score is not None:
            base["risk_score"] = buyer.risk_score
        return base

    def _product_to_decision(self, request: PipelineRequest) -> Dict[str, Any]:
        product = request.product
        if product is None:
            return {}
        base: Dict[str, Any] = {}
        for field in (
            "product_id",
            "sku",
            "category",
            "unit_price",
            "quantity",
            "in_stock",
            "lead_time_days",
            "supplier_id",
            "supplier_rating",
            "has_alternatives",
        ):
            value = getattr(product, field, None)
            if value is not None:
                base[field] = value
        return base

    # ------------------------------------------------------------------
    # Outcome shaping.

    @staticmethod
    def _to_outcome(stage: str, result: StageCallResult) -> StageOutcome:
        return StageOutcome(
            stage=stage,
            status="success" if result.success else "failed",
            latency_ms=result.latency_ms,
            mode=result.mode,
            error_code=result.error_code,
            error_message=result.error_message,
            result=result.result,
        )

    @staticmethod
    def _skipped_outcome(stage: str, *, reason: str) -> StageOutcome:
        return StageOutcome(
            stage=stage,
            status="skipped",
            latency_ms=0.0,
            error_message=reason,
        )

    @staticmethod
    def _compute_overall_status(
        intent: Optional[StageOutcome],
        decision: Optional[StageOutcome],
        llm: Optional[StageOutcome],
        report: Optional[StageOutcome],
    ) -> str:
        outcomes = [o for o in (intent, decision, llm, report) if o is not None]
        if not outcomes:
            return "failed"
        if all(o.status == "success" for o in outcomes):
            return "success"
        if any(o.status == "success" for o in outcomes):
            return "partial"
        return "failed"
