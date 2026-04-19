"""JSON renderer.

Emits a structured report for API consumers and downstream analytics
pipelines. The schema is stable and version-tagged so consumers can
opt-in to breaking changes.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from report.renderers.base import BaseRenderer, RenderContext, RenderOutput


SCHEMA_VERSION = "v1"


class JsonRenderer(BaseRenderer):
    format = "json"
    content_type = "application/json; charset=utf-8"

    def render(self, ctx: RenderContext) -> RenderOutput:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "report_id": ctx.report_id,
            "request_id": ctx.request.request_id,
            "session_id": ctx.request.session_id,
            "audience": ctx.audience,
            "template_id": ctx.template.template_id,
            "language": ctx.language,
            "generated_at": ctx.generated_at.isoformat(),
            "sections": list(ctx.sections),
            "title": ctx.copy.title,
        }

        if ctx.wants("subject"):
            payload["subject"] = {
                "buyer": ctx.buyer.model_dump(mode="json") if ctx.buyer else None,
                "product": ctx.product.model_dump(mode="json") if ctx.product else None,
            }
        if ctx.wants("intent") and ctx.intent is not None:
            payload["intent"] = ctx.intent.model_dump(mode="json")
        if ctx.wants("decision") and ctx.decision is not None:
            decision_data = ctx.decision.model_dump(mode="json")
            if not ctx.request.include_trace:
                decision_data.pop("matched_rules", None)
            payload["decision"] = decision_data
        if ctx.wants("llm") and ctx.llm is not None:
            payload["llm"] = ctx.llm.model_dump(mode="json")

        if ctx.request.metadata:
            payload["metadata"] = dict(ctx.request.metadata)

        content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return RenderOutput.from_text(content, self.content_type)
