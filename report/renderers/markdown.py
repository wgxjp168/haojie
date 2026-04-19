"""Markdown renderer.

Uses a pure-Python string-assembly approach (no Jinja2 dependency) so
the service works in minimal environments. Escaping pipes / newlines in
user content is handled to keep table structure intact.
"""
from __future__ import annotations

from typing import List

from report.renderers.base import BaseRenderer, RenderContext, RenderOutput


class MarkdownRenderer(BaseRenderer):
    format = "markdown"
    content_type = "text/markdown; charset=utf-8"

    def render(self, ctx: RenderContext) -> RenderOutput:
        lines: List[str] = []

        # -------- front matter --------
        lines.append(f"# {ctx.copy.title}")
        lines.append("")
        lines.append(
            f"- **{ctx.copy.report_id}**: `{ctx.report_id}`  "
        )
        lines.append(
            f"- **{ctx.copy.generated_at}**: "
            f"{ctx.generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}  "
        )
        lines.append("")

        # -------- subject --------
        if ctx.wants("subject"):
            self._subject_block(ctx, lines)

        # -------- intent --------
        if ctx.wants("intent") and ctx.intent is not None:
            self._intent_block(ctx, lines)

        # -------- decision --------
        if ctx.wants("decision"):
            self._decision_block(ctx, lines)

        # -------- rationale --------
        if ctx.wants("rationale") and ctx.decision is not None:
            self._rationale_block(ctx, lines)

        # -------- trace --------
        if ctx.wants("trace"):
            self._trace_block(ctx, lines)

        # -------- llm --------
        if ctx.wants("llm") and ctx.llm is not None:
            self._llm_block(ctx, lines)

        # -------- next steps --------
        if ctx.wants("next_steps"):
            self._next_steps_block(ctx, lines)

        # -------- closing --------
        if ctx.wants("closing"):
            if ctx.copy.closing_customer:
                lines.append("---")
                lines.append("")
                lines.append(f"_{ctx.copy.closing_customer}_")
                lines.append("")

        text = "\n".join(lines).rstrip() + "\n"
        return RenderOutput.from_text(text, self.content_type)

    # -------- individual blocks --------

    def _subject_block(self, ctx: RenderContext, lines: List[str]) -> None:
        lines.append(f"## {ctx.copy.section_subject}")
        lines.append("")
        if ctx.buyer is None and ctx.product is None:
            lines.append(f"_{ctx.copy.no_data}_")
            lines.append("")
            return

        if ctx.buyer is not None:
            lines.append(
                f"- **{ctx.copy.field_user}**: "
                f"{self._esc(ctx.buyer.user_id or '-')}  "
            )
            lines.append(
                f"- **{ctx.copy.field_user_type}**: "
                f"{self._esc(ctx.buyer.user_type)}  "
            )
            if ctx.buyer.budget is not None:
                lines.append(
                    f"- **{ctx.copy.field_budget}**: "
                    f"{self.format_amount(ctx.buyer.budget)}  "
                )

        if ctx.product is not None:
            label = ctx.product.name or ctx.product.sku or ctx.product.product_id or "-"
            lines.append(f"- **{ctx.copy.field_product}**: {self._esc(label)}  ")
            if ctx.product.total_amount is not None:
                lines.append(
                    f"- **{ctx.copy.field_amount}**: "
                    f"{self.format_amount(ctx.product.total_amount)}  "
                )

        lines.append("")

    def _intent_block(self, ctx: RenderContext, lines: List[str]) -> None:
        assert ctx.intent is not None
        lines.append(f"## {ctx.copy.section_intent}")
        lines.append("")
        label = ctx.intent.label or ctx.intent.intent
        lines.append(
            f"- **{self._esc(label)}** (`{ctx.intent.intent}`) — "
            f"{ctx.copy.field_confidence}: "
            f"{self.format_percent(ctx.intent.confidence)}"
        )
        if ctx.intent.backend:
            lines.append(f"- backend: `{ctx.intent.backend}`")
        lines.append("")

    def _decision_block(self, ctx: RenderContext, lines: List[str]) -> None:
        lines.append(f"## {ctx.copy.section_decision}")
        lines.append("")
        if ctx.decision is None:
            lines.append(f"_{ctx.copy.no_data}_")
            lines.append("")
            return

        d = ctx.decision
        label = d.label or d.action
        lines.append(
            f"- **{ctx.copy.field_action}**: {self._esc(label)} (`{d.action}`)"
        )
        lines.append(
            f"- **{ctx.copy.field_confidence}**: "
            f"{self.format_percent(d.confidence)}"
        )
        lines.append(
            f"- **{ctx.copy.field_risk}**: {self.format_percent(d.risk_score)}"
        )
        lines.append(
            f"- **{ctx.copy.field_requires_review}**: "
            f"{'✅' if d.requires_review else '—'}"
        )
        if d.strategy:
            lines.append(f"- **{ctx.copy.field_strategy}**: `{d.strategy}`")
        lines.append("")

        if d.candidates:
            lines.append(f"| {ctx.copy.field_action} | {ctx.copy.field_confidence} |")
            lines.append("| --- | --- |")
            for c in d.candidates[:5]:
                label = c.label or c.action
                lines.append(
                    f"| {self._esc(label)} "
                    f"| {self.format_percent(c.confidence)} |"
                )
            lines.append("")

    def _rationale_block(self, ctx: RenderContext, lines: List[str]) -> None:
        assert ctx.decision is not None
        if not ctx.decision.rationale:
            return
        lines.append(f"## {ctx.copy.section_rationale}")
        lines.append("")
        lines.append(self._esc(ctx.decision.rationale))
        lines.append("")

    def _trace_block(self, ctx: RenderContext, lines: List[str]) -> None:
        if not ctx.request.include_trace:
            return
        if ctx.decision is None or not ctx.decision.matched_rules:
            return
        lines.append(f"## {ctx.copy.section_trace}")
        lines.append("")
        for entry in ctx.decision.matched_rules[:20]:
            name = entry.get("name") or entry.get("rule") or "rule"
            action = entry.get("action", "-")
            lines.append(f"- `{self._esc(str(name))}` → `{self._esc(str(action))}`")
        lines.append("")

    def _llm_block(self, ctx: RenderContext, lines: List[str]) -> None:
        assert ctx.llm is not None
        lines.append(f"## {ctx.copy.section_llm}")
        lines.append("")
        if ctx.llm.provider or ctx.llm.model_id:
            lines.append(
                f"> {ctx.llm.provider or '-'} / {ctx.llm.model_id or '-'} "
                f"({ctx.copy.field_confidence}: "
                f"{self.format_percent(ctx.llm.confidence)})"
            )
            lines.append("")
        for paragraph in ctx.llm.content.splitlines() or [ctx.llm.content]:
            lines.append(paragraph)
        lines.append("")

    def _next_steps_block(self, ctx: RenderContext, lines: List[str]) -> None:
        if ctx.decision is None or not ctx.decision.next_steps:
            return
        lines.append(f"## {ctx.copy.section_next_steps}")
        lines.append("")
        for step in ctx.decision.next_steps:
            lines.append(f"1. {self._esc(step)}")
        lines.append("")

    @staticmethod
    def _esc(text: str) -> str:
        """Escape Markdown pipe / newline chars inside table / inline content."""
        if not text:
            return "-"
        return (
            str(text)
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\n", " ")
            .replace("\r", " ")
        )
