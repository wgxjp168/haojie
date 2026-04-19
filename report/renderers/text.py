"""Plain-text renderer.

Ideal for log sinks, email bodies, or CLI output where Markdown/HTML
would add noise. Deterministic, wrapped to 100 columns.
"""
from __future__ import annotations

import textwrap
from typing import List

from report.renderers.base import BaseRenderer, RenderContext, RenderOutput


_WRAP_COLS = 100
_DIVIDER = "-" * _WRAP_COLS


class TextRenderer(BaseRenderer):
    format = "text"
    content_type = "text/plain; charset=utf-8"

    def render(self, ctx: RenderContext) -> RenderOutput:
        lines: List[str] = []
        title = f"{ctx.copy.title} — {ctx.report_id}"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(
            f"{ctx.copy.generated_at}: "
            f"{ctx.generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

        if ctx.wants("subject"):
            self._subject(ctx, lines)
        if ctx.wants("intent") and ctx.intent is not None:
            self._intent(ctx, lines)
        if ctx.wants("decision"):
            self._decision(ctx, lines)
        if ctx.wants("rationale") and ctx.decision and ctx.decision.rationale:
            self._heading(lines, ctx.copy.section_rationale)
            lines.extend(textwrap.wrap(ctx.decision.rationale, width=_WRAP_COLS))
        if ctx.wants("trace") and ctx.request.include_trace:
            self._trace(ctx, lines)
        if ctx.wants("llm") and ctx.llm is not None:
            self._llm(ctx, lines)
        if ctx.wants("next_steps") and ctx.decision and ctx.decision.next_steps:
            self._heading(lines, ctx.copy.section_next_steps)
            for i, step in enumerate(ctx.decision.next_steps, start=1):
                for j, wrapped in enumerate(
                    textwrap.wrap(step, width=_WRAP_COLS - 4)
                ):
                    prefix = f"  {i}. " if j == 0 else "     "
                    lines.append(f"{prefix}{wrapped}")
        if ctx.wants("closing") and ctx.copy.closing_customer:
            lines.append("")
            lines.append(_DIVIDER)
            lines.extend(textwrap.wrap(ctx.copy.closing_customer, width=_WRAP_COLS))

        body = "\n".join(lines).rstrip() + "\n"
        return RenderOutput.from_text(body, self.content_type)

    # ---- helpers ----

    def _heading(self, lines: List[str], heading: str) -> None:
        lines.append("")
        lines.append(heading)
        lines.append("-" * len(heading))

    def _subject(self, ctx: RenderContext, lines: List[str]) -> None:
        self._heading(lines, ctx.copy.section_subject)
        if ctx.buyer is None and ctx.product is None:
            lines.append(f"  {ctx.copy.no_data}")
            return
        if ctx.buyer is not None:
            lines.append(f"  {ctx.copy.field_user}: {ctx.buyer.user_id or '-'}")
            lines.append(f"  {ctx.copy.field_user_type}: {ctx.buyer.user_type}")
            if ctx.buyer.budget is not None:
                lines.append(
                    f"  {ctx.copy.field_budget}: {self.format_amount(ctx.buyer.budget)}"
                )
        if ctx.product is not None:
            label = ctx.product.name or ctx.product.sku or ctx.product.product_id or "-"
            lines.append(f"  {ctx.copy.field_product}: {label}")
            if ctx.product.total_amount is not None:
                lines.append(
                    f"  {ctx.copy.field_amount}: "
                    f"{self.format_amount(ctx.product.total_amount)}"
                )

    def _intent(self, ctx: RenderContext, lines: List[str]) -> None:
        assert ctx.intent is not None
        self._heading(lines, ctx.copy.section_intent)
        label = ctx.intent.label or ctx.intent.intent
        lines.append(
            f"  {label} ({ctx.intent.intent}) — "
            f"{ctx.copy.field_confidence}: "
            f"{self.format_percent(ctx.intent.confidence)}"
        )

    def _decision(self, ctx: RenderContext, lines: List[str]) -> None:
        self._heading(lines, ctx.copy.section_decision)
        if ctx.decision is None:
            lines.append(f"  {ctx.copy.no_data}")
            return
        d = ctx.decision
        label = d.label or d.action
        lines.append(f"  {ctx.copy.field_action}: {label} ({d.action})")
        lines.append(
            f"  {ctx.copy.field_confidence}: {self.format_percent(d.confidence)}"
        )
        lines.append(
            f"  {ctx.copy.field_risk}: {self.format_percent(d.risk_score)}"
        )
        lines.append(
            f"  {ctx.copy.field_requires_review}: "
            f"{'yes' if d.requires_review else 'no'}"
        )
        if d.strategy:
            lines.append(f"  {ctx.copy.field_strategy}: {d.strategy}")
        if d.candidates:
            for c in d.candidates[:5]:
                lbl = c.label or c.action
                lines.append(
                    f"    - {lbl} ({c.action}): "
                    f"{self.format_percent(c.confidence)}"
                )

    def _trace(self, ctx: RenderContext, lines: List[str]) -> None:
        if ctx.decision is None or not ctx.decision.matched_rules:
            return
        self._heading(lines, ctx.copy.section_trace)
        for entry in ctx.decision.matched_rules[:20]:
            name = entry.get("name") or entry.get("rule") or "rule"
            action = entry.get("action", "-")
            lines.append(f"  - {name} -> {action}")

    def _llm(self, ctx: RenderContext, lines: List[str]) -> None:
        assert ctx.llm is not None
        self._heading(lines, ctx.copy.section_llm)
        if ctx.llm.provider or ctx.llm.model_id:
            lines.append(
                f"  [{ctx.llm.provider or '-'}/{ctx.llm.model_id or '-'}]"
            )
        for paragraph in ctx.llm.content.split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=_WRAP_COLS) or [""])
