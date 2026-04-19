"""HTML renderer.

Emits a self-contained, responsive HTML document using a small inline
CSS block. No external CDN dependencies — the output works offline and
is safe to send by email.
"""
from __future__ import annotations

import html
from typing import List

from report.renderers.base import BaseRenderer, RenderContext, RenderOutput


_INLINE_CSS = """
:root { --fg: #111; --muted: #555; --accent: #2563eb; --border: #e5e7eb; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--fg); margin: 0; padding: 24px; background: #fafafa;
}
.wrap { max-width: 880px; margin: 0 auto; background: #fff;
        padding: 32px; border: 1px solid var(--border); border-radius: 12px; }
h1 { margin: 0 0 8px; font-size: 24px; }
h2 { margin-top: 28px; font-size: 18px; border-bottom: 1px solid var(--border);
     padding-bottom: 6px; }
dl { display: grid; grid-template-columns: 170px 1fr; gap: 4px 12px; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
         font-size: 12px; background: #eef2ff; color: var(--accent); }
.badge.warn { background: #fef3c7; color: #92400e; }
.badge.fail { background: #fee2e2; color: #b91c1c; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--muted); }
.llm { border-left: 3px solid var(--accent); padding: 8px 12px; color: var(--muted);
       background: #f8fafc; border-radius: 4px; white-space: pre-wrap; }
ol { padding-left: 20px; }
footer { margin-top: 32px; color: var(--muted); font-size: 13px; }
"""


class HtmlRenderer(BaseRenderer):
    format = "html"
    content_type = "text/html; charset=utf-8"

    def render(self, ctx: RenderContext) -> RenderOutput:
        body: List[str] = []
        body.append("<div class='wrap'>")

        # ---- header ----
        body.append(f"<h1>{self._esc(ctx.copy.title)}</h1>")
        body.append(
            f"<p><strong>{ctx.copy.report_id}:</strong> "
            f"<code>{self._esc(ctx.report_id)}</code> &middot; "
            f"<strong>{ctx.copy.generated_at}:</strong> "
            f"{ctx.generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}</p>"
        )

        if ctx.wants("subject"):
            self._subject(ctx, body)
        if ctx.wants("intent") and ctx.intent is not None:
            self._intent(ctx, body)
        if ctx.wants("decision"):
            self._decision(ctx, body)
        if ctx.wants("rationale") and ctx.decision is not None and ctx.decision.rationale:
            body.append(f"<h2>{self._esc(ctx.copy.section_rationale)}</h2>")
            body.append(f"<p>{self._esc(ctx.decision.rationale)}</p>")
        if ctx.wants("trace"):
            self._trace(ctx, body)
        if ctx.wants("llm") and ctx.llm is not None:
            self._llm(ctx, body)
        if ctx.wants("next_steps") and ctx.decision is not None and ctx.decision.next_steps:
            body.append(f"<h2>{self._esc(ctx.copy.section_next_steps)}</h2><ol>")
            for step in ctx.decision.next_steps:
                body.append(f"<li>{self._esc(step)}</li>")
            body.append("</ol>")
        if ctx.wants("closing") and ctx.copy.closing_customer:
            body.append(f"<footer>{self._esc(ctx.copy.closing_customer)}</footer>")

        body.append("</div>")

        document = (
            "<!DOCTYPE html>\n"
            f"<html lang=\"{self._esc(ctx.language)}\">\n"
            "<head>\n"
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{self._esc(ctx.copy.title)} — {self._esc(ctx.report_id)}</title>\n"
            f"<style>{_INLINE_CSS}</style>\n"
            "</head>\n<body>\n"
            + "\n".join(body)
            + "\n</body>\n</html>\n"
        )
        return RenderOutput.from_text(document, self.content_type)

    # ---- sections ----

    def _subject(self, ctx: RenderContext, body: List[str]) -> None:
        body.append(f"<h2>{self._esc(ctx.copy.section_subject)}</h2>")
        if ctx.buyer is None and ctx.product is None:
            body.append(f"<p><em>{self._esc(ctx.copy.no_data)}</em></p>")
            return
        body.append("<dl>")
        if ctx.buyer is not None:
            body.append(
                f"<dt>{self._esc(ctx.copy.field_user)}</dt>"
                f"<dd>{self._esc(ctx.buyer.user_id or '-')}</dd>"
            )
            body.append(
                f"<dt>{self._esc(ctx.copy.field_user_type)}</dt>"
                f"<dd>{self._esc(ctx.buyer.user_type)}</dd>"
            )
            if ctx.buyer.budget is not None:
                body.append(
                    f"<dt>{self._esc(ctx.copy.field_budget)}</dt>"
                    f"<dd>{self.format_amount(ctx.buyer.budget)}</dd>"
                )
        if ctx.product is not None:
            label = ctx.product.name or ctx.product.sku or ctx.product.product_id or "-"
            body.append(
                f"<dt>{self._esc(ctx.copy.field_product)}</dt>"
                f"<dd>{self._esc(label)}</dd>"
            )
            if ctx.product.total_amount is not None:
                body.append(
                    f"<dt>{self._esc(ctx.copy.field_amount)}</dt>"
                    f"<dd>{self.format_amount(ctx.product.total_amount)}</dd>"
                )
        body.append("</dl>")

    def _intent(self, ctx: RenderContext, body: List[str]) -> None:
        assert ctx.intent is not None
        label = ctx.intent.label or ctx.intent.intent
        body.append(f"<h2>{self._esc(ctx.copy.section_intent)}</h2>")
        body.append(
            f"<p><span class='badge'>{self._esc(label)}</span> "
            f"<code>{self._esc(ctx.intent.intent)}</code> — "
            f"{self._esc(ctx.copy.field_confidence)}: "
            f"{self.format_percent(ctx.intent.confidence)}</p>"
        )

    def _decision(self, ctx: RenderContext, body: List[str]) -> None:
        body.append(f"<h2>{self._esc(ctx.copy.section_decision)}</h2>")
        if ctx.decision is None:
            body.append(f"<p><em>{self._esc(ctx.copy.no_data)}</em></p>")
            return
        d = ctx.decision
        badge_class = "fail" if d.requires_review else ("warn" if not d.confident else "")
        label = d.label or d.action
        body.append(
            f"<p><span class='badge {badge_class}'>{self._esc(label)}</span> "
            f"<code>{self._esc(d.action)}</code></p>"
        )
        body.append("<dl>")
        body.append(
            f"<dt>{self._esc(ctx.copy.field_confidence)}</dt>"
            f"<dd>{self.format_percent(d.confidence)}</dd>"
        )
        body.append(
            f"<dt>{self._esc(ctx.copy.field_risk)}</dt>"
            f"<dd>{self.format_percent(d.risk_score)}</dd>"
        )
        body.append(
            f"<dt>{self._esc(ctx.copy.field_requires_review)}</dt>"
            f"<dd>{'Yes' if d.requires_review else '—'}</dd>"
        )
        if d.strategy:
            body.append(
                f"<dt>{self._esc(ctx.copy.field_strategy)}</dt>"
                f"<dd><code>{self._esc(d.strategy)}</code></dd>"
            )
        body.append("</dl>")

        if d.candidates:
            body.append(
                f"<table><thead><tr><th>{self._esc(ctx.copy.field_action)}</th>"
                f"<th>{self._esc(ctx.copy.field_confidence)}</th></tr></thead><tbody>"
            )
            for c in d.candidates[:5]:
                lbl = c.label or c.action
                body.append(
                    f"<tr><td>{self._esc(lbl)}</td>"
                    f"<td>{self.format_percent(c.confidence)}</td></tr>"
                )
            body.append("</tbody></table>")

    def _trace(self, ctx: RenderContext, body: List[str]) -> None:
        if not ctx.request.include_trace:
            return
        if ctx.decision is None or not ctx.decision.matched_rules:
            return
        body.append(f"<h2>{self._esc(ctx.copy.section_trace)}</h2><ul>")
        for entry in ctx.decision.matched_rules[:20]:
            name = entry.get("name") or entry.get("rule") or "rule"
            action = entry.get("action", "-")
            body.append(
                f"<li><code>{self._esc(str(name))}</code> → "
                f"<code>{self._esc(str(action))}</code></li>"
            )
        body.append("</ul>")

    def _llm(self, ctx: RenderContext, body: List[str]) -> None:
        assert ctx.llm is not None
        body.append(f"<h2>{self._esc(ctx.copy.section_llm)}</h2>")
        meta = f"{ctx.llm.provider or '-'} / {ctx.llm.model_id or '-'}"
        body.append(
            f"<p><small>{self._esc(meta)} · "
            f"{self._esc(ctx.copy.field_confidence)}: "
            f"{self.format_percent(ctx.llm.confidence)}</small></p>"
        )
        body.append(f"<div class='llm'>{self._esc(ctx.llm.content)}</div>")

    @staticmethod
    def _esc(text: str) -> str:
        if text is None:
            return "-"
        return html.escape(str(text), quote=True)
