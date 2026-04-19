"""Rule-based decision strategy.

This is the deterministic, always-available baseline. It encodes the
procurement-policy decision tree as a small DSL of weighted rules, each
of which votes for an action with a numeric score. The highest-scoring
action wins.

Design priorities
-----------------
* **Predictable**: O(rules) per request, well below 1 ms in practice.
* **Auditable**: every rule that fires is recorded in
  ``StrategyOutcome.metadata['matched_rules']`` so downstream tooling can
  explain *why* the engine made a decision.
* **Configurable risk**: rules emit both a vote and a risk delta; the
  cumulative risk is clamped to [0, 1] so the orchestrator can decide
  whether to escalate to a human reviewer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from decision.catalog import (
    ACTION_REGISTRY,
    FALLBACK_ACTION,
    actions_for_audience,
    next_steps_of,
)
from decision.config import DecisionSettings
from decision.features import FeatureVector
from decision.strategies.base import BaseStrategy, StrategyOutcome

Predicate = Callable[[FeatureVector], bool]


@dataclass(frozen=True)
class Rule:
    name: str
    when: Predicate
    action: str
    score: float           # vote magnitude in [0, 1]
    risk_delta: float = 0.0
    rationale_zh: str = ""
    rationale_en: str = ""

    def matches(self, features: FeatureVector) -> bool:
        try:
            return bool(self.when(features))
        except Exception:  # pragma: no cover - defensive
            return False


def _intent_in(*ids: str) -> Predicate:
    bag = frozenset(ids)
    return lambda f: f.intent in bag


def _is_b2b() -> Predicate:
    return lambda f: f.user_type == "b2b"


def _is_b2c() -> Predicate:
    return lambda f: f.user_type == "b2c"


def _and(*preds: Predicate) -> Predicate:
    return lambda f: all(p(f) for p in preds)


def _or(*preds: Predicate) -> Predicate:
    return lambda f: any(p(f) for p in preds)


def _build_default_rules(settings: DecisionSettings) -> List[Rule]:
    """The default policy tree.

    Each rule below answers a specific procurement question. Rules are
    ordered roughly by specificity — but ordering does NOT matter for
    correctness because we always sum the votes per action and pick the
    winner.
    """
    floor = settings.max_auto_approve_amount
    ceiling = settings.require_human_review_above

    return [
        # ---------- cancellation / returns ----------
        Rule(
            name="explicit_cancel",
            when=_intent_in("cancel_order"),
            action="cancel_request",
            score=0.95,
            risk_delta=0.10,
            rationale_zh="用户明确请求取消订单",
            rationale_en="User explicitly asked to cancel the order",
        ),
        Rule(
            name="explicit_return",
            when=_intent_in("return_refund"),
            action="initiate_return",
            score=0.95,
            risk_delta=0.05,
            rationale_zh="用户请求退换货/退款",
            rationale_en="User requested return / refund",
        ),
        Rule(
            name="complaint_escalate",
            when=_intent_in("complaint", "contact_support"),
            action="escalate_to_human",
            score=0.85,
            risk_delta=0.20,
            rationale_zh="用户表达不满，需人工介入",
            rationale_en="User complaint detected — escalate to a human",
        ),
        # ---------- discovery ----------
        Rule(
            name="search_then_recommend",
            when=_intent_in("search_product", "get_recommendation"),
            action="recommend_alternatives",
            score=0.70,
            rationale_zh="用户在浏览/寻找商品",
            rationale_en="User is exploring or looking for products",
        ),
        Rule(
            name="compare_alts",
            when=_intent_in("compare_products"),
            action="compare_alternatives",
            score=0.85,
            rationale_zh="用户希望对比不同商品",
            rationale_en="User wants to compare products",
        ),
        Rule(
            name="show_details",
            when=_intent_in("product_info", "view_reviews"),
            action="show_product_details",
            score=0.80,
            rationale_zh="用户咨询商品详情/评价",
            rationale_en="User is asking for product details / reviews",
        ),
        # ---------- inventory / stock ----------
        Rule(
            name="check_stock_intent",
            when=_intent_in("check_stock"),
            action="check_inventory",
            score=0.85,
            rationale_zh="用户询问库存/交期",
            rationale_en="User is asking about stock or lead time",
        ),
        Rule(
            name="oos_recommend_alts",
            when=lambda f: f.in_stock == 0.0 and f.has_alternatives == 1.0,
            action="recommend_alternatives",
            score=0.65,
            rationale_zh="目标商品缺货，推荐替代品",
            rationale_en="Target SKU is out of stock — recommend alternatives",
        ),
        # ---------- B2C transactional ----------
        Rule(
            name="b2c_place_order",
            when=_and(
                _is_b2c(),
                _intent_in("place_order"),
                # Skip when the buyer is clearly over-budget — the
                # over-budget rule below should win in that case.
                lambda f: not (
                    f.has_budget == 1.0 and f.price_to_budget_ratio >= 1.5
                ),
            ),
            action="proceed_to_checkout",
            score=0.85,
            risk_delta=0.10,
            rationale_zh="C 端用户准备下单结账",
            rationale_en="B2C buyer ready to check out",
        ),
        Rule(
            name="b2c_search_in_stock_add_cart",
            when=_and(
                _is_b2c(),
                _intent_in("inquire_price", "search_product"),
                lambda f: f.in_stock == 1.0,
            ),
            action="add_to_cart",
            score=0.55,
            rationale_zh="C 端用户在询价/搜索且有现货",
            rationale_en="B2C buyer asking price on an in-stock SKU",
        ),
        Rule(
            name="wait_for_promo",
            when=_and(
                _is_b2c(),
                lambda f: f.promotion_active == 0.0
                and f.urgency == "low"
                and f.has_product == 1.0,
            ),
            action="wait_for_promotion",
            score=0.45,
            rationale_zh="非紧急且无促销，建议等待促销",
            rationale_en="Low urgency and no active promo — recommend waiting",
        ),
        # ---------- B2B transactional ----------
        Rule(
            name="b2b_request_quote",
            when=_and(_is_b2b(), _intent_in("request_quote", "inquire_price")),
            action="request_quote",
            score=0.85,
            risk_delta=0.05,
            rationale_zh="B 端采购员请求报价",
            rationale_en="B2B buyer is requesting a formal quote",
        ),
        Rule(
            name="b2b_bulk_order",
            when=_and(_is_b2b(), _intent_in("bulk_order", "place_order")),
            action="bulk_order",
            score=0.80,
            risk_delta=0.20,
            rationale_zh="B 端采购员发起批量采购",
            rationale_en="B2B buyer initiating a bulk order",
        ),
        Rule(
            name="b2b_negotiate",
            when=_and(_is_b2b(), _intent_in("negotiate_terms")),
            action="negotiate_terms",
            score=0.85,
            risk_delta=0.10,
            rationale_zh="B 端采购员希望谈判商务条款",
            rationale_en="B2B buyer wants to negotiate commercial terms",
        ),
        Rule(
            name="b2b_supplier_info",
            when=_and(_is_b2b(), _intent_in("supplier_info")),
            action="contact_supplier",
            score=0.75,
            rationale_zh="B 端采购员需要供应商信息",
            rationale_en="B2B buyer needs supplier information",
        ),
        # ---------- approval ceilings ----------
        Rule(
            name="auto_approve_small_purchase",
            when=lambda f: f.total_amount > 0
            and f.total_amount <= floor
            and f.intent in {"place_order", "bulk_order"}
            and f.in_stock == 1.0
            and f.buyer_risk_score < 0.5,
            action="approve_purchase",
            score=0.75,
            risk_delta=0.15,
            rationale_zh=f"采购金额 <= 自动审批上限 {floor:,.0f}",
            rationale_en=f"Purchase amount within auto-approve ceiling {floor:,.0f}",
        ),
        Rule(
            name="escalate_high_value",
            when=lambda f: f.total_amount >= ceiling,
            action="escalate_to_human",
            score=0.90,
            risk_delta=0.30,
            rationale_zh=f"采购金额 >= 审核门槛 {ceiling:,.0f}，需人工审核",
            rationale_en=f"Purchase amount >= review threshold {ceiling:,.0f} — human review required",
        ),
        # ---------- affordability ----------
        Rule(
            name="reject_over_budget",
            when=lambda f: f.has_budget == 1.0
            and f.total_amount > 0
            and f.price_to_budget_ratio >= 1.5,
            action="reject_purchase",
            score=0.80,
            risk_delta=0.25,
            rationale_zh="超出预算 50% 以上，建议拒绝",
            rationale_en="Total amount exceeds budget by 50% or more",
        ),
        Rule(
            name="defer_marginal_budget",
            when=lambda f: f.has_budget == 1.0
            and 1.0 < f.price_to_budget_ratio < 1.5,
            action="defer_decision",
            score=0.55,
            risk_delta=0.10,
            rationale_zh="略超预算，建议暂缓并寻找替代",
            rationale_en="Slightly over budget — defer and search alternatives",
        ),
        # ---------- buyer-risk floor ----------
        Rule(
            name="high_risk_buyer_review",
            when=lambda f: f.buyer_risk_score >= 0.7
            and f.intent in {"place_order", "bulk_order", "request_quote"},
            action="escalate_to_human",
            score=0.70,
            risk_delta=0.30,
            rationale_zh="买家风险评分较高，建议人工复核",
            rationale_en="Buyer risk score is high — escalate to a human reviewer",
        ),
        Rule(
            name="poor_supplier_warning",
            when=lambda f: f.supplier_rating > 0
            and f.supplier_rating < 2.5
            and f.user_type == "b2b",
            action="contact_supplier",
            score=0.50,
            risk_delta=0.20,
            rationale_zh="供应商评分较低，建议先与供应商沟通",
            rationale_en="Supplier rating is low — contact supplier first",
        ),
        # ---------- conversational catch-alls ----------
        Rule(
            name="greeting",
            when=_intent_in("greeting"),
            action="no_action",
            score=0.55,
            rationale_zh="用户问候，无需采取操作",
            rationale_en="User greeting — no action required",
        ),
        Rule(
            name="farewell",
            when=_intent_in("farewell"),
            action="no_action",
            score=0.55,
            rationale_zh="用户告别，结束对话",
            rationale_en="User farewell — end of conversation",
        ),
        Rule(
            name="unclear_request_more_info",
            when=lambda f: f.intent_confidence > 0
            and f.intent_confidence < 0.4,
            action="request_more_info",
            score=0.55,
            rationale_zh="意图置信度较低，建议追加澄清",
            rationale_en="Intent confidence is low — request clarification",
        ),
    ]


class RuleBasedStrategy(BaseStrategy):
    """The deterministic baseline strategy."""

    name = "rules"

    def __init__(
        self,
        *,
        settings: DecisionSettings,
        rules: Optional[List[Rule]] = None,
    ) -> None:
        self.settings = settings
        self.rules: List[Rule] = list(rules) if rules is not None else _build_default_rules(settings)

    # ------------------------------------------------------------------

    def evaluate(self, features: FeatureVector) -> StrategyOutcome:
        scored: Dict[str, float] = {}
        risks: List[float] = []
        matched: List[Dict[str, str]] = []
        rationales_zh: List[str] = []
        rationales_en: List[str] = []
        is_zh = features.language.lower().startswith("zh")

        for rule in self.rules:
            if not rule.matches(features):
                continue
            scored[rule.action] = scored.get(rule.action, 0.0) + rule.score
            risks.append(rule.risk_delta)
            matched.append({"name": rule.name, "action": rule.action})
            if is_zh and rule.rationale_zh:
                rationales_zh.append(rule.rationale_zh)
            elif rule.rationale_en:
                rationales_en.append(rule.rationale_en)

        if not scored:
            scored = self._fallback_scores(features)
            matched.append({"name": "fallback", "action": FALLBACK_ACTION})
            if is_zh:
                rationales_zh.append("未匹配到任何规则，使用安全兜底")
            else:
                rationales_en.append("No rule matched — using safe fallback")

        # Cap each accumulated vote at 1.0 — keeps the contract that
        # confidence is bounded by [0, 1] without forcing the top score
        # to always equal 1.0 (which would defeat ``confident_threshold``).
        ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        normalised: List[Tuple[str, float]] = []
        for action_id, raw in ordered:
            if action_id not in ACTION_REGISTRY:
                continue
            normalised.append((action_id, round(min(1.0, raw), 4)))
        if not normalised:
            normalised = [(FALLBACK_ACTION, 0.0)]

        # Audience-aware filtering: discard actions whose audience does
        # not include the buyer's user_type, but keep ``no_action`` and
        # ``escalate_to_human`` as universal fallbacks.
        allowed = {a.id for a in actions_for_audience(features.user_type)}
        allowed.update({FALLBACK_ACTION, "escalate_to_human", "request_more_info"})
        filtered = [pair for pair in normalised if pair[0] in allowed] or [
            (FALLBACK_ACTION, 0.0)
        ]

        risk_score = min(1.0, sum(risks)) if risks else 0.0

        rationale_text = "; ".join(rationales_zh if is_zh else rationales_en)
        if not rationale_text:
            rationale_text = "no specific rule fired"

        next_steps = next_steps_of(filtered[0][0], features.language)

        return StrategyOutcome(
            candidates=filtered,
            rationale=rationale_text,
            strategy=self.name,
            risk_score=round(risk_score, 4),
            metadata={"matched_rules": matched, "rule_count": len(self.rules)},
            next_steps=next_steps,
        )

    # ------------------------------------------------------------------

    def _fallback_scores(self, features: FeatureVector) -> Dict[str, float]:
        # Pick something sensible based on intent confidence — if we have
        # *some* signal, request more info; otherwise no_action.
        if features.intent_confidence < 0.3:
            return {"no_action": 0.5, "request_more_info": 0.4}
        return {"request_more_info": 0.5, "no_action": 0.3}
