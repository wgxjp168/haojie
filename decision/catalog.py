"""Decision-action catalog (the procurement-decision taxonomy).

Every action emitted by the engine is a member of this catalog. The split
between B2B and B2C echoes ``intent.taxonomy`` so that downstream
consumers (Part 3.4 report generator, Part 4 channel routing) can rely on
a stable contract. Actions with audience ``("any",)`` apply to either
buyer type.

Action design notes
-------------------
* Each action has a ``severity`` (``low``, ``medium``, ``high``) that
  controls the *requires-review* gate when combined with
  ``risk_review_threshold``.
* Each action has a ``terminal`` flag — terminal actions close the
  current decision loop (e.g. ``approve_purchase``) while non-terminal
  ones expect further interaction (e.g. ``request_more_info``).
* Each action has a default ``next_steps_zh`` / ``next_steps_en`` list
  that the engine emits as a starting point; rule/ML strategies may
  override these on a per-decision basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ActionDefinition:
    id: str
    label_zh: str
    label_en: str
    description: str
    audiences: Tuple[str, ...]
    severity: str  # low | medium | high
    terminal: bool
    next_steps_zh: Tuple[str, ...] = ()
    next_steps_en: Tuple[str, ...] = ()

    @property
    def is_b2b(self) -> bool:
        return "b2b" in self.audiences or "any" in self.audiences

    @property
    def is_b2c(self) -> bool:
        return "b2c" in self.audiences or "any" in self.audiences


_RAW: List[ActionDefinition] = [
    # --- discovery / information gathering ---
    ActionDefinition(
        id="request_more_info",
        label_zh="补充信息",
        label_en="Request more information",
        description="The engine cannot decide yet — ask the buyer for more facts.",
        audiences=("any",),
        severity="low",
        terminal=False,
        next_steps_zh=("提出澄清问题", "等待用户补充关键信息"),
        next_steps_en=("Ask clarifying question", "Wait for additional details"),
    ),
    ActionDefinition(
        id="show_product_details",
        label_zh="展示商品详情",
        label_en="Show product details",
        description="Surface specs, materials, certifications etc. to the buyer.",
        audiences=("any",),
        severity="low",
        terminal=False,
        next_steps_zh=("拉取商品详情卡片", "高亮关键参数"),
        next_steps_en=("Render product detail card", "Highlight key specs"),
    ),
    ActionDefinition(
        id="recommend_alternatives",
        label_zh="推荐替代品",
        label_en="Recommend alternatives",
        description="Show similar / cheaper / faster alternatives.",
        audiences=("any",),
        severity="low",
        terminal=False,
        next_steps_zh=("调用推荐服务", "返回 3-5 个替代候选"),
        next_steps_en=("Call recommender", "Return 3-5 alternatives"),
    ),
    ActionDefinition(
        id="compare_alternatives",
        label_zh="对比替代品",
        label_en="Compare alternatives",
        description="Render a side-by-side comparison of two or more SKUs.",
        audiences=("any",),
        severity="low",
        terminal=False,
        next_steps_zh=("生成对比表", "突出价格/规格差异"),
        next_steps_en=("Generate comparison table", "Highlight spec/price gaps"),
    ),
    ActionDefinition(
        id="check_inventory",
        label_zh="核查库存",
        label_en="Check inventory",
        description="Verify stock / lead time before recommending purchase.",
        audiences=("any",),
        severity="low",
        terminal=False,
        next_steps_zh=("查询 ERP 库存", "返回交期"),
        next_steps_en=("Query ERP for stock", "Return ETA"),
    ),
    # --- transactional (B2C focused) ---
    ActionDefinition(
        id="add_to_cart",
        label_zh="加入购物车",
        label_en="Add to cart",
        description="Recommend adding the SKU to the shopping cart.",
        audiences=("b2c",),
        severity="low",
        terminal=False,
        next_steps_zh=("更新购物车", "提示促销"),
        next_steps_en=("Update cart", "Surface promotion"),
    ),
    ActionDefinition(
        id="proceed_to_checkout",
        label_zh="去结账",
        label_en="Proceed to checkout",
        description="The buyer is ready to checkout the cart.",
        audiences=("b2c",),
        severity="medium",
        terminal=True,
        next_steps_zh=("跳转结算页", "应用最优促销"),
        next_steps_en=("Open checkout", "Apply best promo"),
    ),
    ActionDefinition(
        id="wait_for_promotion",
        label_zh="等待促销",
        label_en="Wait for promotion",
        description="Recommend deferring purchase until an upcoming sale.",
        audiences=("b2c",),
        severity="low",
        terminal=False,
        next_steps_zh=("加入愿望单", "在促销开始时提醒用户"),
        next_steps_en=("Add to wishlist", "Notify when sale starts"),
    ),
    # --- transactional (B2B focused) ---
    ActionDefinition(
        id="request_quote",
        label_zh="请求报价",
        label_en="Request a formal quote",
        description="Submit a formal RFQ to the supplier.",
        audiences=("b2b",),
        severity="medium",
        terminal=False,
        next_steps_zh=("生成 RFQ", "发送给采购员审核"),
        next_steps_en=("Generate RFQ document", "Send to buyer for review"),
    ),
    ActionDefinition(
        id="negotiate_terms",
        label_zh="商务条款谈判",
        label_en="Negotiate terms",
        description="Open a negotiation thread on price / payment / delivery.",
        audiences=("b2b",),
        severity="medium",
        terminal=False,
        next_steps_zh=("起草谈判要点", "发起在线议价"),
        next_steps_en=("Draft negotiation memo", "Open vendor chat"),
    ),
    ActionDefinition(
        id="bulk_order",
        label_zh="发起批量采购",
        label_en="Initiate bulk order",
        description="Place a wholesale / container-load order.",
        audiences=("b2b",),
        severity="high",
        terminal=True,
        next_steps_zh=("生成采购订单", "通知财务和仓储"),
        next_steps_en=("Generate purchase order", "Notify finance & warehouse"),
    ),
    ActionDefinition(
        id="approve_purchase",
        label_zh="批准采购",
        label_en="Approve purchase",
        description="Auto-approve a purchase below the configured ceiling.",
        audiences=("any",),
        severity="medium",
        terminal=True,
        next_steps_zh=("写入采购系统", "推送给财务"),
        next_steps_en=("Persist to procurement system", "Push to finance"),
    ),
    ActionDefinition(
        id="reject_purchase",
        label_zh="拒绝采购",
        label_en="Reject purchase",
        description="Decline the purchase (over-budget, policy violation, …).",
        audiences=("any",),
        severity="high",
        terminal=True,
        next_steps_zh=("回复拒绝理由", "建议替代方案"),
        next_steps_en=("Send rejection rationale", "Suggest alternatives"),
    ),
    ActionDefinition(
        id="defer_decision",
        label_zh="暂缓决策",
        label_en="Defer decision",
        description="Postpone the decision; revisit later.",
        audiences=("any",),
        severity="low",
        terminal=False,
        next_steps_zh=("加入待办列表", "T+1 重新评估"),
        next_steps_en=("Add to follow-up queue", "Re-evaluate next day"),
    ),
    # --- support ---
    ActionDefinition(
        id="escalate_to_human",
        label_zh="转人工审核",
        label_en="Escalate to human reviewer",
        description="Send to a human (account manager / approver) for review.",
        audiences=("any",),
        severity="high",
        terminal=False,
        next_steps_zh=("分配审核人", "附带决策上下文"),
        next_steps_en=("Assign reviewer", "Attach full decision context"),
    ),
    ActionDefinition(
        id="contact_supplier",
        label_zh="联系供应商",
        label_en="Contact supplier",
        description="Open a thread with the supplier (e.g. supplier vetting).",
        audiences=("b2b",),
        severity="medium",
        terminal=False,
        next_steps_zh=("通过 EDI/邮件联系供应商", "记录沟通日志"),
        next_steps_en=("Reach out via EDI/email", "Log the conversation"),
    ),
    ActionDefinition(
        id="cancel_request",
        label_zh="取消请求",
        label_en="Cancel request",
        description="Honour the user's cancellation request.",
        audiences=("any",),
        severity="medium",
        terminal=True,
        next_steps_zh=("撤销订单", "发送取消确认"),
        next_steps_en=("Void the order", "Send cancellation confirmation"),
    ),
    ActionDefinition(
        id="initiate_return",
        label_zh="发起退换货",
        label_en="Initiate return / refund",
        description="Start RMA / refund flow.",
        audiences=("any",),
        severity="medium",
        terminal=True,
        next_steps_zh=("创建退货单", "通知物流揽件"),
        next_steps_en=("Create RMA", "Schedule logistics pickup"),
    ),
    # --- fallback ---
    ActionDefinition(
        id="no_action",
        label_zh="无操作",
        label_en="No action",
        description="No actionable intent detected — keep the conversation open.",
        audiences=("any",),
        severity="low",
        terminal=False,
        next_steps_zh=("继续对话",),
        next_steps_en=("Continue the conversation",),
    ),
]

ACTION_REGISTRY: Dict[str, ActionDefinition] = {a.id: a for a in _RAW}
ACTION_IDS: Tuple[str, ...] = tuple(ACTION_REGISTRY.keys())
FALLBACK_ACTION = "no_action"


def get_action(action_id: str) -> ActionDefinition:
    try:
        return ACTION_REGISTRY[action_id]
    except KeyError as exc:
        raise KeyError(f"unknown action id: {action_id!r}") from exc


def actions_for_audience(audience: str) -> List[ActionDefinition]:
    audience = (audience or "any").lower()
    if audience == "any":
        return list(ACTION_REGISTRY.values())
    return [
        a for a in ACTION_REGISTRY.values()
        if audience in a.audiences or "any" in a.audiences
    ]


def label_of(action_id: str, language: str) -> str:
    a = get_action(action_id)
    return a.label_zh if language.lower().startswith("zh") else a.label_en


def next_steps_of(action_id: str, language: str) -> List[str]:
    a = get_action(action_id)
    if language.lower().startswith("zh"):
        return list(a.next_steps_zh)
    return list(a.next_steps_en)
