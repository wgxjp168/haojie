"""Procurement intent taxonomy.

The taxonomy is split into B2B and B2C sub-spaces because those user types
express needs with different vocabulary and priority. The unified ID space
(``INTENT_IDS``) is what downstream modules consume; ``AUDIENCE`` annotates
which audiences typically use a given intent so we can bias the rule-based
classifier by user_type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class IntentDefinition:
    """One entry in the taxonomy."""

    id: str                     # canonical machine id, e.g. "compare_products"
    label_zh: str               # Chinese display label
    label_en: str               # English display label
    description: str            # human description
    audiences: Tuple[str, ...]  # ("b2b", "b2c"), ("b2b",), ("b2c",) or ("any",)
    keywords_zh: Tuple[str, ...] = ()
    keywords_en: Tuple[str, ...] = ()

    @property
    def is_b2b(self) -> bool:
        return "b2b" in self.audiences or "any" in self.audiences

    @property
    def is_b2c(self) -> bool:
        return "b2c" in self.audiences or "any" in self.audiences


_RAW: List[IntentDefinition] = [
    # --- discovery / information ---
    IntentDefinition(
        id="search_product",
        label_zh="商品搜索",
        label_en="Product search",
        description="User wants to find products by name, category or attributes.",
        audiences=("any",),
        keywords_zh=("搜索", "找", "查找", "有没有", "有什么", "推荐一下", "哪里买"),
        keywords_en=("search", "find", "look for", "show me", "any ", "where to buy"),
    ),
    IntentDefinition(
        id="compare_products",
        label_zh="商品比较",
        label_en="Product comparison",
        description="User wants to compare specs / prices of multiple products.",
        audiences=("any",),
        keywords_zh=("对比", "比较", "哪个更好", "区别", "差别", "优劣"),
        keywords_en=("compare", "vs", "versus", "difference", "which is better"),
    ),
    IntentDefinition(
        id="inquire_price",
        label_zh="价格咨询",
        label_en="Price inquiry",
        description="User asks for the price, quote, discount or promotion.",
        audiences=("any",),
        keywords_zh=("多少钱", "价格", "报价", "折扣", "优惠", "促销", "便宜"),
        keywords_en=("price", "quote", "quotation", "cost", "how much", "discount", "deal"),
    ),
    IntentDefinition(
        id="check_stock",
        label_zh="库存查询",
        label_en="Stock availability",
        description="User asks whether a product is in stock or lead time.",
        audiences=("any",),
        keywords_zh=("库存", "有货", "现货", "到货", "发货时间", "交期"),
        keywords_en=("stock", "in stock", "available", "availability", "lead time", "eta"),
    ),
    IntentDefinition(
        id="product_info",
        label_zh="商品详情",
        label_en="Product details",
        description="User asks for specs, materials, dimensions, certifications.",
        audiences=("any",),
        keywords_zh=("参数", "规格", "材质", "尺寸", "详情", "说明", "认证"),
        keywords_en=("spec", "specification", "material", "dimension", "details", "manual"),
    ),
    IntentDefinition(
        id="view_reviews",
        label_zh="查看评价",
        label_en="View reviews",
        description="User wants to see reviews / ratings / testimonials.",
        audiences=("b2c", "b2b"),
        keywords_zh=("评价", "评论", "口碑", "好评", "差评", "测评"),
        keywords_en=("review", "rating", "testimonial", "feedback"),
    ),
    IntentDefinition(
        id="get_recommendation",
        label_zh="推荐咨询",
        label_en="Recommendation",
        description="User wants a personalised product recommendation.",
        audiences=("any",),
        keywords_zh=("推荐", "建议", "值得买", "该买", "选哪个"),
        keywords_en=("recommend", "suggest", "should I buy", "what to pick"),
    ),
    # --- B2B-centric ---
    IntentDefinition(
        id="request_quote",
        label_zh="请求报价",
        label_en="Request for quotation",
        description="B2B buyer formally requests an RFQ / BOQ quote.",
        audiences=("b2b",),
        keywords_zh=("报价单", "询价", "rfq", "正式报价", "招标"),
        keywords_en=("rfq", "rfp", "request for quote", "tender", "bid"),
    ),
    IntentDefinition(
        id="bulk_order",
        label_zh="批量采购",
        label_en="Bulk order",
        description="Buyer wants to place a bulk / wholesale order.",
        audiences=("b2b",),
        keywords_zh=("批量", "大量", "批发", "大单", "整车", "整柜", "moq"),
        keywords_en=("bulk", "wholesale", "large order", "moq", "container", "pallet"),
    ),
    IntentDefinition(
        id="supplier_info",
        label_zh="供应商信息",
        label_en="Supplier info",
        description="Buyer asks about supplier qualifications / certifications.",
        audiences=("b2b",),
        keywords_zh=("供应商", "厂家", "工厂", "资质", "认证", "iso", "资质证书"),
        keywords_en=("supplier", "vendor", "factory", "iso", "certification", "credentials"),
    ),
    IntentDefinition(
        id="negotiate_terms",
        label_zh="商务条款谈判",
        label_en="Negotiate terms",
        description="Buyer wants to negotiate price, MOQ, payment or delivery terms.",
        audiences=("b2b",),
        keywords_zh=("谈价", "议价", "降价", "账期", "付款方式", "条款", "合同"),
        keywords_en=("negotiate", "net 30", "net 60", "payment terms", "contract", "incoterms"),
    ),
    # --- transactional ---
    IntentDefinition(
        id="place_order",
        label_zh="下单",
        label_en="Place order",
        description="User wants to place an order now.",
        audiences=("any",),
        keywords_zh=("下单", "购买", "买了", "要买", "拍下", "结账", "付款"),
        keywords_en=("buy", "order", "purchase", "checkout", "pay"),
    ),
    IntentDefinition(
        id="track_order",
        label_zh="订单跟踪",
        label_en="Track order",
        description="User asks for the status / tracking of an existing order.",
        audiences=("any",),
        keywords_zh=("订单", "物流", "发货", "快递", "到哪了", "运输", "跟踪"),
        keywords_en=("track", "tracking", "shipment", "delivery status", "where is my order"),
    ),
    IntentDefinition(
        id="cancel_order",
        label_zh="取消订单",
        label_en="Cancel order",
        description="User wants to cancel an order that has not shipped.",
        audiences=("any",),
        keywords_zh=("取消", "不要了", "别发了", "撤单"),
        keywords_en=("cancel", "cancellation", "void my order"),
    ),
    IntentDefinition(
        id="return_refund",
        label_zh="退换货/退款",
        label_en="Return / refund",
        description="User wants to return or get a refund.",
        audiences=("any",),
        keywords_zh=("退货", "退款", "换货", "售后", "退钱"),
        keywords_en=("return", "refund", "exchange", "rma", "money back"),
    ),
    # --- support ---
    IntentDefinition(
        id="complaint",
        label_zh="投诉",
        label_en="Complaint",
        description="User complains about a product, order, or service.",
        audiences=("any",),
        keywords_zh=("投诉", "差评", "气死了", "欺诈", "骗人", "维权"),
        keywords_en=("complaint", "terrible", "fraud", "scam", "dispute"),
    ),
    IntentDefinition(
        id="contact_support",
        label_zh="联系客服",
        label_en="Contact support",
        description="User asks to speak to a human / support agent.",
        audiences=("any",),
        keywords_zh=("客服", "人工", "在吗", "帮忙", "转接"),
        keywords_en=("support", "agent", "human", "help me", "talk to someone"),
    ),
    IntentDefinition(
        id="greeting",
        label_zh="问候",
        label_en="Greeting",
        description="Small talk or greeting with no business intent yet.",
        audiences=("any",),
        keywords_zh=("你好", "在吗", "早上好", "晚上好", "嗨"),
        keywords_en=("hi", "hello", "hey", "good morning", "good evening"),
    ),
    IntentDefinition(
        id="farewell",
        label_zh="告别",
        label_en="Farewell",
        description="User ends the conversation.",
        audiences=("any",),
        keywords_zh=("再见", "拜拜", "谢谢了", "不用了"),
        keywords_en=("bye", "goodbye", "thanks, that's all", "see you"),
    ),
    IntentDefinition(
        id="other",
        label_zh="其他",
        label_en="Other",
        description="Fallback bucket for queries that do not match any known intent.",
        audiences=("any",),
    ),
]

# Public registries ----------------------------------------------------------

INTENT_REGISTRY: Dict[str, IntentDefinition] = {d.id: d for d in _RAW}
INTENT_IDS: Tuple[str, ...] = tuple(INTENT_REGISTRY.keys())
FALLBACK_INTENT = "other"


def get_intent(intent_id: str) -> IntentDefinition:
    try:
        return INTENT_REGISTRY[intent_id]
    except KeyError as exc:
        raise KeyError(f"unknown intent id: {intent_id!r}") from exc


def intents_for_audience(audience: str) -> List[IntentDefinition]:
    """Return intents relevant to a given audience (``b2b``, ``b2c``, ``any``)."""
    audience = (audience or "any").lower()
    if audience == "any":
        return list(INTENT_REGISTRY.values())
    return [d for d in INTENT_REGISTRY.values() if audience in d.audiences or "any" in d.audiences]


def label_of(intent_id: str, language: str) -> str:
    intent = get_intent(intent_id)
    return intent.label_zh if language.lower().startswith("zh") else intent.label_en
