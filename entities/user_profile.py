"""User profile entity capturing preferences and behavioural signals.

The profile is what the downstream AI decision hub consumes as the
user-context half of its prompt. Kept intentionally small so it can be
serialised into a model context without blowing the token budget.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from entities.enums import UserType


@dataclass
class BudgetPreference:
    min_budget: Optional[Decimal] = None
    max_budget: Optional[Decimal] = None
    preferred_budget: Optional[Decimal] = None
    currency: str = "CNY"
    is_flexible: bool = True
    flexibility_pct: float = 0.2

    def is_within_budget(self, price: Decimal) -> bool:
        if self.min_budget is not None and price < self.min_budget:
            return False
        if self.max_budget is not None and price > self.max_budget:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_budget": float(self.min_budget) if self.min_budget is not None else None,
            "max_budget": float(self.max_budget) if self.max_budget is not None else None,
            "preferred_budget": float(self.preferred_budget)
            if self.preferred_budget is not None
            else None,
            "currency": self.currency,
            "is_flexible": self.is_flexible,
            "flexibility_pct": self.flexibility_pct,
        }


@dataclass
class BrandPreference:
    brand_name: str
    category: str
    preference_score: float = 0.5  # 0.0 .. 1.0
    interaction_count: int = 0
    purchase_count: int = 0
    last_interaction: Optional[datetime] = None

    def record_interaction(self, kind: str, rating: Optional[float] = None) -> None:
        self.interaction_count += 1
        if kind == "purchase":
            self.purchase_count += 1
            weight = 0.15
        elif kind == "click":
            weight = 0.05
        else:
            weight = 0.01
        if rating is not None:
            self.preference_score = max(
                0.0, min(1.0, self.preference_score * (1 - weight) + rating * weight)
            )
        self.last_interaction = datetime.now(tz=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand_name": self.brand_name,
            "category": self.category,
            "preference_score": round(self.preference_score, 3),
            "interaction_count": self.interaction_count,
            "purchase_count": self.purchase_count,
            "last_interaction": self.last_interaction.isoformat()
            if self.last_interaction
            else None,
        }


@dataclass
class CategoryPreference:
    category_name: str
    preference_score: float = 0.5
    purchase_count: int = 0
    total_spent: Decimal = Decimal("0")

    def record_purchase(self, amount: Decimal) -> None:
        self.purchase_count += 1
        self.total_spent += amount
        # Saturating score that converges to 1.0 as purchases accumulate.
        self.preference_score = min(1.0, 0.5 + self.purchase_count * 0.05)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_name": self.category_name,
            "preference_score": round(self.preference_score, 3),
            "purchase_count": self.purchase_count,
            "total_spent": float(self.total_spent),
        }


@dataclass
class UsageScenario:
    scenario_name: str
    scenario_type: str  # e.g. office / gaming / study / gift
    priority: int = 5  # 1..10
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "priority": self.priority,
            "description": self.description,
        }


@dataclass
class UserProfile:
    user_id: str
    user_type: UserType
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    budget: BudgetPreference = field(default_factory=BudgetPreference)
    brand_preferences: List[BrandPreference] = field(default_factory=list)
    category_preferences: List[CategoryPreference] = field(default_factory=list)
    usage_scenarios: List[UsageScenario] = field(default_factory=list)

    price_sensitivity: float = 0.5
    quality_preference: float = 0.5
    brand_loyalty: float = 0.5

    total_purchases: int = 0
    total_spent: Decimal = Decimal("0")
    last_purchase_at: Optional[datetime] = None

    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Prime B2B purchasers with tighter price sensitivity and quality bias.
        if self.user_type == UserType.B2B_PURCHASER:
            self.price_sensitivity = 0.7
            self.quality_preference = 0.8
            self.brand_loyalty = 0.6

    def record_purchase(
        self,
        category: str,
        brand: Optional[str],
        amount: Decimal,
        rating: Optional[float] = None,
    ) -> None:
        self.total_purchases += 1
        self.total_spent += amount
        self.last_purchase_at = datetime.now(tz=timezone.utc)

        cat = next(
            (c for c in self.category_preferences if c.category_name == category), None
        )
        if cat is None:
            cat = CategoryPreference(category_name=category)
            self.category_preferences.append(cat)
        cat.record_purchase(amount)

        if brand:
            br = next(
                (
                    b
                    for b in self.brand_preferences
                    if b.brand_name == brand and b.category == category
                ),
                None,
            )
            if br is None:
                br = BrandPreference(brand_name=brand, category=category)
                self.brand_preferences.append(br)
            br.record_interaction("purchase", rating)

        self.updated_at = datetime.now(tz=timezone.utc)

    def top_categories(self, limit: int = 5) -> List[CategoryPreference]:
        return sorted(
            self.category_preferences, key=lambda c: c.preference_score, reverse=True
        )[:limit]

    def top_brands(self, limit: int = 5) -> List[BrandPreference]:
        return sorted(
            self.brand_preferences, key=lambda b: b.preference_score, reverse=True
        )[:limit]

    def get_recommendation_context(self) -> Dict[str, Any]:
        """Small dict to inject into the decision-hub prompt."""
        return {
            "user_id": self.user_id,
            "user_type": self.user_type.value,
            "budget": self.budget.to_dict(),
            "price_sensitivity": self.price_sensitivity,
            "quality_preference": self.quality_preference,
            "brand_loyalty": self.brand_loyalty,
            "top_categories": [c.category_name for c in self.top_categories(3)],
            "top_brands": [b.brand_name for b in self.top_brands(3)],
            "total_purchases": self.total_purchases,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "user_id": self.user_id,
            "user_type": self.user_type.value,
            "budget": self.budget.to_dict(),
            "brand_preferences": [b.to_dict() for b in self.brand_preferences],
            "category_preferences": [c.to_dict() for c in self.category_preferences],
            "usage_scenarios": [s.to_dict() for s in self.usage_scenarios],
            "price_sensitivity": self.price_sensitivity,
            "quality_preference": self.quality_preference,
            "brand_loyalty": self.brand_loyalty,
            "stats": {
                "total_purchases": self.total_purchases,
                "total_spent": float(self.total_spent),
                "last_purchase_at": self.last_purchase_at.isoformat()
                if self.last_purchase_at
                else None,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
