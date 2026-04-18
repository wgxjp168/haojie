"""Tests for the domain entities."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from entities.enums import InputType, SessionStatus, UserRole, UserType
from entities.session import InputItem, Session
from entities.user import User, UserContact
from entities.user_profile import BudgetPreference, UsageScenario, UserProfile


def test_user_defaults_roles_for_b2c():
    user = User.create_b2c("alice", "Alice", "a@example.com", "13800000000")
    assert UserRole.B2C_CONSUMER in user.roles
    assert user.user_type == UserType.B2C_CONSUMER
    assert user.user_id  # auto uuid


def test_user_contact_validation():
    ok = UserContact(email="a@b.com", phone="13800000000")
    assert ok.validate() == []
    bad = UserContact(email="not-an-email", phone="abc")
    errors = bad.validate()
    assert any("email" in e for e in errors)
    assert any("phone" in e for e in errors)


def test_user_has_role():
    user = User.create_b2b(
        "buyer", "Big Buyer", "b@example.com", "13900000000", "Widgets Inc"
    )
    assert user.has_role(UserRole.B2B_PURCHASER)
    assert not user.has_role(UserRole.ADMIN)


def test_session_expires_after_inactivity():
    session = Session(
        session_id="",
        user_id="u1",
        max_inactivity_minutes=1,
    )
    session.last_activity_at = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
    assert session.is_expired()
    assert not session.is_active()


def test_session_touch_updates_activity():
    session = Session(session_id="", user_id="u1")
    old = session.last_activity_at
    session.last_activity_at = old - timedelta(seconds=10)
    session.touch()
    assert session.last_activity_at > old - timedelta(seconds=1)


def test_input_item_status_transitions():
    item = InputItem(
        input_id="i1",
        session_id="s1",
        user_id="u1",
        input_type=InputType.TEXT,
        input_summary="hello",
    )
    item.mark_processing()
    assert item.status.value == "processing"
    item.mark_success({"ok": True}, 12.0)
    assert item.status.value == "success"
    assert item.processing_time_ms == 12.0
    assert item.processed_data == {"ok": True}


def test_session_end_sets_completed():
    session = Session(session_id="", user_id="u1")
    session.end("user_end")
    assert session.status == SessionStatus.COMPLETED
    assert session.ended_at is not None


def test_user_profile_records_purchase():
    profile = UserProfile(user_id="u1", user_type=UserType.B2C_CONSUMER)
    profile.record_purchase(
        category="laptop", brand="Apple", amount=Decimal("6999.00"), rating=0.9
    )
    profile.record_purchase(
        category="laptop", brand="Apple", amount=Decimal("5999.00"), rating=0.95
    )
    assert profile.total_purchases == 2
    assert profile.total_spent == Decimal("12998.00")
    cats = profile.top_categories()
    assert cats[0].category_name == "laptop"
    brands = profile.top_brands()
    assert brands[0].brand_name == "Apple"


def test_user_profile_recommendation_context_shape():
    profile = UserProfile(user_id="u1", user_type=UserType.B2B_PURCHASER)
    profile.budget = BudgetPreference(
        min_budget=Decimal("1000"), max_budget=Decimal("5000")
    )
    profile.usage_scenarios.append(
        UsageScenario(scenario_name="office", scenario_type="office", priority=8)
    )
    ctx = profile.get_recommendation_context()
    assert ctx["user_id"] == "u1"
    assert ctx["user_type"] == "B2B_PURCHASER"
    assert ctx["budget"]["min_budget"] == 1000.0
    # B2B defaults
    assert ctx["price_sensitivity"] == 0.7


def test_budget_within_range():
    b = BudgetPreference(min_budget=Decimal("100"), max_budget=Decimal("500"))
    assert b.is_within_budget(Decimal("200"))
    assert not b.is_within_budget(Decimal("50"))
    assert not b.is_within_budget(Decimal("600"))
