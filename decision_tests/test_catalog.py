from decision.catalog import (
    ACTION_IDS,
    ACTION_REGISTRY,
    FALLBACK_ACTION,
    actions_for_audience,
    get_action,
    label_of,
    next_steps_of,
)


def test_registry_keyed_by_id():
    for k, v in ACTION_REGISTRY.items():
        assert k == v.id


def test_fallback_present():
    assert FALLBACK_ACTION in ACTION_REGISTRY


def test_ids_unique_and_non_empty():
    assert len(ACTION_IDS) == len(set(ACTION_IDS))
    assert all(ACTION_IDS)


def test_b2b_only_actions():
    b2b_actions = {a.id for a in actions_for_audience("b2b")}
    assert "request_quote" in b2b_actions
    assert "bulk_order" in b2b_actions
    # B2C-only action should not appear in the b2b list
    assert "wait_for_promotion" not in b2b_actions


def test_b2c_only_actions():
    b2c = {a.id for a in actions_for_audience("b2c")}
    assert "add_to_cart" in b2c
    assert "wait_for_promotion" in b2c
    assert "request_quote" not in b2c


def test_any_audience_returns_all():
    assert len(actions_for_audience("any")) == len(ACTION_REGISTRY)


def test_label_of_zh_and_en():
    assert label_of("approve_purchase", "zh-CN") == "批准采购"
    assert label_of("approve_purchase", "en") == "Approve purchase"


def test_label_of_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        label_of("bogus", "en")


def test_next_steps_localised():
    zh = next_steps_of("request_quote", "zh-CN")
    en = next_steps_of("request_quote", "en")
    assert zh and en
    assert zh != en


def test_get_action_by_id():
    a = get_action("approve_purchase")
    assert a.id == "approve_purchase"
    assert a.terminal is True
    assert a.severity == "medium"


def test_severity_values():
    for a in ACTION_REGISTRY.values():
        assert a.severity in {"low", "medium", "high"}


def test_audiences_normalised():
    for a in ACTION_REGISTRY.values():
        for aud in a.audiences:
            assert aud in {"any", "b2b", "b2c"}


def test_terminal_actions_set():
    terminal = {a.id for a in ACTION_REGISTRY.values() if a.terminal}
    assert "approve_purchase" in terminal
    assert "request_more_info" not in terminal
