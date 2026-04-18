from intent.rules import RuleBasedClassifier


def test_chinese_price_query_matches_inquire_price():
    rc = RuleBasedClassifier()
    top = rc.classify_best("这款手机多少钱?", language="zh-CN")
    assert top.intent == "inquire_price"
    assert top.score > 0.3


def test_chinese_stock_query_matches_check_stock():
    rc = RuleBasedClassifier()
    top = rc.classify_best("请问这个型号现在有货吗?", language="zh-CN")
    assert top.intent == "check_stock"


def test_english_tracking_query():
    rc = RuleBasedClassifier()
    top = rc.classify_best("where is my order shipment?", language="en")
    assert top.intent == "track_order"


def test_b2b_bulk_order_only_surfaces_for_b2b_audience():
    rc = RuleBasedClassifier()
    # Same text, different audience → the B2B-only intent should score higher
    # for b2b and get a penalty for b2c.
    text = "我们要批量采购 MOQ 1000"
    b2b = rc.classify_best(text, language="zh-CN", user_type="b2b")
    b2c = rc.classify_best(text, language="zh-CN", user_type="b2c")
    assert b2b.intent == "bulk_order"
    assert b2b.score >= b2c.score


def test_unknown_text_falls_back_to_other():
    rc = RuleBasedClassifier()
    top = rc.classify_best("asdf qwerty zzz", language="en")
    assert top.intent == "other"


def test_top_k_returns_multiple_candidates():
    rc = RuleBasedClassifier()
    matches = rc.classify("价格多少 有货吗", language="zh-CN", top_k=3)
    ids = {m.intent for m in matches}
    # Both price and stock keywords present — both should appear.
    assert "inquire_price" in ids
    assert "check_stock" in ids


def test_scores_monotonic_with_more_matches():
    rc = RuleBasedClassifier()
    one = rc.classify_best("价格", language="zh-CN")
    many = rc.classify_best("价格 报价 多少钱 折扣", language="zh-CN")
    assert many.score > one.score
