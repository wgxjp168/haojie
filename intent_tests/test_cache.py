from intent.cache import IntentCache, _LRU


def test_lru_evicts_oldest_first():
    lru = _LRU(2)
    lru.set("a", {"v": 1})
    lru.set("b", {"v": 2})
    lru.set("c", {"v": 3})
    assert lru.get("a") is None
    assert lru.get("b") == {"v": 2}
    assert lru.get("c") == {"v": 3}


def test_lru_zero_capacity_is_noop():
    lru = _LRU(0)
    lru.set("a", {"v": 1})
    assert lru.get("a") is None


def test_intent_cache_without_redis_still_works():
    c = IntentCache(memory_capacity=4, redis_url=None, ttl_seconds=60)
    assert c.ready
    c.set("k", {"top_intent": "x"})
    assert c.get("k") == {"top_intent": "x"}


def test_intent_cache_clear():
    c = IntentCache(memory_capacity=4, redis_url=None, ttl_seconds=60)
    c.set("k", {"top_intent": "x"})
    c.clear()
    assert c.get("k") is None
