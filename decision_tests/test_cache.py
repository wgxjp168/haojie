from decision.cache import DecisionCache, _LRU


def test_lru_capacity_zero_is_noop():
    lru = _LRU(0)
    lru.set("k", {"v": 1})
    assert lru.get("k") is None


def test_lru_evicts_oldest():
    lru = _LRU(2)
    lru.set("a", {"v": 1})
    lru.set("b", {"v": 2})
    lru.set("c", {"v": 3})
    assert lru.get("a") is None
    assert lru.get("b") == {"v": 2}
    assert lru.get("c") == {"v": 3}


def test_lru_get_marks_recent():
    lru = _LRU(2)
    lru.set("a", {"v": 1})
    lru.set("b", {"v": 2})
    lru.get("a")
    lru.set("c", {"v": 3})
    assert lru.get("b") is None
    assert lru.get("a") == {"v": 1}


def test_decision_cache_memory_only():
    cache = DecisionCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    assert cache.ready
    cache.set("k", {"v": 42})
    assert cache.get("k") == {"v": 42}


def test_decision_cache_clear():
    cache = DecisionCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    cache.set("k", {"v": 1})
    cache.clear()
    assert cache.get("k") is None


def test_decision_cache_with_zero_memory_and_no_redis_not_ready():
    cache = DecisionCache(memory_capacity=0, redis_url=None, ttl_seconds=0)
    assert cache.ready is False


def test_decision_cache_get_miss_returns_none():
    cache = DecisionCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    assert cache.get("nope") is None
