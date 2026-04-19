from report.cache import ReportCache, _LRU


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


def test_cache_memory_only_set_get():
    cache = ReportCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    cache.set("k", {"v": 42})
    assert cache.get("k") == {"v": 42}


def test_cache_not_ready_when_zero_memory_no_redis():
    cache = ReportCache(memory_capacity=0, redis_url=None, ttl_seconds=0)
    assert cache.ready is False


def test_cache_miss_returns_none():
    cache = ReportCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    assert cache.get("nope") is None


def test_cache_clear():
    cache = ReportCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    cache.set("k", {"v": 1})
    cache.clear()
    assert cache.get("k") is None
