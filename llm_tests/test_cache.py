from llm.cache import LLMCache, _LRU


def test_lru_capacity_zero_no_op():
    lru = _LRU(0)
    lru.set("k", {"v": 1})
    assert lru.get("k") is None


def test_lru_evicts_oldest():
    lru = _LRU(2)
    lru.set("a", {"v": 1})
    lru.set("b", {"v": 2})
    lru.set("c", {"v": 3})
    assert lru.get("a") is None
    assert lru.get("c") == {"v": 3}


def test_cache_memory_hit():
    cache = LLMCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}


def test_cache_clear():
    cache = LLMCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    cache.set("k", {"v": 1})
    cache.clear()
    assert cache.get("k") is None


def test_cache_not_ready_with_zero_capacity_and_no_redis():
    cache = LLMCache(memory_capacity=0, redis_url=None, ttl_seconds=0)
    assert cache.ready is False


def test_cache_miss_returns_none():
    cache = LLMCache(memory_capacity=4, redis_url=None, ttl_seconds=0)
    assert cache.get("nope") is None
