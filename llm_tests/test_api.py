from fastapi.testclient import TestClient


def _client():
    from llm.main import app

    return TestClient(app)


def test_health_endpoint():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "ilbuyai-llm"
        assert "stub" in body["providers_available"]


def test_completions_health_endpoint():
    with _client() as client:
        resp = client.get("/api/v1/completions/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "stub" in body["providers_available"]
        assert isinstance(body["circuit_states"], dict)


def test_list_models_endpoint():
    with _client() as client:
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        ids = {m["model_id"] for m in body}
        assert "stub-small" in ids
        # Stub is available; paid providers should be reported unavailable
        stub_entry = next(m for m in body if m["model_id"] == "stub-small")
        assert stub_entry["available"] is True
        paid_entry = next(m for m in body if m["model_id"] == "gpt-4o")
        assert paid_entry["available"] is False


def test_completion_endpoint_returns_content():
    with _client() as client:
        resp = client.post(
            "/api/v1/completions",
            json={"prompt": "Hello, world", "strategy": "single"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "stub"
        assert body["content"]
        assert body["cached"] is False


def test_completion_endpoint_rejects_empty_payload():
    with _client() as client:
        resp = client.post("/api/v1/completions", json={})
        assert resp.status_code == 422


def test_completion_endpoint_rejects_unknown_model():
    with _client() as client:
        resp = client.post(
            "/api/v1/completions",
            json={"prompt": "hi", "model_id": "bogus"},
        )
        # LLMError -> mapped to 422 VALIDATION_ERROR
        assert resp.status_code == 422
        assert "code" in resp.json()


def test_completion_endpoint_rejects_invalid_strategy():
    with _client() as client:
        resp = client.post(
            "/api/v1/completions",
            json={"prompt": "hi", "strategy": "magic"},
        )
        assert resp.status_code == 422


def test_batch_endpoint():
    with _client() as client:
        resp = client.post(
            "/api/v1/completions/batch",
            json={"items": [
                {"prompt": "a"},
                {"prompt": "b"},
            ]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2


def test_metrics_endpoint_exposes_prometheus_format():
    with _client() as client:
        # Warm up metrics by calling completion.
        client.post("/api/v1/completions", json={"prompt": "hi"})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "llm_requests_total" in resp.text
        assert "llm_cost_usd_total" in resp.text


def test_completion_caches_repeated_calls():
    with _client() as client:
        first = client.post(
            "/api/v1/completions", json={"prompt": "identical prompt here"}
        )
        assert first.status_code == 200
        assert first.json()["cached"] is False
        second = client.post(
            "/api/v1/completions", json={"prompt": "identical prompt here"}
        )
        assert second.status_code == 200
        assert second.json()["cached"] is True
