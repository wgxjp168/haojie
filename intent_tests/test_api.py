from fastapi.testclient import TestClient


def _client():
    # Import here so conftest env vars take effect first.
    from intent.main import app

    return TestClient(app)


def test_health_endpoint():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "ilbuyai-intent"
        assert "backend" in body


def test_intent_health_endpoint():
    with _client() as client:
        resp = client.get("/api/v1/intents/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["circuit_state"] in {"closed", "open", "half_open"}


def test_list_intents_endpoint():
    with _client() as client:
        resp = client.get("/api/v1/intents")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) >= 15
        ids = {item["id"] for item in body}
        assert {"inquire_price", "check_stock", "compare_products"}.issubset(ids)


def test_predict_endpoint_returns_intent():
    with _client() as client:
        resp = client.post(
            "/api/v1/intents/predict",
            json={"text": "这个多少钱?", "language": "zh-CN"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["top_intent"] == "inquire_price"
        assert body["top_confidence"] > 0
        assert body["backend"] == "rules"


def test_predict_endpoint_validates_empty_text():
    with _client() as client:
        resp = client.post(
            "/api/v1/intents/predict", json={"text": "   ", "language": "en"}
        )
        assert resp.status_code == 422  # pydantic rejects before hitting handler


def test_batch_predict_endpoint():
    with _client() as client:
        resp = client.post(
            "/api/v1/intents/predict/batch",
            json={
                "items": [
                    {"text": "价格多少", "language": "zh-CN"},
                    {"text": "有货吗", "language": "zh-CN"},
                ]
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        intents = [r["top_intent"] for r in body["results"]]
        assert "inquire_price" in intents
        assert "check_stock" in intents


def test_metrics_endpoint_exposes_prometheus_format():
    with _client() as client:
        # Warm up metrics by making a prediction.
        client.post(
            "/api/v1/intents/predict",
            json={"text": "搜索手机", "language": "zh-CN"},
        )
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "intent_requests_total" in resp.text
