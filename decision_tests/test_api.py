from fastapi.testclient import TestClient


def _client():
    from decision.main import app
    return TestClient(app)


def test_health_endpoint():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "ilbuyai-decision"
        assert "strategy" in body


def test_decision_health_endpoint():
    with _client() as client:
        resp = client.get("/api/v1/decisions/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["circuit_state"] in {"closed", "open", "half_open"}


def test_list_actions_endpoint():
    with _client() as client:
        resp = client.get("/api/v1/decisions/actions")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) >= 15
        ids = {a["id"] for a in body}
        assert {"approve_purchase", "request_quote", "no_action"}.issubset(ids)


def test_evaluate_endpoint_returns_decision():
    with _client() as client:
        resp = client.post(
            "/api/v1/decisions/evaluate",
            json={
                "intent": {"intent": "request_quote", "confidence": 0.9},
                "buyer": {"user_type": "b2b"},
                "product": {"unit_price": 100.0, "quantity": 5, "in_stock": True},
                "language": "en",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["action"] == "request_quote"
        assert body["confidence"] > 0


def test_evaluate_endpoint_validates_payload():
    with _client() as client:
        # Missing 'intent' is required.
        resp = client.post("/api/v1/decisions/evaluate", json={})
        assert resp.status_code == 422


def test_evaluate_endpoint_rejects_negative_price():
    with _client() as client:
        resp = client.post(
            "/api/v1/decisions/evaluate",
            json={
                "intent": {"intent": "request_quote", "confidence": 0.9},
                "buyer": {"user_type": "b2b"},
                "product": {"unit_price": -10.0, "quantity": 1},
            },
        )
        assert resp.status_code == 422


def test_batch_endpoint():
    with _client() as client:
        resp = client.post(
            "/api/v1/decisions/evaluate/batch",
            json={
                "items": [
                    {
                        "intent": {"intent": "request_quote", "confidence": 0.9},
                        "buyer": {"user_type": "b2b"},
                        "product": {"unit_price": 100.0, "quantity": 5, "in_stock": True},
                    },
                    {
                        "intent": {"intent": "cancel_order", "confidence": 0.95},
                        "buyer": {"user_type": "b2c"},
                        "product": {},
                    },
                ]
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        actions = [r["action"] for r in body["results"]]
        assert "request_quote" in actions
        assert "cancel_request" in actions


def test_metrics_endpoint_exposes_prometheus_format():
    with _client() as client:
        # Warm up metrics by making a decision.
        client.post(
            "/api/v1/decisions/evaluate",
            json={
                "intent": {"intent": "request_quote", "confidence": 0.9},
                "buyer": {"user_type": "b2b"},
                "product": {"in_stock": True},
            },
        )
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "decision_requests_total" in resp.text


def test_strategy_override_via_request_body():
    with _client() as client:
        resp = client.post(
            "/api/v1/decisions/evaluate",
            json={
                "intent": {"intent": "request_quote", "confidence": 0.9},
                "buyer": {"user_type": "b2b"},
                "product": {},
                "strategy_override": "rules",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["strategy"] == "rules"


def test_invalid_strategy_override_returns_422():
    with _client() as client:
        resp = client.post(
            "/api/v1/decisions/evaluate",
            json={
                "intent": {"intent": "request_quote", "confidence": 0.9},
                "buyer": {"user_type": "b2b"},
                "product": {},
                "strategy_override": "magic",
            },
        )
        assert resp.status_code == 422
