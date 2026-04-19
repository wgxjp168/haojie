from fastapi.testclient import TestClient


def _client():
    from hub.main import app
    return TestClient(app)


def _payload():
    return {
        "request_id": "api-hub-001",
        "text": "这个办公椅多少钱？我想批量采购 50 把。",
        "language": "zh-CN",
        "user_type": "b2b",
        "buyer": {"user_id": "u1", "user_type": "b2b", "budget": 100000.0},
        "product": {
            "product_id": "P001",
            "name": "办公椅",
            "unit_price": 500.0,
            "quantity": 50,
            "in_stock": True,
            "supplier_rating": 4.3,
        },
        "options": {
            "include_llm": False,
            "include_report": True,
            "report_format": "markdown",
            "report_audience": "technical",
        },
    }


def test_root_health():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "ilbuyai-hub"
        assert len(body["stages"]) == 4


def test_hub_health_lists_four_stages():
    with _client() as client:
        resp = client.get("/api/v1/hub/health")
        assert resp.status_code == 200
        body = resp.json()
        assert {s["stage"] for s in body["stages"]} == {"intent", "decision", "llm", "report"}


def test_pipeline_end_to_end():
    with _client() as client:
        resp = client.post("/api/v1/hub/pipeline", json=_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall_status"] == "success"
        assert body["intent"]["status"] == "success"
        assert body["decision"]["status"] == "success"
        assert body["report"]["status"] == "success"
        assert "采购决策报告" in body["report"]["result"]["content"]


def test_pipeline_with_llm_narrative():
    with _client() as client:
        payload = _payload()
        payload["options"]["include_llm"] = True
        resp = client.post("/api/v1/hub/pipeline", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["llm"]["status"] == "success"
        assert body["llm"]["result"]["provider"] == "stub"


def test_pipeline_validates_empty_text():
    with _client() as client:
        payload = _payload()
        payload["text"] = "   "
        resp = client.post("/api/v1/hub/pipeline", json=payload)
        assert resp.status_code == 422


def test_batch_endpoint():
    with _client() as client:
        resp = client.post(
            "/api/v1/hub/pipeline/batch",
            json={"items": [_payload(), _payload()]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2


def test_metrics_endpoint():
    with _client() as client:
        client.post("/api/v1/hub/pipeline", json=_payload())
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "hub_pipeline_total" in resp.text
        assert "hub_stage_total" in resp.text


def test_unsupported_language_returns_422():
    with _client() as client:
        payload = _payload()
        payload["language"] = "fr"
        resp = client.post("/api/v1/hub/pipeline", json=payload)
        assert resp.status_code == 422
