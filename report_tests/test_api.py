from fastapi.testclient import TestClient


def _client():
    from report.main import app
    return TestClient(app)


def _sample_payload():
    return {
        "request_id": "api-test-001",
        "format": "markdown",
        "audience": "technical",
        "language": "zh-CN",
        "buyer": {"user_id": "u1", "user_type": "b2b", "budget": 50000.0},
        "product": {
            "product_id": "P001",
            "name": "桌子",
            "unit_price": 300.0,
            "quantity": 10,
        },
        "intent": {
            "intent": "request_quote",
            "label": "请求报价",
            "confidence": 0.9,
        },
        "decision": {
            "action": "request_quote",
            "label": "请求报价",
            "confidence": 0.85,
            "confident": True,
            "risk_score": 0.1,
            "requires_review": False,
            "rationale": "B 端采购员请求报价",
            "next_steps": ["生成 RFQ"],
        },
    }


def test_root_health():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "ilbuyai-report"


def test_reports_health():
    with _client() as client:
        resp = client.get("/api/v1/reports/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "memory" in body["storage_backends"]


def test_list_templates():
    with _client() as client:
        resp = client.get("/api/v1/reports/templates/list")
        assert resp.status_code == 200
        body = resp.json()
        ids = {t["template_id"] for t in body}
        assert {"executive_summary", "technical_detail", "customer_friendly"} <= ids


def test_generate_endpoint():
    with _client() as client:
        resp = client.post(
            "/api/v1/reports/generate", json=_sample_payload()
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "markdown"
        assert body["content"].startswith("# ")


def test_generate_validates_empty_payload():
    with _client() as client:
        resp = client.post("/api/v1/reports/generate", json={})
        assert resp.status_code == 422


def test_generate_validates_bad_format():
    with _client() as client:
        payload = _sample_payload()
        payload["format"] = "docx"
        resp = client.post("/api/v1/reports/generate", json=payload)
        assert resp.status_code == 422


def test_generate_handles_unknown_template():
    with _client() as client:
        payload = _sample_payload()
        payload["template_id"] = "ghost"
        resp = client.post("/api/v1/reports/generate", json=payload)
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "UNKNOWN_TEMPLATE"


def test_get_and_delete_report_roundtrip():
    with _client() as client:
        payload = _sample_payload()
        payload["request_id"] = "api-rt-002"
        gen = client.post("/api/v1/reports/generate", json=payload)
        assert gen.status_code == 200
        report_id = gen.json()["report_id"]

        fetched = client.get(f"/api/v1/reports/{report_id}")
        assert fetched.status_code == 200
        assert fetched.json()["content"] == gen.json()["content"]

        removed = client.delete(f"/api/v1/reports/{report_id}")
        assert removed.status_code == 200
        assert removed.json()["removed"]["memory"] is True


def test_get_report_missing():
    with _client() as client:
        resp = client.get("/api/v1/reports/no-such-report")
        assert resp.status_code == 404


def test_batch_endpoint():
    with _client() as client:
        resp = client.post(
            "/api/v1/reports/generate/batch",
            json={"items": [_sample_payload(), _sample_payload()]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2


def test_metrics_endpoint():
    with _client() as client:
        client.post("/api/v1/reports/generate", json=_sample_payload())
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "report_requests_total" in resp.text
        assert "report_render_duration_seconds" in resp.text
