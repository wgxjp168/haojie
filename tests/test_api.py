"""HTTP-level tests using FastAPI's TestClient."""
from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"]
    assert body["endpoints"]["docs"] == "/docs"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert "status" in body
    assert "components" in body


def test_supported_formats(client):
    resp = client.get("/api/v1/inputs/supported-formats")
    assert resp.status_code == 200
    body = resp.json()
    assert "image" in body and "voice" in body
    assert "png" in body["image"] or "PNG" in body["image"]


def test_text_input(client):
    resp = client.post(
        "/api/v1/inputs/text",
        json={"text": "我想买一台 iPhone 15，预算8000元"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["input_type"] == "text"
    assert 8000 in body["data"]["extracted"]["prices"]


def test_text_input_validation_error(client):
    resp = client.post("/api/v1/inputs/text", json={"text": ""})
    # Pydantic min_length=1 -> 422
    assert resp.status_code == 422


def test_link_input_unknown_platform_still_accepted(client):
    # Validation only requires a well-formed http URL; platform detection
    # is best-effort.
    resp = client.post(
        "/api/v1/inputs/link",
        json={"url": "https://example.com/some/product"},
    )
    # The processor may time out or fail the title fetch — that's fine,
    # it should report success=true with warnings.
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["is_supported_platform"] is False


def test_link_input_invalid_scheme_rejected(client):
    resp = client.post(
        "/api/v1/inputs/link", json={"url": "ftp://example.com"}
    )
    assert resp.status_code == 400


def test_image_upload(client, tiny_png_bytes):
    files = {"file": ("tiny.png", tiny_png_bytes, "image/png")}
    resp = client.post("/api/v1/inputs/image", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["image_info"]["format"] == "PNG"


def test_voice_upload(client, tiny_wav_bytes):
    files = {"file": ("tiny.wav", tiny_wav_bytes, "audio/wav")}
    resp = client.post(
        "/api/v1/inputs/voice",
        files=files,
        data={"language": "en-US"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["audio_info"]["format"] == "wav"
    assert body["data"]["language"] == "en-US"


def test_session_returned_404_for_unknown(client):
    resp = client.get("/api/v1/sessions/does-not-exist")
    assert resp.status_code == 404


def test_batch_endpoint(client, tiny_png_bytes):
    encoded = base64.b64encode(tiny_png_bytes).decode()
    resp = client.post(
        "/api/v1/inputs/batch",
        json={
            "inputs": [
                {"type": "text", "data": "买笔记本电脑"},
                {"type": "image", "data": encoded},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert body["successful"] >= 1


def test_system_version(client):
    resp = client.get("/api/v1/system/version")
    assert resp.status_code == 200
    assert resp.json()["service"]
