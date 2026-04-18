"""Test configuration — wires a fresh Settings object per test session
and forces the in-memory cache so tests don't need Redis running.
"""
from __future__ import annotations

import io
import os

import pytest

# Set env vars BEFORE Settings is first imported.
os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("APP_DEBUG", "true")
os.environ.setdefault("APP_REDIS_URL", "")  # forces in-memory cache
os.environ.setdefault("APP_CACHE_ENABLED", "false")
os.environ.setdefault("APP_SPEECH_PROVIDER", "local")
os.environ.setdefault("APP_IMAGE_RECOGNITION_PROVIDER", "local")
os.environ.setdefault("APP_STORAGE_BACKEND", "local")
os.environ.setdefault("APP_STORAGE_LOCAL_PATH", "./test_storage")
os.environ.setdefault("APP_API_KEY_REQUIRED", "false")


from config.settings import reload_settings  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def settings():
    return reload_settings()


@pytest.fixture
def tiny_png_bytes() -> bytes:
    """A deterministic 1x1 red PNG — used by image tests."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiny_wav_bytes() -> bytes:
    """Minimal valid RIFF/WAVE header + 0.1s of silence."""
    import struct

    sample_rate = 16000
    duration_ms = 100
    n_samples = sample_rate * duration_ms // 1000
    pcm = b"\x00\x00" * n_samples  # 16-bit silence, mono
    data_chunk = struct.pack("<4sI", b"data", len(pcm)) + pcm
    fmt_chunk = struct.pack(
        "<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16
    )
    riff = b"RIFF" + struct.pack("<I", 4 + len(fmt_chunk) + len(data_chunk)) + b"WAVE"
    return riff + fmt_chunk + data_chunk
