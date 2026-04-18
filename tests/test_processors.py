"""Tests for individual input processors."""
from __future__ import annotations

import pytest

from core.exceptions import UnsupportedFormatError, ValidationError
from processors.base import InputContext
from processors.image_processor import ImageInputProcessor
from processors.link_processor import LinkInputProcessor
from processors.text_processor import TextInputProcessor
from processors.voice_processor import VoiceInputProcessor
from services.image_recognition import (
    ImageRecognitionService,
    LocalImageRecognitionClient,
)
from services.speech import LocalSpeechClient, SpeechRecognitionService


# --- text ---


class TestTextProcessor:
    def setup_method(self):
        self.proc = TextInputProcessor()
        self.ctx = InputContext(user_id="u1")

    def test_validate_ok(self):
        self.proc.validate("I want to buy a laptop, budget 5000 yuan")

    def test_validate_rejects_non_string(self):
        with pytest.raises(ValidationError):
            self.proc.validate(123)

    def test_validate_rejects_empty(self):
        with pytest.raises(ValidationError):
            self.proc.validate("   ")

    def test_validate_rejects_too_long(self):
        with pytest.raises(ValidationError):
            self.proc.validate("a" * 20_000)

    @pytest.mark.asyncio
    async def test_process_returns_structured_data(self):
        out = await self.proc.process(
            "我想购买一台笔记本电脑，预算5000元左右", self.ctx
        )
        assert out["data"]["language"] == "zh"
        assert out["data"]["intent"]["purchase"] is True
        assert out["data"]["signals"]["has_budget_mention"] is True
        assert 5000 in out["data"]["extracted"]["prices"]

    @pytest.mark.asyncio
    async def test_quantity_extraction(self):
        out = await self.proc.process("需要20个办公椅", self.ctx)
        assert out["data"]["extracted"]["quantity"] == 20

    @pytest.mark.asyncio
    async def test_english_intent(self):
        out = await self.proc.process("Please recommend a gaming laptop", self.ctx)
        assert out["data"]["intent"]["recommend"] is True
        assert out["data"]["language"] == "en"


# --- image ---


class TestImageProcessor:
    def setup_method(self):
        # Use the local stub so tests don't need Azure creds.
        svc = ImageRecognitionService()
        svc._client = LocalImageRecognitionClient()
        self.proc = ImageInputProcessor(recognition_service=svc)
        self.ctx = InputContext(user_id="u1")

    def test_validate_rejects_empty(self):
        with pytest.raises(ValidationError):
            self.proc.validate(b"")

    def test_validate_rejects_garbage(self):
        with pytest.raises(ValidationError):
            self.proc.validate(b"not-an-image")

    def test_validate_accepts_png(self, tiny_png_bytes):
        self.proc.validate(tiny_png_bytes)

    @pytest.mark.asyncio
    async def test_process_returns_metadata(self, tiny_png_bytes):
        out = await self.proc.process(tiny_png_bytes, self.ctx)
        info = out["data"]["image_info"]
        assert info["format"] == "PNG"
        assert info["width"] == 1 and info["height"] == 1
        assert info["size_bytes"] == len(tiny_png_bytes)


# --- link ---


class TestLinkProcessor:
    def setup_method(self):
        self.proc = LinkInputProcessor()
        self.ctx = InputContext(user_id="u1")

    def test_validate_rejects_empty(self):
        with pytest.raises(ValidationError):
            self.proc.validate("")

    def test_validate_rejects_non_http(self):
        with pytest.raises(ValidationError):
            self.proc.validate("ftp://example.com")

    def test_platform_jd(self):
        assert self.proc._detect_platform("item.jd.com") == "jd"

    def test_platform_taobao(self):
        assert self.proc._detect_platform("item.taobao.com") == "taobao"

    def test_platform_unknown(self):
        assert self.proc._detect_platform("example.com") is None

    def test_extract_product_id_jd(self):
        from urllib.parse import urlparse

        parsed = urlparse("https://item.jd.com/100012043978.html")
        assert self.proc._extract_product_id(parsed, "jd") == "100012043978"

    def test_extract_product_id_taobao(self):
        from urllib.parse import urlparse

        parsed = urlparse("https://item.taobao.com/item.htm?id=12345&x=1")
        assert self.proc._extract_product_id(parsed, "taobao") == "12345"


# --- voice ---


class TestVoiceProcessor:
    def setup_method(self):
        svc = SpeechRecognitionService()
        svc._client = LocalSpeechClient()
        self.proc = VoiceInputProcessor(speech_service=svc)
        self.ctx = InputContext(user_id="u1")

    def test_validate_rejects_empty(self):
        with pytest.raises(ValidationError):
            self.proc.validate(b"")

    def test_validate_rejects_unknown_format(self):
        with pytest.raises(UnsupportedFormatError):
            self.proc.validate(b"\x00" * 100)

    def test_validate_accepts_wav(self, tiny_wav_bytes):
        self.proc.validate(tiny_wav_bytes)

    @pytest.mark.asyncio
    async def test_process_returns_transcript(self, tiny_wav_bytes):
        out = await self.proc.process(tiny_wav_bytes, self.ctx)
        assert out["data"]["audio_info"]["format"] == "wav"
        assert "local speech stub" in out["data"]["transcript"]
