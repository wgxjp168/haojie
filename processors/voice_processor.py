"""Voice input processor.

Validates the upload (size, magic bytes), then asks the speech
recognition service to transcribe. The transcript is the real output —
it is what the AI decision hub consumes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings
from core.exceptions import (
    ExternalServiceError,
    InputTooLargeError,
    UnsupportedFormatError,
    ValidationError,
)
from core.logging import get_logger
from entities.enums import InputType
from processors.base import BaseInputProcessor, InputContext, ProcessorResult
from services.speech import SpeechRecognitionService, get_speech_service

logger = get_logger(__name__)

_FORMATS = ("wav", "mp3", "m4a", "ogg", "flac", "opus")

_MAGIC_PATTERNS = {
    "wav": (b"RIFF", 0, b"WAVE", 8),
    "mp3_id3": (b"ID3", 0, None, None),
    "mp3_frame": (b"\xff\xfb", 0, None, None),
    "mp3_frame2": (b"\xff\xf3", 0, None, None),
    "mp3_frame3": (b"\xff\xf2", 0, None, None),
    "flac": (b"fLaC", 0, None, None),
    "ogg": (b"OggS", 0, None, None),
    "m4a": (b"ftyp", 4, None, None),
}


def _sniff_format(data: bytes) -> Optional[str]:
    if len(data) < 12:
        return None
    for name, (magic, offset, *_rest) in _MAGIC_PATTERNS.items():
        if data[offset : offset + len(magic)] == magic:
            if name.startswith("mp3"):
                return "mp3"
            if name == "m4a":
                return "m4a"
            return name
    return None


class VoiceInputProcessor(BaseInputProcessor):
    input_type = InputType.VOICE

    def __init__(
        self,
        settings: Optional[Settings] = None,
        speech_service: Optional[SpeechRecognitionService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._speech = speech_service or get_speech_service()

    # --- validation ---

    def validate(self, input_data: Any) -> None:
        if not isinstance(input_data, (bytes, bytearray)):
            raise ValidationError("input must be bytes", field="audio")
        size = len(input_data)
        if size == 0:
            raise ValidationError("empty audio", field="audio")
        max_bytes = self._settings.voice_max_size_mb * 1024 * 1024
        if size > max_bytes:
            raise InputTooLargeError(size, max_bytes)
        fmt = _sniff_format(bytes(input_data))
        if fmt is None:
            raise UnsupportedFormatError("unknown", list(_FORMATS))

    # --- processing ---

    async def process(self, input_data: bytes, context: InputContext) -> ProcessorResult:
        fmt = _sniff_format(input_data) or "unknown"
        language = context.language or self._settings.speech_default_language

        warnings: List[str] = []
        transcript = ""
        confidence = 0.0
        provider = "unavailable"

        try:
            result = await self._speech.recognize(input_data, language=language)
            transcript = result.get("text", "")
            confidence = float(result.get("confidence", 0.0))
            provider = result.get("provider", "unknown")
        except ExternalServiceError as e:
            logger.warning(
                "voice.transcription_failed",
                user_id=context.user_id,
                error=str(e),
            )
            warnings.append(f"transcription failed: {e.message}")

        data: Dict[str, Any] = {
            "audio_info": {
                "format": fmt,
                "size_bytes": len(input_data),
            },
            "transcript": transcript,
            "language": language,
            "confidence": confidence,
            "provider": provider,
        }
        summary = (
            f"voice[{fmt}, {len(input_data)} bytes, {language}]: "
            f"'{transcript[:80]}' (confidence={confidence:.2f})"
        )
        return {"data": data, "warnings": warnings, "summary": summary}

    @property
    def supported_formats(self):  # type: ignore[override]
        return list(_FORMATS)

    async def health_check(self) -> bool:
        return await self._speech.health_check()
