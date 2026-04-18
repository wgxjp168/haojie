"""Image input processor.

Validates the upload (size, format), extracts basic metadata with Pillow,
and then delegates recognition (objects / brands / text) to the
:class:`ImageRecognitionService`. Recognition failures are captured as
warnings so the caller still gets the image metadata.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional

from PIL import Image, UnidentifiedImageError

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
from services.image_recognition import ImageRecognitionService, get_image_recognition_service

logger = get_logger(__name__)

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "BMP")


class ImageInputProcessor(BaseInputProcessor):
    input_type = InputType.IMAGE

    def __init__(
        self,
        settings: Optional[Settings] = None,
        recognition_service: Optional[ImageRecognitionService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._recognition = recognition_service or get_image_recognition_service()

    # --- validation ---

    def validate(self, input_data: Any) -> None:
        if not isinstance(input_data, (bytes, bytearray)):
            raise ValidationError("input must be bytes", field="image")
        size = len(input_data)
        if size == 0:
            raise ValidationError("empty image", field="image")
        max_bytes = self._settings.image_max_size_mb * 1024 * 1024
        if size > max_bytes:
            raise InputTooLargeError(size, max_bytes)
        # Header validation — Pillow will identify the format.
        try:
            with Image.open(io.BytesIO(input_data)) as img:
                img.verify()
                fmt = (img.format or "").upper()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValidationError(f"invalid image data: {e}", field="image") from e
        if fmt not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(fmt, list(SUPPORTED_FORMATS))

    # --- processing ---

    async def process(self, input_data: bytes, context: InputContext) -> ProcessorResult:
        meta = self._extract_metadata(input_data)
        warnings = []

        try:
            recognition = await self._recognition.recognize(input_data)
        except ExternalServiceError as e:
            logger.warning(
                "image.recognition_failed",
                user_id=context.user_id,
                error=str(e),
            )
            warnings.append(f"recognition failed: {e.message}")
            recognition = {
                "objects": [],
                "tags": [],
                "brands": [],
                "description": "",
                "provider": "unavailable",
            }

        data: Dict[str, Any] = {
            "image_info": meta,
            "recognition": recognition,
        }
        summary = (
            f"image[{meta['format']} {meta['width']}x{meta['height']}, "
            f"{meta['size_bytes']} bytes]: "
            f"{len(recognition.get('objects', []))} objects, "
            f"{len(recognition.get('brands', []))} brands"
        )
        return {"data": data, "warnings": warnings, "summary": summary}

    # --- helpers ---

    @staticmethod
    def _extract_metadata(data: bytes) -> Dict[str, Any]:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            return {
                "format": (img.format or "UNKNOWN").upper(),
                "width": width,
                "height": height,
                "mode": img.mode,
                "size_bytes": len(data),
                "aspect_ratio": round(width / height, 3) if height else 0.0,
                "is_animated": getattr(img, "is_animated", False),
            }

    @property
    def supported_formats(self):  # type: ignore[override]
        return [f.lower() for f in SUPPORTED_FORMATS]

    async def health_check(self) -> bool:
        return await self._recognition.health_check()
