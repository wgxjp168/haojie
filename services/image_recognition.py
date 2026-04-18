"""Image recognition service with a pluggable provider."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx

from config.settings import ImageRecognitionProvider, Settings, get_settings
from core.exceptions import ExternalServiceError
from core.logging import get_logger
from core.metrics import EXTERNAL_API_CALLS_TOTAL, EXTERNAL_API_LATENCY_SECONDS

logger = get_logger(__name__)


class ImageRecognitionClient(ABC):
    service_name: str = "image"

    @abstractmethod
    async def recognize(self, image_data: bytes) -> Dict[str, Any]: ...

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:  # pragma: no cover
        return None


class LocalImageRecognitionClient(ImageRecognitionClient):
    """Stub client — returns empty structured results."""

    service_name = "image_local"

    async def recognize(self, image_data: bytes) -> Dict[str, Any]:
        return {
            "objects": [],
            "text": [],
            "brands": [],
            "tags": [],
            "confidence": 0.0,
            "provider": "local",
        }


class AzureImageRecognitionClient(ImageRecognitionClient):
    """Azure Computer Vision — Analyze Image API."""

    service_name = "azure_image"

    def __init__(self, settings: Settings) -> None:
        if not settings.azure_cv_endpoint or not settings.azure_cv_key:
            raise ValueError(
                "Azure Computer Vision requires azure_cv_endpoint and azure_cv_key"
            )
        self._endpoint = settings.azure_cv_endpoint.rstrip("/")
        self._key = settings.azure_cv_key
        self._client = httpx.AsyncClient(timeout=30.0)

    async def recognize(self, image_data: bytes) -> Dict[str, Any]:
        url = f"{self._endpoint}/vision/v3.2/analyze"
        params = {
            "visualFeatures": "Categories,Tags,Description,Objects,Brands",
            "details": "Landmarks",
        }
        resp = await self._client.post(
            url,
            params=params,
            headers={
                "Ocp-Apim-Subscription-Key": self._key,
                "Content-Type": "application/octet-stream",
            },
            content=image_data,
        )
        if resp.status_code != 200:
            raise ExternalServiceError(
                self.service_name,
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                resp.status_code,
            )
        body = resp.json()
        return {
            "objects": [
                {"name": o.get("object"), "confidence": o.get("confidence")}
                for o in body.get("objects", [])
            ],
            "tags": [
                {"name": t.get("name"), "confidence": t.get("confidence")}
                for t in body.get("tags", [])
            ],
            "brands": [
                {"name": b.get("name"), "confidence": b.get("confidence")}
                for b in body.get("brands", [])
            ],
            "description": (body.get("description", {}).get("captions") or [{}])[0].get(
                "text", ""
            ),
            "provider": "azure",
            "raw": body,
        }

    async def health_check(self) -> bool:
        try:
            # 1x1 png
            tiny = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff"
                b"\xff?\x00\x05\xfe\x02\xfe\xa3\x16#\xf0\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            await self.recognize(tiny)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


class ImageRecognitionService:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._client: ImageRecognitionClient = self._build_client()

    def _build_client(self) -> ImageRecognitionClient:
        provider = self._settings.image_recognition_provider
        if provider == ImageRecognitionProvider.AZURE:
            return AzureImageRecognitionClient(self._settings)
        return LocalImageRecognitionClient()

    async def recognize(self, image_data: bytes) -> Dict[str, Any]:
        started = time.perf_counter()
        status = "success"
        try:
            return await self._client.recognize(image_data)
        except ExternalServiceError:
            status = "error"
            raise
        except Exception as e:
            status = "error"
            raise ExternalServiceError(self._client.service_name, str(e)) from e
        finally:
            duration = time.perf_counter() - started
            EXTERNAL_API_CALLS_TOTAL.labels(
                service=self._client.service_name, status=status
            ).inc()
            EXTERNAL_API_LATENCY_SECONDS.labels(
                service=self._client.service_name
            ).observe(duration)

    async def health_check(self) -> bool:
        return await self._client.health_check()

    async def close(self) -> None:
        await self._client.close()


_image_service: Optional[ImageRecognitionService] = None


def get_image_recognition_service() -> ImageRecognitionService:
    global _image_service
    if _image_service is None:
        _image_service = ImageRecognitionService()
    return _image_service
