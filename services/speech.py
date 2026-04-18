"""Speech-to-text service with a pluggable provider.

The provider is chosen via ``Settings.speech_provider``. The ``local``
provider is a deterministic stub used in tests and dev; real providers
(Azure, Google) call out to their respective APIs.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx

from config.settings import Settings, SpeechProvider, get_settings
from core.exceptions import ExternalServiceError
from core.logging import get_logger
from core.metrics import EXTERNAL_API_CALLS_TOTAL, EXTERNAL_API_LATENCY_SECONDS

logger = get_logger(__name__)


class SpeechClient(ABC):
    service_name: str = "speech"

    @abstractmethod
    async def recognize(
        self,
        audio_data: bytes,
        language: str,
        sample_rate: int = 16_000,
    ) -> Dict[str, Any]: ...

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:  # pragma: no cover — no resources by default
        return None


class LocalSpeechClient(SpeechClient):
    """Stub that returns a canned transcript. Useful for dev/tests."""

    service_name = "speech_local"

    async def recognize(
        self,
        audio_data: bytes,
        language: str,
        sample_rate: int = 16_000,
    ) -> Dict[str, Any]:
        # Deterministic "transcript" so tests can assert on it.
        return {
            "text": "[local speech stub] received audio of %d bytes" % len(audio_data),
            "language": language,
            "confidence": 0.5,
            "provider": "local",
        }


class AzureSpeechClient(SpeechClient):
    """Azure Cognitive Services - Speech to Text."""

    service_name = "azure_speech"

    def __init__(self, settings: Settings) -> None:
        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise ValueError("Azure Speech requires azure_speech_key and azure_speech_region")
        self._key = settings.azure_speech_key
        self._region = settings.azure_speech_region
        self._client = httpx.AsyncClient(timeout=30.0)
        self._token: Optional[str] = None
        self._token_expiry: float = 0

    async def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        url = f"https://{self._region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        resp = await self._client.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": self._key,
                "Content-Length": "0",
            },
        )
        if resp.status_code != 200:
            raise ExternalServiceError(
                self.service_name,
                f"token request failed: HTTP {resp.status_code}",
                resp.status_code,
            )
        self._token = resp.text
        self._token_expiry = time.time() + 540  # 9 minutes (Azure gives 10)
        return self._token

    async def recognize(
        self,
        audio_data: bytes,
        language: str,
        sample_rate: int = 16_000,
    ) -> Dict[str, Any]:
        token = await self._get_token()
        url = (
            f"https://{self._region}.stt.speech.microsoft.com"
            "/speech/recognition/conversation/cognitiveservices/v1"
        )
        resp = await self._client.post(
            url,
            params={"language": language, "format": "detailed"},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"audio/wav; codec=audio/pcm; samplerate={sample_rate}",
                "Accept": "application/json",
            },
            content=audio_data,
        )
        if resp.status_code != 200:
            raise ExternalServiceError(
                self.service_name,
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                resp.status_code,
            )
        body = resp.json()
        if body.get("RecognitionStatus") != "Success":
            raise ExternalServiceError(
                self.service_name, f"recognition status: {body.get('RecognitionStatus')}"
            )
        n_best = body.get("NBest") or []
        best = n_best[0] if n_best else {}
        return {
            "text": best.get("Display") or body.get("DisplayText", ""),
            "language": language,
            "confidence": float(best.get("Confidence", 0.0)),
            "provider": "azure",
            "raw": body,
        }

    async def health_check(self) -> bool:
        try:
            await self._get_token()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


class SpeechRecognitionService:
    """Façade used by the voice processor."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._client: SpeechClient = self._build_client()

    def _build_client(self) -> SpeechClient:
        provider = self._settings.speech_provider
        if provider == SpeechProvider.AZURE:
            return AzureSpeechClient(self._settings)
        # TODO(p2): wire GoogleSpeechClient when the GCP creds are provisioned.
        return LocalSpeechClient()

    async def recognize(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        sample_rate: int = 16_000,
    ) -> Dict[str, Any]:
        lang = language or self._settings.speech_default_language
        started = time.perf_counter()
        status = "success"
        try:
            result = await self._client.recognize(audio_data, lang, sample_rate)
            return result
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


_speech_service: Optional[SpeechRecognitionService] = None


def get_speech_service() -> SpeechRecognitionService:
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechRecognitionService()
    return _speech_service
