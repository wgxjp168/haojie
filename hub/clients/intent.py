"""Intent (Part 3.1) stage client — embedded + HTTP."""
from __future__ import annotations

from typing import Any, Dict

from hub.clients.base import BaseStageClient
from hub.core.exceptions import UpstreamFailedError


class IntentClient(BaseStageClient):
    stage = "intent"

    @property
    def _http_path(self) -> str:
        return "/api/v1/intents/predict"

    async def _call_embedded(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Import lazily so embedded mode does not force intent-service imports
        # on hub instances configured for http-only.
        try:
            from intent.schemas import IntentRequest
            from intent.service import get_intent_service
        except Exception as exc:  # pragma: no cover - misconfig
            raise UpstreamFailedError(
                f"intent embedded mode unavailable: {exc}",
                details={"stage": self.stage},
            ) from exc

        try:
            request = IntentRequest(**payload)
        except Exception as exc:
            raise UpstreamFailedError(
                f"intent request validation failed: {exc}",
                details={"stage": self.stage, "payload": _redact(payload)},
            ) from exc

        service = get_intent_service()
        try:
            response = service.predict(request)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamFailedError(
                f"intent service call failed: {exc}",
                details={"stage": self.stage},
            ) from exc
        return response.model_dump(mode="json")


def _redact(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Avoid leaking the full user text into error details."""
    safe = dict(payload)
    text = safe.get("text")
    if isinstance(text, str) and len(text) > 80:
        safe["text"] = text[:80] + "..."
    return safe
