"""LLM (Part 3.3) stage client — embedded + HTTP."""
from __future__ import annotations

from typing import Any, Dict

from hub.clients.base import BaseStageClient
from hub.core.exceptions import UpstreamFailedError


class LLMClient(BaseStageClient):
    stage = "llm"

    @property
    def _http_path(self) -> str:
        return "/api/v1/completions"

    async def _call_embedded(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from llm.schemas import CompletionRequest
            from llm.service import get_llm_service
        except Exception as exc:  # pragma: no cover - misconfig
            raise UpstreamFailedError(
                f"llm embedded mode unavailable: {exc}",
                details={"stage": self.stage},
            ) from exc

        try:
            request = CompletionRequest(**payload)
        except Exception as exc:
            raise UpstreamFailedError(
                f"llm request validation failed: {exc}",
                details={"stage": self.stage},
            ) from exc

        service = get_llm_service()
        try:
            response = await service.complete(request)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamFailedError(
                f"llm service call failed: {exc}",
                details={"stage": self.stage},
            ) from exc
        return response.model_dump(mode="json")
