"""Decision (Part 3.2) stage client — embedded + HTTP."""
from __future__ import annotations

from typing import Any, Dict

from hub.clients.base import BaseStageClient
from hub.core.exceptions import UpstreamFailedError


class DecisionClient(BaseStageClient):
    stage = "decision"

    @property
    def _http_path(self) -> str:
        return "/api/v1/decisions/evaluate"

    async def _call_embedded(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from decision.schemas import DecisionRequest
            from decision.service import get_decision_service
        except Exception as exc:  # pragma: no cover - misconfig
            raise UpstreamFailedError(
                f"decision embedded mode unavailable: {exc}",
                details={"stage": self.stage},
            ) from exc

        try:
            request = DecisionRequest(**payload)
        except Exception as exc:
            raise UpstreamFailedError(
                f"decision request validation failed: {exc}",
                details={"stage": self.stage},
            ) from exc

        service = get_decision_service()
        try:
            response = service.decide(request)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamFailedError(
                f"decision service call failed: {exc}",
                details={"stage": self.stage},
            ) from exc
        return response.model_dump(mode="json")
