"""Report (Part 3.4) stage client — embedded + HTTP."""
from __future__ import annotations

from typing import Any, Dict

from hub.clients.base import BaseStageClient
from hub.core.exceptions import UpstreamFailedError


class ReportClient(BaseStageClient):
    stage = "report"

    @property
    def _http_path(self) -> str:
        return "/api/v1/reports/generate"

    async def _call_embedded(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from report.schemas import ReportRequest
            from report.service import get_report_service
        except Exception as exc:  # pragma: no cover - misconfig
            raise UpstreamFailedError(
                f"report embedded mode unavailable: {exc}",
                details={"stage": self.stage},
            ) from exc

        try:
            request = ReportRequest(**payload)
        except Exception as exc:
            raise UpstreamFailedError(
                f"report request validation failed: {exc}",
                details={"stage": self.stage},
            ) from exc

        service = get_report_service()
        try:
            response = service.generate(request)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamFailedError(
                f"report service call failed: {exc}",
                details={"stage": self.stage},
            ) from exc
        return response.model_dump(mode="json")
