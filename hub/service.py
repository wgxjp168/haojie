"""Public hub service — composes clients + pipeline orchestrator."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Dict, List, Optional

from hub.clients import (
    BaseStageClient,
    DecisionClient,
    IntentClient,
    LLMClient,
    ReportClient,
)
from hub.config import ClientMode, HubSettings, get_hub_settings
from hub.core.exceptions import ValidationHubError
from hub.core.logging import get_logger
from hub.pipeline import PipelineOrchestrator
from hub.schemas import (
    BatchPipelineRequest,
    BatchPipelineResponse,
    HealthResponse,
    PipelineRequest,
    PipelineResponse,
    StageHealth,
)

log = get_logger(__name__)


_STAGES = ("intent", "decision", "llm", "report")


def _build_default_clients(settings: HubSettings) -> Dict[str, BaseStageClient]:
    common = dict(
        max_retries=settings.http_max_retries,
        retry_backoff_seconds=settings.http_retry_backoff_seconds,
        failure_threshold=settings.cb_failure_threshold,
        recovery_seconds=settings.cb_recovery_seconds,
        api_key=settings.downstream_api_key,
    )
    return {
        "intent": IntentClient(
            mode=settings.intent_mode,
            base_url=settings.intent_base_url,
            timeout_seconds=settings.intent_timeout_seconds,
            **common,
        ),
        "decision": DecisionClient(
            mode=settings.decision_mode,
            base_url=settings.decision_base_url,
            timeout_seconds=settings.decision_timeout_seconds,
            **common,
        ),
        "llm": LLMClient(
            mode=settings.llm_mode,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            **common,
        ),
        "report": ReportClient(
            mode=settings.report_mode,
            base_url=settings.report_base_url,
            timeout_seconds=settings.report_timeout_seconds,
            **common,
        ),
    }


class HubService:
    """Top-level orchestrator exposed by the FastAPI app."""

    def __init__(
        self,
        *,
        settings: Optional[HubSettings] = None,
        clients: Optional[Dict[str, BaseStageClient]] = None,
        orchestrator: Optional[PipelineOrchestrator] = None,
    ) -> None:
        self.settings = settings or get_hub_settings()
        self.clients = clients or _build_default_clients(self.settings)
        if orchestrator is None:
            orchestrator = PipelineOrchestrator(
                settings=self.settings,
                intent_client=self.clients["intent"],
                decision_client=self.clients["decision"],
                llm_client=self.clients["llm"],
                report_client=self.clients["report"],
            )
        self.orchestrator = orchestrator
        self._started_at = time.time()

    # ------------------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    async def run_pipeline(self, request: PipelineRequest) -> PipelineResponse:
        self._validate_language(request)
        return await asyncio.wait_for(
            self.orchestrator.run(request),
            timeout=self.settings.pipeline_timeout_seconds,
        )

    async def run_batch(
        self, batch: BatchPipelineRequest
    ) -> BatchPipelineResponse:
        if len(batch.items) > self.settings.max_batch_size:
            raise ValidationHubError(
                f"batch size {len(batch.items)} exceeds max_batch_size "
                f"{self.settings.max_batch_size}",
                details={"max": self.settings.max_batch_size},
            )
        start = time.perf_counter()
        # Fan-out concurrently — each pipeline is already internally sequenced.
        results = await asyncio.gather(
            *(self.run_pipeline(item) for item in batch.items),
            return_exceptions=False,
        )
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return BatchPipelineResponse(
            results=list(results), total=len(results), latency_ms=latency_ms
        )

    def health(self) -> HealthResponse:
        stages: List[StageHealth] = []
        for name in _STAGES:
            client = self.clients[name]
            endpoint = client.base_url if client.mode == ClientMode.HTTP else None
            stages.append(
                StageHealth(
                    stage=name,
                    mode=client.mode.value,
                    available=client.available,
                    circuit_state=client.circuit_state,
                    endpoint=endpoint,
                )
            )
        return HealthResponse(
            status="ok",
            version=self.settings.version,
            uptime_seconds=round(self.uptime_seconds, 3),
            stages=stages,
        )

    # ------------------------------------------------------------------

    def _validate_language(self, request: PipelineRequest) -> None:
        if (
            request.language is not None
            and request.language not in self.settings.supported_languages
        ):
            raise ValidationHubError(
                f"language {request.language!r} not supported",
                details={
                    "supported": self.settings.supported_languages,
                },
            )


# -------- singleton helpers (used by FastAPI lifespan) --------

_service_lock = threading.Lock()
_service_instance: Optional[HubService] = None


def get_hub_service() -> HubService:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = HubService()
    return _service_instance


def reset_hub_service() -> None:
    global _service_instance
    with _service_lock:
        _service_instance = None
