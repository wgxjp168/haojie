"""Multi-modal input manager.

Orchestrates the full path for a single input:

  validate → process → attach session context → cache/store → return

Features:

- Per-input-type circuit breakers to protect against failing dependencies.
- Per-user in-process rate limiter (on top of the HTTP middleware).
- Optional cache of processed results keyed by input hash.
- Optional storage of the raw payload to object storage.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cache.manager import CacheManager, get_cache_manager
from config.settings import Settings, get_settings
from core.exceptions import (
    ErrorCode,
    ExternalServiceError,
    InputProcessorError,
    ProcessingFailedError,
    RateLimitExceededError,
    UnsupportedFormatError,
    ValidationError,
)
from core.logging import get_logger
from core.metrics import (
    INPUT_PROCESSING_SECONDS,
    INPUT_REQUESTS_TOTAL,
    INPUT_SIZE_BYTES,
)
from entities.enums import DeviceType, InputType, PriorityLevel, ProcessingMode, ProcessingStatus
from entities.session import InputItem
from managers.session_manager import SessionManager, get_session_manager
from processors.base import BaseInputProcessor, InputContext
from processors.image_processor import ImageInputProcessor
from processors.link_processor import LinkInputProcessor
from processors.text_processor import TextInputProcessor
from processors.voice_processor import VoiceInputProcessor
from storage.base import StorageManager, get_storage_manager

logger = get_logger(__name__)


# --- request / response / config ---


@dataclass
class ProcessingConfig:
    mode: ProcessingMode = ProcessingMode.SYNC
    priority: PriorityLevel = PriorityLevel.NORMAL
    timeout_seconds: int = 30
    validate: bool = True
    store_raw: bool = False  # keep raw input in object storage
    cache_result: bool = True
    cache_ttl_seconds: Optional[int] = None
    language: Optional[str] = None


@dataclass
class InputRequest:
    input_data: Any
    input_type: InputType
    user_id: str
    session_id: Optional[str] = None
    config: ProcessingConfig = field(default_factory=ProcessingConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def hash_key(self) -> str:
        h = hashlib.sha256()
        h.update(self.input_type.value.encode())
        if isinstance(self.input_data, (bytes, bytearray)):
            h.update(self.input_data)
        elif isinstance(self.input_data, str):
            h.update(self.input_data.encode("utf-8"))
        else:
            h.update(repr(self.input_data).encode("utf-8"))
        if self.config.language:
            h.update(self.config.language.encode())
        return h.hexdigest()


@dataclass
class ProcessingResult:
    success: bool
    input_id: str
    session_id: str
    user_id: str
    input_type: InputType
    started_at: datetime
    completed_at: datetime
    processing_time_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    cached: bool = False
    storage_key: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "input_id": self.input_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "input_type": self.input_type.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "data": self.data,
            "warnings": self.warnings,
            "error": self.error,
            "cached": self.cached,
            "storage_key": self.storage_key,
            "summary": self.summary,
        }


# --- helpers: circuit breaker, rate limiter ---


class CircuitBreaker:
    """Tiny asyncio-compatible circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: int = 30,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self._recovery:
                # half-open: allow the next call through.
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                logger.warning("circuit_breaker.opened", failures=self._failures)


class AsyncRateLimiter:
    """Fixed-window rate limiter, keyed by an arbitrary string."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._buckets: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        if self._max <= 0:
            return True
        now = time.monotonic()
        cutoff = now - self._window
        async with self._lock:
            bucket = self._buckets.setdefault(key, [])
            # evict old
            self._buckets[key] = [t for t in bucket if t > cutoff]
            if len(self._buckets[key]) >= self._max:
                return False
            self._buckets[key].append(now)
            return True


# --- manager ---


class MultiModalInputManager:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[CacheManager] = None,
        storage: Optional[StorageManager] = None,
        session_manager: Optional[SessionManager] = None,
        processors: Optional[Dict[InputType, BaseInputProcessor]] = None,
    ):
        self._settings = settings or get_settings()
        self._cache = cache or get_cache_manager()
        self._storage = storage or get_storage_manager()
        self._sessions = session_manager or get_session_manager()
        self._processors: Dict[InputType, BaseInputProcessor] = processors or {
            InputType.TEXT: TextInputProcessor(self._settings),
            InputType.IMAGE: ImageInputProcessor(self._settings),
            InputType.LINK: LinkInputProcessor(self._settings),
            InputType.VOICE: VoiceInputProcessor(self._settings),
        }
        self._breakers: Dict[InputType, CircuitBreaker] = {
            t: CircuitBreaker() for t in self._processors
        }
        self._rate_limiter = AsyncRateLimiter(
            max_requests=self._settings.rate_limit_per_minute,
            window_seconds=60,
        )

    # --- public API ---

    @property
    def processors(self) -> Dict[InputType, BaseInputProcessor]:
        return self._processors

    async def process(self, request: InputRequest) -> ProcessingResult:
        started_at = datetime.now(tz=timezone.utc)
        start_perf = time.perf_counter()
        input_type = request.input_type

        processor = self._processors.get(input_type)
        if processor is None:
            raise UnsupportedFormatError(
                input_type.value, [t.value for t in self._processors]
            )

        # Rate limit by user.
        if not await self._rate_limiter.allow(request.user_id):
            raise RateLimitExceededError(
                self._settings.rate_limit_per_minute, 60, retry_after=60
            )

        # Circuit breaker for this processor.
        breaker = self._breakers[input_type]
        if not await breaker.allow():
            raise ExternalServiceError(
                f"processor_{input_type.value}", "circuit breaker open"
            )

        INPUT_REQUESTS_TOTAL.labels(
            input_type=input_type.value, status="started"
        ).inc()
        if isinstance(request.input_data, (bytes, bytearray, str)):
            size = (
                len(request.input_data)
                if isinstance(request.input_data, (bytes, bytearray))
                else len(request.input_data.encode("utf-8"))
            )
            INPUT_SIZE_BYTES.labels(input_type=input_type.value).observe(size)

        # Ensure we have a session.
        session = None
        if request.session_id:
            session = await self._sessions.get_session(request.session_id)
        if session is None:
            session = await self._sessions.create_session(
                user_id=request.user_id, device_type=DeviceType.API
            )
        request.session_id = session.session_id

        # Try cache.
        cache_key = request.hash_key()
        if request.config.cache_result:
            cached_payload = await self._cache.get(
                cache_key, namespace=f"input:{input_type.value}"
            )
            if cached_payload is not None:
                result = self._result_from_cache(request, cached_payload, started_at, start_perf)
                logger.info(
                    "input.cache_hit",
                    input_type=input_type.value,
                    user_id=request.user_id,
                )
                INPUT_REQUESTS_TOTAL.labels(
                    input_type=input_type.value, status="cached"
                ).inc()
                return result

        # Validate & process.
        storage_key: Optional[str] = None
        try:
            if request.config.validate:
                processor.validate(request.input_data)

            if request.config.store_raw and isinstance(
                request.input_data, (bytes, bytearray)
            ):
                try:
                    storage_key = await self._storage.save_input(
                        bytes(request.input_data),
                        input_type.value,
                        request.user_id,
                        metadata={"input_id": request.input_id, "request_id": request.request_id},
                    )
                except Exception as e:
                    logger.warning("storage.save_failed", error=str(e))

            context = InputContext(
                user_id=request.user_id,
                session_id=request.session_id,
                request_id=request.request_id,
                language=request.config.language,
                user_profile=request.metadata.get("user_profile"),
                metadata=request.metadata,
            )

            raw_result = await asyncio.wait_for(
                processor.process(request.input_data, context),
                timeout=request.config.timeout_seconds,
            )

            await breaker.record_success()
        except (ValidationError, UnsupportedFormatError):
            INPUT_REQUESTS_TOTAL.labels(
                input_type=input_type.value, status="invalid"
            ).inc()
            raise
        except asyncio.TimeoutError as e:
            await breaker.record_failure()
            INPUT_REQUESTS_TOTAL.labels(
                input_type=input_type.value, status="timeout"
            ).inc()
            raise InputProcessorError(
                code=ErrorCode.PROCESSING_TIMEOUT,
                message=f"processing timed out after {request.config.timeout_seconds}s",
            ) from e
        except InputProcessorError:
            await breaker.record_failure()
            INPUT_REQUESTS_TOTAL.labels(
                input_type=input_type.value, status="failed"
            ).inc()
            raise
        except Exception as e:  # unexpected — log & wrap
            await breaker.record_failure()
            logger.exception(
                "input.unexpected_error",
                input_type=input_type.value,
                user_id=request.user_id,
            )
            INPUT_REQUESTS_TOTAL.labels(
                input_type=input_type.value, status="failed"
            ).inc()
            raise ProcessingFailedError(str(e)) from e

        completed_at = datetime.now(tz=timezone.utc)
        duration_ms = (time.perf_counter() - start_perf) * 1_000
        INPUT_PROCESSING_SECONDS.labels(input_type=input_type.value).observe(
            duration_ms / 1_000
        )
        INPUT_REQUESTS_TOTAL.labels(
            input_type=input_type.value, status="success"
        ).inc()

        result = ProcessingResult(
            success=True,
            input_id=request.input_id,
            session_id=request.session_id,
            user_id=request.user_id,
            input_type=input_type,
            started_at=started_at,
            completed_at=completed_at,
            processing_time_ms=duration_ms,
            data=raw_result.get("data", {}),
            warnings=list(raw_result.get("warnings", [])),
            storage_key=storage_key,
            summary=raw_result.get("summary"),
        )

        # Append to session.
        await self._sessions.add_input(
            request.session_id,
            InputItem(
                input_id=request.input_id,
                session_id=request.session_id,
                user_id=request.user_id,
                input_type=input_type,
                input_summary=result.summary or input_type.value,
                status=ProcessingStatus.SUCCESS,
                processed_data=result.data,
                processing_time_ms=duration_ms,
                processed_at=completed_at,
                metadata={"storage_key": storage_key} if storage_key else {},
            ),
        )

        if request.config.cache_result:
            await self._cache.set(
                cache_key,
                {
                    "data": result.data,
                    "warnings": result.warnings,
                    "summary": result.summary,
                },
                namespace=f"input:{input_type.value}",
                ttl=request.config.cache_ttl_seconds,
            )

        return result

    async def process_batch(
        self, requests: List[InputRequest]
    ) -> List[ProcessingResult]:
        """Process many requests concurrently with bounded parallelism."""
        sem = asyncio.Semaphore(10)

        async def _one(req: InputRequest) -> ProcessingResult:
            async with sem:
                try:
                    return await self.process(req)
                except InputProcessorError as e:
                    return ProcessingResult(
                        success=False,
                        input_id=req.input_id,
                        session_id=req.session_id or "",
                        user_id=req.user_id,
                        input_type=req.input_type,
                        started_at=datetime.now(tz=timezone.utc),
                        completed_at=datetime.now(tz=timezone.utc),
                        processing_time_ms=0.0,
                        error=e.to_dict(),
                    )

        return await asyncio.gather(*(_one(r) for r in requests))

    def _result_from_cache(
        self,
        request: InputRequest,
        cached_payload: Dict[str, Any],
        started_at: datetime,
        start_perf: float,
    ) -> ProcessingResult:
        completed_at = datetime.now(tz=timezone.utc)
        duration_ms = (time.perf_counter() - start_perf) * 1_000
        return ProcessingResult(
            success=True,
            input_id=request.input_id,
            session_id=request.session_id or "",
            user_id=request.user_id,
            input_type=request.input_type,
            started_at=started_at,
            completed_at=completed_at,
            processing_time_ms=duration_ms,
            data=cached_payload.get("data", {}),
            warnings=list(cached_payload.get("warnings", [])),
            summary=cached_payload.get("summary"),
            cached=True,
        )

    # --- health ---

    async def health_check(self) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}
        for itype, proc in self._processors.items():
            try:
                checks[itype.value] = {"healthy": await proc.health_check()}
            except Exception as e:
                checks[itype.value] = {"healthy": False, "error": str(e)}
        checks["cache"] = {"healthy": await self._cache.ping()}
        checks["storage"] = {"healthy": await self._storage.health_check()}
        overall = all(c.get("healthy") for c in checks.values())
        return {"overall": overall, "components": checks}

    async def close(self) -> None:
        for proc in self._processors.values():
            try:
                await proc.close()
            except Exception:
                pass


# --- singleton accessor ---


_input_manager: Optional[MultiModalInputManager] = None


def get_input_manager() -> MultiModalInputManager:
    global _input_manager
    if _input_manager is None:
        _input_manager = MultiModalInputManager()
    return _input_manager
