"""LLM completion router.

Turns a ``CompletionRequest`` + configured strategy into a single
successful ``CompletionResponse`` by orchestrating one or more provider
calls. The router is aware of:
* per-provider circuit breakers,
* per-provider rate limits,
* the model catalog (pricing, context-window),
* the service-level default fallback chain.

It is *not* aware of HTTP/auth/caching — those live in ``LLMService``.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from llm.catalog import MODEL_CATALOG, ModelInfo, get_model
from llm.circuit_breaker import CircuitBreakerRegistry
from llm.config import LLMSettings, LLMStrategy
from llm.core.exceptions import (
    BudgetExceededError,
    CircuitOpenError,
    LLMError,
    NoProviderAvailableError,
    ProviderFailedError,
    ProviderTimeoutError,
    RateLimitedError,
    UnknownModelError,
)
from llm.core.logging import get_logger
from llm.core.metrics import (
    LLM_COST_TOTAL,
    LLM_DURATION,
    LLM_FALLBACK_TOTAL,
    LLM_PROVIDER_FAILURES,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
)
from llm.providers.base import BaseProvider, ProviderResult
from llm.rate_limiter import RateLimiterRegistry
from llm.schemas import (
    ChatMessage,
    CompletionAttempt,
    CompletionRequest,
    CompletionResponse,
    TokenUsage,
)

log = get_logger(__name__)


@dataclass
class _Candidate:
    """One completion candidate (successful provider result)."""

    model: ModelInfo
    result: ProviderResult
    cost_usd: float


class LLMRouter:
    """Strategy-aware completion orchestrator."""

    def __init__(
        self,
        *,
        settings: LLMSettings,
        providers: Dict[str, BaseProvider],
        breakers: Optional[CircuitBreakerRegistry] = None,
        rate_limiters: Optional[RateLimiterRegistry] = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.breakers = breakers or CircuitBreakerRegistry(
            failure_threshold=settings.cb_failure_threshold,
            recovery_seconds=settings.cb_recovery_seconds,
        )
        self.rate_limiters = rate_limiters or RateLimiterRegistry(
            rate_per_minute=settings.rate_limit_per_minute,
            burst=settings.rate_limit_burst,
        )

    # -------------------------------------------------------------- public

    async def run(self, request: CompletionRequest) -> CompletionResponse:
        start = time.perf_counter()
        strategy = self._resolve_strategy(request)
        targets = self._resolve_targets(request, strategy)

        if not targets:
            raise NoProviderAvailableError(
                "no configured targets resolved for this request",
                details={
                    "requested_model": request.model_id,
                    "fallback_chain": self.settings.fallback_chain,
                },
            )

        messages = request.normalized_messages()
        self._guard_prompt_size(messages)

        sampling = self._resolve_sampling(request)
        timeout = float(
            request.timeout_seconds or self.settings.request_timeout_seconds
        )

        attempts: List[CompletionAttempt] = []

        if strategy == LLMStrategy.SINGLE:
            candidate = await self._run_single(
                targets[0], messages, sampling, timeout, attempts
            )
            return self._build_response(
                request, candidate, attempts, strategy, start
            )

        if strategy == LLMStrategy.FALLBACK:
            candidate = await self._run_fallback(
                targets, messages, sampling, timeout, attempts
            )
            return self._build_response(
                request, candidate, attempts, strategy, start
            )

        # Parallel strategies all race providers concurrently.
        parallel_targets = targets[: self.settings.parallel_fanout]

        if strategy == LLMStrategy.PARALLEL_ANY:
            candidate = await self._run_parallel_any(
                parallel_targets, messages, sampling, timeout, attempts
            )
            return self._build_response(
                request, candidate, attempts, strategy, start
            )

        if strategy == LLMStrategy.PARALLEL_VOTE:
            candidate = await self._run_parallel_vote(
                parallel_targets, messages, sampling, timeout, attempts
            )
            return self._build_response(
                request, candidate, attempts, strategy, start
            )

        # CONFIDENCE
        candidate = await self._run_confidence(
            parallel_targets, messages, sampling, timeout, attempts
        )
        return self._build_response(request, candidate, attempts, strategy, start)

    # ---------------------------------------------------------- strategies

    async def _run_single(
        self,
        model: ModelInfo,
        messages: List[ChatMessage],
        sampling: Dict,
        timeout: float,
        attempts: List[CompletionAttempt],
    ) -> _Candidate:
        candidate = await self._attempt(model, messages, sampling, timeout, attempts)
        if candidate is None:
            raise self._attempts_to_error(attempts, "single strategy failed")
        return candidate

    async def _run_fallback(
        self,
        targets: List[ModelInfo],
        messages: List[ChatMessage],
        sampling: Dict,
        timeout: float,
        attempts: List[CompletionAttempt],
    ) -> _Candidate:
        previous: Optional[ModelInfo] = None
        for model in targets:
            candidate = await self._attempt(model, messages, sampling, timeout, attempts)
            if candidate is not None:
                return candidate
            if previous is not None:
                LLM_FALLBACK_TOTAL.labels(
                    from_provider=previous.provider,
                    to_provider=model.provider,
                    reason=attempts[-1].error_code or "unknown",
                ).inc()
            previous = model

        raise self._attempts_to_error(attempts, "all fallback targets failed")

    async def _run_parallel_any(
        self,
        targets: List[ModelInfo],
        messages: List[ChatMessage],
        sampling: Dict,
        timeout: float,
        attempts: List[CompletionAttempt],
    ) -> _Candidate:
        tasks = {
            asyncio.create_task(
                self._attempt(m, messages, sampling, timeout, attempts)
            ): m
            for m in targets
        }

        winner: Optional[_Candidate] = None
        try:
            for coro in asyncio.as_completed(tasks):
                candidate = await coro
                if candidate is not None:
                    winner = candidate
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Await cancellations so we don't leak warnings.
            await asyncio.gather(*tasks, return_exceptions=True)

        if winner is None:
            raise self._attempts_to_error(attempts, "no provider produced a response")
        return winner

    async def _run_parallel_vote(
        self,
        targets: List[ModelInfo],
        messages: List[ChatMessage],
        sampling: Dict,
        timeout: float,
        attempts: List[CompletionAttempt],
    ) -> _Candidate:
        candidates = await self._run_all(targets, messages, sampling, timeout, attempts)
        if not candidates:
            raise self._attempts_to_error(attempts, "no successful votes collected")

        # Vote on the trimmed text to avoid trivial whitespace splitting
        # the vote. Ties fall to the highest-confidence candidate.
        tally: Counter[str] = Counter()
        by_key: Dict[str, List[_Candidate]] = {}
        for cand in candidates:
            key = cand.result.content.strip()
            tally[key] += 1
            by_key.setdefault(key, []).append(cand)

        top_key, _ = max(
            tally.items(),
            key=lambda kv: (kv[1], max(c.result.confidence for c in by_key[kv[0]])),
        )
        best = max(by_key[top_key], key=lambda c: c.result.confidence)
        return best

    async def _run_confidence(
        self,
        targets: List[ModelInfo],
        messages: List[ChatMessage],
        sampling: Dict,
        timeout: float,
        attempts: List[CompletionAttempt],
    ) -> _Candidate:
        candidates = await self._run_all(targets, messages, sampling, timeout, attempts)
        if not candidates:
            raise self._attempts_to_error(
                attempts, "no confident candidate available"
            )
        return max(candidates, key=lambda c: c.result.confidence)

    async def _run_all(
        self,
        targets: List[ModelInfo],
        messages: List[ChatMessage],
        sampling: Dict,
        timeout: float,
        attempts: List[CompletionAttempt],
    ) -> List[_Candidate]:
        tasks = [
            self._attempt(m, messages, sampling, timeout, attempts) for m in targets
        ]
        results = await asyncio.gather(*tasks)
        return [c for c in results if c is not None]

    # ------------------------------------------------------------- internals

    async def _attempt(
        self,
        model: ModelInfo,
        messages: List[ChatMessage],
        sampling: Dict,
        timeout: float,
        attempts: List[CompletionAttempt],
    ) -> Optional[_Candidate]:
        """Make one provider attempt and record a CompletionAttempt."""
        provider = self.providers.get(model.provider)
        start = time.perf_counter()

        if provider is None or not provider.available:
            attempts.append(
                CompletionAttempt(
                    provider=model.provider,
                    model_id=model.model_id,
                    remote_name=model.remote_name,
                    status="skipped",
                    latency_ms=0.0,
                    error_code="provider_unavailable",
                    error_message=f"provider {model.provider!r} not configured",
                )
            )
            LLM_PROVIDER_FAILURES.labels(
                provider=model.provider, reason="unavailable"
            ).inc()
            return None

        breaker = self.breakers.get(model.provider)
        try:
            breaker.ensure_allowed()
            self.rate_limiters.ensure_allowed(model.provider)
        except CircuitOpenError as exc:
            attempts.append(
                CompletionAttempt(
                    provider=model.provider,
                    model_id=model.model_id,
                    remote_name=model.remote_name,
                    status="skipped",
                    latency_ms=0.0,
                    error_code=exc.code.value,
                    error_message=exc.message,
                )
            )
            return None
        except RateLimitedError as exc:
            attempts.append(
                CompletionAttempt(
                    provider=model.provider,
                    model_id=model.model_id,
                    remote_name=model.remote_name,
                    status="skipped",
                    latency_ms=0.0,
                    error_code=exc.code.value,
                    error_message=exc.message,
                )
            )
            return None

        try:
            result = await provider.complete(
                model=model,
                messages=messages,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                max_tokens=sampling["max_tokens"],
                stop=sampling["stop"],
                timeout_seconds=timeout,
            )
        except ProviderTimeoutError as exc:
            breaker.record_failure()
            latency_ms = (time.perf_counter() - start) * 1000.0
            attempts.append(
                CompletionAttempt(
                    provider=model.provider,
                    model_id=model.model_id,
                    remote_name=model.remote_name,
                    status="failure",
                    latency_ms=round(latency_ms, 3),
                    error_code=exc.code.value,
                    error_message=exc.message,
                )
            )
            LLM_PROVIDER_FAILURES.labels(
                provider=model.provider, reason="timeout"
            ).inc()
            LLM_REQUESTS_TOTAL.labels(
                provider=model.provider, model=model.model_id, status="timeout"
            ).inc()
            return None
        except LLMError as exc:
            breaker.record_failure()
            latency_ms = (time.perf_counter() - start) * 1000.0
            reason = self._classify_failure(exc)
            attempts.append(
                CompletionAttempt(
                    provider=model.provider,
                    model_id=model.model_id,
                    remote_name=model.remote_name,
                    status="failure",
                    latency_ms=round(latency_ms, 3),
                    error_code=exc.code.value,
                    error_message=exc.message,
                )
            )
            LLM_PROVIDER_FAILURES.labels(
                provider=model.provider, reason=reason
            ).inc()
            LLM_REQUESTS_TOTAL.labels(
                provider=model.provider, model=model.model_id, status="error"
            ).inc()
            return None
        except Exception as exc:  # noqa: BLE001
            breaker.record_failure()
            latency_ms = (time.perf_counter() - start) * 1000.0
            attempts.append(
                CompletionAttempt(
                    provider=model.provider,
                    model_id=model.model_id,
                    remote_name=model.remote_name,
                    status="failure",
                    latency_ms=round(latency_ms, 3),
                    error_code="INTERNAL_ERROR",
                    error_message=str(exc),
                )
            )
            LLM_PROVIDER_FAILURES.labels(
                provider=model.provider, reason="unknown"
            ).inc()
            LLM_REQUESTS_TOTAL.labels(
                provider=model.provider, model=model.model_id, status="error"
            ).inc()
            return None

        breaker.record_success()
        cost_usd = model.estimated_cost_usd(
            result.tokens.prompt_tokens, result.tokens.completion_tokens
        )

        # Budget guardrail: refuse to emit an attempt whose actual cost
        # crossed the per-request ceiling. We still return the candidate
        # so downstream can attribute the cost, but the metric tracks it.
        if cost_usd > self.settings.max_request_cost_usd:
            log.warning(
                "llm.router.budget_exceeded",
                provider=model.provider,
                model=model.model_id,
                cost=cost_usd,
                cap=self.settings.max_request_cost_usd,
            )

        LLM_REQUESTS_TOTAL.labels(
            provider=model.provider, model=model.model_id, status="ok"
        ).inc()
        LLM_DURATION.labels(
            provider=model.provider, model=model.model_id
        ).observe(result.latency_ms / 1000.0)
        LLM_TOKENS_TOTAL.labels(
            provider=model.provider, model=model.model_id, kind="prompt"
        ).inc(result.tokens.prompt_tokens)
        LLM_TOKENS_TOTAL.labels(
            provider=model.provider, model=model.model_id, kind="completion"
        ).inc(result.tokens.completion_tokens)
        LLM_TOKENS_TOTAL.labels(
            provider=model.provider, model=model.model_id, kind="total"
        ).inc(result.tokens.total_tokens)
        LLM_COST_TOTAL.labels(
            provider=model.provider, model=model.model_id
        ).inc(cost_usd)

        attempts.append(
            CompletionAttempt(
                provider=model.provider,
                model_id=model.model_id,
                remote_name=model.remote_name,
                status="success",
                latency_ms=result.latency_ms,
                tokens=result.tokens,
                cost_usd=cost_usd,
            )
        )
        return _Candidate(model=model, result=result, cost_usd=cost_usd)

    def _classify_failure(self, exc: LLMError) -> str:
        if isinstance(exc, ProviderTimeoutError):
            return "timeout"
        if isinstance(exc, ProviderFailedError):
            return "http_error"
        return "unknown"

    def _attempts_to_error(
        self, attempts: List[CompletionAttempt], message: str
    ) -> LLMError:
        skipped = [a for a in attempts if a.status == "skipped"]
        failed = [a for a in attempts if a.status == "failure"]
        if failed:
            last = failed[-1]
            return ProviderFailedError(
                f"{message}: {last.error_code or 'unknown'} from {last.provider}",
                details={"attempts": [a.model_dump() for a in attempts]},
            )
        if skipped:
            return NoProviderAvailableError(
                f"{message}: every target was skipped",
                details={"attempts": [a.model_dump() for a in attempts]},
            )
        return NoProviderAvailableError(message)

    # ------------------------------------------------------ target resolution

    def _resolve_strategy(self, request: CompletionRequest) -> LLMStrategy:
        raw = (request.strategy or self.settings.default_strategy.value).lower()
        try:
            return LLMStrategy(raw)
        except ValueError:
            return self.settings.default_strategy

    def _resolve_targets(
        self, request: CompletionRequest, strategy: LLMStrategy
    ) -> List[ModelInfo]:
        if request.target_model_ids:
            ids = [m for m in request.target_model_ids if m]
        elif request.model_id:
            ids = [request.model_id]
            if strategy in {
                LLMStrategy.FALLBACK,
                LLMStrategy.PARALLEL_ANY,
                LLMStrategy.PARALLEL_VOTE,
                LLMStrategy.CONFIDENCE,
            }:
                # Pad with defaults that aren't already in the list.
                for extra in self.settings.fallback_chain:
                    if extra not in ids:
                        ids.append(extra)
        else:
            ids = list(self.settings.fallback_chain)

        models: List[ModelInfo] = []
        for model_id in ids:
            try:
                models.append(get_model(model_id))
            except KeyError as exc:
                raise UnknownModelError(
                    f"model {model_id!r} is not in the catalog",
                    details={"model_id": model_id},
                ) from exc
        return models

    def _resolve_sampling(self, request: CompletionRequest) -> Dict:
        return {
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.settings.default_temperature
            ),
            "top_p": (
                request.top_p
                if request.top_p is not None
                else self.settings.default_top_p
            ),
            "max_tokens": (
                request.max_tokens
                if request.max_tokens is not None
                else self.settings.max_completion_tokens
            ),
            "stop": request.stop,
        }

    def _guard_prompt_size(self, messages: Sequence[ChatMessage]) -> None:
        total_chars = sum(len(m.content) for m in messages)
        if total_chars > self.settings.max_prompt_chars:
            raise BudgetExceededError(
                f"prompt length {total_chars} exceeds max "
                f"{self.settings.max_prompt_chars}",
                details={"max_prompt_chars": self.settings.max_prompt_chars},
            )

    # ---------------------------------------------------------- response

    def _build_response(
        self,
        request: CompletionRequest,
        candidate: _Candidate,
        attempts: List[CompletionAttempt],
        strategy: LLMStrategy,
        start: float,
    ) -> CompletionResponse:
        latency_ms = (time.perf_counter() - start) * 1000.0
        confident = candidate.result.confidence >= self.settings.confident_threshold
        return CompletionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            content=candidate.result.content,
            provider=candidate.model.provider,
            model_id=candidate.model.model_id,
            strategy=strategy.value,
            finish_reason=candidate.result.finish_reason,
            confidence=round(candidate.result.confidence, 4),
            confident=confident,
            tokens=candidate.result.tokens,
            cost_usd=candidate.cost_usd,
            latency_ms=round(latency_ms, 3),
            attempts=list(attempts),
            cached=False,
            metadata={"raw": candidate.result.raw} if candidate.result.raw else {},
        )
