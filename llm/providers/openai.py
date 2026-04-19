"""OpenAI / Azure-OpenAI HTTP provider (optional dependency: httpx).

The adapter speaks the OpenAI Chat Completions REST API directly via
``httpx`` so no first-party SDK is required. When ``httpx`` is not
installed the provider reports ``available=False`` and is skipped by
the router.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from llm.catalog import ModelInfo
from llm.core.exceptions import (
    ProviderFailedError,
    ProviderTimeoutError,
    UnauthorizedError,
)
from llm.providers.base import BaseProvider, ProviderResult
from llm.schemas import ChatMessage, TokenUsage

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dep
    httpx = None  # type: ignore


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible chat completions."""

    name = "openai"

    def __init__(self, *, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials=credentials)
        self._api_key = (credentials or {}).get("api_key")
        self._base_url = (credentials or {}).get("base_url") or "https://api.openai.com/v1"
        self._organization = (credentials or {}).get("organization")

    @property
    def available(self) -> bool:
        return bool(httpx is not None and self._api_key)

    async def complete(
        self,
        *,
        model: ModelInfo,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        timeout_seconds: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ProviderResult:
        if not self.available:
            raise ProviderFailedError(
                "OpenAI provider unavailable (missing httpx or api_key)"
            )

        start = time.perf_counter()

        payload: Dict[str, Any] = {
            "model": model.remote_name,
            "messages": [self._fmt(m) for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": min(max_tokens, model.max_output_tokens),
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._organization:
            headers["OpenAI-Organization"] = self._organization

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=timeout_seconds
            ) as client:
                resp = await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:  # pragma: no cover - needs network
            raise ProviderTimeoutError(
                f"OpenAI timeout after {timeout_seconds}s",
                details={"model": model.remote_name},
            ) from exc
        except httpx.RequestError as exc:  # pragma: no cover - needs network
            raise ProviderFailedError(
                f"OpenAI request failed: {exc}",
                details={"model": model.remote_name},
            ) from exc

        if resp.status_code == 401:
            raise UnauthorizedError("OpenAI authentication failed")
        if resp.status_code == 429:
            raise ProviderFailedError(
                "OpenAI rate-limited by provider",
                details={"status": 429},
            )
        if resp.status_code >= 400:
            raise ProviderFailedError(
                f"OpenAI returned HTTP {resp.status_code}",
                details={"body": resp.text[:500]},
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason", "stop")
            usage = data.get("usage") or {}
            tokens = TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderFailedError(
                f"OpenAI response parse error: {exc}",
                details={"status": resp.status_code},
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        # Confidence: OpenAI doesn't expose one, so we derive from
        # finish_reason (``stop`` > ``length`` > others) and token ratio.
        confidence = 0.9 if finish_reason == "stop" else 0.6

        return ProviderResult(
            content=content,
            tokens=tokens,
            finish_reason=finish_reason,
            confidence=confidence,
            latency_ms=round(latency_ms, 3),
            raw={"model": model.remote_name},
        )

    @staticmethod
    def _fmt(m: ChatMessage) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name:
            msg["name"] = m.name
        return msg


class AzureOpenAIProvider(OpenAIProvider):
    """Azure-hosted OpenAI deployment.

    Uses Azure's deployment-name routing: ``{endpoint}/openai/deployments/
    {deployment}/chat/completions?api-version=...``. The ``remote_name``
    field in the catalog is treated as the *deployment name*.
    """

    name = "azure"

    def __init__(self, *, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials=credentials)
        creds = credentials or {}
        self._api_key = creds.get("api_key")
        self._endpoint = (creds.get("endpoint") or "").rstrip("/")
        self._api_version = creds.get("api_version") or "2024-02-01"

    @property
    def available(self) -> bool:
        return bool(httpx is not None and self._api_key and self._endpoint)

    async def complete(
        self,
        *,
        model: ModelInfo,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        timeout_seconds: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ProviderResult:
        if not self.available:
            raise ProviderFailedError(
                "Azure provider unavailable (missing httpx, api_key, or endpoint)"
            )

        start = time.perf_counter()

        url = (
            f"{self._endpoint}/openai/deployments/{model.remote_name}"
            f"/chat/completions?api-version={self._api_version}"
        )
        payload = {
            "messages": [self._fmt(m) for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": min(max_tokens, model.max_output_tokens),
        }
        if stop:
            payload["stop"] = stop

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"api-key": self._api_key, "Content-Type": "application/json"},
                )
        except httpx.TimeoutException as exc:  # pragma: no cover - needs network
            raise ProviderTimeoutError(
                f"Azure timeout after {timeout_seconds}s",
            ) from exc
        except httpx.RequestError as exc:  # pragma: no cover - needs network
            raise ProviderFailedError(f"Azure request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ProviderFailedError(
                f"Azure returned HTTP {resp.status_code}",
                details={"body": resp.text[:500]},
            )

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        latency_ms = (time.perf_counter() - start) * 1000.0

        return ProviderResult(
            content=choice["message"]["content"] or "",
            tokens=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            ),
            finish_reason=choice.get("finish_reason", "stop"),
            confidence=0.9 if choice.get("finish_reason") == "stop" else 0.6,
            latency_ms=round(latency_ms, 3),
            raw={"deployment": model.remote_name},
        )
