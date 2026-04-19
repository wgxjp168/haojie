"""Anthropic Messages API provider (optional dependency: httpx)."""
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


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API client."""

    name = "anthropic"

    def __init__(self, *, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials=credentials)
        creds = credentials or {}
        self._api_key = creds.get("api_key")
        self._base_url = creds.get("base_url") or "https://api.anthropic.com/v1"

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
                "Anthropic provider unavailable (missing httpx or api_key)"
            )

        start = time.perf_counter()

        # Split system prompt from the conversation — Anthropic wants it
        # as a top-level ``system`` field, not in ``messages``.
        system_prompt_parts: List[str] = []
        conversation: List[Dict[str, str]] = []

        for m in messages:
            if m.role == "system":
                system_prompt_parts.append(m.content)
            elif m.role == "tool":
                # Anthropic handles tool results differently; we flatten
                # into a user turn for simplicity in v1.
                conversation.append({"role": "user", "content": m.content})
            else:
                conversation.append({"role": m.role, "content": m.content})

        payload: Dict[str, Any] = {
            "model": model.remote_name,
            "max_tokens": min(max_tokens, model.max_output_tokens),
            "temperature": temperature,
            "top_p": top_p,
            "messages": conversation,
        }
        if system_prompt_parts:
            payload["system"] = "\n\n".join(system_prompt_parts)
        if stop:
            payload["stop_sequences"] = stop

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=timeout_seconds
            ) as client:
                resp = await client.post(
                    "/messages", json=payload, headers=headers
                )
        except httpx.TimeoutException as exc:  # pragma: no cover - needs network
            raise ProviderTimeoutError(
                f"Anthropic timeout after {timeout_seconds}s"
            ) from exc
        except httpx.RequestError as exc:  # pragma: no cover - needs network
            raise ProviderFailedError(f"Anthropic request failed: {exc}") from exc

        if resp.status_code == 401:
            raise UnauthorizedError("Anthropic authentication failed")
        if resp.status_code >= 400:
            raise ProviderFailedError(
                f"Anthropic returned HTTP {resp.status_code}",
                details={"body": resp.text[:500]},
            )

        try:
            data = resp.json()
            parts = data.get("content") or []
            content = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            finish_reason = data.get("stop_reason", "stop")
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("input_tokens", 0))
            completion_tokens = int(usage.get("output_tokens", 0))
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderFailedError(
                f"Anthropic response parse error: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        confidence = 0.9 if finish_reason in {"stop", "end_turn"} else 0.6

        return ProviderResult(
            content=content,
            tokens=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason=finish_reason,
            confidence=confidence,
            latency_ms=round(latency_ms, 3),
            raw={"model": model.remote_name},
        )
