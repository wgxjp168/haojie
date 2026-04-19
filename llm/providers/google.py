"""Google Generative Language / Gemini provider (optional dependency: httpx)."""
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


class GoogleProvider(BaseProvider):
    """Google Gemini ``generateContent`` API client."""

    name = "google"

    def __init__(self, *, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials=credentials)
        creds = credentials or {}
        self._api_key = creds.get("api_key")
        self._base_url = (
            creds.get("base_url") or "https://generativelanguage.googleapis.com/v1"
        )

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
                "Google provider unavailable (missing httpx or api_key)"
            )

        start = time.perf_counter()

        # Flatten messages; Gemini uses ``role: user|model`` and an
        # optional ``systemInstruction`` field.
        system_text: List[str] = []
        contents: List[Dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_text.append(m.content)
            else:
                role = "user" if m.role in ("user", "tool") else "model"
                contents.append(
                    {"role": role, "parts": [{"text": m.content}]}
                )

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": min(max_tokens, model.max_output_tokens),
            },
        }
        if system_text:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_text)}]
            }
        if stop:
            payload["generationConfig"]["stopSequences"] = stop

        url = (
            f"/models/{model.remote_name}:generateContent?key={self._api_key}"
        )

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=timeout_seconds
            ) as client:
                resp = await client.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                )
        except httpx.TimeoutException as exc:  # pragma: no cover - needs network
            raise ProviderTimeoutError(
                f"Google timeout after {timeout_seconds}s"
            ) from exc
        except httpx.RequestError as exc:  # pragma: no cover - needs network
            raise ProviderFailedError(f"Google request failed: {exc}") from exc

        if resp.status_code == 401 or resp.status_code == 403:
            raise UnauthorizedError("Google authentication failed")
        if resp.status_code >= 400:
            raise ProviderFailedError(
                f"Google returned HTTP {resp.status_code}",
                details={"body": resp.text[:500]},
            )

        try:
            data = resp.json()
            candidates = data.get("candidates") or []
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            content = "".join(p.get("text", "") for p in parts)
            finish_reason = (candidates[0].get("finishReason") if candidates else "STOP") or "STOP"
            usage = data.get("usageMetadata") or {}
            prompt_tokens = int(usage.get("promptTokenCount", 0))
            completion_tokens = int(usage.get("candidatesTokenCount", 0))
            total = int(usage.get("totalTokenCount", prompt_tokens + completion_tokens))
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise ProviderFailedError(
                f"Google response parse error: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        confidence = 0.9 if finish_reason == "STOP" else 0.6

        return ProviderResult(
            content=content,
            tokens=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total,
            ),
            finish_reason=finish_reason,
            confidence=confidence,
            latency_ms=round(latency_ms, 3),
            raw={"model": model.remote_name},
        )
