"""Link input processor.

Parses and validates a product URL from a known e-commerce platform,
extracts the product-id-like identifier, and (best effort) fetches the
page title. We do NOT scrape product details here — that happens later
in the data acquisition layer. Keeping this fast means the user gets a
quick acknowledgement that their link was accepted.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from config.settings import Settings, get_settings
from core.exceptions import ValidationError
from core.logging import get_logger
from entities.enums import InputType
from processors.base import BaseInputProcessor, InputContext, ProcessorResult

logger = get_logger(__name__)


_SUPPORTED_PLATFORMS: Dict[str, str] = {
    "taobao.com": "taobao",
    "tmall.com": "tmall",
    "jd.com": "jd",
    "yangkeduo.com": "pinduoduo",
    "pinduoduo.com": "pinduoduo",
    "1688.com": "1688",
    "vip.com": "vip",
    "suning.com": "suning",
    "douyin.com": "douyin",
}


_JD_ID_RE = re.compile(r"/(\d{5,})\.html")


class LinkInputProcessor(BaseInputProcessor):
    input_type = InputType.LINK

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    # --- validation ---

    def validate(self, input_data: Any) -> None:
        if not isinstance(input_data, str) or not input_data.strip():
            raise ValidationError("link must be a non-empty string", field="url")
        url = input_data.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValidationError(
                "link must use http or https", field="url", value=parsed.scheme
            )
        if not parsed.netloc:
            raise ValidationError("link has no host", field="url", value=url)

    # --- processing ---

    async def process(self, input_data: str, context: InputContext) -> ProcessorResult:
        url = input_data.strip()
        parsed = urlparse(url)
        platform = self._detect_platform(parsed.netloc)
        product_id = self._extract_product_id(parsed, platform)

        page_title: Optional[str] = None
        warnings: List[str] = []
        try:
            page_title = await self._fetch_title(url)
        except Exception as e:
            warnings.append(f"page fetch failed: {e}")

        data: Dict[str, Any] = {
            "original_url": url,
            "host": parsed.netloc,
            "path": parsed.path,
            "query": {k: v[0] for k, v in parse_qs(parsed.query).items()},
            "platform": platform,
            "is_supported_platform": platform is not None,
            "product_id": product_id,
            "page_title": page_title,
        }
        summary = (
            f"link[{platform or 'unknown'}]: "
            f"product_id={product_id or 'n/a'}"
        )
        return {"data": data, "warnings": warnings, "summary": summary}

    # --- helpers ---

    @staticmethod
    def _detect_platform(netloc: str) -> Optional[str]:
        host = netloc.lower()
        for domain, name in _SUPPORTED_PLATFORMS.items():
            if host == domain or host.endswith("." + domain):
                return name
        return None

    @staticmethod
    def _extract_product_id(parsed_url, platform: Optional[str]) -> Optional[str]:
        if platform is None:
            return None
        query = parse_qs(parsed_url.query)
        if platform in ("taobao", "tmall"):
            values = query.get("id")
            return values[0] if values else None
        if platform == "jd":
            match = _JD_ID_RE.search(parsed_url.path)
            return match.group(1) if match else None
        if platform == "pinduoduo":
            for key in ("goods_id", "id"):
                if key in query:
                    return query[key][0]
            return None
        if platform == "1688":
            match = re.search(r"offer/(\d+)\.html", parsed_url.path)
            return match.group(1) if match else None
        return None

    async def _fetch_title(self, url: str) -> Optional[str]:
        headers = {"User-Agent": self._settings.link_user_agent}
        timeout = httpx.Timeout(self._settings.link_timeout)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                resp = await client.get(url)
        except asyncio.TimeoutError as e:
            raise RuntimeError("timeout") from e
        if resp.status_code >= 400:
            raise RuntimeError(f"status {resp.status_code}")
        # Pull the <title> tag out of the first 64 KiB only — no full parser.
        body = resp.text[:65_536]
        match = re.search(r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        title = re.sub(r"\s+", " ", match.group(1)).strip()
        return title[:500] or None
