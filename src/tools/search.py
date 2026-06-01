"""Shared async web search for OpenAI and Anthropic providers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


async def web_search(query: str, num_results: int = 5) -> str:
    """Call Google Custom Search API and return formatted plain-text results.

    Falls back to empty string if GOOGLE_CUSTOM_SEARCH_API_KEY / GOOGLE_CUSTOM_SEARCH_CX
    are not set — the model then relies on training knowledge only.
    """
    api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
    if not api_key or not cx:
        logger.warning(
            "GOOGLE_CUSTOM_SEARCH_API_KEY or GOOGLE_CUSTOM_SEARCH_CX not set; "
            "skipping web search — model will use training knowledge only."
        )
        return ""

    try:
        import httpx

        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": api_key, "cx": cx, "q": query, "num": num_results}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("items", [])
        if not items:
            return "No search results found."

        lines = []
        for item in items:
            title = item.get("title", "")
            snippet = item.get("snippet", "").replace("\n", " ")
            link = item.get("link", "")
            lines.append(f"- {title}: {snippet} ({link})")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Web search failed for query %r: %s", query, e)
        return ""
