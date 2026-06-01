"""Anthropic provider — Messages API with optional tool-use search loop."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "name": "google_search",
    "description": "Search the web for current information about a place or activity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
}


class AnthropicProvider:
    """Anthropic Messages API with optional google_search tool-use loop."""

    def __init__(
        self,
        name: str,
        instruction: str,
        enable_search: bool = False,
        model: str = "claude-haiku-4-5-20251001",
        max_tool_rounds: int = 10,
    ) -> None:
        try:
            import anthropic as _anthropic  # noqa: F401 — verify package is installed
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        self._name = name
        self._model = model
        self._instruction = instruction
        self._tools = [_SEARCH_TOOL_SCHEMA] if enable_search else []
        self._max_tool_rounds = max_tool_rounds
        self._call_count = 0
        self._client = None  # created lazily on first ask() so missing key fails at call time

    async def ask(self, prompt: str) -> tuple[str, str]:
        """Messages API with automatic tool-use loop for google_search."""
        from src.tools.search import web_search

        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()

        self._call_count += 1
        logger.info("[%s] call #%d prompt:\n%s", self._name, self._call_count, prompt)

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        try:
            for _ in range(self._max_tool_rounds):
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "system": self._instruction,
                    "messages": messages,
                    "max_tokens": 8096,
                }
                if self._tools:
                    kwargs["tools"] = self._tools

                response = await self._client.messages.create(**kwargs)

                if response.stop_reason == "tool_use":
                    tool_uses = [b for b in response.content if b.type == "tool_use"]
                    # Append the full assistant turn (preserves tool_use blocks)
                    messages.append({"role": "assistant", "content": response.content})
                    tool_results = []
                    for tu in tool_uses:
                        result = await web_search(tu.input.get("query", ""))
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": result or "No results found.",
                        })
                    messages.append({"role": "user", "content": tool_results})
                else:
                    text = next(
                        (b.text for b in response.content if hasattr(b, "text")), ""
                    )
                    return (text, "") if text else ("", "No text in response.")

            return "", "Max tool rounds exceeded."
        except Exception as e:
            return "", f"Anthropic error: {e}"
