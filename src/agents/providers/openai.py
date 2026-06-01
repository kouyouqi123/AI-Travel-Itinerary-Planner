"""OpenAI provider — Chat Completions with optional function-calling search loop."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "Search the web for current information about a place or activity.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


class OpenAIProvider:
    """OpenAI Chat Completions with optional google_search function-call loop."""

    def __init__(
        self,
        name: str,
        instruction: str,
        enable_search: bool = False,
        model: str = "gpt-4o-mini",
        max_tool_rounds: int = 10,
    ) -> None:
        ## todo: move this elsewhere. it should already be installed? 
        try:
            import openai as _openai  # noqa: F401 — verify package is installed
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self._name = name
        self._model = model
        self._instruction = instruction
        self._tools = [_SEARCH_TOOL_SCHEMA] if enable_search else []
        self._max_tool_rounds = max_tool_rounds
        self._call_count = 0
        self._client = None  # created lazily on first ask() so missing key fails at call time

    async def ask(self, prompt: str) -> tuple[str, str]:
        """Chat Completions with automatic tool-call loop for google_search."""
        from src.tools.search import web_search

        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()  # reads from OPENAI_API_KEY env var

        self._call_count += 1
        logger.info("[%s] call #%d prompt:\n%s", self._name, self._call_count, prompt)

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        try:
            for _ in range(self._max_tool_rounds):
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": self._instruction},
                        *messages,
                    ],
                }
                if self._tools:
                    kwargs["tools"] = self._tools

                response = await self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    # Append the assistant turn with its tool calls
                    messages.append(choice.message.model_dump(exclude_unset=True))
                    for tc in choice.message.tool_calls:
                        args = json.loads(tc.function.arguments)
                        result = await web_search(args.get("query", ""))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result or "No results found.",
                        })
                else:
                    return choice.message.content or "", ""

            return "", "Max tool rounds exceeded."
        except Exception as e:
            return "", f"OpenAI error: {e}"
