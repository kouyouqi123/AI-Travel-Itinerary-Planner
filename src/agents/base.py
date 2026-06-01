"""Shared infrastructure for all LLM agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentProvider(Protocol):
    """All providers implement this single method.

    System instruction is set at construction time; ask() receives only the
    user-facing prompt. Returns (text, error) — error is empty on success.
    """

    async def ask(self, prompt: str) -> tuple[str, str]: ...


def create_provider(
    provider: str,
    instruction: str,
    enable_search: bool = False,
    model: str | None = None,
) -> AgentProvider:
    """Factory: return the right provider for the given provider string.

    provider: "openai" | "gemini" | "anthropic"
    enable_search: whether the provider should register a web-search tool
    model: overrides the per-provider default model
    """
    if provider == "gemini":
        from google.adk.tools import google_search as _google_search
        from src.agents.providers.gemini import GeminiProvider
        kwargs = {"tools": [_google_search]} if enable_search else {}
        if model:
            kwargs["model"] = model
        return GeminiProvider(name=provider, instruction=instruction, **kwargs)

    if provider == "openai":
        from src.agents.providers.openai import OpenAIProvider
        kwargs = {"enable_search": enable_search}
        if model:
            kwargs["model"] = model
        return OpenAIProvider(name=provider, instruction=instruction, **kwargs)

    if provider == "anthropic":
        from src.agents.providers.anthropic import AnthropicProvider
        kwargs = {"enable_search": enable_search}
        if model:
            kwargs["model"] = model
        return AnthropicProvider(name=provider, instruction=instruction, **kwargs)

    raise ValueError(f"Unknown provider {provider!r}. Choose: openai, gemini, anthropic.")


def _extract_json(text: str) -> list[dict] | None:
    """Extract a JSON array from agent response text, handling code fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass
    i = text.find("[")
    if i == -1:
        return None
    depth = 0
    for j, c in enumerate(text[i:], i):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[i : j + 1])
                    return data if isinstance(data, list) else None
                except json.JSONDecodeError:
                    return None
    return None


def _extract_json_dict(text: str) -> dict | None:
    """Extract a JSON object from agent response text, handling code fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    i = text.find("{")
    if i == -1:
        return None
    depth = 0
    for j, c in enumerate(text[i:], i):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[i : j + 1])
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
