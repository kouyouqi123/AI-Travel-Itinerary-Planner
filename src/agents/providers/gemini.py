"""Gemini provider — wraps Google ADK Agent + InMemoryRunner."""

from __future__ import annotations

import logging
import os
from typing import Any

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

logger = logging.getLogger(__name__)


def _retry_config(attempts: int = 5, exp_base: int = 7) -> types.HttpRetryOptions:
    return types.HttpRetryOptions(
        attempts=attempts,
        exp_base=exp_base,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )


def _extract_text(response: Any) -> str:
    """Extract text from an ADK run_debug response (list[Event] or fallback shapes)."""
    if isinstance(response, list):
        for event in reversed(response):
            if hasattr(event, "is_final_response") and event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            return part.text
        for event in reversed(response):
            if hasattr(event, "content") and event.content and hasattr(event.content, "parts"):
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
    if isinstance(response, str):
        return response
    if hasattr(response, "content") and response.content:
        c = response.content
        if isinstance(c, str):
            return c
        if hasattr(c, "parts") and c.parts:
            for part in c.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
    return ""


class GeminiProvider:
    """Uses Google ADK with optional google_search tool."""

    def __init__(
        self,
        name: str,
        instruction: str,
        tools: list = [],
        model: str = "gemini-2.5-flash",
        retry_attempts: int = 5,
        retry_exp_base: int = 7,
    ) -> None:
        if os.getenv("GOOGLE_API_KEY"):
            os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
        self._agent = Agent(
            name=name,
            model=Gemini(
                model=model,
                retry_options=_retry_config(retry_attempts, retry_exp_base),
            ),
            instruction=instruction,
            tools=tools,
        )
        self._runner = InMemoryRunner(agent=self._agent)
        self._call_count = 0

    async def ask(self, prompt: str) -> tuple[str, str]:
        """Send a prompt; returns (text, error). Each call uses a fresh session.

        A unique session_id per call prevents conversation history from bleeding
        across unrelated queries when the agent is reused as a singleton.
        """
        session_id = f"session_{self._call_count}"
        self._call_count += 1
        logger.info("[%s] call #%d prompt:\n%s", self._agent.name, self._call_count, prompt)
        try:
            response = await self._runner.run_debug(prompt, session_id=session_id)
            text = _extract_text(response)
            return (text, "") if text else ("", "No text in agent response.")
        except Exception as e:
            return "", f"Agent error: {e}"
        finally:
            # Singletons live for the process lifetime; delete each one-shot session
            # immediately so InMemoryRunner doesn't accumulate stale session objects.
            try:
                await self._runner.session_service.delete_session(
                    app_name=self._runner.app_name,
                    user_id="debug_user_id",
                    session_id=session_id,
                )
            except Exception:
                pass
