"""Judge backed by a local or remote Ollama server."""

from __future__ import annotations

from typing import Any

from rag_sentinel.exceptions import JudgeConfigurationError, JudgeResponseError
from rag_sentinel.llm.base import BaseHTTPJudge, Message


class OllamaJudge(BaseHTTPJudge):
    """Uses ``POST /api/chat`` with ``format=<json schema>`` for constrained decoding."""

    async def _complete(self, messages: list[Message], json_schema: dict[str, Any]) -> str:
        options: dict[str, Any] = {"temperature": self._temperature}
        if self._seed is not None:
            options["seed"] = self._seed
        body = await self._post_json(
            "/api/chat",
            {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "format": json_schema,
                "options": options,
            },
        )
        try:
            content: str = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            msg = "unexpected Ollama response shape (missing message.content)"
            raise JudgeResponseError(msg) from exc
        return content

    async def healthcheck(self) -> None:
        body = await self._get_json("/api/tags")
        available = {m.get("name", "") for m in body.get("models", [])}
        # Ollama reports "llama3.1:8b"; users often configure "llama3.1" meaning ":latest".
        wanted = {self._model, f"{self._model}:latest"}
        if not available & wanted:
            msg = (
                f"model {self._model!r} is not available on the Ollama server. "
                f"Run `ollama pull {self._model}`. Available: {sorted(available) or 'none'}"
            )
            raise JudgeConfigurationError(msg)
