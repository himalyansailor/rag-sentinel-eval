"""Judge for OpenAI-compatible servers such as vLLM, TGI, LM Studio or llama.cpp server."""

from __future__ import annotations

from typing import Any

from rag_sentinel.exceptions import JudgeConfigurationError, JudgeResponseError
from rag_sentinel.llm.base import BaseHTTPJudge, Message


class OpenAICompatibleJudge(BaseHTTPJudge):
    """Uses ``POST /v1/chat/completions`` with ``response_format={"type": "json_schema"}``.

    ``base_url`` is the server root (e.g. ``http://localhost:8000``), without ``/v1``.
    """

    async def _complete(self, messages: list[Message], json_schema: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": str(json_schema.get("title", "response")),
                    "schema": json_schema,
                },
            },
        }
        if self._seed is not None:
            payload["seed"] = self._seed
        body = await self._post_json("/v1/chat/completions", payload)
        try:
            content: str | None = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = "unexpected OpenAI-compatible response shape (missing choices[0].message)"
            raise JudgeResponseError(msg) from exc
        if not content:
            msg = "judge returned an empty completion"
            raise JudgeResponseError(msg)
        return content

    async def healthcheck(self) -> None:
        body = await self._get_json("/v1/models")
        available = {m.get("id", "") for m in body.get("data", [])}
        if self._model not in available:
            msg = (
                f"model {self._model!r} is not served by the OpenAI-compatible server. "
                f"Available: {sorted(available) or 'none'}"
            )
            raise JudgeConfigurationError(msg)
