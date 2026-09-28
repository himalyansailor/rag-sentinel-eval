"""Shared HTTP judge implementation (see docs/LLD.md §5).

Subclasses only translate a chat request into a provider-specific HTTP call. Retries, JSON
extraction, schema validation and error mapping live here, once.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Self

import httpx
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_sentinel.exceptions import (
    JudgeConfigurationError,
    JudgeResponseError,
    JudgeUnavailableError,
)
from rag_sentinel.log import get_logger
from rag_sentinel.protocols import SchemaT

log = get_logger(__name__)

Message = dict[str, str]

DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous, impartial evaluator of retrieval-augmented generation systems. "
    "Follow the instructions exactly and respond only with JSON matching the requested schema."
)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def extract_json(raw: str) -> str:
    """Strip Markdown code fences that some models wrap around JSON output."""
    match = _FENCE_RE.match(raw)
    return match.group(1) if match else raw.strip()


class BaseHTTPJudge(ABC):
    """Base class for judges served over HTTP (Ollama, vLLM, any OpenAI-compatible server)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        seed: int | None = 42,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        backoff_s: tuple[float, float] = (1.0, 10.0),
    ) -> None:
        self._model = model
        self._backoff_min, self._backoff_max = backoff_s
        self._max_retries = max_retries
        self._temperature = temperature
        self._seed = seed
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers=headers,
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self, prompt: str, schema: type[SchemaT], *, system: str | None = None
    ) -> SchemaT:
        """Ask the model and return its answer parsed into ``schema``.

        Raises:
            JudgeUnavailableError: the server stayed unreachable/overloaded through all retries.
            JudgeConfigurationError: the server rejected the request (not retried).
            JudgeResponseError: the output never validated against ``schema``.
        """
        messages: list[Message] = [
            {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        json_schema = schema.model_json_schema()
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=self._backoff_min, max=self._backoff_max),
            retry=retry_if_exception_type((JudgeUnavailableError, JudgeResponseError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                attempt_no = attempt.retry_state.attempt_number
                raw = await self._complete(messages, json_schema)
                try:
                    return schema.model_validate_json(extract_json(raw))
                except ValidationError as exc:
                    log.warning(
                        "judge.response.invalid",
                        model=self._model,
                        schema=schema.__name__,
                        attempt=attempt_no,
                        errors=exc.error_count(),
                        preview=raw[:200],
                    )
                    msg = f"{self._model} returned output that does not match {schema.__name__}"
                    raise JudgeResponseError(msg) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` and return the decoded JSON body, mapping failures to JudgeErrors."""
        try:
            response = await self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            log.warning("judge.request.timeout", model=self._model, path=path)
            msg = f"judge request to {path} timed out"
            raise JudgeUnavailableError(msg) from exc
        except httpx.TransportError as exc:
            log.warning("judge.request.transport_error", model=self._model, error=str(exc))
            msg = f"cannot reach judge at {self._client.base_url}: {exc}"
            raise JudgeUnavailableError(msg) from exc
        return self._decode(response)

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(path)
        except httpx.TransportError as exc:
            msg = f"cannot reach judge at {self._client.base_url}: {exc}"
            raise JudgeUnavailableError(msg) from exc
        return self._decode(response)

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        status = response.status_code
        if status == httpx.codes.TOO_MANY_REQUESTS or status >= httpx.codes.INTERNAL_SERVER_ERROR:
            log.warning("judge.request.server_error", model=self._model, status=status)
            msg = f"judge returned HTTP {status}: {response.text[:200]}"
            raise JudgeUnavailableError(msg)
        if status >= httpx.codes.BAD_REQUEST:
            msg = f"judge rejected the request (HTTP {status}): {response.text[:300]}"
            raise JudgeConfigurationError(msg)
        try:
            body: dict[str, Any] = response.json()
        except json.JSONDecodeError as exc:
            msg = "judge returned a non-JSON HTTP body"
            raise JudgeResponseError(msg) from exc
        return body

    @abstractmethod
    async def _complete(self, messages: list[Message], json_schema: dict[str, Any]) -> str:
        """Send one chat completion constrained to ``json_schema``; return the raw text."""

    @abstractmethod
    async def healthcheck(self) -> None:
        """Fail fast with a helpful error if the server or model is unavailable."""

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
