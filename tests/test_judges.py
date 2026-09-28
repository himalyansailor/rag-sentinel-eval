from __future__ import annotations

import json

import httpx
import pytest
import respx

from rag_sentinel.config import JudgeSettings
from rag_sentinel.exceptions import (
    JudgeConfigurationError,
    JudgeResponseError,
    JudgeUnavailableError,
)
from rag_sentinel.llm.base import extract_json
from rag_sentinel.llm.factory import build_judge
from rag_sentinel.llm.ollama import OllamaJudge
from rag_sentinel.llm.openai_compat import OpenAICompatibleJudge
from rag_sentinel.metrics.schemas import StatementList

OLLAMA = "http://ollama.test"
VLLM = "http://vllm.test"


def ollama_judge(max_retries: int = 2) -> OllamaJudge:
    return OllamaJudge(
        base_url=OLLAMA, model="llama3.1:8b", max_retries=max_retries, backoff_s=(0, 0)
    )


def ollama_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def test_extract_json_strips_code_fences() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('  {"a": 1} ') == '{"a": 1}'


@respx.mock
async def test_ollama_sends_schema_constrained_request() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=ollama_reply('{"statements": ["x"]}')
    )
    async with ollama_judge() as judge:
        result = await judge.generate("prompt", StatementList)

    assert result.statements == ["x"]
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is False
    assert body["format"]["title"] == "StatementList"
    assert body["options"] == {"temperature": 0.0, "seed": 42}
    assert body["messages"][0]["role"] == "system"


@respx.mock
async def test_invalid_output_is_retried_then_succeeds() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").mock(
        side_effect=[ollama_reply("not json"), ollama_reply('```json\n{"statements": []}\n```')]
    )
    async with ollama_judge() as judge:
        result = await judge.generate("p", StatementList)
    assert result.statements == []
    assert route.call_count == 2


@respx.mock
async def test_invalid_output_raises_after_retries() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").mock(return_value=ollama_reply('{"nope": 1}'))
    async with ollama_judge(max_retries=1) as judge:
        with pytest.raises(JudgeResponseError):
            await judge.generate("p", StatementList)
    assert route.call_count == 2


@respx.mock
async def test_server_errors_are_retried_then_raise_unavailable() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(503, text="busy"))
    async with ollama_judge(max_retries=2) as judge:
        with pytest.raises(JudgeUnavailableError):
            await judge.generate("p", StatementList)
    assert route.call_count == 3


@respx.mock
async def test_client_errors_fail_fast() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "model not found"})
    )
    async with ollama_judge() as judge:
        with pytest.raises(JudgeConfigurationError, match="404"):
            await judge.generate("p", StatementList)
    assert route.call_count == 1


@respx.mock
async def test_connection_errors_map_to_unavailable() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    async with ollama_judge(max_retries=0) as judge:
        with pytest.raises(JudgeUnavailableError, match="cannot reach judge"):
            await judge.generate("p", StatementList)


@respx.mock
async def test_ollama_healthcheck_requires_pulled_model() -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
    )
    async with ollama_judge() as judge:
        with pytest.raises(JudgeConfigurationError, match=r"ollama pull llama3\.1:8b"):
            await judge.healthcheck()


@respx.mock
async def test_ollama_healthcheck_accepts_latest_tag() -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "mistral:latest"}]})
    )
    async with OllamaJudge(base_url=OLLAMA, model="mistral") as judge:
        await judge.healthcheck()


@respx.mock
async def test_openai_compatible_judge_uses_response_format_and_auth() -> None:
    route = respx.post(f"{VLLM}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": '{"statements": ["a", "b"]}'}}]}
        )
    )
    settings = JudgeSettings(
        provider="openai_compatible", base_url=VLLM, model="Qwen/Qwen2.5-7B", api_key="s3cret"
    )
    judge = build_judge(settings)
    assert isinstance(judge, OpenAICompatibleJudge)
    async with judge:
        result = await judge.generate("p", StatementList)

    assert result.statements == ["a", "b"]
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer s3cret"
    body = json.loads(request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "StatementList"


@respx.mock
async def test_openai_compatible_healthcheck() -> None:
    respx.get(f"{VLLM}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "other-model"}]})
    )
    async with OpenAICompatibleJudge(base_url=VLLM, model="Qwen/Qwen2.5-7B") as judge:
        with pytest.raises(JudgeConfigurationError, match="not served"):
            await judge.healthcheck()


@respx.mock
async def test_openai_compatible_empty_completion_is_a_response_error() -> None:
    respx.post(f"{VLLM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
    )
    async with OpenAICompatibleJudge(
        base_url=VLLM, model="m", max_retries=0, backoff_s=(0, 0)
    ) as judge:
        with pytest.raises(JudgeResponseError, match="empty"):
            await judge.generate("p", StatementList)


def test_factory_defaults_to_ollama() -> None:
    assert isinstance(build_judge(JudgeSettings()), OllamaJudge)
