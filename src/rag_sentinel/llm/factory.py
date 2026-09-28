"""Construct judges from settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_sentinel.llm.ollama import OllamaJudge
from rag_sentinel.llm.openai_compat import OpenAICompatibleJudge

if TYPE_CHECKING:
    from rag_sentinel.config import JudgeSettings
    from rag_sentinel.llm.base import BaseHTTPJudge


def build_judge(settings: JudgeSettings) -> BaseHTTPJudge:
    """Create the configured judge. The caller owns it and must ``await judge.aclose()``."""
    judge_cls: type[BaseHTTPJudge] = OllamaJudge
    headers: dict[str, str] | None = None
    if settings.provider == "openai_compatible":
        judge_cls = OpenAICompatibleJudge
        if settings.api_key is not None:
            headers = {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"}
    return judge_cls(
        base_url=settings.base_url,
        model=settings.model,
        timeout_s=settings.timeout_s,
        max_retries=settings.max_retries,
        temperature=settings.temperature,
        seed=settings.seed,
        headers=headers,
    )
