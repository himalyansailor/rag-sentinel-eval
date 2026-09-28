"""Structural interfaces that decouple the evaluator from concrete adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from rag_sentinel.models import RAGResponse

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class JudgeLLM(Protocol):
    """An LLM that answers prompts with JSON validated against a Pydantic schema."""

    @property
    def model_name(self) -> str: ...

    async def generate(
        self, prompt: str, schema: type[SchemaT], *, system: str | None = None
    ) -> SchemaT: ...

    async def healthcheck(self) -> None: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class RAGPipeline(Protocol):
    """Any system under test: implement this to evaluate your own RAG application."""

    async def aquery(self, question: str) -> RAGResponse: ...

    def describe(self) -> str: ...
