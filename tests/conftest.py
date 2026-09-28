"""Shared offline test doubles. Nothing here touches the network or a model server."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from rag_sentinel.config import PipelineSettings, get_settings
from rag_sentinel.llm.embeddings import HashingEmbeddings
from rag_sentinel.metrics.schemas import (
    GeneratedQuestions,
    StatementList,
    Verdict,
    VerdictList,
)
from rag_sentinel.models import (
    EvalRecord,
    EvalSample,
    MetricName,
    MetricResult,
    RAGResponse,
    RetrievedContext,
    RunConfig,
    RunReport,
    SampleResult,
)
from rag_sentinel.pipeline import LangChainRAGPipeline

SchemaT = TypeVar("SchemaT", bound=BaseModel)
ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
GOLDEN_SET = ROOT / "data" / "golden" / "golden_set.jsonl"


class ScriptedJudge:
    """A JudgeLLM that returns queued responses in order and records every prompt."""

    def __init__(self, responses: list[BaseModel | Exception] | None = None) -> None:
        self.responses: deque[BaseModel | Exception] = deque(responses or [])
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted-judge"

    def queue(self, *responses: BaseModel | Exception) -> None:
        self.responses.extend(responses)

    async def generate(
        self, prompt: str, schema: type[SchemaT], *, system: str | None = None
    ) -> SchemaT:
        self.prompts.append(prompt)
        if not self.responses:
            msg = f"ScriptedJudge ran out of responses (asked for {schema.__name__})"
            raise AssertionError(msg)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, schema), f"expected {schema.__name__}, got {type(response)}"
        return response

    async def healthcheck(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class PatternJudge:
    """A stateless judge for concurrent tests: answers by schema, always 'supported'."""

    @property
    def model_name(self) -> str:
        return "pattern-judge"

    async def generate(
        self, prompt: str, schema: type[SchemaT], *, system: str | None = None
    ) -> SchemaT:
        payload: Any
        if schema is StatementList:
            payload = StatementList(statements=["Claim one.", "Claim two."])
        elif schema is Verdict:
            payload = Verdict(reason="ok", verdict="yes")
        elif schema is VerdictList:
            count = int(prompt.split(" total)", maxsplit=1)[0].rsplit("(", 1)[1])
            payload = VerdictList(verdicts=[Verdict(reason="ok", verdict="yes")] * count)
        elif schema is GeneratedQuestions:
            payload = GeneratedQuestions(
                questions=["What is the refund window?"], noncommittal=False
            )
        else:  # pragma: no cover
            raise AssertionError(schema)
        return payload  # type: ignore[no-any-return]

    async def healthcheck(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def chunk_verdicts(*labels: int) -> list[Verdict]:
    """Individual verdicts, as returned by per-chunk judging (context precision)."""
    return [Verdict(reason=f"r{i}", verdict="yes" if v else "no") for i, v in enumerate(labels)]


def verdicts(*labels: int) -> VerdictList:
    return VerdictList(
        verdicts=[
            Verdict(reason=f"r{i}", verdict="yes" if v else "no") for i, v in enumerate(labels)
        ]
    )


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(__import__("os").environ):
        if key.startswith("SENTINEL_"):
            monkeypatch.delenv(key)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def record() -> EvalRecord:
    return EvalRecord(
        question="How long do monthly customers have to request a refund?",
        answer="Monthly customers can request a full refund within 14 days. Refunds take 5 days.",
        contexts=[
            "Customers on monthly plans can request a full refund within 14 days.",
            "Approved refunds are returned within 5 to 10 business days.",
        ],
        ground_truth="Monthly customers can get a full refund within 14 days of first payment.",
    )


@pytest.fixture
def scripted_judge() -> ScriptedJudge:
    return ScriptedJudge()


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture(scope="session")
def offline_pipeline() -> LangChainRAGPipeline:
    settings = PipelineSettings(corpus_dir=CORPUS_DIR, generator="extractive", top_k=3)
    return LangChainRAGPipeline.from_settings(settings, HashingEmbeddings())


def make_report(
    aggregates: dict[MetricName, float | None],
    *,
    errors: int = 0,
    total: int = 4,
    metrics: list[MetricName] | None = None,
    run_id: str = "run-1",
    dataset: str = "golden_set",
) -> RunReport:
    samples = []
    for i in range(total):
        sample = EvalSample(id=f"s{i}", question=f"Question {i}?")
        if i < errors:
            samples.append(SampleResult(sample=sample, error="PipelineError: boom"))
            continue
        response = RAGResponse(
            question=sample.question,
            answer="An answer.",
            contexts=[RetrievedContext(content="ctx", source="a.md", rank=1, score=0.9)],
            latency_ms=10.0,
        )
        results = {name: MetricResult(name=name, score=value) for name, value in aggregates.items()}
        samples.append(SampleResult(sample=sample, response=response, metrics=results))
    return RunReport(
        run_id=run_id,
        dataset=dataset,
        config=RunConfig(
            judge_model="j",
            generator_model="g",
            embedding_model="e",
            top_k=3,
            metrics=metrics if metrics is not None else list(aggregates),
        ),
        samples=samples,
        aggregates=aggregates,
    )
