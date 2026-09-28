from __future__ import annotations

import pytest

from rag_sentinel.datasets import load_golden_set
from rag_sentinel.evaluator import Evaluator, aggregate
from rag_sentinel.exceptions import PipelineError
from rag_sentinel.llm.embeddings import HashingEmbeddings
from rag_sentinel.metrics import AnswerRelevance, ContextRecall, Faithfulness
from rag_sentinel.models import (
    EvalSample,
    MetricName,
    MetricResult,
    Provenance,
    RAGResponse,
    RunConfig,
    SampleResult,
)
from rag_sentinel.pipeline import LangChainRAGPipeline

from .conftest import GOLDEN_SET, PatternJudge

CONFIG = RunConfig(
    judge_model="pattern-judge",
    generator_model="extractive",
    embedding_model="hashing",
    top_k=3,
    metrics=[MetricName.FAITHFULNESS, MetricName.ANSWER_RELEVANCE, MetricName.CONTEXT_RECALL],
)


class FlakyPipeline:
    """Fails for questions containing 'boom', delegates otherwise."""

    def __init__(self, inner: LangChainRAGPipeline) -> None:
        self.inner = inner

    async def aquery(self, question: str) -> RAGResponse:
        if "boom" in question:
            raise PipelineError("generator crashed")
        return await self.inner.aquery(question)

    def describe(self) -> str:
        return "flaky"


def test_aggregate_ignores_none_scores() -> None:
    sample = EvalSample(id="a", question="q")
    results = [
        SampleResult(
            sample=sample,
            metrics={MetricName.FAITHFULNESS: MetricResult(name=MetricName.FAITHFULNESS, score=s)},
        )
        for s in (1.0, 0.5, None)
    ]
    agg = aggregate(results, [MetricName.FAITHFULNESS, MetricName.CONTEXT_RECALL])
    assert agg == {MetricName.FAITHFULNESS: 0.75, MetricName.CONTEXT_RECALL: None}


def test_evaluator_validates_arguments(offline_pipeline: LangChainRAGPipeline) -> None:
    with pytest.raises(ValueError, match="metric"):
        Evaluator(offline_pipeline, [])
    with pytest.raises(ValueError, match="max_concurrency"):
        Evaluator(offline_pipeline, [Faithfulness(PatternJudge())], max_concurrency=0)


async def test_end_to_end_run_on_golden_set(offline_pipeline: LangChainRAGPipeline) -> None:
    judge = PatternJudge()
    evaluator = Evaluator(
        offline_pipeline,
        [Faithfulness(judge), AnswerRelevance(judge, HashingEmbeddings()), ContextRecall(judge)],
        max_concurrency=3,
    )
    samples = load_golden_set(GOLDEN_SET)
    progress: list[int] = []

    report = await evaluator.run(
        samples,
        dataset="golden_set",
        config=CONFIG,
        provenance=Provenance(git_ref="feature/x"),
        on_progress=lambda done, total: progress.append(done),
    )

    assert [s.sample.id for s in report.samples] == [s.id for s in samples]  # order preserved
    assert report.aggregates[MetricName.FAITHFULNESS] == 1.0
    assert report.aggregates[MetricName.CONTEXT_RECALL] == 1.0
    assert 0.0 <= (report.aggregates[MetricName.ANSWER_RELEVANCE] or 0.0) <= 1.0
    assert report.error_count == 0
    assert sorted(progress) == list(range(1, len(samples) + 1))
    assert report.provenance.git_ref == "feature/x"
    assert len(report.run_id) == 12


async def test_pipeline_failures_are_isolated_per_sample(
    offline_pipeline: LangChainRAGPipeline,
) -> None:
    evaluator = Evaluator(FlakyPipeline(offline_pipeline), [Faithfulness(PatternJudge())])
    samples = [
        EvalSample(id="ok", question="How long is the refund window?"),
        EvalSample(id="bad", question="boom"),
    ]
    report = await evaluator.run(samples, dataset="d", config=CONFIG)

    ok, bad = report.samples
    assert ok.error is None
    assert ok.metrics[MetricName.FAITHFULNESS].score == 1.0
    assert bad.error == "PipelineError: generator crashed"
    assert bad.metrics == {}
    assert report.error_count == 1
    assert report.error_rate == 0.5
    assert report.aggregates[MetricName.FAITHFULNESS] == 1.0
