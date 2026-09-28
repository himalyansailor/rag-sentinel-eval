"""Evaluation orchestration (see docs/LLD.md §8).

The evaluator runs every golden sample through the pipeline and the selected metrics with
bounded concurrency, isolates failures per sample and per metric, and aggregates the scores.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from statistics import fmean
from typing import TYPE_CHECKING

import structlog

from rag_sentinel.log import get_logger
from rag_sentinel.models import (
    EvalRecord,
    EvalSample,
    MetricName,
    MetricResult,
    Provenance,
    RunConfig,
    RunReport,
    SampleResult,
)

if TYPE_CHECKING:
    from rag_sentinel.metrics.base import Metric
    from rag_sentinel.protocols import RAGPipeline

log = get_logger(__name__)

ProgressCallback = Callable[[int, int], None]


def aggregate(
    samples: Iterable[SampleResult], metric_names: Iterable[MetricName]
) -> dict[MetricName, float | None]:
    """Mean of non-``None`` scores per metric; ``None`` if no sample produced a score."""
    sample_list = list(samples)
    result: dict[MetricName, float | None] = {}
    for name in metric_names:
        scores = [
            m.score
            for s in sample_list
            if (m := s.metrics.get(name)) is not None and m.score is not None
        ]
        result[name] = round(fmean(scores), 4) if scores else None
    return result


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class Evaluator:
    """Scores a :class:`RAGPipeline` against a golden dataset with a set of metrics."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        metrics: Sequence[Metric],
        *,
        max_concurrency: int = 4,
    ) -> None:
        if not metrics:
            msg = "at least one metric is required"
            raise ValueError(msg)
        if max_concurrency < 1:
            msg = "max_concurrency must be >= 1"
            raise ValueError(msg)
        self.pipeline = pipeline
        self.metrics = list(metrics)
        self.max_concurrency = max_concurrency

    @property
    def metric_names(self) -> list[MetricName]:
        return [m.name for m in self.metrics]

    async def score_record(self, record: EvalRecord) -> dict[MetricName, MetricResult]:
        """Run all metrics concurrently on one record. Never raises (metrics isolate errors)."""
        results = await asyncio.gather(*(m.score(record) for m in self.metrics))
        return {r.name: r for r in results}

    async def evaluate_sample(self, sample: EvalSample) -> SampleResult:
        """Query the pipeline and score the response. Never raises."""
        with structlog.contextvars.bound_contextvars(sample_id=sample.id):
            try:
                response = await self.pipeline.aquery(sample.question)
            except Exception as exc:  # isolation boundary for the system under test
                log.warning("sample.failed", stage="pipeline", error=str(exc))
                return SampleResult(sample=sample, error=f"{type(exc).__name__}: {exc}")

            metrics = await self.score_record(EvalRecord.from_response(sample, response))
            log.info(
                "sample.completed",
                latency_ms=response.latency_ms,
                contexts=len(response.contexts),
                **{name.value: result.score for name, result in metrics.items()},
            )
            return SampleResult(sample=sample, response=response, metrics=metrics)

    async def run(
        self,
        samples: Sequence[EvalSample],
        *,
        dataset: str,
        config: RunConfig,
        provenance: Provenance | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> RunReport:
        """Evaluate ``samples`` and return a :class:`RunReport` (sample order is preserved)."""
        run_id = new_run_id()
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        done = 0

        async def bounded(sample: EvalSample) -> SampleResult:
            nonlocal done
            async with semaphore:
                result = await self.evaluate_sample(sample)
            done += 1
            if on_progress is not None:
                on_progress(done, len(samples))
            return result

        with structlog.contextvars.bound_contextvars(run_id=run_id):
            log.info(
                "run.started",
                dataset=dataset,
                samples=len(samples),
                metrics=[m.value for m in self.metric_names],
                pipeline=self.pipeline.describe(),
                max_concurrency=self.max_concurrency,
            )
            results = list(await asyncio.gather(*(bounded(s) for s in samples)))
            report = RunReport(
                run_id=run_id,
                dataset=dataset,
                provenance=provenance or Provenance(),
                config=config,
                samples=results,
                aggregates=aggregate(results, self.metric_names),
                duration_s=round(time.perf_counter() - started, 2),
            )
            log.info(
                "run.completed",
                aggregates={k.value: v for k, v in report.aggregates.items()},
                error_rate=round(report.error_rate, 3),
                duration_s=report.duration_s,
            )
        return report
