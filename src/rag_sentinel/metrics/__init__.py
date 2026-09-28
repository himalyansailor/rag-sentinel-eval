"""RAGAS-style metrics and their registry (see docs/LLD.md §6)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from rag_sentinel.metrics.answer_relevance import AnswerRelevance
from rag_sentinel.metrics.base import Metric
from rag_sentinel.metrics.context_precision import ContextPrecision
from rag_sentinel.metrics.context_recall import ContextRecall
from rag_sentinel.metrics.faithfulness import Faithfulness
from rag_sentinel.models import MetricName

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from rag_sentinel.config import EvaluationSettings
    from rag_sentinel.protocols import JudgeLLM

METRIC_REGISTRY: dict[MetricName, type[Metric]] = {
    MetricName.FAITHFULNESS: Faithfulness,
    MetricName.ANSWER_RELEVANCE: AnswerRelevance,
    MetricName.CONTEXT_PRECISION: ContextPrecision,
    MetricName.CONTEXT_RECALL: ContextRecall,
}


def build_metrics(
    names: Iterable[MetricName],
    *,
    judge: JudgeLLM,
    embeddings: Embeddings,
    settings: EvaluationSettings,
) -> list[Metric]:
    """Instantiate metrics in a stable (registry) order, de-duplicating names."""
    wanted = set(names)
    metrics: list[Metric] = []
    for name, cls in METRIC_REGISTRY.items():
        if name not in wanted:
            continue
        if cls is AnswerRelevance:
            metrics.append(
                AnswerRelevance(judge, embeddings, n_questions=settings.answer_relevance_questions)
            )
        else:
            metrics.append(cls(judge))
    return metrics


__all__ = [
    "METRIC_REGISTRY",
    "AnswerRelevance",
    "ContextPrecision",
    "ContextRecall",
    "Faithfulness",
    "Metric",
    "build_metrics",
]
