"""Context Recall: did retrieval surface everything the reference answer needs?"""

from __future__ import annotations

from typing import ClassVar

from rag_sentinel.metrics import prompts
from rag_sentinel.metrics.base import Metric, judge_verdicts
from rag_sentinel.metrics.text import split_sentences
from rag_sentinel.models import EvalRecord, MetricDetail, MetricName, MetricResult


class ContextRecall(Metric):
    """``attributable_reference_sentences / total_reference_sentences`` (LLD §6.2)."""

    name: ClassVar[MetricName] = MetricName.CONTEXT_RECALL
    requires_ground_truth: ClassVar[bool] = True

    async def _score(self, record: EvalRecord) -> MetricResult:
        sentences = split_sentences(record.ground_truth or "")
        if not sentences:
            return MetricResult(name=self.name, reason="ground truth has no sentences")
        if not record.contexts:
            return MetricResult(
                name=self.name,
                score=0.0,
                reason="no contexts were retrieved",
                details=[MetricDetail(item=s, score=0.0, reason="no context") for s in sentences],
            )

        verdicts = await judge_verdicts(
            self.judge,
            prompts.attribute_sentences(record.question, record.contexts, sentences),
            len(sentences),
        )
        attributed = sum(v.value for v in verdicts)
        return MetricResult(
            name=self.name,
            score=attributed / len(sentences),
            reason=f"{attributed}/{len(sentences)} reference sentences found in the context",
            details=[
                MetricDetail(item=s, score=float(v.value), reason=v.reason)
                for s, v in zip(sentences, verdicts, strict=True)
            ],
        )
