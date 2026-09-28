"""Context Precision: are the useful chunks ranked at the top of the retrieved list?"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from rag_sentinel.metrics import prompts
from rag_sentinel.metrics.base import Metric
from rag_sentinel.metrics.schemas import Verdict
from rag_sentinel.metrics.text import average_precision
from rag_sentinel.models import EvalRecord, MetricDetail, MetricName, MetricResult


class ContextPrecision(Metric):
    """Average precision over per-chunk usefulness verdicts (LLD §6.2).

    Each chunk is judged in its own call (concurrently), which keeps verdicts attached to the
    right chunk. Uses the ground truth as the reference when available, otherwise the generated
    answer (the "without reference" variant); the reason field records which one was used.
    """

    name: ClassVar[MetricName] = MetricName.CONTEXT_PRECISION

    async def _score(self, record: EvalRecord) -> MetricResult:
        if not record.contexts:
            return MetricResult(name=self.name, reason="no contexts were retrieved")

        has_reference = bool((record.ground_truth or "").strip())
        reference = record.ground_truth if has_reference and record.ground_truth else record.answer
        verdicts = await asyncio.gather(
            *(
                self.judge.generate(
                    prompts.judge_chunk_usefulness(record.question, reference, chunk), Verdict
                )
                for chunk in record.contexts
            )
        )
        labels = [v.value for v in verdicts]
        return MetricResult(
            name=self.name,
            score=average_precision(labels),
            reason=(
                f"{sum(labels)}/{len(labels)} chunks useful "
                f"(reference: {'ground truth' if has_reference else 'generated answer'})"
            ),
            details=[
                MetricDetail(
                    item=f"chunk #{rank}: {ctx[:160]}", score=float(v.value), reason=v.reason
                )
                for rank, (ctx, v) in enumerate(
                    zip(record.contexts, verdicts, strict=True), start=1
                )
            ],
        )
