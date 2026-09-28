"""Metric base class and shared judge helpers (see docs/LLD.md §6.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from rag_sentinel.exceptions import JudgeResponseError
from rag_sentinel.log import get_logger
from rag_sentinel.metrics.schemas import Verdict, VerdictList
from rag_sentinel.models import EvalRecord, MetricName, MetricResult

if TYPE_CHECKING:
    from rag_sentinel.protocols import JudgeLLM

log = get_logger(__name__)

#: Extra attempts when the judge returns the wrong number of verdicts.
VERDICT_COUNT_RETRIES = 2


async def judge_verdicts(judge: JudgeLLM, prompt: str, expected: int) -> list[Verdict]:
    """Ask for one verdict per item and insist on getting exactly ``expected`` of them.

    Schema-constrained decoding guarantees the *shape*, not the *length*; small models
    occasionally merge or drop items, which would silently misalign verdicts with inputs.
    """
    got = 0
    for attempt in range(1, VERDICT_COUNT_RETRIES + 2):
        result = await judge.generate(prompt, VerdictList)
        got = len(result.verdicts)
        if got == expected:
            return result.verdicts
        log.warning("judge.verdicts.count_mismatch", expected=expected, got=got, attempt=attempt)
    msg = f"judge returned {got} verdicts, expected {expected}"
    raise JudgeResponseError(msg)


class Metric(ABC):
    """A RAG quality metric scored in ``[0, 1]`` (higher is better).

    ``score`` is a template method: it handles applicability and isolates failures so that one
    bad judge response can never abort a whole evaluation run.
    """

    name: ClassVar[MetricName]
    requires_ground_truth: ClassVar[bool] = False

    def __init__(self, judge: JudgeLLM) -> None:
        self.judge = judge

    async def score(self, record: EvalRecord) -> MetricResult:
        if self.requires_ground_truth and not (record.ground_truth or "").strip():
            return MetricResult(name=self.name, reason="skipped: sample has no ground truth")
        try:
            return await self._score(record)
        except Exception as exc:  # isolation boundary: any failure becomes a scored error
            log.exception("metric.failed", metric=self.name.value, error=str(exc))
            return MetricResult(name=self.name, error=f"{type(exc).__name__}: {exc}")

    @abstractmethod
    async def _score(self, record: EvalRecord) -> MetricResult:
        """Compute the metric. May raise; :meth:`score` converts exceptions to results."""
