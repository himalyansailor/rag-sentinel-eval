"""Domain models shared by every layer (see docs/LLD.md §2).

These are pure data containers: no I/O, no imports from the rest of the package except enums.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MetricName(StrEnum):
    """Identifiers of the built-in metrics. Values are stable storage keys."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalSample(_Frozen):
    """One row of a golden dataset."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedContext(_Frozen):
    """A chunk returned by the retriever, in rank order (1 = best)."""

    content: str
    source: str
    rank: int = Field(ge=1)
    score: float | None = None


class RAGResponse(_Frozen):
    """What a RAG pipeline returns for a single question."""

    question: str
    answer: str
    contexts: list[RetrievedContext]
    latency_ms: float = Field(ge=0)


class EvalRecord(_Frozen):
    """The minimal view of a sample that metrics consume."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None

    @classmethod
    def from_response(cls, sample: EvalSample, response: RAGResponse) -> EvalRecord:
        return cls(
            question=sample.question,
            answer=response.answer,
            contexts=[c.content for c in response.contexts],
            ground_truth=sample.ground_truth,
        )


class MetricDetail(_Frozen):
    """Item-level evidence behind a score, e.g. one statement and whether it is supported."""

    item: str
    score: float
    reason: str = ""


class MetricResult(_Frozen):
    """Outcome of one metric on one sample.

    ``score`` is ``None`` when the metric was not applicable (``reason`` explains why) or
    failed (``error`` is set).
    """

    name: MetricName
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = ""
    details: list[MetricDetail] = Field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SampleResult(_Frozen):
    """Pipeline output and metric results for one golden sample."""

    sample: EvalSample
    response: RAGResponse | None = None
    metrics: dict[MetricName, MetricResult] = Field(default_factory=dict)
    error: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None or any(not m.ok for m in self.metrics.values())


class Provenance(_Frozen):
    """Where a run came from; used for baselines and dashboard filtering."""

    git_sha: str | None = None
    git_ref: str | None = None
    trigger: str = "local"
    ci_run_url: str | None = None


class RunConfig(_Frozen):
    """Configuration snapshot stored with each run so results stay interpretable."""

    judge_model: str
    generator_model: str
    embedding_model: str
    top_k: int
    metrics: list[MetricName]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunReport(_Frozen):
    """Everything produced by one evaluation run."""

    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    dataset: str
    provenance: Provenance = Field(default_factory=Provenance)
    config: RunConfig
    samples: list[SampleResult] = Field(default_factory=list)
    aggregates: dict[MetricName, float | None] = Field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.samples if s.has_error)

    @property
    def error_rate(self) -> float:
        return self.error_count / self.sample_count if self.samples else 0.0


class GateViolation(_Frozen):
    metric: str
    threshold: float
    actual: float | None
    message: str


class GateResult(_Frozen):
    passed: bool
    violations: list[GateViolation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
