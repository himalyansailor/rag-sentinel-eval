"""Answer Relevance: does the answer address the question that was asked?"""

from __future__ import annotations

from statistics import fmean
from typing import TYPE_CHECKING, ClassVar

from rag_sentinel.metrics import prompts
from rag_sentinel.metrics.base import Metric
from rag_sentinel.metrics.schemas import GeneratedQuestions
from rag_sentinel.metrics.text import clamp01, cosine_similarity
from rag_sentinel.models import EvalRecord, MetricDetail, MetricName, MetricResult

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from rag_sentinel.protocols import JudgeLLM


class AnswerRelevance(Metric):
    """Mean cosine similarity between the question and questions regenerated from the answer.

    If the answer were a perfect response, the regenerated questions would paraphrase the
    original. Noncommittal answers score 0 (LLD §6.2).
    """

    name: ClassVar[MetricName] = MetricName.ANSWER_RELEVANCE

    def __init__(self, judge: JudgeLLM, embeddings: Embeddings, n_questions: int = 3) -> None:
        super().__init__(judge)
        self.embeddings = embeddings
        self.n_questions = n_questions

    async def _score(self, record: EvalRecord) -> MetricResult:
        generated = await self.judge.generate(
            prompts.generate_questions(record.answer, self.n_questions), GeneratedQuestions
        )
        if generated.noncommittal:
            return MetricResult(name=self.name, score=0.0, reason="answer is noncommittal")
        questions = [q.strip() for q in generated.questions if q.strip()]
        if not questions:
            return MetricResult(name=self.name, score=0.0, reason="no questions could be derived")

        vectors = await self.embeddings.aembed_documents([record.question, *questions])
        original, *candidates = vectors
        similarities = [clamp01(cosine_similarity(original, c)) for c in candidates]
        return MetricResult(
            name=self.name,
            score=clamp01(fmean(similarities)),
            reason=f"mean similarity over {len(questions)} regenerated questions",
            details=[
                MetricDetail(item=q, score=s, reason="cosine similarity to the original question")
                for q, s in zip(questions, similarities, strict=True)
            ],
        )
