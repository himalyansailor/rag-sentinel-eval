"""Faithfulness: the share of answer statements that the retrieved context supports."""

from __future__ import annotations

import re
from typing import ClassVar

from rag_sentinel.metrics import prompts
from rag_sentinel.metrics.base import Metric, judge_verdicts
from rag_sentinel.metrics.schemas import StatementList
from rag_sentinel.models import EvalRecord, MetricDetail, MetricName, MetricResult

_WORD_RE = re.compile(r"\w")


class Faithfulness(Metric):
    """``supported_statements / total_statements`` (LLD §6.2).

    A score below 1.0 means the answer contains claims the retrieved context does not back up,
    i.e. hallucinations from the point of view of the RAG system.
    """

    name: ClassVar[MetricName] = MetricName.FAITHFULNESS

    async def _score(self, record: EvalRecord) -> MetricResult:
        extracted = await self.judge.generate(
            prompts.extract_statements(record.question, record.answer), StatementList
        )
        # Drop empty or punctuation-only items such as "..." that small models sometimes emit.
        statements = [s.strip() for s in extracted.statements if _WORD_RE.search(s)]
        if not statements:
            return MetricResult(name=self.name, reason="answer contains no verifiable statements")

        verdicts = await judge_verdicts(
            self.judge, prompts.verify_statements(record.contexts, statements), len(statements)
        )
        details = [
            MetricDetail(item=statement, score=float(v.value), reason=v.reason)
            for statement, v in zip(statements, verdicts, strict=True)
        ]
        supported = sum(v.value for v in verdicts)
        return MetricResult(
            name=self.name,
            score=supported / len(statements),
            reason=f"{supported}/{len(statements)} statements supported by the context",
            details=details,
        )
