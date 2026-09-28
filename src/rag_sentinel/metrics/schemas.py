"""Pydantic schemas the judge must return. They double as JSON schemas for constrained decoding."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StatementList(BaseModel):
    """Atomic claims extracted from an answer."""

    statements: list[str] = Field(description="Atomic, self-contained factual statements.")


class Verdict(BaseModel):
    """A binary judgement with a short justification.

    The reason comes first so the model reasons before labelling. The label is a word, not a
    digit: on qwen2.5:3b, "yes"/"no" matched the model's own rationale on 12/14 real statements
    versus 4/14 with 0/1, where it often wrote "the context states X" and then answered 0.
    """

    reason: str = Field(description="One-sentence justification.")
    verdict: Literal["yes", "no"] = Field(description='"yes" = supported / useful, "no" = not.')

    @property
    def value(self) -> int:
        """1 for "yes", 0 for "no" — the numeric form used in scores."""
        return 1 if self.verdict == "yes" else 0


class VerdictList(BaseModel):
    """One verdict per numbered input item, in the same order."""

    verdicts: list[Verdict]


class GeneratedQuestions(BaseModel):
    """Questions that the answer would be a response to (Answer Relevance)."""

    questions: list[str]
    noncommittal: bool = Field(
        description="True if the answer is evasive or vague, e.g. 'I don't know'."
    )
