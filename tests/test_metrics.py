from __future__ import annotations

import pytest

from rag_sentinel.config import EvaluationSettings
from rag_sentinel.exceptions import JudgeUnavailableError
from rag_sentinel.llm.embeddings import HashingEmbeddings
from rag_sentinel.metrics import (
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    build_metrics,
    prompts,
)
from rag_sentinel.metrics.schemas import GeneratedQuestions, StatementList
from rag_sentinel.metrics.text import average_precision, cosine_similarity, split_sentences
from rag_sentinel.models import EvalRecord, MetricName

from .conftest import ScriptedJudge, chunk_verdicts, verdicts

# ------------------------------------------------------------------ text utils ---


def test_split_sentences_handles_punctuation_newlines_and_bullets() -> None:
    text = "First one. Second one!\n- Third one\n\nUptime is 99.95% monthly."
    assert split_sentences(text) == [
        "First one.",
        "Second one!",
        "Third one",
        "Uptime is 99.95% monthly.",
    ]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([1, 1, 1], 1.0),
        ([0, 0], 0.0),
        ([1, 0], 1.0),
        ([0, 1], 0.5),
        ([1, 0, 1], (1 + 2 / 3) / 2),
        ([], 0.0),
    ],
)
def test_average_precision(labels: list[int], expected: float) -> None:
    assert average_precision(labels) == pytest.approx(expected)


def test_cosine_similarity_edge_cases() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


# ---------------------------------------------------------------- faithfulness ---


async def test_faithfulness_scores_supported_share(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(
        StatementList(
            statements=["Refund window is 14 days.", "Refunds take 5 days.", "  ", "..."]
        ),
        verdicts(1, 0),
    )
    result = await Faithfulness(scripted_judge).score(record)

    assert result.ok
    assert result.score == pytest.approx(0.5)
    assert [d.score for d in result.details] == [1.0, 0.0]
    assert "2 total" in scripted_judge.prompts[1]  # blank and "..." dropped before NLI


async def test_faithfulness_without_statements_is_not_scored(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(StatementList(statements=[]))
    result = await Faithfulness(scripted_judge).score(record)
    assert result.score is None
    assert result.ok
    assert "no verifiable statements" in result.reason


async def test_faithfulness_retries_on_verdict_count_mismatch(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(StatementList(statements=["a.", "b."]), verdicts(1), verdicts(1, 1))
    result = await Faithfulness(scripted_judge).score(record)
    assert result.score == pytest.approx(1.0)


async def test_faithfulness_gives_up_after_repeated_mismatch(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(
        StatementList(statements=["a.", "b."]), verdicts(1), verdicts(1), verdicts(1)
    )
    result = await Faithfulness(scripted_judge).score(record)
    assert result.score is None
    assert result.error is not None
    assert "expected 2" in result.error


async def test_metric_isolates_judge_failures(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(JudgeUnavailableError("down"))
    result = await Faithfulness(scripted_judge).score(record)
    assert not result.ok
    assert result.error == "JudgeUnavailableError: down"


# ------------------------------------------------------------ answer relevance ---


async def test_answer_relevance_uses_embedding_similarity(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(
        GeneratedQuestions(
            questions=[record.question, "What is the capital of France?"], noncommittal=False
        )
    )
    result = await AnswerRelevance(scripted_judge, HashingEmbeddings(), n_questions=2).score(record)

    assert result.score is not None
    identical, unrelated = (d.score for d in result.details)
    assert identical == pytest.approx(1.0)
    assert unrelated < 0.3
    assert result.score == pytest.approx((identical + unrelated) / 2)


async def test_answer_relevance_noncommittal_scores_zero(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(GeneratedQuestions(questions=["?"], noncommittal=True))
    result = await AnswerRelevance(scripted_judge, HashingEmbeddings()).score(record)
    assert result.score == 0.0
    assert "noncommittal" in result.reason


# ----------------------------------------------------------- context precision ---


async def test_context_precision_rewards_relevant_chunks_ranked_first(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(*chunk_verdicts(0, 1))
    result = await ContextPrecision(scripted_judge).score(record)
    assert result.score == pytest.approx(0.5)
    assert "ground truth" in result.reason


async def test_context_precision_falls_back_to_answer_as_reference(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(*chunk_verdicts(1, 1))
    no_gt = record.model_copy(update={"ground_truth": None})
    result = await ContextPrecision(scripted_judge).score(no_gt)
    assert result.score == pytest.approx(1.0)
    assert "generated answer" in result.reason
    assert record.answer in scripted_judge.prompts[0]


async def test_context_precision_without_contexts_is_not_scored(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    result = await ContextPrecision(scripted_judge).score(
        record.model_copy(update={"contexts": []})
    )
    assert result.score is None
    assert scripted_judge.prompts == []


# -------------------------------------------------------------- context recall ---


async def test_context_recall_counts_attributable_sentences(scripted_judge: ScriptedJudge) -> None:
    rec = EvalRecord(
        question="q",
        answer="a",
        contexts=["ctx"],
        ground_truth="Fact one. Fact two. Fact three.",
    )
    scripted_judge.queue(verdicts(1, 1, 0))
    result = await ContextRecall(scripted_judge).score(rec)
    assert result.score == pytest.approx(2 / 3)
    assert len(result.details) == 3


async def test_context_recall_skips_without_ground_truth(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    result = await ContextRecall(scripted_judge).score(
        record.model_copy(update={"ground_truth": " "})
    )
    assert result.score is None
    assert result.ok
    assert result.reason.startswith("skipped")


async def test_context_recall_is_zero_without_contexts(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    result = await ContextRecall(scripted_judge).score(record.model_copy(update={"contexts": []}))
    assert result.score == 0.0


# -------------------------------------------------------------------- registry ---


def test_build_metrics_is_ordered_and_deduplicated(scripted_judge: ScriptedJudge) -> None:
    metrics = build_metrics(
        [
            MetricName.CONTEXT_RECALL,
            MetricName.FAITHFULNESS,
            MetricName.FAITHFULNESS,
            MetricName.ANSWER_RELEVANCE,
        ],
        judge=scripted_judge,
        embeddings=HashingEmbeddings(),
        settings=EvaluationSettings(answer_relevance_questions=5),
    )
    assert [m.name for m in metrics] == [
        MetricName.FAITHFULNESS,
        MetricName.ANSWER_RELEVANCE,
        MetricName.CONTEXT_RECALL,
    ]
    relevance = metrics[1]
    assert isinstance(relevance, AnswerRelevance)
    assert relevance.n_questions == 5


# --------------------------------------------------------------------- prompts ---


@pytest.mark.parametrize(
    "prompt",
    [
        prompts.verify_statements(["ctx"], ["a.", "b.", "c."]),
        prompts.attribute_sentences("q", ["ctx"], ["a.", "b.", "c."]),
    ],
)
def test_verdict_prompts_have_example_and_single_count(prompt: str) -> None:
    # The worked example is what makes 3B judges reliable; the only "(N total)" marker must be
    # the real one, because callers and fakes read the expected count from it.
    assert "Example" in prompt
    assert prompt.count(" total)") == 1
    assert "(3 total)" in prompt
    assert '{"verdicts": [{"reason": "...", "verdict": "yes" or "no"}]}' in prompt
    assert "{{" not in prompt
    assert '"verdict": 0' not in prompt  # labels are words, never digits


async def test_context_precision_judges_each_chunk_separately(
    record: EvalRecord, scripted_judge: ScriptedJudge
) -> None:
    scripted_judge.queue(*chunk_verdicts(1, 0))
    result = await ContextPrecision(scripted_judge).score(record)

    assert result.score == pytest.approx(1.0)  # the only useful chunk is ranked first
    assert len(scripted_judge.prompts) == 2
    for prompt, chunk in zip(scripted_judge.prompts, record.contexts, strict=True):
        assert chunk in prompt
        other = next(c for c in record.contexts if c != chunk)
        assert other not in prompt  # one chunk per prompt: nothing to misattribute
    assert [d.score for d in result.details] == [1.0, 0.0]


def test_chunk_usefulness_prompt_shape() -> None:
    prompt = prompts.judge_chunk_usefulness("q", "ref", "the chunk")
    assert "Example" in prompt
    assert '{"reason": "...", "verdict": "yes" or "no"}' in prompt
    assert "{{" not in prompt
