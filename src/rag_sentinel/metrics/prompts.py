"""Prompt builders for every judge step.

Design rules (tuned for small open-weight judges, validated against qwen2.5:3b):

* every verdict step asks one plain yes/no question and shows a worked example — without the
  example, 3B judges quoted the supporting fact and still answered "no";
* list-style steps number every item and state the count; context precision judges one
  chunk per call instead, because small models misattribute verdicts across long chunks;
* the rationale field precedes the verdict so the model reasons before labelling.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_sentinel.metrics.text import numbered

_VERDICT_FORMAT = '{"verdicts": [{"reason": "...", "verdict": "yes" or "no"}]}'


def _contexts_block(contexts: Sequence[str]) -> str:
    return "\n\n".join(contexts) if contexts else "(no context retrieved)"


def _support_example() -> str:
    # Deliberately unrelated to any real corpus, and without a "(N total)" count, so it can
    # never be mistaken for the real task.
    return """Example
CONTEXT: The office opens at 9am. Parking is free for employees.
ITEMS:
[1] Employees can park for free.
[2] The office closes at 5pm.
Answer: {"verdicts": [
  {"reason": "The context says parking is free for employees.", "verdict": "yes"},
  {"reason": "The context never states a closing time.", "verdict": "no"}
]}"""


def extract_statements(question: str, answer: str) -> str:
    return f"""Break the ANSWER into atomic, self-contained factual statements.

Rules:
- Each statement expresses exactly one fact.
- Replace pronouns with the entities they refer to so each statement stands alone.
- Do not add information that is not in the ANSWER.
- Ignore filler, greetings and statements about the assistant itself.
- If the ANSWER contains no factual claims (e.g. "I don't know"), return an empty list.

QUESTION: {question}

ANSWER: {answer}

Respond with JSON: {{"statements": ["...", "..."]}}"""


def verify_statements(contexts: Sequence[str], statements: Sequence[str]) -> str:
    return f"""Task: for each numbered STATEMENT, answer "Does the CONTEXT contain this
information?"
- verdict "yes": the context states the same fact, even if worded differently.
- verdict "no": the fact is missing from the context or contradicts it.
Judge only the statement's own claim. Do not use outside knowledge.

{_support_example()}

Now do the task.
CONTEXT:
{_contexts_block(contexts)}

STATEMENTS ({len(statements)} total):
{numbered(statements)}

Return exactly {len(statements)} verdicts, in statement order.
Respond with JSON: {_VERDICT_FORMAT}"""


def generate_questions(answer: str, n: int) -> str:
    return f"""Write {n} different questions that the ANSWER below would be a direct response to.
Also decide whether the ANSWER is noncommittal: evasive, vague or refusing (for example
"I don't know" or "I'm not sure"). A specific answer is not noncommittal.

ANSWER: {answer}

Respond with JSON: {{"questions": ["...", "..."], "noncommittal": true or false}}"""


def judge_chunk_usefulness(question: str, reference: str, chunk: str) -> str:
    """One chunk per call. Batching all chunks into one prompt made qwen2.5:3b attach verdicts
    to the wrong chunk number (31/40 correct vs 39/40 per chunk, see LLD §6.5)."""
    return f"""Task: answer "Does this CHUNK contain information that appears in the REFERENCE
ANSWER?"
- verdict "yes": at least one fact of the reference answer is stated in the chunk.
- verdict "no": nothing in the chunk is used by the reference answer.

Example
REFERENCE ANSWER: Employees can park for free.
CHUNK: Parking is free for employees. The cafeteria serves lunch from noon.
Answer: {{"reason": "The chunk states that parking is free for employees.", "verdict": "yes"}}

Now do the task.
QUESTION: {question}

REFERENCE ANSWER: {reference}

CHUNK:
{chunk}

Respond with JSON: {{"reason": "...", "verdict": "yes" or "no"}}"""


def attribute_sentences(question: str, contexts: Sequence[str], sentences: Sequence[str]) -> str:
    return f"""Task: for each numbered SENTENCE of the reference answer, answer "Does the CONTEXT
contain this information?"
- verdict "yes": the context states the same fact, even if worded differently.
- verdict "no": the fact is missing from the context or contradicts it.
Judge only the sentence's own claim. Do not use outside knowledge.

{_support_example()}

Now do the task.
QUESTION: {question}

CONTEXT:
{_contexts_block(contexts)}

SENTENCES ({len(sentences)} total):
{numbered(sentences)}

Return exactly {len(sentences)} verdicts, in sentence order.
Respond with JSON: {_VERDICT_FORMAT}"""
