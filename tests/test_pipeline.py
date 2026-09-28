from __future__ import annotations

from pathlib import Path

import pytest

from rag_sentinel.config import PipelineSettings
from rag_sentinel.exceptions import PipelineError
from rag_sentinel.llm.embeddings import HashingEmbeddings
from rag_sentinel.pipeline import LangChainRAGPipeline, extractive_answer, load_corpus
from rag_sentinel.protocols import RAGPipeline

from .conftest import CORPUS_DIR


def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    emb = HashingEmbeddings(dim=64)
    a, b = emb.embed_documents(["refund within 14 days", "refund within 14 days"])
    assert a == b
    assert sum(v * v for v in a) == pytest.approx(1.0)
    assert emb.embed_query("") == [0.0] * 64


def test_load_corpus_reads_markdown_with_sources() -> None:
    docs = load_corpus(CORPUS_DIR)
    assert {d.metadata["source"] for d in docs} >= {"refund_policy.md", "sla.md"}


def test_load_corpus_rejects_missing_or_empty_dirs(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="not found"):
        load_corpus(tmp_path / "missing")
    with pytest.raises(PipelineError, match=r"no \.md"):
        load_corpus(tmp_path)


def test_extractive_answer_picks_overlapping_sentences() -> None:
    answer = extractive_answer(
        {
            "question": "How long is the refund window?",
            "context": "",
            "chunks": ["# Title", "The refund window is 14 days. Weather is nice."],
        }
    )
    assert answer == "The refund window is 14 days."


def test_extractive_answer_admits_ignorance() -> None:
    answer = extractive_answer({"question": "zebra?", "context": "", "chunks": ["Nothing here."]})
    assert answer.startswith("I don't know")


async def test_pipeline_retrieves_relevant_context(offline_pipeline: LangChainRAGPipeline) -> None:
    assert isinstance(offline_pipeline, RAGPipeline)
    response = await offline_pipeline.aquery(
        "How long do monthly plan customers have to request a full refund?"
    )
    assert response.contexts[0].source == "refund_policy.md"
    assert [c.rank for c in response.contexts] == [1, 2, 3]
    assert "14 days" in response.answer
    assert response.latency_ms >= 0
    assert "extractive" in offline_pipeline.describe()


async def test_pipeline_rejects_empty_question(offline_pipeline: LangChainRAGPipeline) -> None:
    with pytest.raises(PipelineError):
        await offline_pipeline.aquery("   ")


def test_two_pipelines_do_not_share_collections() -> None:
    settings = PipelineSettings(corpus_dir=CORPUS_DIR, generator="extractive")
    a = LangChainRAGPipeline.from_settings(settings, HashingEmbeddings())
    b = LangChainRAGPipeline.from_settings(settings, HashingEmbeddings())
    assert a._store._collection.name != b._store._collection.name
    assert a._store._collection.count() == b._store._collection.count()


def test_settings_reject_overlap_larger_than_chunk() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        PipelineSettings(chunk_size=200, chunk_overlap=200)
