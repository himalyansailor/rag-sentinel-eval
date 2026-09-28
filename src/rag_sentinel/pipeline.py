"""Reference RAG pipeline: LangChain + in-memory ChromaDB (see docs/LLD.md §7).

This is the *system under test* that ships with the project so the evaluator works out of the
box. To evaluate your own application, implement :class:`rag_sentinel.protocols.RAGPipeline`.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypedDict

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_sentinel.exceptions import PipelineError
from rag_sentinel.log import get_logger
from rag_sentinel.metrics.text import split_sentences
from rag_sentinel.models import RAGResponse, RetrievedContext

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from rag_sentinel.config import PipelineSettings

log = get_logger(__name__)

CORPUS_SUFFIXES = frozenset({".md", ".txt"})

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise support assistant. Answer the question using ONLY the "
            "context excerpts below. Reply in one to three plain sentences of your own; do not "
            "copy headings, markdown or excerpt separators. If the context does not contain "
            'the answer, reply exactly: "I don\'t know based on the available documents."\n\n'
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class GeneratorInput(TypedDict):
    question: str
    context: str
    chunks: list[str]


# --------------------------------------------------------------------------- indexing ---


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every ``.md``/``.txt`` file under ``corpus_dir`` (recursively, sorted)."""
    if not corpus_dir.is_dir():
        msg = f"corpus directory not found: {corpus_dir}"
        raise PipelineError(msg)
    documents: list[Document] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in CORPUS_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            source = path.relative_to(corpus_dir).as_posix()
            documents.append(Document(page_content=text, metadata={"source": source}))
    if not documents:
        msg = f"no .md or .txt documents found in {corpus_dir}"
        raise PipelineError(msg)
    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _chunk_id(chunk: Document, index: int) -> str:
    key = f"{chunk.metadata.get('source', '')}:{index}:{chunk.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def build_vector_store(
    chunks: list[Document], embeddings: Embeddings, collection_name: str
) -> Chroma:
    """Index chunks into a fresh in-memory Chroma collection (cosine distance)."""
    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
    )
    store = Chroma(
        # EphemeralClient shares process-wide state: suffix the name so pipelines never collide.
        collection_name=f"{collection_name}-{uuid.uuid4().hex[:8]}",
        embedding_function=embeddings,
        client=client,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks, ids=[_chunk_id(c, i) for i, c in enumerate(chunks)])
    return store


# ------------------------------------------------------------------------- generation ---

_WORD_RE = re.compile(r"[a-z0-9]+")
# fmt: off
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "our", "the", "to", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
})
# fmt: on


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def extractive_answer(inputs: GeneratorInput, max_sentences: int = 2) -> str:
    """Deterministic offline "generator": the context sentences that best overlap the question.

    It never invents content, so it is faithful by construction and a useful baseline.
    """
    question_words = _content_words(inputs["question"])
    candidates: list[tuple[int, int, str]] = []
    for chunk_rank, chunk in enumerate(inputs["chunks"]):
        for sentence in split_sentences(chunk):
            if sentence.startswith("#"):
                continue
            overlap = len(question_words & _content_words(sentence))
            if overlap:
                candidates.append((overlap, -chunk_rank, sentence))
    if not candidates:
        return "I don't know based on the available documents."
    best = sorted(candidates, key=lambda c: (c[0], c[1]), reverse=True)[:max_sentences]
    return " ".join(sentence for _, _, sentence in best)


def build_generator(settings: PipelineSettings) -> Runnable[GeneratorInput, str]:
    """Return the answer-generation runnable selected by ``settings.generator``."""
    if settings.generator == "extractive":
        return RunnableLambda(extractive_answer)
    from langchain_ollama import ChatOllama  # noqa: PLC0415 - optional heavy import

    llm = ChatOllama(
        model=settings.generator_model,
        base_url=settings.base_url,
        temperature=0.0,
        seed=7,
    )
    chain: Runnable[Any, str] = ANSWER_PROMPT | llm | StrOutputParser()
    return chain


def format_context(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(chunks)


# --------------------------------------------------------------------------- pipeline ---


class LangChainRAGPipeline:
    """Retrieve top-k chunks from Chroma, then generate an answer with an LCEL runnable."""

    def __init__(
        self,
        *,
        vector_store: Chroma,
        generator: Runnable[GeneratorInput, str],
        top_k: int = 4,
        generator_label: str = "custom",
        chunk_count: int = 0,
    ) -> None:
        self._store = vector_store
        self._generator = generator
        self._top_k = top_k
        self._generator_label = generator_label
        self._chunk_count = chunk_count

    @classmethod
    def from_settings(cls, settings: PipelineSettings, embeddings: Embeddings) -> Self:
        """Load the corpus, chunk it, index it and wire up the generator."""
        started = time.perf_counter()
        try:
            documents = load_corpus(settings.corpus_dir)
            chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
            store = build_vector_store(chunks, embeddings, settings.collection_name)
            generator = build_generator(settings)
        except PipelineError:
            raise
        except Exception as exc:
            msg = f"failed to build the RAG pipeline: {exc}"
            raise PipelineError(msg) from exc
        log.info(
            "pipeline.built",
            documents=len(documents),
            chunks=len(chunks),
            generator=settings.generator,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        label = (
            "extractive"
            if settings.generator == "extractive"
            else f"ollama/{settings.generator_model}"
        )
        return cls(
            vector_store=store,
            generator=generator,
            top_k=settings.top_k,
            generator_label=label,
            chunk_count=len(chunks),
        )

    @property
    def generator_label(self) -> str:
        return self._generator_label

    def describe(self) -> str:
        return (
            f"LangChainRAGPipeline(chroma, chunks={self._chunk_count}, top_k={self._top_k}, "
            f"generator={self._generator_label})"
        )

    async def retrieve(self, question: str) -> list[RetrievedContext]:
        try:
            hits = await self._store.asimilarity_search_with_score(question, k=self._top_k)
        except Exception as exc:
            msg = f"retrieval failed: {exc}"
            raise PipelineError(msg) from exc
        return [
            RetrievedContext(
                content=doc.page_content,
                source=str(doc.metadata.get("source", "unknown")),
                rank=rank,
                score=round(1.0 - float(distance), 4),  # cosine distance -> similarity
            )
            for rank, (doc, distance) in enumerate(hits, start=1)
        ]

    async def aquery(self, question: str) -> RAGResponse:
        """Answer ``question``. Raises :class:`PipelineError` on any failure."""
        if not question.strip():
            msg = "question must not be empty"
            raise PipelineError(msg)
        started = time.perf_counter()
        contexts = await self.retrieve(question)
        chunks = [c.content for c in contexts]
        try:
            answer = await self._generator.ainvoke(
                {"question": question, "context": format_context(chunks), "chunks": chunks}
            )
        except Exception as exc:
            msg = f"generation failed: {exc}"
            raise PipelineError(msg) from exc
        return RAGResponse(
            question=question,
            answer=answer.strip(),
            contexts=contexts,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
