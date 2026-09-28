"""Embedding providers. Everything speaks LangChain's :class:`Embeddings` interface."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from rag_sentinel.config import EmbeddingSettings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings via signed feature hashing.

    Unigrams and bigrams are hashed into ``dim`` buckets and the vector is L2-normalised, so
    cosine similarity approximates lexical overlap. Good enough for offline tests and demos;
    use a real embedding model for meaningful evaluation.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            msg = "dim must be positive"
            raise ValueError(msg)
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        vector = [0.0] * self.dim
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: EmbeddingSettings) -> Embeddings:
    """Create the configured embeddings provider."""
    if settings.provider == "hashing":
        return HashingEmbeddings()
    from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415 - optional heavy import

    embeddings: Embeddings = OllamaEmbeddings(model=settings.model, base_url=settings.base_url)
    return embeddings


def embedding_model_label(settings: EmbeddingSettings) -> str:
    """Human-readable identifier stored with each run."""
    return "hashing-384" if settings.provider == "hashing" else f"ollama/{settings.model}"
