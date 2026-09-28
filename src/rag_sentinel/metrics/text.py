"""Small numeric and text utilities used by metrics. Pure functions, unit-tested directly."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation and newlines; drop empty fragments."""
    parts = (p.strip(" \t-*•") for p in _SENTENCE_BOUNDARY.split(text))
    return [p for p in parts if p]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 if either has zero norm."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def average_precision(relevance: Sequence[int]) -> float:
    """Average precision of a ranked list of binary relevance labels (rank 1 first).

    ``AP = sum_k(precision@k * rel_k) / sum_k(rel_k)``; 0.0 when nothing is relevant.
    """
    hits = 0
    total = 0.0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def numbered(items: Sequence[str]) -> str:
    """Render items as ``[1] ...`` lines, the format every prompt uses."""
    return "\n".join(f"[{i}] {item}" for i, item in enumerate(items, start=1))
