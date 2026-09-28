"""Golden dataset loading.

Format: JSON Lines, one object per line::

    {"id": "refund-01", "question": "...", "ground_truth": "...", "metadata": {"topic": "billing"}}

A JSON array of the same objects is accepted too. Blank lines and lines starting with ``//``
are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_sentinel.exceptions import DatasetError
from rag_sentinel.models import EvalSample


def _parse_rows(path: Path, text: str) -> list[tuple[int, Any]]:
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"{path}: invalid JSON: {exc}"
            raise DatasetError(msg) from exc
        if not isinstance(data, list):
            msg = f"{path}: expected a JSON array of samples"
            raise DatasetError(msg)
        return list(enumerate(data, start=1))

    rows: list[tuple[int, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            rows.append((line_no, json.loads(stripped)))
        except json.JSONDecodeError as exc:
            msg = f"{path}:{line_no}: invalid JSON: {exc.msg}"
            raise DatasetError(msg) from exc
    return rows


def load_golden_set(path: Path, *, limit: int | None = None) -> list[EvalSample]:
    """Load and validate a golden dataset.

    Raises:
        DatasetError: file missing, malformed, empty, or containing duplicate ids.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read dataset {path}: {exc}"
        raise DatasetError(msg) from exc

    samples: list[EvalSample] = []
    seen: set[str] = set()
    for line_no, row in _parse_rows(path, text):
        try:
            sample = EvalSample.model_validate(row)
        except ValidationError as exc:
            msg = f"{path}:{line_no}: invalid sample: {exc.errors()[0]['msg']}"
            raise DatasetError(msg) from exc
        if sample.id in seen:
            msg = f"{path}:{line_no}: duplicate sample id {sample.id!r}"
            raise DatasetError(msg)
        seen.add(sample.id)
        samples.append(sample)

    if not samples:
        msg = f"dataset {path} contains no samples"
        raise DatasetError(msg)
    return samples[:limit] if limit is not None else samples


def dataset_name(path: Path) -> str:
    """Stable dataset identifier used to group runs (the file stem)."""
    return path.stem
