"""Application settings (see docs/LLD.md §3).

All configuration comes from environment variables prefixed ``SENTINEL_`` (nested groups use
``__``, e.g. ``SENTINEL_JUDGE__MODEL``) or from a local ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_sentinel.models import MetricName

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def _validate_threshold_map(value: dict[str, float]) -> dict[str, float]:
    valid = {m.value for m in MetricName}
    for name, threshold in value.items():
        if name not in valid:
            msg = f"unknown metric {name!r}; expected one of {sorted(valid)}"
            raise ValueError(msg)
        if not 0.0 <= threshold <= 1.0:
            msg = f"threshold for {name!r} must be within [0, 1], got {threshold}"
            raise ValueError(msg)
    return value


class JudgeSettings(BaseModel):
    provider: Literal["ollama", "openai_compatible"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = 42


class EmbeddingSettings(BaseModel):
    provider: Literal["ollama", "hashing"] = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"


class PipelineSettings(BaseModel):
    corpus_dir: Path = Path("data/corpus")
    collection_name: str = "sentinel"
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, ge=1, le=50)
    generator: Literal["ollama", "extractive"] = "ollama"
    generator_model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size"
            raise ValueError(msg)
        return self


class EvaluationSettings(BaseModel):
    max_concurrency: int = Field(default=4, ge=1, le=64)
    answer_relevance_questions: int = Field(default=3, ge=1, le=10)


class GateSettings(BaseModel):
    thresholds: dict[str, float] = Field(default_factory=lambda: {"faithfulness": 0.85})
    warn_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "answer_relevance": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.7,
        }
    )
    max_error_rate: float = Field(default=0.2, ge=0.0, le=1.0)

    _check_thresholds = field_validator("thresholds", "warn_thresholds")(_validate_threshold_map)


class StorageSettings(BaseModel):
    db_path: Path = Path(".sentinel/history.db")


class Settings(BaseSettings):
    """Root settings object. Obtain it through :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: LogLevel = "INFO"
    log_json: bool = False
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached; call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
