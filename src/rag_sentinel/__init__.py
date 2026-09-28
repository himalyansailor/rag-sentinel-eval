"""rag-sentinel-eval: CI quality gates and regression tracking for RAG pipelines."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rag-sentinel-eval")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
