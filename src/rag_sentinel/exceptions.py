"""Error taxonomy for rag-sentinel (see docs/LLD.md §4).

Every error raised deliberately by the package derives from :class:`SentinelError`, so callers
can catch one base class and still map specific failures to exit codes.
"""

from __future__ import annotations


class SentinelError(Exception):
    """Base class for all rag-sentinel errors."""


class ConfigurationError(SentinelError):
    """Invalid settings or command-line arguments."""


class DatasetError(SentinelError):
    """The golden dataset could not be read or is invalid."""


class PipelineError(SentinelError):
    """The RAG pipeline failed to build, retrieve or generate."""


class StorageError(SentinelError):
    """The run-history store failed."""


class JudgeError(SentinelError):
    """Base class for failures of the LLM judge."""


class JudgeUnavailableError(JudgeError):
    """The judge could not be reached, timed out, or returned a retryable server error."""


class JudgeConfigurationError(JudgeError):
    """The judge rejected the request (unknown model, bad credentials, bad request)."""


class JudgeResponseError(JudgeError):
    """The judge answered, but not with valid, schema-conforming output."""
