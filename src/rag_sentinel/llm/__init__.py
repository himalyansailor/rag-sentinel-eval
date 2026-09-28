"""LLM adapters: judges and embeddings."""

from rag_sentinel.llm.base import BaseHTTPJudge
from rag_sentinel.llm.embeddings import HashingEmbeddings, build_embeddings
from rag_sentinel.llm.factory import build_judge
from rag_sentinel.llm.ollama import OllamaJudge
from rag_sentinel.llm.openai_compat import OpenAICompatibleJudge

__all__ = [
    "BaseHTTPJudge",
    "HashingEmbeddings",
    "OllamaJudge",
    "OpenAICompatibleJudge",
    "build_embeddings",
    "build_judge",
]
