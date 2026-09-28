# Dashboard + CLI image. The judge (Ollama/vLLM) runs as a separate service.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN useradd --create-home --uid 1000 sentinel

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
RUN mkdir -p /app/.sentinel && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/rag_sentinel/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
