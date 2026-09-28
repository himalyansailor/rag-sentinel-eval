.DEFAULT_GOAL := help
RUN := poetry run

.PHONY: help install lint format typecheck test check models eval eval-offline demo dashboard clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the project with dev dependencies
	poetry install
	$(RUN) pre-commit install

lint: ## Ruff lint + format check
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format: ## Auto-format and fix lint
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

typecheck: ## Strict mypy
	$(RUN) mypy src

test: ## Offline unit tests with coverage
	$(RUN) pytest --cov

check: lint typecheck test ## Everything CI's quality job runs

models: ## Pull the default Ollama models
	ollama pull qwen2.5:3b
	ollama pull llama3.2:3b
	ollama pull nomic-embed-text

eval: ## Evaluate the golden set with Ollama and enforce the gate
	$(RUN) rag-sentinel evaluate --output-markdown reports/summary.md --output-json reports/report.json

eval-offline: ## Pipeline smoke run without a generator LLM (judge still required)
	SENTINEL_PIPELINE__GENERATOR=extractive SENTINEL_EMBEDDINGS__PROVIDER=hashing $(RUN) rag-sentinel evaluate --limit 3

demo: ## Seed synthetic history and open the dashboard
	$(RUN) rag-sentinel seed-demo
	$(RUN) rag-sentinel dashboard

dashboard: ## Open the Streamlit dashboard
	$(RUN) rag-sentinel dashboard

clean: ## Remove caches and local outputs
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.xml htmlcov reports dist
