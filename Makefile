.DEFAULT_GOAL := help
PYTHON ?= python
TEST_DATABASE_URL ?= postgresql://rag:rag@localhost:5432/rag

.PHONY: help install lint format test test-integration eval eval-live corpus run up down

help: ## Show targets
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | awk -F ":.*## " '{printf "  %-17s %s\n", $$1, $$2}'

install: ## Install runtime and dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

lint: ## Ruff lint and format check
	ruff check .
	ruff format --check .

format: ## Apply ruff fixes and formatting
	ruff check --fix .
	ruff format .

test: ## Unit and API tests (no network, no database)
	$(PYTHON) -m pytest

test-integration: ## Include the pgvector round trip (needs `make up` or a Postgres)
	TEST_DATABASE_URL=$(TEST_DATABASE_URL) $(PYTHON) -m pytest

eval: ## Mock evals (offline, deterministic) -> evals/RESULTS.md
	$(PYTHON) -m evals.run

eval-live: ## Live evals with OpenAI embeddings and Claude (needs API keys)
	$(PYTHON) -m evals.run --mode live

corpus: ## Rebuild the eval PDFs from evals/corpus/source
	$(PYTHON) -m evals.build_corpus

run: ## Dev server with reload
	uvicorn app.main:app --reload --port 8000

up: ## API + pgvector Postgres via docker compose
	docker compose up --build

down: ## Stop the compose stack
	docker compose down
