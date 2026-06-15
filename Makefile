# ==============================================================================
# Predictive Maintenance MLOps — Developer Makefile
#
# Usage: make <target>
# All commands run inside the 'ncm' conda environment.
# ==============================================================================

.DEFAULT_GOAL := help
CONDA_RUN := conda run -n ncm
PYTHON := $(CONDA_RUN) python
PIP := $(CONDA_RUN) pip

# --- Setup -------------------------------------------------------------------

.PHONY: install
install: ## Install missing production + dev dependencies into ncm env
	$(PIP) install xgboost>=2.1 mlflow>=2.17 "dvc>=3.56" evidently>=0.5 \
		gunicorn>=23.0 pandera>=0.21 ruff>=0.8 mypy>=1.13 pytest-cov>=5.0

# --- Code Quality ------------------------------------------------------------

.PHONY: lint
lint: ## Run ruff linter on all source code
	$(CONDA_RUN) ruff check src/ api/ tests/

.PHONY: lint-fix
lint-fix: ## Auto-fix linting issues
	$(CONDA_RUN) ruff check --fix src/ api/ tests/

.PHONY: typecheck
typecheck: ## Run mypy type checker
	$(CONDA_RUN) mypy src/ api/ --ignore-missing-imports

.PHONY: format
format: ## Format code with ruff
	$(CONDA_RUN) ruff format src/ api/ tests/

# --- Testing -----------------------------------------------------------------

.PHONY: test
test: ## Run full test suite with coverage
	$(CONDA_RUN) pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing

.PHONY: test-fast
test-fast: ## Run tests without coverage (faster)
	$(CONDA_RUN) pytest tests/ -v --tb=short

# --- ML Pipeline -------------------------------------------------------------

.PHONY: dvc-repro
dvc-repro: ## Reproduce the full DVC pipeline (prepare → validate → featurize → train → evaluate)
	$(CONDA_RUN) dvc repro

.PHONY: train
train: ## Run training stage only
	$(PYTHON) -m src.train --config configs/params.yaml

.PHONY: evaluate
evaluate: ## Run evaluation stage only
	$(PYTHON) -m src.evaluate --config configs/params.yaml

.PHONY: drift-report
drift-report: ## Generate 3-layer drift monitoring report
	$(PYTHON) -m src.monitor --config configs/params.yaml

.PHONY: drift-simulate
drift-simulate: ## Generate drift report with simulated concept drift
	$(PYTHON) -m src.monitor --config configs/params.yaml --simulate-drift

.PHONY: retrain-check
retrain-check: ## Check if retraining is needed based on drift signals
	$(PYTHON) -m src.retrain_check --config configs/params.yaml

.PHONY: model-card
model-card: ## Auto-generate MODEL_CARD.md from latest training run
	$(PYTHON) -m src.model_card --config configs/params.yaml

# --- Serving -----------------------------------------------------------------

.PHONY: serve
serve: ## Start FastAPI dev server with hot reload
	$(CONDA_RUN) uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: serve-prod
serve-prod: ## Start production server (gunicorn + uvicorn workers)
	$(CONDA_RUN) gunicorn -w 2 -k uvicorn.workers.UvicornWorker \
		api.main:app --bind 0.0.0.0:8000 --timeout 120

# --- Docker ------------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build production Docker image
	docker build -f docker/Dockerfile -t pred-maint:latest .

.PHONY: docker-run
docker-run: ## Run Docker container locally
	docker run -p 8000:8000 --name pred-maint pred-maint:latest

.PHONY: docker-stop
docker-stop: ## Stop and remove Docker container
	docker stop pred-maint && docker rm pred-maint

# --- MLflow ------------------------------------------------------------------

.PHONY: mlflow-ui
mlflow-ui: ## Launch MLflow tracking UI
	$(CONDA_RUN) mlflow ui --host 0.0.0.0 --port 5000

# --- Cleanup -----------------------------------------------------------------

.PHONY: clean
clean: ## Remove all generated artifacts (keeps raw data)
	rm -rf data/processed/ data/features/ models/ metrics/ reports/
	rm -rf mlruns/ mlartifacts/
	rm -rf __pycache__ src/__pycache__ api/__pycache__ tests/__pycache__
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -f MODEL_CARD.md

# --- Help --------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo "Predictive Maintenance MLOps — Available Targets"
	@echo "================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
