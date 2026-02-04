.PHONY: setup install test clean download preprocess features label train evaluate backtest report help

PY?=python
SHELL := /bin/bash

help:  ## Show this help message
	@echo "Market Event AI - Trading System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Create virtual environment and install dependencies
	$(PY) -m venv .venv && \
	. .venv/bin/activate && \
	pip install -U pip && \
	pip install -e .

install:  ## Install package in development mode
	pip install -e .

download:  ## Download all data sources
	market-event-ai download --source all --start-date 2016-01-01 --end-date 2020-12-31

preprocess:  ## Preprocess raw data
	market-event-ai preprocess

features:  ## Extract features
	market-event-ai features

label:  ## Generate labels
	market-event-ai label --task classification --threshold 0.02

train:  ## Train model (default: xgboost)
	market-event-ai train --model-type xgboost

evaluate:  ## Evaluate model
	market-event-ai evaluate --model-type xgboost

backtest:  ## Run backtest
	market-event-ai backtest --model-type xgboost

report:  ## Generate report
	market-event-ai report --model-type xgboost

pipeline:  ## Run complete pipeline (download -> report)
	make download && \
	make preprocess && \
	make features && \
	make label && \
	make train && \
	make evaluate && \
	make backtest && \
	make report

test:  ## Run tests
	pytest -v

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	pytest tests/integration/ -v

test-coverage:  ## Run tests with coverage
	pytest --cov=market_event_ai --cov-report=html --cov-report=term

lint:  ## Run linters
	black src/ tests/ --check
	ruff check src/ tests/

format:  ## Format code
	black src/ tests/
	ruff check src/ tests/ --fix

clean:  ## Clean generated files
	rm -rf data/processed/* data/features/* data/labels/* data/models/* data/backtests/* data/reports/*
	rm -rf db/*.db
	rm -rf .pytest_cache __pycache__ */__pycache__ **/__pycache__
	rm -rf .coverage htmlcov/
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean  ## Clean everything including downloaded data
	rm -rf data/raw/*
	rm -rf .venv/

info:  ## Show system information
	market-event-ai info

# Legacy targets for compatibility
serve:  ## Run API (if implemented)
	@echo "API not implemented in this version"

docker-build:  ## Build Docker image (if needed)
	docker build -t market-event-ai:latest -f docker/Dockerfile .

docker-run:  ## Run Docker container
	docker run --rm -p 8000:8000 market-event-ai:latest
