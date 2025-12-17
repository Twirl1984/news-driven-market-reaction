.PHONY: setup ingest build train serve test docker-build docker-run kind-create helm-install helm-uninstall clean

PY?=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

ingest:
	$(PY) -m src.ingest_to_sqlite

build:
	$(PY) -m src.build_dataset

train:
	$(PY) -m src.train_transformer

serve:
	MODEL_DIR=models/transformer_reg uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t news-market-reaction:latest -f docker/Dockerfile .

docker-run:
	docker run --rm -p 8000:8000 -e MODEL_DIR=/app/models/transformer_reg news-market-reaction:latest

kind-create:
	kind create cluster --name nmr

helm-install:
	helm upgrade --install nmr ./helm --set image.repository=news-market-reaction --set image.tag=latest

helm-uninstall:
	helm uninstall nmr || true

clean:
	rm -rf data/processed db/*.db models/transformer_reg .pytest_cache __pycache__ */__pycache__
