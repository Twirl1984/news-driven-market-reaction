# News → Market Reaction Predictor (Option A: Kaggle paired dataset)

This repo is a hands-on MVP for:
- **Deep learning with Transformers** (headline → next-day log return)
- **Data engineering** (CSV → SQLite → parquet)
- **MLOps basics** (reproducible training, tests, FastAPI inference, Docker, Helm/K8s)

## 0) Dataset
Download the Kaggle dataset (example):
- "S&P 500 with Financial News Headlines (2008–2024)"

Put the CSV into:
- `data/raw/sp500_headlines.csv`

Expected minimal columns (rename in code if needed):
- `date` (YYYY-MM-DD or timestamp)
- `headline` (string)
- `close` (float close price for the same date)

## 1) Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 1) ingest CSV to SQLite
python -m src.ingest_to_sqlite

# 2) build modeling table (parquet)
python -m src.build_dataset

# 3) train transformer regressor
python -m src.train_transformer

# 4) run API
MODEL_DIR=models/transformer_reg uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Test prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"headline":"Company warns of weaker demand in Q3"}'
```

## 2) Docker
```bash
docker build -t news-market-reaction:latest -f docker/Dockerfile .
docker run --rm -p 8000:8000 -e MODEL_DIR=/app/models/transformer_reg \
  -v "$(pwd)/models:/app/models" news-market-reaction:latest
```

## 3) Kubernetes (kind/minikube) + Helm
This is a lightweight local stand-in for OpenShift-like workflows.

Prereqs:
- kind OR minikube
- kubectl
- helm

Build local image, then load it into kind:
```bash
docker build -t news-market-reaction:latest -f docker/Dockerfile .
kind create cluster --name nmr
kind load docker-image news-market-reaction:latest --name nmr
helm upgrade --install nmr ./helm --set image.repository=news-market-reaction --set image.tag=latest
kubectl port-forward svc/nmr 8000:8000
```

## 4) Notes / next steps
- Replace `distilbert-base-uncased` with a finance-tuned model if desired.
- Add time-series features (last-5-day returns, volatility) and fuse with text embeddings.
- Add "policy" (prescriptive layer): e.g. alert/monitor/ignore based on predicted return & confidence.
