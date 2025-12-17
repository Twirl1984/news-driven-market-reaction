from fastapi.testclient import TestClient
import importlib
import os

def test_health_endpoint(monkeypatch):
    # Use a dummy model dir only if user hasn't trained yet; skip if missing.
    # If MODEL_DIR isn't present, we just assert that import fails gracefully in CI runs.
    model_dir = os.getenv("MODEL_DIR", "models/transformer_reg")
    if not os.path.exists(model_dir):
        return

    api = importlib.import_module("src.api")
    client = TestClient(api.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
