import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = os.getenv("MODEL_DIR", "models/transformer_reg")

app = FastAPI(title="News Market Reaction API", version="0.1.0")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

class PredictRequest(BaseModel):
    headline: str = Field(..., min_length=3, max_length=500)

class PredictResponse(BaseModel):
    predicted_log_return_next_day: float
    direction: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = tok(req.headline, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        y = model(**x).logits.squeeze().item()
    direction = "up" if y >= 0 else "down"
    return {"predicted_log_return_next_day": float(y), "direction": direction}
