from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.config import PATHS

def load():
    tok = AutoTokenizer.from_pretrained(PATHS.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(PATHS.model_dir)
    model.eval()
    return tok, model

def predict_one(headline: str) -> float:
    tok, model = load()
    x = tok(headline, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        y = model(**x).logits.squeeze().item()
    return float(y)

if __name__ == "__main__":
    print(predict_one("Company warns of weaker demand in Q3"))
