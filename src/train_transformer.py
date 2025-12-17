import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from src.config import PATHS

DATE_COL = "date"
TEXT_COL = "headline"
LABEL_COL = "log_return_next_day"

DEFAULT_MODEL = os.getenv("HF_MODEL", "distilbert-base-uncased")

def time_split(df: pd.DataFrame, frac_train: float = 0.8):
    df = df.sort_values(DATE_COL)
    cut = int(len(df) * frac_train)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def compute_mae(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = preds.squeeze()
    return float(np.mean(np.abs(preds - labels)))

def main():
    if not PATHS.processed_parquet.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {PATHS.processed_parquet}. Run: python -m src.build_dataset"
        )

    df = pd.read_parquet(PATHS.processed_parquet)
    train_df, val_df = time_split(df, 0.8)

    train_ds = Dataset.from_pandas(train_df[[TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "label"}))
    val_ds   = Dataset.from_pandas(val_df[[TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "label"}))

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    def tok_fn(batch):
        return tok(batch[TEXT_COL], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds   = val_ds.map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL,
        num_labels=1,
        problem_type="regression",
    )

    args = TrainingArguments(
        output_dir=str(PATHS.model_dir),
        learning_rate=float(os.getenv("LR", "2e-5")),
        per_device_train_batch_size=int(os.getenv("BATCH", "16")),
        per_device_eval_batch_size=int(os.getenv("BATCH", "16")),
        num_train_epochs=float(os.getenv("EPOCHS", "2")),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        return {"mae": compute_mae(np.array(preds), np.array(labels))}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    PATHS.model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(PATHS.model_dir))
    tok.save_pretrained(str(PATHS.model_dir))
    print(f"Saved model to: {PATHS.model_dir}")

if __name__ == "__main__":
    main()
