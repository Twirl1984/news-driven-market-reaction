import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from transformers import AutoModel, AutoTokenizer

DATASET_PATH = Path("data/processed/dataset.parquet")

MODEL_DIR = Path("models/embedding_ridge")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
LOCAL_MODEL_PATH = Path("models/distilbert-base-uncased")

MAX_LEN = 64
RANDOM_STATE = 42


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_transformer(model_name: str, local_model_path: Path):
    use_local = local_model_path.exists()
    src = local_model_path if use_local else model_name
    tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=use_local)
    model = AutoModel.from_pretrained(src, local_files_only=use_local)
    model.eval()
    return tokenizer, model, use_local, str(src)


def embed_to_memmap(
    texts: pd.Series,
    tokenizer,
    model,
    out_path: Path,
    batch_size: int,
    max_len: int,
    dtype: np.dtype,
) -> Tuple[Path, Tuple[int, int], str]:
    """
    Embed texts in batches and store to a memmap on disk.
    Returns: (path, shape, dtype_str)
    """
    n = len(texts)
    hidden_dim = int(model.config.hidden_size)
    shape = (n, hidden_dim)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    X_mm = np.memmap(out_path, mode="w+", dtype=dtype, shape=shape)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts.iloc[start:end].astype(str).tolist()

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        with torch.inference_mode():
            out = model(**enc)
            pooled = mean_pooling(out, enc["attention_mask"])
            pooled_np = pooled.cpu().numpy().astype(dtype, copy=False)

        X_mm[start:end, :] = pooled_np

        if (start // batch_size) % 50 == 0:
            print(f"  embedded {end}/{n}")

    X_mm.flush()
    return out_path, shape, str(dtype)


def open_memmap(path: Path, shape: Tuple[int, int], dtype: np.dtype, mode: str = "r"):
    return np.memmap(path, mode=mode, dtype=dtype, shape=shape)


@dataclass
class ModelMetadata:
    created_utc: str
    dataset_path: str
    dataset_sha256: Optional[str]
    dataset_rows: int
    split_idx: int
    train_rows: int
    val_rows: int
    text_column: str
    target_column: str

    model_name: str
    transformer_source: str
    used_local_model_path: bool
    max_len: int
    batch_size: int
    dtype: str
    hidden_dim: int

    x_train_path: str
    x_train_shape: Tuple[int, int]
    x_val_path: str
    x_val_shape: Tuple[int, int]

    ridge_alpha: float
    ridge_random_state: int

    github_run_id: Optional[str] = None
    github_run_number: Optional[str] = None
    github_sha: Optional[str] = None
    github_ref: Optional[str] = None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_metrics(y_train: np.ndarray, y_val: np.ndarray, preds: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_val, preds))

    pred_zero = np.zeros_like(y_val)
    mae_zero = float(mean_absolute_error(y_val, pred_zero))

    pred_mean = np.full_like(y_val, fill_value=float(np.mean(y_train)))
    mae_mean = float(mean_absolute_error(y_val, pred_mean))

    sign_true = np.sign(y_val)
    sign_pred = np.sign(preds)
    mask = (sign_true != 0)

    if mask.any():
        dir_acc = float((sign_true[mask] == sign_pred[mask]).mean())
        n_nonzero = int(mask.sum())
    else:
        dir_acc = None
        n_nonzero = 0

    return {
        "mae": mae,
        "mae_zero": mae_zero,
        "mae_mean": mae_mean,
        "directional_acc_nonzero": dir_acc,
        "n_val": int(len(y_val)),
        "n_val_nonzero": n_nonzero,
    }


def run_alpha_sweep(
    model_dir: Path,
    alpha_grid: list,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
) -> dict:
    print(f"Running alpha sweep on {len(alpha_grid)} values...")
    results = []
    best = None

    for a in alpha_grid:
        reg = Ridge(alpha=float(a), random_state=RANDOM_STATE)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_val)
        m = compute_metrics(y_train, y_val, preds)

        row = {"alpha": float(a), **m}
        results.append(row)

        if best is None or row["mae"] < best["mae"]:
            best = row

        dir_str = (
            f"{row['directional_acc_nonzero']:.3%}"
            if row["directional_acc_nonzero"] is not None
            else "NA"
        )

        print(
            f"alpha={a:<8g}  MAE={row['mae']:.6f}  "
            f"MAE0={row['mae_zero']:.6f}  MEAN={row['mae_mean']:.6f}  "
            f"DIR={dir_str}"
        )

    results_sorted = sorted(results, key=lambda r: r["mae"])

    out = {
        "created_utc": utc_now_iso(),
        "alpha_grid": [float(a) for a in alpha_grid],
        "best": best,
        "results_sorted_by_mae": results_sorted,
    }

    out_path = model_dir / "alpha_sweep.json"
    write_json(out_path, out)
    print(f"\nSaved sweep results to: {out_path}")

    print("\nTop 5 alphas by MAE:")
    for r in results_sorted[:5]:
        print(f"  alpha={r['alpha']:<8g}  MAE={r['mae']:.6f}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Batched DistilBERT embeddings + Ridge (memmap) with metrics/metadata and alpha sweep."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--embed-only", action="store_true", help="Only create memmap embeddings; do not train Ridge.")
    group.add_argument("--train-only", action="store_true", help="Only train Ridge using existing memmap embeddings.")

    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size (lower if RAM is tight).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], help="Memmap dtype.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha.")
    parser.add_argument("--max-len", type=int, default=MAX_LEN, help="Tokenizer max length.")
    parser.add_argument("--text-col", type=str, default="Title", help="Text column in dataset.")
    parser.add_argument("--target-col", type=str, default="log_return_next_day", help="Target column in dataset.")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Path to dataset parquet.")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Output directory.")

    parser.add_argument(
        "--alpha-sweep",
        action="store_true",
        help="Run a mini grid search over Ridge alpha and write alpha_sweep.json.",
    )
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100",
        help="Comma-separated list of alphas for sweep.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if args.dtype == "float32" else np.float16

    x_train_path = model_dir / "X_train.memmap"
    x_val_path = model_dir / "X_val.memmap"
    split_info_path = model_dir / "split_info.json"
    embedding_info_path = model_dir / "embedding_info.json"

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path).sort_values("Date").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # Save split info (useful for reproducibility)
    split_info = {
        "created_utc": utc_now_iso(),
        "dataset_path": str(dataset_path),
        "dataset_rows": int(len(df)),
        "split_idx": int(split_idx),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "text_column": args.text_col,
        "target_column": args.target_col,
    }
    write_json(split_info_path, split_info)

    tokenizer = model = None
    used_local = False
    transformer_source = None
    hidden_dim = None

    # ---- EMBED STEP ----
    if not args.train_only:
        tokenizer, model, used_local, transformer_source = load_transformer(MODEL_NAME, LOCAL_MODEL_PATH)
        hidden_dim = int(model.config.hidden_size)

        print("Embedding train set to memmap...")
        _, x_train_shape, dtype_str = embed_to_memmap(
            train_df[args.text_col],
            tokenizer,
            model,
            x_train_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            dtype=dtype,
        )

        print("Embedding validation set to memmap...")
        _, x_val_shape, _ = embed_to_memmap(
            val_df[args.text_col],
            tokenizer,
            model,
            x_val_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            dtype=dtype,
        )

        write_json(
            embedding_info_path,
            {
                "created_utc": utc_now_iso(),
                "x_train_path": str(x_train_path),
                "x_train_shape": list(x_train_shape),
                "x_val_path": str(x_val_path),
                "x_val_shape": list(x_val_shape),
                "dtype": dtype_str,
                "hidden_dim": int(hidden_dim),
                "batch_size": int(args.batch_size),
                "max_len": int(args.max_len),
                "model_name": MODEL_NAME,
                "transformer_source": str(transformer_source),
                "used_local_model_path": bool(used_local),
            },
        )

        if args.embed_only:
            print("Embed-only mode: done.")
            return

    # ---- TRAIN STEP ----
    # Load embedding info if needed
    if args.train_only:
        if not embedding_info_path.exists():
            raise FileNotFoundError(
                f"Missing {embedding_info_path}. Run once without --train-only or with --embed-only to create memmaps."
            )
        emb_info = json.loads(embedding_info_path.read_text(encoding="utf-8"))
        x_train_shape = tuple(emb_info["x_train_shape"])
        x_val_shape = tuple(emb_info["x_val_shape"])
        dtype = np.float32 if emb_info["dtype"] == "float32" else np.float16
        hidden_dim = int(emb_info["hidden_dim"])
        transformer_source = emb_info.get("transformer_source", MODEL_NAME)
        used_local = bool(emb_info.get("used_local_model_path", False))
    else:
        emb_info = json.loads(embedding_info_path.read_text(encoding="utf-8"))
        x_train_shape = tuple(emb_info["x_train_shape"])
        x_val_shape = tuple(emb_info["x_val_shape"])

    if not x_train_path.exists() or not x_val_path.exists():
        raise FileNotFoundError("Memmap files not found. Run embedding step first (no --train-only).")

    X_train = open_memmap(x_train_path, x_train_shape, dtype, mode="r")
    X_val = open_memmap(x_val_path, x_val_shape, dtype, mode="r")

    y_train = train_df[args.target_col].to_numpy()
    y_val = val_df[args.target_col].to_numpy()

    # ---- ALPHA SWEEP (optional) ----
    sweep_out = None
    if args.alpha_sweep:
        alpha_grid = []
        for s in args.alpha_grid.split(","):
            s = s.strip()
            if s:
                alpha_grid.append(float(s))

        sweep_out = run_alpha_sweep(
            model_dir=model_dir,
            alpha_grid=alpha_grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        # Use best alpha for final model fit + saving
        args.alpha = float(sweep_out["best"]["alpha"])
        print(f"\nUsing best alpha={args.alpha:g} for final training...")

    # ---- FINAL TRAIN (single alpha) ----
    print("Training final Ridge...")
    reg = Ridge(alpha=args.alpha, random_state=RANDOM_STATE)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_val)
    metrics = compute_metrics(y_train, y_val, preds)

    print(f"Validation MAE: {metrics['mae']:.6f}")
    print(f"Baseline MAE (predict 0):    {metrics['mae_zero']:.6f}")
    print(f"Baseline MAE (predict mean): {metrics['mae_mean']:.6f}")
    if metrics["directional_acc_nonzero"] is not None:
        print(
            f"Directional Accuracy (non-zero y): {metrics['directional_acc_nonzero']:.3%} "
            f"(n={metrics['n_val_nonzero']})"
        )
    else:
        print("Directional Accuracy: skipped (all y_val are zero).")

    # Save ridge model
    joblib.dump(reg, model_dir / "regressor.joblib")

    # Save tokenizer/model only if available (we embedded this run)
    if tokenizer is not None and model is not None:
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

    # Write metrics JSON (include sweep summary if present)
    metrics_out = {
        "created_utc": utc_now_iso(),
        **metrics,
        "ridge_alpha": float(args.alpha),
        "ridge_random_state": int(RANDOM_STATE),
        "alpha_sweep_used": bool(args.alpha_sweep),
        "alpha_sweep_best": sweep_out["best"] if sweep_out is not None else None,
    }
    write_json(model_dir / "metrics.json", metrics_out)

    # Write metadata JSON
    dataset_hash = sha256_file(dataset_path) if dataset_path.exists() else None
    metadata = ModelMetadata(
        created_utc=utc_now_iso(),
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_hash,
        dataset_rows=int(len(df)),
        split_idx=int(split_idx),
        train_rows=int(len(train_df)),
        val_rows=int(len(val_df)),
        text_column=args.text_col,
        target_column=args.target_col,
        model_name=MODEL_NAME,
        transformer_source=str(transformer_source) if transformer_source else MODEL_NAME,
        used_local_model_path=bool(used_local),
        max_len=int(args.max_len),
        batch_size=int(args.batch_size),
        dtype="float32" if dtype == np.float32 else "float16",
        hidden_dim=int(hidden_dim) if hidden_dim is not None else int(x_train_shape[1]),
        x_train_path=str(x_train_path),
        x_train_shape=tuple(int(x) for x in x_train_shape),
        x_val_path=str(x_val_path),
        x_val_shape=tuple(int(x) for x in x_val_shape),
        ridge_alpha=float(args.alpha),
        ridge_random_state=int(RANDOM_STATE),
        github_run_id=os.getenv("GITHUB_RUN_ID"),
        github_run_number=os.getenv("GITHUB_RUN_NUMBER"),
        github_sha=os.getenv("GITHUB_SHA"),
        github_ref=os.getenv("GITHUB_REF"),
    )
    write_json(model_dir / "model_metadata.json", asdict(metadata))

    print(f"\nSaved artifacts to: {model_dir}")
    print("Files:")
    print(f"  - {x_train_path.name}, {x_val_path.name}")
    print("  - regressor.joblib")
    print("  - metrics.json")
    print("  - model_metadata.json")
    print("  - split_info.json")
    print("  - embedding_info.json")
    if args.alpha_sweep:
        print("  - alpha_sweep.json")


if __name__ == "__main__":
    main()
