import numpy as np
import pandas as pd
from src.config import PATHS

DATE_COL = "Date"
HEADLINE_COL = "Title"
CLOSE_COL = "CP"

def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, HEADLINE_COL, CLOSE_COL])
    df = df.sort_values(DATE_COL)

    df["log_close"] = np.log(df[CLOSE_COL].astype(float))
    df["log_return_next_day"] = df["log_close"].shift(-1) - df["log_close"]

    # Keep modeling columns
    out = df[[DATE_COL, HEADLINE_COL, CLOSE_COL, "log_return_next_day"]].dropna()
    out[HEADLINE_COL] = out[HEADLINE_COL].astype(str)
    return out

def main():
    if not PATHS.raw_csv.exists():
        raise FileNotFoundError(f"Missing raw CSV at {PATHS.raw_csv}")

    df = pd.read_csv(PATHS.raw_csv)
    out = build(df)

    PATHS.processed_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PATHS.processed_parquet, index=False)
    print(f"Wrote processed dataset: {PATHS.processed_parquet} rows={len(out):,}")

if __name__ == "__main__":
    main()
