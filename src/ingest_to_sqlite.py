import sqlite3
import pandas as pd
from src.config import PATHS

# Columns expected in CSV (adjust these if your CSV differs)
DATE_COL = "date"
HEADLINE_COL = "headline"
CLOSE_COL = "close"

def main():
    csv_path = PATHS.raw_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {csv_path}. Put your Kaggle CSV at data/raw/sp500_headlines.csv "
            "or set RAW_CSV env var."
        )

    df = pd.read_csv(csv_path)
    # Normalize columns
    for col in (DATE_COL, HEADLINE_COL, CLOSE_COL):
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column '{col}'. Found columns: {list(df.columns)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, HEADLINE_COL, CLOSE_COL]).copy()
    df[HEADLINE_COL] = df[HEADLINE_COL].astype(str)

    PATHS.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PATHS.sqlite_db) as con:
        df.to_sql("sp500_news", con, if_exists="replace", index=False)

    print(f"Ingested {len(df):,} rows into SQLite: {PATHS.sqlite_db} (table sp500_news)")

if __name__ == "__main__":
    main()
