from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Paths:
    raw_csv: Path = Path(os.getenv("RAW_CSV", "data/raw/sp500_headlines.csv"))
    sqlite_db: Path = Path(os.getenv("SQLITE_DB", "db/news.db"))
    processed_parquet: Path = Path(os.getenv("PROCESSED_PARQUET", "data/processed/dataset.parquet"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "models/transformer_reg"))

PATHS = Paths()
