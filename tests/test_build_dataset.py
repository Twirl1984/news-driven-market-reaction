import pandas as pd
from src.build_dataset import build

def test_build_dataset_creates_target():
    df = pd.DataFrame({
        "date": ["2024-01-01","2024-01-02","2024-01-03"],
        "headline": ["a","b","c"],
        "close": [100.0, 101.0, 99.0],
    })
    out = build(df)
    assert "log_return_next_day" in out.columns
    assert len(out) == 2  # last row drops due to shift(-1)
