# Quick Start Guide

Get Market Event AI up and running in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- pip or conda
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Twirl1984/news-driven-market-reaction.git
cd news-driven-market-reaction
git checkout Trump  # Use the Trump branch
```

### 2. Set Up Environment

#### Option A: Using venv (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

#### Option B: Using conda

```bash
conda create -n market-event-ai python=3.9
conda activate market-event-ai
pip install -e .
```

#### Option C: Using Make

```bash
make setup
source .venv/bin/activate
```

### 3. Configure (Optional)

```bash
# Copy environment template
cp .env.example .env

# Edit if needed (defaults work fine)
nano .env

# Review assets configuration
cat config/assets.yaml
```

## Running Your First Backtest

### Quick Pipeline (5 minutes)

Run the complete pipeline with a short time period:

```bash
# Download data (2016-2017 only)
market-event-ai download --source all --start-date 2016-01-01 --end-date 2016-12-31

# Preprocess
market-event-ai preprocess

# Extract features
market-event-ai features

# Generate labels
market-event-ai label --task classification

# Train model
market-event-ai train --model-type xgboost

# Evaluate
market-event-ai evaluate

# Backtest
market-event-ai backtest

# Generate report
market-event-ai report
```

### Alternative: Use Make

```bash
# Run complete pipeline
make pipeline
```

### Faster Test (2 minutes)

If you want to test even faster with sample data only:

```bash
# Generate sample data (no internet required)
market-event-ai download --source trump_tweets
market-event-ai download --source gdelt --start-date 2016-01-01 --end-date 2016-03-31

# Note: Financial data requires internet connection to yfinance
# For testing without internet, skip financial download and use sample data
```

## Viewing Results

### Check System Status

```bash
market-event-ai info
```

Expected output:
```
=== Market Event AI Configuration ===

Paths:
  Raw Data:       data/raw
  Processed Data: data/processed
  ...

Model Configuration:
  Model Type:      xgboost
  ...
```

### View Backtest Results

```bash
# View summary report
cat data/reports/xgboost/summary.md

# Check metrics
cat data/backtests/xgboost/metrics.json

# View equity curve (if image viewer available)
open data/reports/xgboost/equity_curve.png  # macOS
xdg-open data/reports/xgboost/equity_curve.png  # Linux
```

### Explore Data

```bash
# Check downloaded data
ls -lh data/raw/

# Check processed data
ls -lh data/processed/

# Check features
ls -lh data/features/

# Check trained models
ls -lh data/models/xgboost/
```

## Testing

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest --cov=market_event_ai --cov-report=html
```

### Expected Test Output

```
tests/unit/test_config.py::test_settings_initialization PASSED
tests/unit/test_config.py::test_paths_creation PASSED
tests/unit/test_config.py::test_load_assets_config PASSED
tests/unit/test_downloaders.py::test_trump_tweets_downloader PASSED
tests/unit/test_downloaders.py::test_gdelt_downloader PASSED
tests/unit/test_preprocessors.py::test_text_cleaner PASSED
tests/unit/test_preprocessors.py::test_sentiment_extraction PASSED

===== 7 passed in 1.24s =====
```

## Common Commands

### Help

```bash
# Main help
market-event-ai --help

# Command-specific help
market-event-ai download --help
market-event-ai train --help
market-event-ai backtest --help
```

### Different Models

```bash
# Train different models
market-event-ai train --model-type logistic
market-event-ai train --model-type random_forest
market-event-ai train --model-type lightgbm

# Evaluate each
market-event-ai evaluate --model-type logistic
market-event-ai backtest --model-type logistic
market-event-ai report --model-type logistic
```

### Different Time Periods

```bash
# 2016-2020 (full period)
market-event-ai download --source all --start-date 2016-01-01 --end-date 2020-12-31

# 2024-present
market-event-ai download --source all --start-date 2024-01-01 --end-date 2024-12-31
```

### Different Label Configurations

```bash
# Higher threshold (more conservative)
market-event-ai label --task classification --threshold 0.05

# Multiple horizons
market-event-ai label --horizon 1 --horizon 5 --horizon 20
```

## Directory Structure

After running the pipeline, you'll have:

```
news-driven-market-reaction/
├── data/
│   ├── raw/               # Downloaded raw data
│   │   ├── trump_tweets.json
│   │   ├── gdelt_events.csv
│   │   └── financial_data.parquet
│   ├── processed/         # Cleaned data
│   │   ├── tweets_processed.csv
│   │   ├── gdelt_processed.csv
│   │   └── financial_processed.parquet
│   ├── features/          # Engineered features
│   │   └── features.parquet
│   ├── labels/            # Training labels
│   │   └── labeled_data_classification.parquet
│   ├── models/            # Trained models
│   │   └── xgboost/
│   │       ├── model.joblib
│   │       ├── metadata.json
│   │       └── evaluation.json
│   ├── backtests/         # Backtest results
│   │   └── xgboost/
│   │       ├── equity_curve.csv
│   │       ├── trades.csv
│   │       └── metrics.json
│   └── reports/           # Final reports
│       └── xgboost/
│           ├── summary.md
│           ├── equity_curve.png
│           └── trades_analysis.png
```

## Troubleshooting

### Import Errors

```bash
# Reinstall in editable mode
pip install -e .
```

### Missing Data

```bash
# Re-download
market-event-ai download --source all --force
```

### Tests Failing

```bash
# Check Python version
python --version  # Should be 3.9+

# Check installation
pip list | grep market-event-ai

# Reinstall dependencies
pip install -e . --force-reinstall
```

### Network Issues (yfinance)

If you can't download financial data due to network restrictions:

```bash
# The system uses sample data for events
# For financial data, you may need to:
# 1. Use a different network
# 2. Configure proxy settings
# 3. Use pre-downloaded data
```

## Next Steps

### 1. Experiment with Different Configurations

Edit `.env` to change:
- Trading costs and slippage
- Signal thresholds
- Walk-forward window size
- Model hyperparameters

### 2. Add More Assets

Edit `config/assets.yaml` to:
- Enable more ETFs
- Add individual stocks
- Configure trading rules per asset class

### 3. Extend the System

See `ARCHITECTURE.md` for:
- Adding new data sources
- Implementing new models
- Creating custom features
- Building new metrics

### 4. Read the Documentation

- `README.md` - Full documentation
- `ARCHITECTURE.md` - System design
- `LICENSE` - MIT License terms

## Getting Help

### Check Logs

The system logs to console. Use `--verbose` for more detail:

```bash
market-event-ai --verbose download --source all
```

### Run Info Command

```bash
market-event-ai info
```

### Check GitHub Issues

Visit: https://github.com/Twirl1984/news-driven-market-reaction/issues

## What's Next?

Congratulations! You've successfully:
- ✅ Installed Market Event AI
- ✅ Run your first backtest
- ✅ Generated trading signals
- ✅ Evaluated performance

Now you can:
- Experiment with different time periods
- Try different models
- Adjust trading parameters
- Add your own features
- Extend the system

Happy Trading! 📈
