# Market Event AI CLI Documentation

## Overview

The Market Event AI CLI provides a comprehensive command-line interface for managing the entire machine learning trading pipeline, from data acquisition to backtesting and performance reporting.

## Installation

```bash
# Install the package in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

```bash
# Display configuration
market-event-ai info

# Download data
market-event-ai download --source all

# Preprocess data
market-event-ai preprocess

# Generate features
market-event-ai features

# Generate labels
market-event-ai label

# Train a model
market-event-ai train --model-type xgboost

# Run backtest
market-event-ai backtest --model-path models/xgboost_return_1d.joblib
```

## Commands

### Global Options

```
-c, --config PATH    Path to configuration file (YAML)
-v, --verbose        Enable verbose logging (DEBUG level)
--help               Show this message and exit
```

### info

Display system configuration and status.

```bash
market-event-ai info
```

Shows:
- Data paths configuration
- Model configuration
- Trading/backtesting parameters
- Logging level

### download

Download raw data from various sources (Trump tweets, GDELT events, financial data).

```bash
# Download all sources
market-event-ai download

# Download specific source
market-event-ai download --source trump_tweets
market-event-ai download --source gdelt
market-event-ai download --source financial

# Download with date range
market-event-ai download --start-date 2016-01-01 --end-date 2020-12-31

# Force re-download
market-event-ai download --force
```

**Options:**
- `-s, --source [trump_tweets|gdelt|financial|all]` - Data source to download (default: all)
- `--start-date YYYY-MM-DD` - Start date for data download
- `--end-date YYYY-MM-DD` - End date for data download
- `-f, --force` - Force re-download even if data exists

### preprocess

Preprocess raw data for feature engineering.

```bash
# Preprocess with default settings
market-event-ai preprocess

# Skip text cleaning
market-event-ai preprocess --no-clean-text

# Keep duplicates
market-event-ai preprocess --keep-duplicates

# Custom input/output directories
market-event-ai preprocess --input-dir data/raw --output-dir data/processed
```

**Options:**
- `--input-dir PATH` - Input directory with raw data
- `--output-dir PATH` - Output directory for preprocessed data
- `--clean-text/--no-clean-text` - Apply text cleaning (default: True)
- `--remove-duplicates/--keep-duplicates` - Remove duplicates (default: True)

### features

Generate features from preprocessed data.

```bash
# Generate all features
market-event-ai features

# Generate specific feature set
market-event-ai features --feature-set sentiment
market-event-ai features --feature-set embeddings

# Custom embedding model
market-event-ai features --embedding-model roberta-base --batch-size 64

# Custom directories
market-event-ai features --input-dir data/processed --output-dir data/features
```

**Options:**
- `--input-dir PATH` - Input directory with preprocessed data
- `--output-dir PATH` - Output directory for features
- `-f, --feature-set [text|sentiment|embeddings|technical|all]` - Feature set (default: all)
- `--embedding-model TEXT` - Model for embeddings (default: distilbert-base-uncased)
- `--batch-size INTEGER` - Batch size for embedding generation (default: 32)

### label

Generate labels/targets for model training.

```bash
# Generate all label types
market-event-ai label

# Specific label type
market-event-ai label --label-type returns
market-event-ai label --label-type direction

# Multiple horizons
market-event-ai label --horizon 1 --horizon 5 --horizon 20

# Direction labels with threshold
market-event-ai label --label-type direction --min-threshold 0.02
```

**Options:**
- `--input-dir PATH` - Input directory with features
- `--output-dir PATH` - Output directory for labels
- `-h, --horizon INTEGER` - Prediction horizon in days (can specify multiple, default: 1, 5, 20)
- `--label-type [returns|direction|volatility|all]` - Type of labels (default: all)
- `--min-threshold FLOAT` - Minimum threshold for binary labels

### train

Train machine learning models.

```bash
# Train with default model
market-event-ai train

# Train specific model type
market-event-ai train --model-type xgboost
market-event-ai train --model-type lightgbm
market-event-ai train --model-type neural_net

# With hyperparameter optimization
market-event-ai train --model-type xgboost --hyperopt --n-trials 100

# Custom target and cross-validation
market-event-ai train --target return_5d --cv-folds 10
```

**Options:**
- `-m, --model-type [xgboost|lightgbm|random_forest|neural_net|transformer]` - Model type
- `--input-dir PATH` - Input directory with features and labels
- `--output-dir PATH` - Output directory for trained models
- `-t, --target TEXT` - Target variable to predict (default: return_1d)
- `--cv-folds INTEGER` - Number of CV folds (default: 5)
- `--hyperopt/--no-hyperopt` - Enable hyperparameter optimization (default: False)
- `--n-trials INTEGER` - Number of optimization trials (default: 100)

### evaluate

Evaluate trained models on test data.

```bash
# Evaluate a model
market-event-ai evaluate --model-path models/xgboost_return_1d.joblib

# Evaluate with specific test data
market-event-ai evaluate --model-path models/model.joblib --test-data data/test.parquet

# Specific metrics
market-event-ai evaluate --model-path models/model.joblib --metrics classification

# Custom output directory
market-event-ai evaluate --model-path models/model.joblib --output-dir reports/eval
```

**Options:**
- `--model-path PATH` - Path to trained model (required)
- `--test-data PATH` - Path to test data (optional)
- `-m, --metrics [all|regression|classification]` - Metrics to compute (default: all)
- `--output-dir PATH` - Output directory for results

### backtest

Run backtesting on trained models.

```bash
# Basic backtest
market-event-ai backtest --model-path models/xgboost_return_1d.joblib

# Custom date range
market-event-ai backtest --model-path models/model.joblib \
  --start-date 2018-01-01 --end-date 2020-12-31

# Different strategy
market-event-ai backtest --model-path models/model.joblib --strategy long_only

# Custom capital and rebalancing
market-event-ai backtest --model-path models/model.joblib \
  --initial-capital 1000000 --rebalance-freq weekly

# Disable walk-forward analysis
market-event-ai backtest --model-path models/model.joblib --no-walk-forward
```

**Options:**
- `--model-path PATH` - Path to trained model (required)
- `--start-date YYYY-MM-DD` - Backtest start date
- `--end-date YYYY-MM-DD` - Backtest end date
- `--initial-capital FLOAT` - Initial capital (default: 100000)
- `--strategy [long_short|long_only|market_neutral]` - Trading strategy (default: long_short)
- `--rebalance-freq [daily|weekly|monthly]` - Rebalancing frequency (default: daily)
- `--walk-forward/--no-walk-forward` - Walk-forward analysis (default: True)
- `--output-dir PATH` - Output directory for results

### report

Generate comprehensive reports.

```bash
# Generate full report
market-event-ai report

# Specific report type
market-event-ai report --report-type backtest
market-event-ai report --report-type evaluation

# Different format
market-event-ai report --format pdf
market-event-ai report --format markdown

# Custom input and output
market-event-ai report --input-dir data/backtests --output reports/analysis.html

# Without plots
market-event-ai report --no-plots
```

**Options:**
- `--input-dir PATH` - Input directory with results
- `-t, --report-type [evaluation|backtest|full]` - Report type (default: full)
- `-f, --format [html|pdf|markdown]` - Output format (default: html)
- `-o, --output PATH` - Output path for report
- `--include-plots/--no-plots` - Include visualizations (default: True)

## Configuration File

You can use a YAML configuration file to override default settings:

```yaml
# config.yaml
data:
  random_seed: 42
  
model:
  model_type: xgboost
  embedding_model: bert-base-uncased
  
trading:
  initial_capital: 500000
  trading_cost_bps: 10
  slippage_bps: 5
  signal_threshold: 0.02
  backtest_start_date: "2016-01-01"
  backtest_end_date: "2020-12-31"

paths:
  data_raw: data/raw
  data_processed: data/processed
  models: models
```

Use it with:
```bash
market-event-ai --config config.yaml download
```

## Environment Variables

All configuration can be overridden via environment variables:

```bash
# Data paths
export DATA_RAW_DIR=data/raw
export DATA_PROCESSED_DIR=data/processed
export DATA_FEATURES_DIR=data/features
export DATA_LABELS_DIR=data/labels
export MODELS_DIR=data/models
export BACKTESTS_DIR=data/backtests
export REPORTS_DIR=data/reports

# Model configuration
export MODEL_TYPE=xgboost
export EMBEDDING_MODEL=distilbert-base-uncased
export RANDOM_SEED=42

# Trading configuration
export INITIAL_CAPITAL=100000
export TRADING_COST_BPS=10
export SLIPPAGE_BPS=5
export SIGNAL_THRESHOLD=0.02

# Backtesting
export BACKTEST_START_DATE=2016-01-01
export BACKTEST_END_DATE=2020-12-31
export WALK_FORWARD_WINDOW_DAYS=252

# Logging
export LOG_LEVEL=INFO
```

## Complete Workflow Example

```bash
# 1. Check configuration
market-event-ai info

# 2. Download all data
market-event-ai download --source all --start-date 2015-01-01 --end-date 2020-12-31

# 3. Preprocess
market-event-ai preprocess

# 4. Generate features
market-event-ai features --feature-set all

# 5. Generate labels
market-event-ai label --horizon 1 --horizon 5 --horizon 20

# 6. Train model with hyperparameter optimization
market-event-ai train --model-type xgboost --hyperopt --n-trials 50

# 7. Evaluate model
market-event-ai evaluate --model-path models/xgboost_return_1d.joblib

# 8. Run backtest
market-event-ai backtest --model-path models/xgboost_return_1d.joblib

# 9. Generate report
market-event-ai report --report-type full --format html
```

## Error Handling

The CLI provides clear error messages and uses exit codes:
- `0` - Success
- `1` - Error (with details logged)

Enable verbose logging for debugging:
```bash
market-event-ai --verbose download
```

## Tips

1. **Use `--force` carefully**: Re-downloading or reprocessing large datasets can be time-consuming
2. **Start small**: Test with a small date range before processing full datasets
3. **Monitor resources**: Some operations (embeddings, training) can be memory-intensive
4. **Save configurations**: Use config files to ensure reproducible experiments
5. **Check logs**: All operations are logged for auditing and debugging

## Getting Help

For command-specific help:
```bash
market-event-ai <command> --help
```

For general help:
```bash
market-event-ai --help
```
