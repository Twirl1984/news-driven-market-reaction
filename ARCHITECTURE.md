# Architecture Documentation

## System Overview

Market Event AI is a complete end-to-end machine learning trading system that analyzes political events and generates trading signals for ETFs.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────┤
│  Trump Tweets  │  GDELT Events  │  Financial Data (yfinance)    │
│  (Archived)    │  (Political)   │  (ETF Prices & Volume)        │
└────────┬────────────────┬───────────────────────┬────────────────┘
         │                │                       │
         ▼                ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                              │
├─────────────────────────────────────────────────────────────────┤
│  • Download raw data                                             │
│  • Store in data/raw/                                            │
│  • Schemas: Event, FinancialData                                 │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                                │
├─────────────────────────────────────────────────────────────────┤
│  Events:                  │  Financial:                          │
│  • Text cleaning          │  • Calculate returns                 │
│  • Remove URLs/mentions   │  • Calculate volatility              │
│  • Sentiment extraction   │  • Technical indicators (SMA)        │
│  • Date alignment         │  • Missing data handling             │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
├─────────────────────────────────────────────────────────────────┤
│  • Aggregate events by day                                       │
│  • Event counts, sentiment stats                                 │
│  • Decay features (EMA)                                          │
│  • Sentiment momentum                                            │
│  • Merge with financial data                                     │
│  • Interaction features                                          │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LABEL GENERATION                              │
├─────────────────────────────────────────────────────────────────┤
│  • Calculate future returns (forward-looking)                    │
│  • Classification: LONG (>threshold), FLAT (<=threshold)         │
│  • Regression: Continuous returns                                │
│  • Data leakage checks                                           │
│  • Temporal validation                                           │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                │
├─────────────────────────────────────────────────────────────────┤
│  Models:                  │  Validation:                         │
│  • Logistic Regression    │  • Walk-forward CV                   │
│  • Random Forest          │  • Time-series splits                │
│  • XGBoost (default)      │  • Embargo period                    │
│  • LightGBM               │  • No lookahead bias                 │
│                           │                                      │
│  Metrics: Accuracy, F1, Precision, Recall                        │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKTESTING                                 │
├─────────────────────────────────────────────────────────────────┤
│  Portfolio Simulation:                                           │
│  • Initial capital: $100,000                                     │
│  • Transaction costs (10 bps)                                    │
│  • Slippage (5 bps)                                              │
│  • Position sizing (100% in LONG, 0% in FLAT)                    │
│                                                                  │
│  Trading Logic:                                                  │
│  • LONG signal → Buy/Hold position                               │
│  • FLAT signal → Close position                                  │
│  • Trade execution at close prices                               │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PERFORMANCE EVALUATION                          │
├─────────────────────────────────────────────────────────────────┤
│  Metrics:                                                        │
│  • CAGR (Compound Annual Growth Rate)                            │
│  • Sharpe Ratio (risk-adjusted returns)                          │
│  • Maximum Drawdown                                              │
│  • Hit Rate                                                      │
│  • Turnover                                                      │
│  • Benchmark comparison (Buy & Hold)                             │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      REPORTING                                   │
├─────────────────────────────────────────────────────────────────┤
│  Outputs:                                                        │
│  • Markdown summary report                                       │
│  • Equity curve plot                                             │
│  • Drawdown analysis                                             │
│  • Trade distribution charts                                     │
│  • CSV files (trades, equity)                                    │
│  • JSON metrics                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Ingestion Phase
- **Input**: Raw data sources (APIs, files)
- **Processing**: Download and store
- **Output**: `data/raw/`
  - `trump_tweets.json`
  - `gdelt_events.csv`
  - `financial_data.parquet`

### 2. Preprocessing Phase
- **Input**: `data/raw/`
- **Processing**: Clean, normalize, extract sentiment
- **Output**: `data/processed/`
  - `tweets_processed.csv`
  - `gdelt_processed.csv`
  - `financial_processed.parquet`

### 3. Feature Engineering Phase
- **Input**: `data/processed/`
- **Processing**: Aggregate events, create features, align with market data
- **Output**: `data/features/`
  - `features.parquet` (combined dataset)

### 4. Label Generation Phase
- **Input**: `data/features/`
- **Processing**: Calculate future returns, create classification labels
- **Output**: `data/labels/`
  - `labeled_data_classification.parquet`
  - `labeled_data_regression.parquet`

### 5. Model Training Phase
- **Input**: `data/labels/`
- **Processing**: Train models with walk-forward CV
- **Output**: `data/models/{model_type}/`
  - `model.joblib`
  - `metadata.json`
  - `evaluation.json`

### 6. Backtesting Phase
- **Input**: `data/models/`, `data/labels/`
- **Processing**: Simulate trading, calculate metrics
- **Output**: `data/backtests/{model_type}/`
  - `equity_curve.csv`
  - `trades.csv`
  - `metrics.json`

### 7. Reporting Phase
- **Input**: `data/backtests/`
- **Processing**: Generate visualizations and reports
- **Output**: `data/reports/{model_type}/`
  - `summary.md`
  - `equity_curve.png`
  - `trades_analysis.png`

## Key Design Principles

### 1. No Lookahead Bias
- Features use only past data (t-1, t-2, ...)
- Labels use only future data (t+1, t+2, ...)
- Walk-forward validation with embargo period
- Strict temporal ordering

### 2. Reproducibility
- Fixed random seeds (42)
- Deterministic data processing
- Pinned dependency versions
- Version-controlled configuration

### 3. Modularity
- Clear separation of concerns
- Each module has single responsibility
- Easy to extend with new data sources
- Easy to add new models

### 4. Realistic Simulation
- Transaction costs modeled (10 bps default)
- Slippage modeled (5 bps default)
- No fractional shares
- Position limits enforced

### 5. Professional Standards
- Type hints throughout
- Comprehensive logging
- Error handling
- Unit and integration tests
- Documentation

## Technology Stack

### Core Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Classical ML models
- **xgboost**: Gradient boosting
- **lightgbm**: Gradient boosting
- **yfinance**: Financial data
- **textblob**: Sentiment analysis

### Infrastructure
- **Click**: CLI framework
- **Python-dotenv**: Configuration
- **PyYAML**: Configuration files
- **pytest**: Testing
- **matplotlib/seaborn**: Visualization

## Configuration

### Environment Variables (`.env`)
- Data directories
- API keys (optional)
- Model parameters
- Trading parameters
- Backtesting parameters

### Assets Configuration (`config/assets.yaml`)
- ETF/stock tickers
- Asset classes
- Trading rules per asset class
- Enable/disable assets

## CLI Commands

```bash
market-event-ai info         # Show configuration
market-event-ai download     # Download data
market-event-ai preprocess   # Preprocess data
market-event-ai features     # Extract features
market-event-ai label        # Generate labels
market-event-ai train        # Train models
market-event-ai evaluate     # Evaluate models
market-event-ai backtest     # Run backtests
market-event-ai report       # Generate reports
```

## Extension Points

### Adding New Data Sources
1. Create downloader class in `data/downloaders.py`
2. Implement `download()` method
3. Add to CLI command
4. Update preprocessing

### Adding New Models
1. Create trainer class in `models/trainers.py`
2. Implement `create_model()` and `train()` methods
3. Add to CLI choices
4. Update evaluation

### Adding New Features
1. Extend `FeatureEngineer` in `features/extractors.py`
2. Add feature generation methods
3. Update feature list
4. Test for leakage

### Adding New Metrics
1. Extend `Backtester` in `portfolio/backtesters.py`
2. Implement metric calculation
3. Add to results output
4. Update reports

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Test individual components
- Mock external dependencies
- Fast execution
- High coverage

### Integration Tests (`tests/integration/`)
- Test component interactions
- Use temporary directories
- End-to-end workflows
- Validate outputs

## Performance Considerations

### Memory
- Use parquet for large datasets
- Process in chunks if needed
- Clear intermediate results

### Speed
- Vectorized operations (pandas/numpy)
- Parallel processing where applicable
- Efficient data structures
- Caching when appropriate

### Scalability
- Modular design allows scaling
- Can process multiple assets in parallel
- Can extend to larger datasets
- Can add more complex models
