"""Configuration settings for Market Event AI."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Paths:
    """Data paths configuration."""
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    
    # Data directories
    data_raw: Path = field(default_factory=lambda: Path(os.getenv("DATA_RAW_DIR", "data/raw")))
    data_processed: Path = field(default_factory=lambda: Path(os.getenv("DATA_PROCESSED_DIR", "data/processed")))
    data_features: Path = field(default_factory=lambda: Path(os.getenv("DATA_FEATURES_DIR", "data/features")))
    data_labels: Path = field(default_factory=lambda: Path(os.getenv("DATA_LABELS_DIR", "data/labels")))
    models: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "data/models")))
    backtests: Path = field(default_factory=lambda: Path(os.getenv("BACKTESTS_DIR", "data/backtests")))
    reports: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", "data/reports")))
    
    # Database
    sqlite_db: Path = field(default_factory=lambda: Path(os.getenv("SQLITE_DB", "db/market_events.db")))
    
    # Config
    assets_config: Path = field(default_factory=lambda: Path("config/assets.yaml"))
    
    def create_dirs(self):
        """Create all necessary directories."""
        for path in [self.data_raw, self.data_processed, self.data_features, 
                     self.data_labels, self.models, self.backtests, self.reports]:
            path.mkdir(parents=True, exist_ok=True)
        self.sqlite_db.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data source configuration."""
    trump_tweets_url: str = field(default_factory=lambda: os.getenv("TRUMP_TWEETS_URL", ""))
    gdelt_base_url: str = field(default_factory=lambda: os.getenv("GDELT_BASE_URL", "http://data.gdeltproject.org/gdeltv2"))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "xgboost"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "distilbert-base-uncased"))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))


@dataclass
class TradingConfig:
    """Trading and backtesting configuration."""
    trading_cost_bps: float = field(default_factory=lambda: float(os.getenv("TRADING_COST_BPS", "10")))
    slippage_bps: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", "5")))
    initial_capital: float = field(default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "100000")))
    signal_threshold: float = field(default_factory=lambda: float(os.getenv("SIGNAL_THRESHOLD", "0.02")))
    
    # Backtesting
    backtest_start_date: str = field(default_factory=lambda: os.getenv("BACKTEST_START_DATE", "2016-01-01"))
    backtest_end_date: str = field(default_factory=lambda: os.getenv("BACKTEST_END_DATE", "2020-12-31"))
    walk_forward_window_days: int = field(default_factory=lambda: int(os.getenv("WALK_FORWARD_WINDOW_DAYS", "252")))
    embargo_days: int = field(default_factory=lambda: int(os.getenv("EMBARGO_DAYS", "5")))


@dataclass
class Settings:
    """Main application settings."""
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        """Initialize settings."""
        self.paths.create_dirs()
    
    def load_assets_config(self):
        """Load assets configuration from YAML."""
        if self.paths.assets_config.exists():
            with open(self.paths.assets_config, 'r') as f:
                return yaml.safe_load(f)
        return {"etfs": [], "stocks": [], "trading_rules": {}}


# Global settings instance
settings = Settings()
