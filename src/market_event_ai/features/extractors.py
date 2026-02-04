"""Feature extraction modules."""
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class EventAggregator:
    """Aggregate events per day for alignment with market data."""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    
    def aggregate_daily_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate events by date."""
        logger.info(f"Aggregating {len(events_df)} events by date...")
        
        # Ensure date column
        events_df['date'] = pd.to_datetime(events_df['timestamp_utc']).dt.date
        
        # Group by date
        daily = events_df.groupby('date').agg({
            'event_id': 'count',
            'polarity': ['mean', 'std', 'min', 'max'],
            'subjectivity': ['mean', 'std'],
            'text_clean': lambda x: ' '.join(x)
        }).reset_index()
        
        # Flatten column names
        daily.columns = ['date', 'event_count', 
                        'polarity_mean', 'polarity_std', 'polarity_min', 'polarity_max',
                        'subjectivity_mean', 'subjectivity_std', 'text_combined']
        
        # Fill NaN in std columns
        daily['polarity_std'] = daily['polarity_std'].fillna(0)
        daily['subjectivity_std'] = daily['subjectivity_std'].fillna(0)
        
        logger.info(f"Aggregated to {len(daily)} daily records")
        return daily
    
    def create_decay_features(self, daily_events: pd.DataFrame, decay_days: int = 5) -> pd.DataFrame:
        """Create decay features for event impact."""
        logger.info("Creating decay features...")
        
        daily_events = daily_events.sort_values('date').reset_index(drop=True)
        
        # Create exponential decay for event count
        for days in [1, 3, 5]:
            alpha = 2 / (days + 1)  # EMA alpha
            daily_events[f'event_count_decay_{days}d'] = (
                daily_events['event_count']
                .ewm(alpha=alpha, adjust=False)
                .mean()
                .shift(1)  # Shift to avoid lookahead
            )
        
        # Create sentiment momentum
        daily_events['polarity_momentum_3d'] = (
            daily_events['polarity_mean']
            .rolling(3, min_periods=1)
            .mean()
            .shift(1)
        )
        
        return daily_events


class FeatureEngineer:
    """Extract features for model training."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aggregator = EventAggregator()
    
    def create_features(self, events_file: Path, financial_file: Path) -> Path:
        """Create feature matrix by aligning events with market data."""
        logger.info("Creating feature matrix...")
        
        # Load data
        if events_file.suffix == '.csv':
            events = pd.read_csv(events_file)
        else:
            events = pd.read_parquet(events_file)
        
        financial = pd.read_parquet(financial_file)
        
        # Aggregate events by date
        daily_events = self.aggregator.aggregate_daily_events(events)
        
        # Create decay features
        daily_events = self.aggregator.create_decay_features(daily_events)
        
        # Ensure dates are proper type
        daily_events['date'] = pd.to_datetime(daily_events['date'])
        financial['date'] = pd.to_datetime(financial['date'])
        
        # Merge with financial data
        features = financial.merge(
            daily_events,
            on='date',
            how='left'
        )
        
        # Fill missing event features (days with no events)
        event_cols = [col for col in daily_events.columns if col != 'date']
        features[event_cols] = features[event_cols].fillna(0)
        
        # Add technical indicators (already in financial data)
        # Add interaction features
        features['event_volume_interaction'] = features['event_count'] * np.log1p(features['volume'])
        features['sentiment_return_interaction'] = features['polarity_mean'] * features['log_return'].shift(1)
        
        # Drop rows with NaN in critical features
        features = features.dropna(subset=['close', 'log_return'])
        
        # Save features
        output_file = self.output_dir / "features.parquet"
        features.to_parquet(output_file, index=False)
        
        logger.info(f"Created feature matrix with shape {features.shape}")
        logger.info(f"Saved to {output_file}")
        
        return output_file


def extract_features():
    """Main feature extraction pipeline."""
    logger.info("Starting feature extraction...")
    
    engineer = FeatureEngineer(settings.paths.data_features)
    
    # Use preprocessed data
    events_file = settings.paths.data_processed / "tweets_processed.csv"
    if not events_file.exists():
        events_file = settings.paths.data_processed / "gdelt_processed.csv"
    
    financial_file = settings.paths.data_processed / "financial_processed.parquet"
    
    if not events_file.exists() or not financial_file.exists():
        raise FileNotFoundError("Missing preprocessed data files")
    
    features_file = engineer.create_features(events_file, financial_file)
    
    logger.info("Feature extraction completed")
    return features_file


# Wrapper functions for CLI
def generate_text_features(input_dir: Path, output_dir: Path):
    """Generate text features (CLI wrapper)."""
    logger.info("Text features are generated as part of the main feature extraction")
    # This is included in the main feature extraction
    return None


def generate_sentiment_features(input_dir: Path, output_dir: Path):
    """Generate sentiment features (CLI wrapper)."""
    logger.info("Sentiment features are already computed during preprocessing")
    # Sentiment is computed during preprocessing
    return None


def generate_embeddings(input_dir: Path, output_dir: Path, model_name: str, batch_size: int):
    """Generate text embeddings (CLI wrapper)."""
    logger.info(f"Text embeddings using {model_name} (not implemented in MVP)")
    # Optional for MVP - can be added later
    return None


def generate_technical_indicators(input_dir: Path, output_dir: Path):
    """Generate technical indicators (CLI wrapper)."""
    logger.info("Technical indicators are generated during financial preprocessing")
    # Technical indicators are part of financial preprocessing
    return None


def combine_all_features(input_dir: Path, output_dir: Path):
    """Combine all features (CLI wrapper)."""
    logger.info("Starting combined feature extraction...")
    
    engineer = FeatureEngineer(output_dir)
    
    # Find preprocessed data
    events_file = input_dir / "tweets_processed.csv"
    if not events_file.exists():
        events_file = input_dir / "gdelt_processed.csv"
    
    financial_file = input_dir / "financial_processed.parquet"
    
    if not events_file.exists() or not financial_file.exists():
        raise FileNotFoundError(f"Missing preprocessed data files in {input_dir}")
    
    features_file = engineer.create_features(events_file, financial_file)
    
    logger.info("Feature combination completed")
    return features_file
