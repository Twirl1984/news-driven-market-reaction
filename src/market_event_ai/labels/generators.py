"""Label generation for supervised learning."""
import logging
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import numpy as np

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate labels for trading signals."""
    
    def __init__(self, threshold: float = 0.02, horizon: int = 1):
        """
        Initialize label generator.
        
        Args:
            threshold: Return threshold for classification (default: 2%)
            horizon: Forward-looking horizon in days (default: 1)
        """
        self.threshold = threshold
        self.horizon = horizon
    
    def create_classification_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create classification labels based on future returns.
        
        Labels:
            1 = LONG (future return > threshold)
            0 = FLAT (|future return| <= threshold)
            -1 = SHORT (future return < -threshold) [optional]
        """
        logger.info(f"Creating classification labels with threshold={self.threshold}, horizon={self.horizon}")
        
        df = df.copy()
        df = df.sort_values(['ticker', 'date'])
        
        # Calculate future returns
        df['future_return'] = df.groupby('ticker')['log_return'].shift(-self.horizon)
        
        # Create labels
        df['label'] = 0  # Default: FLAT
        df.loc[df['future_return'] > self.threshold, 'label'] = 1  # LONG
        df.loc[df['future_return'] < -self.threshold, 'label'] = -1  # SHORT (optional)
        
        # For this system, convert SHORT to FLAT (only LONG/FLAT strategy)
        df.loc[df['label'] == -1, 'label'] = 0
        
        # Create signal names
        df['signal'] = df['label'].map({1: 'LONG', 0: 'FLAT', -1: 'SHORT'})
        
        logger.info(f"Label distribution:\n{df['signal'].value_counts()}")
        
        return df
    
    def create_regression_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regression labels (future returns)."""
        logger.info(f"Creating regression labels with horizon={self.horizon}")
        
        df = df.copy()
        df = df.sort_values(['ticker', 'date'])
        
        # Calculate future returns
        df['target_return'] = df.groupby('ticker')['log_return'].shift(-self.horizon)
        
        return df
    
    def check_data_leakage(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Check for potential data leakage.
        
        Returns:
            Dict with leakage checks results
        """
        logger.info("Checking for data leakage...")
        
        checks = {}
        
        # Check 1: Future data in features
        feature_cols = [col for col in df.columns 
                       if col not in ['label', 'signal', 'future_return', 'target_return', 'date', 'ticker']]
        
        # Simple check: ensure features are computed on past data only
        # More sophisticated checks would involve examining shift operations
        checks['has_future_features'] = any('future' in col.lower() for col in feature_cols)
        
        # Check 2: NaN in labels
        if 'label' in df.columns:
            nan_labels = df['label'].isna().sum()
            checks['has_nan_labels'] = nan_labels > 0
            if nan_labels > 0:
                logger.warning(f"Found {nan_labels} NaN labels (expected at end due to horizon)")
        
        # Check 3: Ensure timestamps are sorted
        checks['is_sorted'] = df['date'].is_monotonic_increasing or \
                             df.groupby('ticker')['date'].apply(lambda x: x.is_monotonic_increasing).all()
        
        logger.info(f"Leakage checks: {checks}")
        
        return checks


def generate_labels(task: str = 'classification', threshold: float = None, horizon: int = 1):
    """
    Main label generation pipeline.
    
    Args:
        task: 'classification' or 'regression'
        threshold: Return threshold for classification
        horizon: Forward-looking horizon in days
    """
    logger.info(f"Starting label generation for {task} task...")
    
    # Load features
    features_file = settings.paths.data_features / "features.parquet"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    df = pd.read_parquet(features_file)
    
    # Use threshold from settings if not provided
    if threshold is None:
        threshold = settings.trading.signal_threshold
    
    # Generate labels
    label_gen = LabelGenerator(threshold=threshold, horizon=horizon)
    
    if task == 'classification':
        df = label_gen.create_classification_labels(df)
    elif task == 'regression':
        df = label_gen.create_regression_labels(df)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Check for leakage
    leakage_checks = label_gen.check_data_leakage(df)
    
    if leakage_checks.get('has_future_features', False):
        logger.error("CRITICAL: Future data detected in features!")
        raise ValueError("Data leakage detected: future features found")
    
    # Drop rows with NaN labels (at the end due to forward-looking horizon)
    if task == 'classification':
        df = df.dropna(subset=['label', 'future_return'])
    else:
        df = df.dropna(subset=['target_return'])
    
    # Save labeled data
    output_file = settings.paths.data_labels / f"labeled_data_{task}.parquet"
    df.to_parquet(output_file, index=False)
    
    logger.info(f"Generated {len(df)} labeled samples")
    logger.info(f"Saved to {output_file}")
    
    return output_file


# Wrapper functions for CLI
def generate_return_labels(input_dir: Path, output_dir: Path, horizons: list):
    """Generate return labels (CLI wrapper)."""
    logger.info(f"Generating return labels for horizons: {horizons}")
    
    # Load features
    features_file = input_dir / "features.parquet"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    df = pd.read_parquet(features_file)
    
    # Generate for each horizon
    for horizon in horizons:
        label_gen = LabelGenerator(threshold=0.0, horizon=horizon)
        df_labeled = label_gen.create_regression_labels(df.copy())
        
        # Save
        output_file = output_dir / f"return_labels_h{horizon}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_labeled.to_parquet(output_file, index=False)
        logger.info(f"Saved return labels for horizon {horizon} to {output_file}")
    
    return output_dir


def generate_direction_labels(input_dir: Path, output_dir: Path, horizons: list, threshold: float):
    """Generate direction labels (CLI wrapper)."""
    logger.info(f"Generating direction labels for horizons: {horizons}, threshold: {threshold}")
    
    # Load features
    features_file = input_dir / "features.parquet"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    df = pd.read_parquet(features_file)
    
    # Generate for each horizon
    for horizon in horizons:
        label_gen = LabelGenerator(threshold=threshold, horizon=horizon)
        df_labeled = label_gen.create_classification_labels(df.copy())
        
        # Save
        output_file = output_dir / f"direction_labels_h{horizon}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_labeled.to_parquet(output_file, index=False)
        logger.info(f"Saved direction labels for horizon {horizon} to {output_file}")
    
    # Also save as default for training
    default_output = output_dir / "labeled_data_classification.parquet"
    df_labeled.to_parquet(default_output, index=False)
    
    return output_dir


def generate_volatility_labels(input_dir: Path, output_dir: Path, horizons: list):
    """Generate volatility labels (CLI wrapper)."""
    logger.info(f"Generating volatility labels for horizons: {horizons}")
    
    # Load features
    features_file = input_dir / "features.parquet"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    df = pd.read_parquet(features_file)
    df = df.sort_values(['ticker', 'date'])
    
    # Generate for each horizon
    for horizon in horizons:
        # Calculate forward volatility
        df['target_volatility'] = df.groupby('ticker')['log_return'].transform(
            lambda x: x.rolling(horizon, min_periods=1).std().shift(-horizon)
        )
        
        # Save
        output_file = output_dir / f"volatility_labels_h{horizon}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved volatility labels for horizon {horizon} to {output_file}")
    
    return output_dir
