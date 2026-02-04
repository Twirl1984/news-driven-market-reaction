"""Model training modules."""
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import joblib

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """Walk-forward cross-validation splitter."""
    
    def __init__(self, n_splits: int = 5, embargo_days: int = 5):
        """
        Initialize splitter.
        
        Args:
            n_splits: Number of splits
            embargo_days: Days to embargo between train and test to prevent leakage
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
    
    def split(self, df: pd.DataFrame) -> list:
        """
        Generate walk-forward splits.
        
        Returns:
            List of (train_idx, test_idx) tuples
        """
        logger.info(f"Creating {self.n_splits} walk-forward splits with {self.embargo_days} day embargo")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        dates = df['date'].unique()
        n_dates = len(dates)
        
        # Calculate split points
        test_size = n_dates // (self.n_splits + 1)
        
        splits = []
        for i in range(self.n_splits):
            # Train: from start to split point
            train_end_idx = (i + 1) * test_size
            
            # Apply embargo
            embargo_end_idx = train_end_idx + self.embargo_days
            
            # Test: after embargo to next split
            test_start_idx = embargo_end_idx
            test_end_idx = min(train_end_idx + test_size, n_dates)
            
            if test_start_idx >= test_end_idx:
                continue
            
            # Convert date indices to row indices
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            
            train_idx = df[df['date'].isin(train_dates)].index.tolist()
            test_idx = df[df['date'].isin(test_dates)].index.tolist()
            
            splits.append((train_idx, test_idx))
            
            logger.info(f"Split {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
        
        return splits


class ModelTrainer:
    """Train classification models for trading signals."""
    
    def __init__(self, model_type: str, random_seed: int = 42):
        """
        Initialize trainer.
        
        Args:
            model_type: 'logistic', 'random_forest', 'xgboost', 'lightgbm'
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def create_model(self):
        """Create model instance."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_seed,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_seed,
                eval_metric='logloss'
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_seed,
                class_weight='balanced',
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare feature matrix and labels.
        
        Returns:
            X: Feature matrix
            y: Labels
        """
        # Exclude non-feature columns
        exclude_cols = [
            'date', 'ticker', 'timestamp', 'asset_id', 'asset_class',
            'label', 'signal', 'future_return', 'target_return',
            'text_raw', 'text_clean', 'text_combined', 'metadata'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['label'].values
        
        # Fill any remaining NaNs
        X = X.fillna(0)
        
        # Replace inf values
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features: {feature_cols[:10]}... (showing first 10)")
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model with walk-forward validation.
        
        Returns:
            Dict with model, metrics, and metadata
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare data
        X, y = self.prepare_features(df)
        
        # Walk-forward split
        splitter = WalkForwardSplitter(
            n_splits=5,
            embargo_days=settings.trading.embargo_days
        )
        splits = splitter.split(df)
        
        # Train on all data (for final model)
        model = self.create_model()
        model.fit(X, y)
        
        # Calculate metrics from cross-validation
        cv_metrics = []
        for train_idx, test_idx in splits:
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_test, y_test = X.iloc[test_idx], y[test_idx]
            
            fold_model = self.create_model()
            fold_model.fit(X_train, y_train)
            
            # Predict
            y_pred = fold_model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            fold_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            cv_metrics.append(fold_metrics)
        
        # Average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in cv_metrics])
            for metric in cv_metrics[0].keys()
        }
        
        logger.info(f"Cross-validation metrics: {avg_metrics}")
        
        return {
            'model': model,
            'metrics': avg_metrics,
            'feature_names': X.columns.tolist(),
            'model_type': self.model_type
        }


def train_model(model_type: str = None):
    """
    Main training pipeline.
    
    Args:
        model_type: Type of model to train
    """
    logger.info("Starting model training...")
    
    # Use model type from settings if not provided
    if model_type is None:
        model_type = settings.model.model_type
    
    # Load labeled data
    data_file = settings.paths.data_labels / "labeled_data_classification.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Labeled data not found: {data_file}")
    
    df = pd.read_parquet(data_file)
    
    logger.info(f"Loaded {len(df)} labeled samples")
    logger.info(f"Training {model_type} model...")
    
    # Train model
    trainer = ModelTrainer(model_type=model_type, random_seed=settings.model.random_seed)
    results = trainer.train(df)
    
    # Save model
    model_dir = settings.paths.models / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "model.joblib"
    joblib.dump(results['model'], model_file)
    
    # Save metadata
    metadata = {
        'model_type': results['model_type'],
        'metrics': results['metrics'],
        'feature_names': results['feature_names'],
        'random_seed': settings.model.random_seed
    }
    
    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Metrics: {results['metrics']}")
    
    return model_dir
