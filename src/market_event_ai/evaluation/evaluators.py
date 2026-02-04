"""Model evaluation modules."""
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

from market_event_ai.config.settings import settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained models."""
    
    def __init__(self, model_path: Path):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model directory
        """
        self.model_path = model_path
        self.model = joblib.load(model_path / "model.joblib")
        
        with open(model_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
    
    def evaluate(self, data: pd.DataFrame) -> dict:
        """
        Evaluate model on data.
        
        Args:
            data: Labeled data with features
        
        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating model on {len(data)} samples...")
        
        # Prepare features
        feature_names = self.metadata['feature_names']
        X = data[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        y_true = data['label'].values
        
        # Predict
        y_pred = self.model.predict(X)
        y_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(feature_names, importance.tolist()))
            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance,
            'num_samples': len(data),
            'num_features': len(feature_names)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return results


def evaluate_model(model_type: str = None):
    """
    Main evaluation pipeline.
    
    Args:
        model_type: Type of model to evaluate
    """
    logger.info("Starting model evaluation...")
    
    # Use model type from settings if not provided
    if model_type is None:
        model_type = settings.model.model_type
    
    # Load model
    model_dir = settings.paths.models / model_type
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")
    
    # Load test data
    data_file = settings.paths.data_labels / "labeled_data_classification.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Labeled data not found: {data_file}")
    
    data = pd.read_parquet(data_file)
    
    # Evaluate
    evaluator = ModelEvaluator(model_dir)
    results = evaluator.evaluate(data)
    
    # Save results
    output_file = model_dir / "evaluation.json"
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = results.copy()
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_file}")
    
    return results
