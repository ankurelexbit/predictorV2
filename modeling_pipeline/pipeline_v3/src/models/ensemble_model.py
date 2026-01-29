"""
Ensemble Model

Combines multiple models using weighted averaging with optimized weights.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
from typing import Dict, List

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model combining multiple base models.
    
    Uses weighted averaging with optimized weights.
    """
    
    def __init__(self):
        """Initialize ensemble model."""
        self.models = {}
        self.weights = {}
        self.calibrators = {}
        self.is_calibrated = False
    
    def add_model(self, name: str, model, weight: float = None):
        """
        Add a base model to the ensemble.
        
        Args:
            name: Model name
            model: Model object with predict_proba method
            weight: Initial weight (if None, will be optimized)
        """
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
        logger.info(f"Added model: {name}")
    
    def optimize_weights(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        metric: str = 'log_loss'
    ):
        """
        Optimize ensemble weights on validation data.
        
        Args:
            df: Validation features
            y_true: True labels
            metric: Metric to optimize ('log_loss')
        """
        logger.info("Optimizing ensemble weights...")
        
        # Get predictions from all models
        model_preds = {}
        for name, model in self.models.items():
            try:
                model_preds[name] = model.predict_proba(df, calibrated=True)
            except TypeError:
                # Model doesn't support calibrated parameter
                model_preds[name] = model.predict_proba(df)
        
        # Objective function
        def objective(weights):
            # Normalize weights
            weights = weights / weights.sum()
            
            # Weighted average
            ensemble_pred = np.zeros_like(list(model_preds.values())[0])
            for i, name in enumerate(self.models.keys()):
                ensemble_pred += weights[i] * model_preds[name]
            
            # Compute metric
            return log_loss(y_true, ensemble_pred)
        
        # Initial weights (equal)
        n_models = len(self.models)
        x0 = np.ones(n_models) / n_models
        
        # Bounds (all weights between 0 and 1)
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        # Store optimized weights
        opt_weights = result.x / result.x.sum()
        for i, name in enumerate(self.models.keys()):
            self.weights[name] = opt_weights[i]
        
        logger.info("Optimized weights:")
        for name, weight in self.weights.items():
            logger.info(f"  {name}: {weight:.4f}")
        
        logger.info(f"Optimized log loss: {result.fun:.4f}")
    
    def predict_proba(
        self,
        df: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """
        Get ensemble predictions.
        
        Args:
            df: Features DataFrame
            calibrated: Whether to apply calibration
        
        Returns:
            Array of shape (n_samples, 3)
        """
        # Get predictions from all models
        ensemble_pred = None
        
        for name, model in self.models.items():
            try:
                pred = model.predict_proba(df, calibrated=True)
            except TypeError:
                pred = model.predict_proba(df)
            
            weight = self.weights.get(name, 1.0 / len(self.models))
            
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred
        
        # Apply calibration if available
        if calibrated and self.is_calibrated:
            calibrated_probs = np.zeros_like(ensemble_pred)
            
            for class_idx in range(3):
                class_probs = ensemble_pred[:, class_idx]
                calibrator = self.calibrators[class_idx]
                
                if isinstance(calibrator, IsotonicRegression):
                    calibrated_probs[:, class_idx] = calibrator.predict(class_probs)
                else:
                    calibrated_probs[:, class_idx] = calibrator.predict_proba(
                        class_probs.reshape(-1, 1)
                    )[:, 1]
            
            # Renormalize
            calibrated_probs = np.clip(calibrated_probs, 0.001, 0.999)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            
            return calibrated_probs
        
        return ensemble_pred
    
    def calibrate(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """Fit calibration on validation data."""
        raw_probs = self.predict_proba(df, calibrated=False)
        
        for class_idx in range(3):
            class_probs = raw_probs[:, class_idx]
            class_labels = (y_true == class_idx).astype(int)
            
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
            
            calibrator.fit(class_probs.reshape(-1, 1), class_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
        logger.info(f"Calibration fitted using {method} method")
    
    def save(self, path: Path):
        """Save ensemble model."""
        joblib.dump({
            'weights': self.weights,
            'calibrators': self.calibrators,
            'is_calibrated': self.is_calibrated,
            'model_names': list(self.models.keys())
        }, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: Path, models: Dict):
        """Load ensemble model."""
        data = joblib.dump(path)
        self.weights = data['weights']
        self.calibrators = data['calibrators']
        self.is_calibrated = data['is_calibrated']
        self.models = models
        logger.info(f"Ensemble loaded from {path}")
