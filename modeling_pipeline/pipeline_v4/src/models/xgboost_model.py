"""
XGBoost Football Model Wrapper.

Wraps XGBoost classifier with football-specific functionality:
- Custom objective for multi-class (Home/Draw/Away)
- Sklearn-compatible interface
- Calibration support
- Feature importance analysis
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Optional, List, Union
import joblib
import logging

logger = logging.getLogger(__name__)

class XGBoostFootballModel(BaseEstimator, ClassifierMixin):
    """
    XGBoost wrapper for football match prediction (Home/Draw/Away).
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize model.
        
        Args:
            params: XGBoost parameters
        """
        self.params = params or {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.03,
            'max_depth': 4,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
            eval_set: Optional[list] = None, verbose: bool = False):
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target values (0=Away, 1=Draw, 2=Home)
            eval_set: List of (X, y) tuples for validation
            verbose: Whether to print training progress
        """
        self.feature_names = X.columns.tolist()
        
        # Initialize XGBoost Classifier
        self.model = xgb.XGBClassifier(**self.params)
        
        # Fit model
        evals = eval_set if eval_set else []
        self.model.fit(
            X, y,
            eval_set=evals,
            verbose=verbose
        )
        
        return self
    
    def calibrate(self, X_val: pd.DataFrame, y_val: np.ndarray, method: str = 'isotonic'):
        """
        Calibrate probabilities using validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            method: 'isotonic' or 'sigmoid'
        """
        logger.info(f"Calibrating model using {method} regression...")
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        # Get uncalibrated probabilities
        probs = self.model.predict_proba(X_val)
        self.calibrated_model = []
        
        # Train calibrator for each class (One-vs-Rest)
        n_classes = probs.shape[1]
        for i in range(n_classes):
            # Target for this class
            y_i = (y_val == i).astype(int)
            p_i = probs[:, i]
            
            if method == 'isotonic':
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(p_i, y_i)
                self.calibrated_model.append(iso)
            else:
                # Sigmoid calibration (Platt scaling)
                lr = LogisticRegression(C=1.0, solver='lbfgs')
                lr.fit(p_i.reshape(-1, 1), y_i)
                self.calibrated_model.append(lr)
                
        logger.info("Calibration complete")
        
    def predict_proba(self, X: pd.DataFrame, calibrated: bool = False) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature DataFrame
            calibrated: Whether to use calibrated model (if available)
            
        Returns:
            Array of probabilities shape (n_samples, 3)
        """
        # Get base probabilities
        probs = self.model.predict_proba(X)
        
        if calibrated and self.calibrated_model:
            calibrated_probs = np.zeros_like(probs)
            for i, calibrator in enumerate(self.calibrated_model):
                if hasattr(calibrator, 'predict_proba'):
                    # Logistic Regression
                    calibrated_probs[:, i] = calibrator.predict_proba(probs[:, i].reshape(-1, 1))[:, 1]
                else:
                    # Isotonic
                    calibrated_probs[:, i] = calibrator.predict(probs[:, i])
            
            # Normalize to sum to 1
            row_sums = calibrated_probs.sum(axis=1)
            calibrated_probs = calibrated_probs / row_sums[:, np.newaxis]
            return calibrated_probs
        
        return probs
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.model:
            raise ValueError("Model not trained yet")
            
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        return joblib.load(filepath)
