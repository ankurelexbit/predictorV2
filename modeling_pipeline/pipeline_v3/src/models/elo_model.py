"""
Elo Probability Model

Converts Elo ratings to 1X2 probabilities with isotonic calibration.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from typing import Dict

logger = logging.getLogger(__name__)


class EloProbabilityModel:
    """
    Wrapper for Elo-based probability predictions.
    
    This model converts Elo ratings (already computed in feature engineering)
    into calibrated 1X2 probabilities.
    """
    
    def __init__(self, home_advantage: float = 25):
        """
        Initialize Elo probability model.
        
        Args:
            home_advantage: Elo points added to home team rating
        """
        self.home_advantage = home_advantage
        self.calibrators = {}  # One per class
        self.is_calibrated = False
    
    def predict_proba(
        self,
        features_df: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """
        Get 1X2 probabilities from Elo ratings.

        Args:
            features_df: DataFrame with home_elo, away_elo columns
            calibrated: Whether to apply calibration

        Returns:
            Array of shape (n_samples, 3) with probabilities [away, draw, home]
        """
        # Calculate probabilities from raw Elo ratings
        home_rating = features_df['home_elo'].values + self.home_advantage
        away_rating = features_df['away_elo'].values

        elo_diff = home_rating - away_rating

        # Base expected score using Elo formula
        exp_home = 1 / (1 + 10 ** (-elo_diff / 400))

        # Draw probability estimation
        # Higher when teams are closely matched, lower when big gap
        base_draw_rate = 0.32
        elo_draw_factor = 1 - (np.abs(elo_diff) / 600)
        elo_draw_factor = np.clip(elo_draw_factor, 0.5, 1.5)

        p_draw = base_draw_rate * elo_draw_factor

        # Allocate remaining probability
        remaining = 1 - p_draw
        p_home = remaining * exp_home
        p_away = remaining * (1 - exp_home)

        # Stack into (n_samples, 3) array with order [away, draw, home]
        probs = np.column_stack([p_away, p_draw, p_home])

        if calibrated and self.is_calibrated:
            probs = self._calibrate(probs)

        # Ensure valid probabilities
        probs = np.clip(probs, 0.001, 0.999)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs
    
    def calibrate(
        self,
        features_df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """
        Fit calibration on validation data.
        
        Args:
            features_df: Features with Elo ratings
            y_true: True labels (0=away, 1=draw, 2=home)
            method: 'isotonic' or 'sigmoid'
        """
        raw_probs = self.predict_proba(features_df, calibrated=False)
        
        # Calibrate each class separately
        for class_idx in range(3):
            class_probs = raw_probs[:, class_idx]
            class_labels = (y_true == class_idx).astype(int)
            
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                # Platt scaling (logistic regression)
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
            
            calibrator.fit(class_probs.reshape(-1, 1), class_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
        logger.info(f"Calibration fitted using {method} method")
    
    def _calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        calibrated = np.zeros_like(probs)
        
        for class_idx in range(3):
            class_probs = probs[:, class_idx]
            calibrator = self.calibrators[class_idx]
            
            if isinstance(calibrator, IsotonicRegression):
                calibrated[:, class_idx] = calibrator.predict(class_probs)
            else:
                calibrated[:, class_idx] = calibrator.predict_proba(
                    class_probs.reshape(-1, 1)
                )[:, 1]
        
        # Renormalize
        calibrated = np.clip(calibrated, 0.001, 0.999)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated
    
    def save(self, path: Path):
        """Save model to file."""
        joblib.dump({
            'home_advantage': self.home_advantage,
            'calibrators': self.calibrators,
            'is_calibrated': self.is_calibrated
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        data = joblib.load(path)
        self.home_advantage = data['home_advantage']
        self.calibrators = data['calibrators']
        self.is_calibrated = data['is_calibrated']
        logger.info(f"Model loaded from {path}")
