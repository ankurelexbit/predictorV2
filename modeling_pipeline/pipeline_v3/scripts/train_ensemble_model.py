"""
Ensemble Model Training - Phase 8

Trains and combines multiple models:
1. XGBoost (primary - 60% weight)
2. LightGBM (secondary - 30% weight)
3. Neural Network (experimental - 10% weight)

Includes probability calibration for better predictions.

Expected impact: -0.003 to -0.008 log loss improvement
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.feature_selector import FeatureSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of XGBoost, LightGBM, and Neural Network."""
    
    def __init__(
        self,
        xgb_weight: float = 0.6,
        lgb_weight: float = 0.3,
        nn_weight: float = 0.1,
        calibrate: bool = True
    ):
        """
        Initialize ensemble model.
        
        Args:
            xgb_weight: Weight for XGBoost predictions
            lgb_weight: Weight for LightGBM predictions
            nn_weight: Weight for Neural Network predictions
            calibrate: Whether to calibrate probabilities
        """
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.nn_weight = nn_weight
        self.calibrate = calibrate
        
        # Normalize weights
        total = xgb_weight + lgb_weight + nn_weight
        self.xgb_weight /= total
        self.lgb_weight /= total
        self.nn_weight /= total
        
        self.xgb_model = None
        self.lgb_model = None
        self.nn_model = None
        
        logger.info(f"Ensemble weights: XGB={self.xgb_weight:.2f}, "
                   f"LGB={self.lgb_weight:.2f}, NN={self.nn_weight:.2f}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        logger.info("\n" + "="*60)
        logger.info("Training Ensemble Models")
        logger.info("="*60)
        
        # Train XGBoost
        logger.info("\n1. Training XGBoost...")
        self.xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Train LightGBM
        logger.info("\n2. Training LightGBM...")
        self.lgb_model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.lgb_model.fit(X_train, y_train)
        
        # Train Neural Network
        logger.info("\n3. Training Neural Network...")
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            verbose=False
        )
        self.nn_model.fit(X_train, y_train)
        
        # Calibrate if requested
        if self.calibrate and X_val is not None and y_val is not None:
            logger.info("\n4. Calibrating probabilities...")
            self._calibrate_models(X_val, y_val)
        
        logger.info("\n✅ Ensemble training complete")
    
    def _calibrate_models(self, X_val, y_val):
        """Calibrate model probabilities using validation set."""
        logger.info("Calibrating XGBoost...")
        self.xgb_model = CalibratedClassifierCV(
            self.xgb_model, method='isotonic', cv='prefit'
        )
        self.xgb_model.fit(X_val, y_val)
        
        logger.info("Calibrating LightGBM...")
        self.lgb_model = CalibratedClassifierCV(
            self.lgb_model, method='isotonic', cv='prefit'
        )
        self.lgb_model.fit(X_val, y_val)
        
        logger.info("Calibrating Neural Network...")
        self.nn_model = CalibratedClassifierCV(
            self.nn_model, method='isotonic', cv='prefit'
        )
        self.nn_model.fit(X_val, y_val)
    
    def predict_proba(self, X):
        """
        Predict probabilities using weighted ensemble.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability predictions (n_samples, n_classes)
        """
        # Get predictions from each model
        xgb_pred = self.xgb_model.predict_proba(X)
        lgb_pred = self.lgb_model.predict_proba(X)
        nn_pred = self.nn_model.predict_proba(X)
        
        # Weighted average
        ensemble_pred = (
            self.xgb_weight * xgb_pred +
            self.lgb_weight * lgb_pred +
            self.nn_weight * nn_pred
        )
        
        return ensemble_pred
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X, y, name=""):
        """Evaluate ensemble and individual models."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation{' - ' + name if name else ''}")
        logger.info(f"{'='*60}")
        
        # Individual model predictions
        xgb_pred = self.xgb_model.predict_proba(X)
        lgb_pred = self.lgb_model.predict_proba(X)
        nn_pred = self.nn_model.predict_proba(X)
        ensemble_pred = self.predict_proba(X)
        
        # Calculate log loss
        xgb_loss = log_loss(y, xgb_pred)
        lgb_loss = log_loss(y, lgb_pred)
        nn_loss = log_loss(y, nn_pred)
        ensemble_loss = log_loss(y, ensemble_pred)
        
        logger.info(f"XGBoost Log Loss:     {xgb_loss:.4f}")
        logger.info(f"LightGBM Log Loss:    {lgb_loss:.4f}")
        logger.info(f"Neural Net Log Loss:  {nn_loss:.4f}")
        logger.info(f"Ensemble Log Loss:    {ensemble_loss:.4f}")
        
        # Show improvement
        best_individual = min(xgb_loss, lgb_loss, nn_loss)
        improvement = best_individual - ensemble_loss
        logger.info(f"\nEnsemble Improvement: {improvement:.4f}")
        
        return ensemble_loss


def main():
    """Main ensemble training pipeline."""
    # Paths
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'training_data.csv'
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading training data...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    target_col = 'result'
    exclude_cols = [target_col, 'fixture_id', 'date', 'home_team_id', 'away_team_id', 'league_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    
    # Load feature selector if available
    selector_path = models_dir / 'feature_selector.pkl'
    if selector_path.exists():
        logger.info("Loading feature selector...")
        selector = FeatureSelector.load(selector_path)
        X = selector.transform(X)
        logger.info(f"Using {len(selector.selected_features)} selected features")
    else:
        logger.warning("No feature selector found - using all features")
    
    # Split data: train (60%), validation (20%), test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train ensemble
    ensemble = EnsembleModel(
        xgb_weight=0.6,
        lgb_weight=0.3,
        nn_weight=0.1,
        calibrate=True
    )
    
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_loss = ensemble.evaluate(X_train, y_train, "Train Set")
    val_loss = ensemble.evaluate(X_val, y_val, "Validation Set")
    test_loss = ensemble.evaluate(X_test, y_test, "Test Set")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Train Log Loss:      {train_loss:.4f}")
    logger.info(f"Validation Log Loss: {val_loss:.4f}")
    logger.info(f"Test Log Loss:       {test_loss:.4f}")
    
    # Save ensemble
    logger.info("\nSaving ensemble model...")
    joblib.dump(ensemble, models_dir / 'ensemble_model.pkl')
    logger.info(f"Saved to {models_dir / 'ensemble_model.pkl'}")
    
    logger.info("\n✅ Ensemble training complete!")


if __name__ == '__main__':
    main()
