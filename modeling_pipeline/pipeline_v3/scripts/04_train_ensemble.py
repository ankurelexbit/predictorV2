#!/usr/bin/env python3
"""
04 - Train Ensemble Model
==========================

Combine all models into an ensemble.

Usage:
    python scripts/04_train_ensemble.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.elo_model import EloProbabilityModel
from src.models.dixon_coles_model import DixonColesModel
from src.models.xgboost_model import XGBoostFootballModel
from src.models.ensemble_model import EnsembleModel
from sklearn.metrics import log_loss, accuracy_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess data."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    mask = df['home_score'].notna() & df['away_score'].notna()
    df = df[mask].copy()
    
    conditions = [
        (df['home_score'] < df['away_score']),
        (df['home_score'] == df['away_score']),
        (df['home_score'] > df['away_score'])
    ]
    df['target'] = np.select(conditions, [0, 1, 2], default=-1)
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """Split data into Train, Validation, Test."""
    train_mask = df['match_date'] < '2024-01-01'
    val_mask = (df['match_date'] >= '2024-01-01') & (df['match_date'] < '2025-01-01')
    test_mask = df['match_date'] >= '2025-01-01'
    
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, name: str = "Model"):
    """Evaluate prediction quality."""
    metrics = {
        'log_loss': log_loss(y_true, y_pred),
        'accuracy': accuracy_score(y_true, np.argmax(y_pred, axis=1))
    }
    
    logger.info(f"{name}:")
    logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.1%}")
    
    return metrics


def main():
    """Train and evaluate ensemble model."""
    data_path = 'data/csv/training_data_complete.csv'
    
    # 1. Load Data
    df = load_data(data_path)
    train_df, val_df, test_df = split_data(df)
    
    y_val = val_df['target'].values.astype(int)
    y_test = test_df['target'].values.astype(int)
    
    logger.info(f"Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 2. Load Base Models
    logger.info("=" * 60)
    logger.info("LOADING BASE MODELS")
    logger.info("=" * 60)
    
    elo_model = EloProbabilityModel()
    elo_model.load(Path('models/baseline_elo.joblib'))
    
    dc_model = DixonColesModel()
    dc_model.load(Path('models/dixon_coles.joblib'))
    
    xgb_model = XGBoostFootballModel()
    xgb_model.load(Path('models/xgboost_model.joblib'))
    
    # 3. Evaluate Individual Models on Test Set
    logger.info("=" * 60)
    logger.info("INDIVIDUAL MODEL PERFORMANCE (Test Set)")
    logger.info("=" * 60)
    
    elo_test_probs = elo_model.predict_proba(test_df, calibrated=True)
    elo_metrics = evaluate_predictions(y_test, elo_test_probs, "Baseline Elo")
    
    dc_test_probs = dc_model.predict_proba(test_df)
    dc_metrics = evaluate_predictions(y_test, dc_test_probs, "Dixon-Coles")
    
    xgb_test_probs = xgb_model.predict_proba(test_df, calibrated=True)
    xgb_metrics = evaluate_predictions(y_test, xgb_test_probs, "XGBoost")
    
    # 4. Create Ensemble
    logger.info("=" * 60)
    logger.info("CREATING ENSEMBLE")
    logger.info("=" * 60)
    
    ensemble = EnsembleModel()
    ensemble.add_model('elo', elo_model)
    ensemble.add_model('dixon_coles', dc_model)
    ensemble.add_model('xgboost', xgb_model)
    
    # 5. Optimize Weights on Validation Set
    ensemble.optimize_weights(val_df, y_val)
    
    # 6. Calibrate Ensemble
    logger.info("\nCalibrating ensemble...")
    ensemble.calibrate(val_df, y_val, method='isotonic')
    
    # 7. Evaluate Ensemble on Test Set
    logger.info("=" * 60)
    logger.info("ENSEMBLE PERFORMANCE (Test Set)")
    logger.info("=" * 60)
    
    ensemble_test_probs = ensemble.predict_proba(test_df, calibrated=True)
    ensemble_metrics = evaluate_predictions(y_test, ensemble_test_probs, "Ensemble")
    
    # 8. Save Ensemble
    model_path = Path('models/ensemble.joblib')
    ensemble.save(model_path)
    
    # 9. Final Summary
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline Elo:   Log Loss = {elo_metrics['log_loss']:.4f}")
    logger.info(f"Dixon-Coles:    Log Loss = {dc_metrics['log_loss']:.4f}")
    logger.info(f"XGBoost:        Log Loss = {xgb_metrics['log_loss']:.4f}")
    logger.info(f"Ensemble:       Log Loss = {ensemble_metrics['log_loss']:.4f}")
    logger.info("")
    logger.info(f"Ensemble saved to: {model_path}")
    logger.info("\n✅ Model generation complete!")


if __name__ == "__main__":
    main()
