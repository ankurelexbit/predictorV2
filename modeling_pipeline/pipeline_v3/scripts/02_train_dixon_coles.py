#!/usr/bin/env python3
"""
02 - Train Dixon-Coles Model
=============================

Train and evaluate the Dixon-Coles model with hyperparameter tuning.

Usage:
    python scripts/02_train_dixon_coles.py [--tune]
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dixon_coles_model import DixonColesModel
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
    
    # Filter to matches with results
    mask = df['home_score'].notna() & df['away_score'].notna()
    df = df[mask].copy()
    
    # Create target variable (0=away, 1=draw, 2=home)
    conditions = [
        (df['home_score'] < df['away_score']),
        (df['home_score'] == df['away_score']),
        (df['home_score'] > df['away_score'])
    ]
    df['target'] = np.select(conditions, [0, 1, 2], default=-1)
    
    # Ensure date is datetime
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """Split data into Train, Validation, Test."""
    train_mask = df['match_date'] < '2024-01-01'
    val_mask = (df['match_date'] >= '2024-01-01') & (df['match_date'] < '2025-01-01')
    test_mask = df['match_date'] >= '2025-01-01'
    
    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()
    
    logger.info(f"Data Splits:")
    logger.info(f"  Train: {len(train)} rows")
    logger.info(f"  Val:   {len(val)} rows")
    logger.info(f"  Test:  {len(test)} rows")
    
    return train, val, test


def tune_hyperparameters(train_df: pd.DataFrame, val_df: pd.DataFrame, y_val: np.ndarray, n_trials: int = 5):
    """Tune time_decay hyperparameter."""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    
    # Grid of time_decay values to try
    decay_values = [0.001, 0.0015, 0.0018, 0.002, 0.0025]
    
    best_score = float('inf')
    best_decay = None
    
    for decay in decay_values:
        logger.info(f"Trying time_decay={decay:.4f}")
        
        model = DixonColesModel(time_decay=decay, max_goals=10)
        model.fit(train_df, verbose=False)
        
        val_probs = model.predict_proba(val_df)
        score = log_loss(y_val, val_probs)
        
        logger.info(f"  Validation Log Loss: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_decay = decay
            logger.info(f"  *** New best!")
    
    logger.info(f"\nBest time_decay: {best_decay:.4f} (Log Loss: {best_score:.4f})")
    return best_decay


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, name: str = "Model"):
    """Evaluate prediction quality."""
    metrics = {
        'log_loss': log_loss(y_true, y_pred),
        'accuracy': accuracy_score(y_true, np.argmax(y_pred, axis=1))
    }
    
    logger.info(f"{name} Metrics:")
    logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def main():
    """Train and evaluate Dixon-Coles model."""
    parser = argparse.ArgumentParser(description="Dixon-Coles Model")
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning")
    args = parser.parse_args()
    
    data_path = 'data/csv/training_data_complete.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # 1. Load Data
    df = load_data(data_path)
    logger.info(f"Loaded {len(df)} matches")
    
    # 2. Split Data
    train_df, val_df, test_df = split_data(df)
    
    # 3. Get targets
    y_train = train_df['target'].values.astype(int)
    y_val = val_df['target'].values.astype(int)
    y_test = test_df['target'].values.astype(int)
    
    # 4. Hyperparameter Tuning (optional)
    if args.tune:
        best_decay = tune_hyperparameters(train_df, val_df, y_val)
    else:
        best_decay = 0.0018  # Default
    
    # 5. Train Final Model
    logger.info("=" * 60)
    logger.info("TRAINING DIXON-COLES MODEL")
    logger.info("=" * 60)
    
    model = DixonColesModel(time_decay=best_decay, max_goals=10)
    model.fit(train_df, verbose=True)
    
    # 6. Evaluate Uncalibrated
    logger.info("=" * 60)
    logger.info("UNCALIBRATED PREDICTIONS")
    logger.info("=" * 60)
    
    val_probs_raw = model.predict_proba(val_df)
    evaluate_predictions(y_val, val_probs_raw, "Validation (Raw)")
    
    # 7. Calibrate
    logger.info("=" * 60)
    logger.info("CALIBRATING ON VALIDATION SET")
    logger.info("=" * 60)
    
    model.calibrate(val_df, y_val, method='isotonic')
    
    val_probs_cal = model.predict_proba(val_df)
    evaluate_predictions(y_val, val_probs_cal, "Validation (Calibrated)")
    
    # 8. Final Test Evaluation
    logger.info("=" * 60)
    logger.info("FINAL TEST SET EVALUATION (2025)")
    logger.info("=" * 60)
    
    test_probs = model.predict_proba(test_df)
    test_metrics = evaluate_predictions(y_test, test_probs, "Test Set")
    
    # 9. Sample Predictions
    logger.info("-" * 60)
    logger.info("Sample Predictions:")
    sample = test_df.head(5)
    sample_probs = test_probs[:5]
    
    for i, (_, row) in enumerate(sample.iterrows()):
        probs = sample_probs[i]
        p_away = probs[0]
        p_draw = probs[1]
        p_home = probs[2]
        
        target_map = {0: 'Away', 1: 'Draw', 2: 'Home'}
        actual = target_map[row['target']]
        
        logger.info(
            f"{row['match_date'].date()} | {row['home_team_id']} vs {row['away_team_id']} | "
            f"Pred: H {p_home:.2f} D {p_draw:.2f} A {p_away:.2f} | Actual: {actual}"
        )
    
    # 10. Show Top Teams
    logger.info("-" * 60)
    logger.info("Top 10 Teams by Attack Strength:")
    strengths = model.get_team_strengths()
    logger.info("\n" + strengths.head(10).to_string(index=False))
    
    # 11. Save Model
    model_path = Path('models/dixon_coles.joblib')
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dixon-Coles Test Log Loss: {test_metrics['log_loss']:.4f}")
    logger.info(f"Dixon-Coles Test Accuracy: {test_metrics['accuracy']:.1%}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("")
    logger.info("Next: Run 03_train_xgboost.py")


if __name__ == "__main__":
    main()
