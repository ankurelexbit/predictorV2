#!/usr/bin/env python3
"""
01 - Train Baseline Elo Model
==============================

Train and evaluate the Elo-based baseline model.

Usage:
    python scripts/01_train_baseline_elo.py
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
    
    # Filter to matches with results and Elo ratings
    mask = (
        df['home_score'].notna() &
        df['away_score'].notna() &
        df['home_elo'].notna() &
        df['away_elo'].notna()
    )
    df = df[mask].copy()
    
    # Create target variable (0=away, 1=draw, 2=home)
    conditions = [
        (df['home_score'] < df['away_score']),  # Away Win
        (df['home_score'] == df['away_score']), # Draw
        (df['home_score'] > df['away_score'])   # Home Win
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
    logger.info(f"  Train: {len(train)} rows ({train['match_date'].min().date()} to {train['match_date'].max().date()})")
    logger.info(f"  Val:   {len(val)} rows ({val['match_date'].min().date()} to {val['match_date'].max().date()})")
    logger.info(f"  Test:  {len(test)} rows ({test['match_date'].min().date()} to {test['match_date'].max().date()})")
    
    return train, val, test


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
    """Train and evaluate Elo baseline model."""
    data_path = 'data/csv/training_data_complete.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # 1. Load Data
    df = load_data(data_path)
    logger.info(f"Loaded {len(df)} matches with Elo ratings")
    
    # 2. Split Data
    train_df, val_df, test_df = split_data(df)
    
    # 3. Initialize Model
    model = EloProbabilityModel(home_advantage=25)
    
    # 4. Get targets
    y_train = train_df['target'].values.astype(int)
    y_val = val_df['target'].values.astype(int)
    y_test = test_df['target'].values.astype(int)
    
    # 5. Evaluate Uncalibrated
    logger.info("=" * 60)
    logger.info("UNCALIBRATED ELO PREDICTIONS")
    logger.info("=" * 60)
    
    val_probs_raw = model.predict_proba(val_df, calibrated=False)
    evaluate_predictions(y_val, val_probs_raw, "Validation (Raw)")
    
    # 6. Calibrate on Validation Set
    logger.info("=" * 60)
    logger.info("CALIBRATING ON VALIDATION SET")
    logger.info("=" * 60)
    
    model.calibrate(val_df, y_val, method='isotonic')
    
    val_probs_cal = model.predict_proba(val_df, calibrated=True)
    evaluate_predictions(y_val, val_probs_cal, "Validation (Calibrated)")
    
    # 7. Final Test Evaluation
    logger.info("=" * 60)
    logger.info("FINAL TEST SET EVALUATION (2025)")
    logger.info("=" * 60)
    
    test_probs = model.predict_proba(test_df, calibrated=True)
    test_metrics = evaluate_predictions(y_test, test_probs, "Test Set")
    
    # 8. Sample Predictions
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
    
    # 9. Save Model
    model_path = Path('models/baseline_elo.joblib')
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Elo Model Test Log Loss: {test_metrics['log_loss']:.4f}")
    logger.info(f"Elo Model Test Accuracy: {test_metrics['accuracy']:.1%}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("")
    logger.info("Next: Run 02_train_dixon_coles.py")


if __name__ == "__main__":
    main()
