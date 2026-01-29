#!/usr/bin/env python3
"""
03 - Train XGBoost Model
=========================

Train and evaluate XGBoost model with hyperparameter tuning.

Usage:
    python scripts/03_train_xgboost.py [--tune] [--n-trials 20]
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

from src.models.xgboost_model import XGBoostFootballModel, tune_xgboost
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
    
    # Rename columns to match script expectations or update logic
    column_map = {
        'home_goals': 'home_score',
        'away_goals': 'away_score',
        'starting_at': 'match_date'
    }
    df.rename(columns=column_map, inplace=True)
    
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
    # Handle both match_date and starting_at columns
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        date_col = 'match_date'
    elif 'starting_at' in df.columns:
        df['match_date'] = pd.to_datetime(df['starting_at'], errors='coerce')
        date_col = 'match_date'
    else:
        logger.error("No date column found (match_date or starting_at)")
        return None
    
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
    """Train and evaluate XGBoost model."""
    parser = argparse.ArgumentParser(description="XGBoost Model")
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning")
    parser.add_argument('--n-trials', type=int, default=20, help="Number of tuning trials")
    args = parser.parse_args()
    
    data_path = 'data/csv/training_data_complete_v3_expanded.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # 1. Load Data
    df = load_data(data_path)
    logger.info(f"Loaded {len(df)} matches")

    # 1b. Drop Zero-Variance Columns (Points Features)
    zero_var_cols = [col for col in df.columns if df[col].nunique() <= 1 and col not in ['fixture_id', 'target', 'league_id', 'season_id']]
    if zero_var_cols:
        logger.info(f"Dropping {len(zero_var_cols)} constant columns: {zero_var_cols}")
        df.drop(columns=zero_var_cols, inplace=True)

    # 1c. Drop Data Leakage Columns (Targets)
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result']
    to_drop = [c for c in leakage_cols if c in df.columns]
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} leakage columns: {to_drop}")
        df.drop(columns=to_drop, inplace=True)
    
    # 2. Split Data
    train_df, val_df, test_df = split_data(df)
    
    # 3. Get targets
    y_train = train_df['target'].values.astype(int)
    y_val = val_df['target'].values.astype(int)
    y_test = test_df['target'].values.astype(int)
    
    # 4. Hyperparameter Tuning (optional)
    if args.tune:
        logger.info("=" * 60)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("=" * 60)
        
        tune_results = tune_xgboost(train_df, val_df, n_trials=args.n_trials)
        
        logger.info(f"\nBest parameters:")
        for param, value in tune_results['best_params'].items():
            if param not in ['objective', 'num_class', 'random_state', 'n_jobs']:
                logger.info(f"  {param}: {value}")
        logger.info(f"Best log loss: {tune_results['best_score']:.4f}")
        
        params = tune_results['best_params']
    else:
        params = None  # Use defaults
    
    # 5. Train Final Model
    logger.info("=" * 60)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("=" * 60)
    
    model = XGBoostFootballModel(params=params)
    model.fit(train_df, val_df, verbose=True)
    
    # 6. Feature Importance
    importance_df = model.get_feature_importance()
    logger.info("\nTop 15 features by importance:")
    logger.info("\n" + importance_df.head(15).to_string(index=False))
    
    # 7. Evaluate Uncalibrated
    logger.info("=" * 60)
    logger.info("UNCALIBRATED PREDICTIONS")
    logger.info("=" * 60)
    
    val_probs_raw = model.predict_proba(val_df, calibrated=False)
    evaluate_predictions(y_val, val_probs_raw, "Validation (Raw)")
    
    # 8. Calibrate
    logger.info("=" * 60)
    logger.info("CALIBRATING ON VALIDATION SET")
    logger.info("=" * 60)
    
    model.calibrate(val_df, y_val, method='isotonic')
    
    val_probs_cal = model.predict_proba(val_df, calibrated=True)
    evaluate_predictions(y_val, val_probs_cal, "Validation (Calibrated)")
    
    # 9. Final Test Evaluation
    logger.info("=" * 60)
    logger.info("FINAL TEST SET EVALUATION (2025)")
    logger.info("=" * 60)
    
    test_probs = model.predict_proba(test_df, calibrated=True)
    test_metrics = evaluate_predictions(y_test, test_probs, "Test Set")
    
    # 10. Sample Predictions
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
    
    # 11. Save Model
    model_path = Path('models/xgboost_model.joblib')
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"XGBoost Test Log Loss: {test_metrics['log_loss']:.4f}")
    logger.info(f"XGBoost Test Accuracy: {test_metrics['accuracy']:.1%}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("")
    logger.info("Next: Run 04_train_ensemble.py to combine all models")


if __name__ == "__main__":
    main()
