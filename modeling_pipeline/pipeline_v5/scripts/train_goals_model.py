#!/usr/bin/env python3
"""
Train Poisson Goals Model for Advanced Markets
================================================

Trains CatBoost + LightGBM regressors (Poisson objective) to predict expected
home/away goals. These lambda parameters feed into Poisson distributions to
derive O/U, BTTS, Handicap, and Correct Score probabilities.

Usage:
    python3 scripts/train_goals_model.py --data data/training_data.csv --version 1.0.0
    python3 scripts/train_goals_model.py --data data/training_data.csv  # auto-version
"""

import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.production_config import META_COLS
from src.goals import PoissonGoalsModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str):
    """Load and prepare training data with chronological split."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Require home_score and away_score
    if 'home_score' not in df.columns or 'away_score' not in df.columns:
        raise ValueError("Training data must contain 'home_score' and 'away_score' columns")

    # Drop rows with missing scores
    before = len(df)
    df = df.dropna(subset=['home_score', 'away_score']).reset_index(drop=True)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with missing scores")

    # Feature columns (same as outcome model)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Chronological split: 70/15/15
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Data: {n} fixtures")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%) "
                f"[{train_df['match_date'].min().date()} to {train_df['match_date'].max().date()}]")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%) "
                f"[{val_df['match_date'].min().date()} to {val_df['match_date'].max().date()}]")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%) "
                f"[{test_df['match_date'].min().date()} to {test_df['match_date'].max().date()}]")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Avg home goals: {df['home_score'].mean():.2f}, away: {df['away_score'].mean():.2f}")

    return train_df, val_df, test_df, feature_cols


def evaluate_goals(model: PoissonGoalsModel, X: pd.DataFrame,
                   y_home: np.ndarray, y_away: np.ndarray) -> dict:
    """Evaluate goal prediction accuracy."""
    lh, la = model.predict(X)

    # MAE
    home_mae = np.mean(np.abs(lh - y_home))
    away_mae = np.mean(np.abs(la - y_away))

    # Mean predicted vs actual
    home_bias = np.mean(lh) - np.mean(y_home)
    away_bias = np.mean(la) - np.mean(y_away)

    return {
        'home_mae': round(float(home_mae), 4),
        'away_mae': round(float(away_mae), 4),
        'home_bias': round(float(home_bias), 4),
        'away_bias': round(float(away_bias), 4),
        'mean_pred_home': round(float(np.mean(lh)), 3),
        'mean_actual_home': round(float(np.mean(y_home)), 3),
        'mean_pred_away': round(float(np.mean(la)), 3),
        'mean_actual_away': round(float(np.mean(y_away)), 3),
    }


def evaluate_poisson_calibration(model: PoissonGoalsModel, X: pd.DataFrame,
                                 y_home: np.ndarray, y_away: np.ndarray) -> dict:
    """Check Poisson calibration: group by predicted lambda, compare to actual mean."""
    lh, la = model.predict(X)

    results = {}
    for name, pred_lambda, actual in [('home', lh, y_home), ('away', la, y_away)]:
        # Bucket by predicted lambda
        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
        labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0+']
        bucket_idx = np.digitize(pred_lambda, bins) - 1
        bucket_idx = np.clip(bucket_idx, 0, len(labels) - 1)

        calibration = []
        for i, label in enumerate(labels):
            mask = bucket_idx == i
            count = int(mask.sum())
            if count > 0:
                calibration.append({
                    'bucket': label,
                    'count': count,
                    'mean_predicted': round(float(np.mean(pred_lambda[mask])), 3),
                    'mean_actual': round(float(np.mean(actual[mask])), 3),
                    'diff': round(float(np.mean(pred_lambda[mask]) - np.mean(actual[mask])), 3),
                })

        results[name] = calibration

    return results


def evaluate_derived_markets(model: PoissonGoalsModel, X: pd.DataFrame,
                             y_home: np.ndarray, y_away: np.ndarray) -> dict:
    """Evaluate derived market accuracy (O/U 2.5, BTTS)."""
    lh, la = model.predict(X)
    actual_total = y_home + y_away
    actual_btts = (y_home >= 1) & (y_away >= 1)

    n = len(y_home)
    ou25_correct = 0
    btts_correct = 0
    ou25_probs = []
    btts_probs = []

    for i in range(n):
        markets = PoissonGoalsModel.derive_markets(float(lh[i]), float(la[i]))

        # O/U 2.5
        pred_over = markets['over_2_5_prob'] > 0.5
        actual_over = actual_total[i] > 2.5
        if pred_over == actual_over:
            ou25_correct += 1
        ou25_probs.append(markets['over_2_5_prob'])

        # BTTS
        pred_btts = markets['btts_prob'] > 0.5
        actual_b = actual_btts[i]
        if pred_btts == actual_b:
            btts_correct += 1
        btts_probs.append(markets['btts_prob'])

    ou25_acc = ou25_correct / n
    btts_acc = btts_correct / n

    # Calibration for O/U 2.5
    ou25_probs = np.array(ou25_probs)
    actual_over_arr = (actual_total > 2.5).astype(float)

    # Bucket calibration
    ou_cal = []
    for lo, hi, label in [(0, 0.3, '<30%'), (0.3, 0.45, '30-45%'), (0.45, 0.55, '45-55%'),
                           (0.55, 0.7, '55-70%'), (0.7, 1.01, '>70%')]:
        mask = (ou25_probs >= lo) & (ou25_probs < hi)
        count = int(mask.sum())
        if count > 0:
            ou_cal.append({
                'bucket': label,
                'count': count,
                'mean_predicted': round(float(ou25_probs[mask].mean()), 3),
                'actual_rate': round(float(actual_over_arr[mask].mean()), 3),
            })

    return {
        'over_2_5_accuracy': round(float(ou25_acc), 4),
        'btts_accuracy': round(float(btts_acc), 4),
        'actual_over_2_5_rate': round(float((actual_total > 2.5).mean()), 4),
        'actual_btts_rate': round(float(actual_btts.mean()), 4),
        'over_2_5_calibration': ou_cal,
    }


def get_next_goals_version(model_dir: Path) -> str:
    """Get next version for goals model."""
    import re
    existing = list(model_dir.glob("goals_model_v*.joblib"))
    if not existing:
        return "1.0.0"

    versions = []
    for f in existing:
        match = re.search(r'goals_model_v(\d+)\.(\d+)\.(\d+)\.joblib', f.name)
        if match:
            versions.append(tuple(map(int, match.groups())))

    if versions:
        latest = max(versions)
        return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"

    return "1.0.0"


def main():
    parser = argparse.ArgumentParser(description='Train Poisson goals model for advanced markets')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--version', help='Model version (default: auto-increment)')
    parser.add_argument('--output-dir', default='models/production', help='Model output directory')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("POISSON GOALS MODEL TRAINING")
    logger.info("=" * 80)

    # Load data
    train_df, val_df, test_df, feature_cols = load_data(args.data)

    X_train = train_df[feature_cols]
    y_home_train = train_df['home_score'].values.astype(float)
    y_away_train = train_df['away_score'].values.astype(float)

    X_val = val_df[feature_cols]
    y_home_val = val_df['home_score'].values.astype(float)
    y_away_val = val_df['away_score'].values.astype(float)

    X_test = test_df[feature_cols]
    y_home_test = test_df['home_score'].values.astype(float)
    y_away_test = test_df['away_score'].values.astype(float)

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    model = PoissonGoalsModel()
    model.train(X_train, y_home_train, y_away_train,
                X_val, y_home_val, y_away_val)

    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION — Goal Prediction (Test Set)")
    logger.info("=" * 80)

    goal_metrics = evaluate_goals(model, X_test, y_home_test, y_away_test)

    logger.info(f"Home goals — MAE: {goal_metrics['home_mae']:.4f}, "
                f"Predicted: {goal_metrics['mean_pred_home']:.3f}, "
                f"Actual: {goal_metrics['mean_actual_home']:.3f}, "
                f"Bias: {goal_metrics['home_bias']:+.4f}")
    logger.info(f"Away goals — MAE: {goal_metrics['away_mae']:.4f}, "
                f"Predicted: {goal_metrics['mean_pred_away']:.3f}, "
                f"Actual: {goal_metrics['mean_actual_away']:.3f}, "
                f"Bias: {goal_metrics['away_bias']:+.4f}")

    # Poisson calibration
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION — Poisson Calibration")
    logger.info("=" * 80)

    cal = evaluate_poisson_calibration(model, X_test, y_home_test, y_away_test)

    for side in ['home', 'away']:
        logger.info(f"\n  {side.upper()} goals calibration:")
        logger.info(f"  {'Bucket':<10} {'Count':>6} {'Predicted':>10} {'Actual':>10} {'Diff':>8}")
        logger.info(f"  {'-'*44}")
        for row in cal[side]:
            logger.info(f"  {row['bucket']:<10} {row['count']:>6} "
                        f"{row['mean_predicted']:>10.3f} {row['mean_actual']:>10.3f} "
                        f"{row['diff']:>+8.3f}")

    # Derived markets
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION — Derived Markets (Test Set)")
    logger.info("=" * 80)

    market_metrics = evaluate_derived_markets(model, X_test, y_home_test, y_away_test)

    logger.info(f"Over 2.5 — Accuracy: {market_metrics['over_2_5_accuracy']*100:.1f}% "
                f"(actual over rate: {market_metrics['actual_over_2_5_rate']*100:.1f}%)")
    logger.info(f"BTTS     — Accuracy: {market_metrics['btts_accuracy']*100:.1f}% "
                f"(actual BTTS rate: {market_metrics['actual_btts_rate']*100:.1f}%)")

    logger.info(f"\n  O/U 2.5 calibration:")
    logger.info(f"  {'Bucket':<10} {'Count':>6} {'Predicted':>10} {'Actual':>10}")
    logger.info(f"  {'-'*38}")
    for row in market_metrics['over_2_5_calibration']:
        logger.info(f"  {row['bucket']:<10} {row['count']:>6} "
                    f"{row['mean_predicted']:>10.3f} {row['actual_rate']:>10.3f}")

    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)

    model_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    version = args.version or get_next_goals_version(model_dir)
    model_path = model_dir / f"goals_model_v{version}.joblib"
    model.save(str(model_path))

    # Save metadata
    all_metrics = {
        'goal_metrics': goal_metrics,
        'market_metrics': {
            'over_2_5_accuracy': market_metrics['over_2_5_accuracy'],
            'btts_accuracy': market_metrics['btts_accuracy'],
        },
    }

    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'PoissonGoalsModel (CatBoost+LightGBM Poisson regressors)',
        'features': len(feature_cols),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'test_metrics': all_metrics,
    }

    metadata_path = model_dir / f"goals_model_v{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")

    # Update LATEST_GOALS pointer
    latest_path = model_dir / "LATEST_GOALS"
    with open(latest_path, 'w') as f:
        f.write(f"goals_model_v{version}.joblib")
    logger.info(f"LATEST_GOALS → goals_model_v{version}.joblib")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model:     goals_model_v{version}.joblib")
    logger.info(f"Features:  {len(feature_cols)}")
    logger.info(f"Home MAE:  {goal_metrics['home_mae']:.4f}")
    logger.info(f"Away MAE:  {goal_metrics['away_mae']:.4f}")
    logger.info(f"O/U 2.5:   {market_metrics['over_2_5_accuracy']*100:.1f}% accuracy")
    logger.info(f"BTTS:      {market_metrics['btts_accuracy']*100:.1f}% accuracy")
    logger.info(f"\nUsage: python3 scripts/predict_live.py --days-ahead 7")


if __name__ == '__main__':
    main()
