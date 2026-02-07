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
        markets = model.derive_markets(float(lh[i]), float(la[i]))

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


def tune_hyperparameters(X_train: pd.DataFrame, y_home_train: np.ndarray, y_away_train: np.ndarray,
                         X_val: pd.DataFrame, y_home_val: np.ndarray, y_away_val: np.ndarray,
                         feature_cols: list, n_trials: int = 100) -> tuple:
    """Tune CatBoost and LightGBM hyperparameters independently using Optuna.

    Optimizes Poisson deviance (negative log-likelihood) on validation set,
    averaged over home + away goals predictions.

    Returns:
        (best_cat_params, best_lgb_params) dicts ready for PoissonGoalsModel.train()
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return None, None

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor, early_stopping

    X_train_clean = X_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val_clean = X_val[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # =========================================================================
    # 1. Tune CatBoost
    # =========================================================================
    logger.info(f"\n[1/2] Tuning CatBoost ({n_trials} trials)...")

    def catboost_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 800),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            'loss_function': 'Poisson',
            'random_seed': 42,
            'verbose': 0,
        }

        total_mae = 0.0
        for y_train, y_val in [(y_home_train, y_home_val), (y_away_train, y_away_val)]:
            model = CatBoostRegressor(**params)
            model.fit(X_train_clean, y_train,
                      eval_set=(X_val_clean, y_val),
                      early_stopping_rounds=50, verbose=False)
            preds = np.maximum(model.predict(X_val_clean), 0.05)
            total_mae += np.mean(np.abs(preds - y_val))

        return total_mae / 2  # Average MAE across home + away

    cb_study = optuna.create_study(direction='minimize')
    cb_study.optimize(catboost_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"  Best CatBoost MAE: {cb_study.best_value:.4f}")
    logger.info(f"  Best params: {cb_study.best_params}")

    best_cat_params = {
        'iterations': cb_study.best_params['iterations'],
        'depth': cb_study.best_params['depth'],
        'learning_rate': cb_study.best_params['learning_rate'],
        'l2_leaf_reg': cb_study.best_params['l2_leaf_reg'],
        'min_data_in_leaf': cb_study.best_params['min_data_in_leaf'],
        'loss_function': 'Poisson',
        'verbose': 0,
    }

    # =========================================================================
    # 2. Tune LightGBM
    # =========================================================================
    logger.info(f"\n[2/2] Tuning LightGBM ({n_trials} trials)...")

    def lightgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'objective': 'poisson',
            'random_state': 42,
            'verbose': -1,
        }

        total_mae = 0.0
        for y_train, y_val in [(y_home_train, y_home_val), (y_away_train, y_away_val)]:
            model = LGBMRegressor(**params)
            model.fit(X_train_clean, y_train,
                      eval_set=[(X_val_clean, y_val)],
                      callbacks=[early_stopping(50, verbose=False)])
            preds = np.maximum(model.predict(X_val_clean), 0.05)
            total_mae += np.mean(np.abs(preds - y_val))

        return total_mae / 2

    lgb_study = optuna.create_study(direction='minimize')
    lgb_study.optimize(lightgbm_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"  Best LightGBM MAE: {lgb_study.best_value:.4f}")
    logger.info(f"  Best params: {lgb_study.best_params}")

    best_lgb_params = {
        'n_estimators': lgb_study.best_params['n_estimators'],
        'max_depth': lgb_study.best_params['max_depth'],
        'learning_rate': lgb_study.best_params['learning_rate'],
        'reg_lambda': lgb_study.best_params['reg_lambda'],
        'reg_alpha': lgb_study.best_params['reg_alpha'],
        'num_leaves': lgb_study.best_params['num_leaves'],
        'min_child_samples': lgb_study.best_params['min_child_samples'],
        'objective': 'poisson',
        'verbose': -1,
    }

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TUNING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"CatBoost best avg MAE:  {cb_study.best_value:.4f}")
    logger.info(f"LightGBM best avg MAE:  {lgb_study.best_value:.4f}")

    return best_cat_params, best_lgb_params


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
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials per model (default: 100)')

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

    # Optionally tune hyperparameters
    best_cat_params = None
    best_lgb_params = None

    if args.tune:
        logger.info("\n" + "=" * 80)
        logger.info(f"HYPERPARAMETER TUNING ({args.trials} trials per model)")
        logger.info("=" * 80)

        best_cat_params, best_lgb_params = tune_hyperparameters(
            X_train, y_home_train, y_away_train,
            X_val, y_home_val, y_away_val,
            feature_cols, n_trials=args.trials,
        )

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING" + (" (with tuned params)" if args.tune else ""))
    logger.info("=" * 80)

    model = PoissonGoalsModel()
    model.train(X_train, y_home_train, y_away_train,
                X_val, y_home_val, y_away_val,
                cat_params=best_cat_params, lgb_params=best_lgb_params)

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
        'dixon_coles_rho': model.rho,
    }

    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'PoissonGoalsModel (CatBoost+LightGBM Poisson regressors)',
        'tuned': args.tune,
        'tuning_trials': args.trials if args.tune else 0,
        'features': len(feature_cols),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'test_metrics': all_metrics,
    }
    if best_cat_params:
        metadata['catboost_params'] = {k: v for k, v in best_cat_params.items() if k != 'verbose'}
    if best_lgb_params:
        metadata['lightgbm_params'] = {k: v for k, v in best_lgb_params.items() if k != 'verbose'}

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
    logger.info(f"DC rho:    {model.rho:.4f}")
    logger.info(f"Home MAE:  {goal_metrics['home_mae']:.4f}")
    logger.info(f"Away MAE:  {goal_metrics['away_mae']:.4f}")
    logger.info(f"O/U 2.5:   {market_metrics['over_2_5_accuracy']*100:.1f}% accuracy")
    logger.info(f"BTTS:      {market_metrics['btts_accuracy']*100:.1f}% accuracy")
    logger.info(f"\nUsage: python3 scripts/predict_live.py --days-ahead 7")


if __name__ == '__main__':
    main()
