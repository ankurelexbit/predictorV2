#!/usr/bin/env python3
"""
Train Production Model for V5 Pipeline
=======================================

Trains a CatBoost + LightGBM ensemble model with optional hyperparameter tuning.

Features:
- Multiclass classification (Home/Draw/Away)
- Balanced class weights
- Optional hyperparameter tuning with Optuna
- Automatic model versioning
- Comprehensive evaluation metrics

Usage:
    # Train with default parameters
    python3 scripts/train_model.py --data data/training_data.csv

    # Train with hyperparameter tuning
    python3 scripts/train_model.py --data data/training_data.csv --tune --trials 50

    # Specify output version
    python3 scripts/train_model.py --data data/training_data.csv --version 1.0.0
"""

import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.production_config import TRAINING_CONFIG, HYPERPARAM_SEARCH, META_COLS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def load_data(data_path: str):
    """Load and prepare training data."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)

    # Create target variable
    df['target'] = df['result'].map({'A': 0, 'D': 1, 'H': 2})

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Chronological split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    logger.info(f"  Train period: {train_df['match_date'].min().date()} to {train_df['match_date'].max().date()}")
    logger.info(f"  Test period:  {test_df['match_date'].min().date()} to {test_df['match_date'].max().date()}")

    return train_df, val_df, test_df, feature_cols


def train_ensemble(X_train, y_train, X_val, y_val, catboost_params=None, lightgbm_params=None):
    """Train CatBoost + LightGBM ensemble."""

    # Use config params if not provided
    cb_params = catboost_params or TRAINING_CONFIG['catboost']
    lgb_params = lightgbm_params or TRAINING_CONFIG['lightgbm']

    logger.info("\nTraining CatBoost...")
    catboost = CatBoostClassifier(
        iterations=cb_params.get('iterations', 500),
        depth=cb_params.get('depth', 6),
        learning_rate=cb_params.get('learning_rate', 0.03),
        l2_leaf_reg=cb_params.get('l2_leaf_reg', 5),
        min_data_in_leaf=cb_params.get('min_data_in_leaf', 1),
        auto_class_weights=cb_params.get('auto_class_weights', 'Balanced'),
        random_seed=RANDOM_STATE,
        verbose=False
    )
    catboost.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=cb_params.get('early_stopping_rounds', 50),
        verbose=False
    )
    logger.info(f"  CatBoost best iteration: {catboost.best_iteration_}")

    logger.info("Training LightGBM...")
    lightgbm = LGBMClassifier(
        n_estimators=lgb_params.get('n_estimators', 500),
        max_depth=lgb_params.get('max_depth', 6),
        learning_rate=lgb_params.get('learning_rate', 0.03),
        reg_lambda=lgb_params.get('reg_lambda', 5),
        reg_alpha=lgb_params.get('reg_alpha', 0),
        num_leaves=lgb_params.get('num_leaves', 31),
        min_child_samples=lgb_params.get('min_child_samples', 20),
        class_weight=lgb_params.get('class_weight', 'balanced'),
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lightgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    logger.info(f"  LightGBM best iteration: {lightgbm.best_iteration_}")

    return catboost, lightgbm


def evaluate_model(catboost, lightgbm, X_test, y_test, y_result):
    """Evaluate ensemble on test set."""
    # Get ensemble predictions
    probs_cat = catboost.predict_proba(X_test)
    probs_lgb = lightgbm.predict_proba(X_test)
    probs_ensemble = (probs_cat + probs_lgb) / 2

    preds = probs_ensemble.argmax(axis=1)

    # Overall metrics
    accuracy = accuracy_score(y_test, preds)
    logloss = log_loss(y_test, probs_ensemble)

    # Per-outcome metrics
    result_map = {'H': 2, 'A': 0, 'D': 1}
    y_numeric = np.array([result_map[r] for r in y_result])

    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'predictions': {
            'home': int((preds == 2).sum()),
            'away': int((preds == 0).sum()),
            'draw': int((preds == 1).sum())
        }
    }

    # Win rate per outcome
    for label, name in [(2, 'home'), (0, 'away'), (1, 'draw')]:
        mask = preds == label
        if mask.sum() > 0:
            correct = (y_test[mask] == label).sum()
            metrics[f'{name}_win_rate'] = float(correct / mask.sum())
        else:
            metrics[f'{name}_win_rate'] = 0.0

    return metrics, probs_ensemble


def evaluate_with_thresholds(probs, y_result, thresholds, odds_filter=None):
    """Evaluate with confidence thresholds and optional odds filter."""
    result_map = {'H': 2, 'A': 0, 'D': 1}
    y_numeric = np.array([result_map[r] for r in y_result])

    results = {
        'home': {'preds': 0, 'correct': 0},
        'away': {'preds': 0, 'correct': 0},
        'draw': {'preds': 0, 'correct': 0}
    }

    for i in range(len(probs)):
        p_away, p_draw, p_home = probs[i]
        true_label = y_numeric[i]

        candidates = []
        if p_home >= thresholds['home']:
            candidates.append(('home', p_home, 2))
        if p_away >= thresholds['away']:
            candidates.append(('away', p_away, 0))
        if p_draw >= thresholds['draw']:
            candidates.append(('draw', p_draw, 1))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            outcome, prob, label = best
            results[outcome]['preds'] += 1
            if label == true_label:
                results[outcome]['correct'] += 1

    return results


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """Tune hyperparameters using Optuna - separate tuning per model."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return None, None

    # Each model gets full trial count
    # =========================================================================
    # 1. Tune CatBoost independently
    # =========================================================================
    logger.info(f"\n[1/2] Tuning CatBoost ({n_trials} trials)...")

    def catboost_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 800),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            'auto_class_weights': 'Balanced',
            'random_seed': RANDOM_STATE,
            'verbose': False
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=50, verbose=False)

        probs = model.predict_proba(X_val)
        return log_loss(y_val, probs)

    cb_study = optuna.create_study(direction='minimize')
    cb_study.optimize(catboost_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"  Best CatBoost log loss: {cb_study.best_value:.4f}")
    logger.info(f"  Best params: {cb_study.best_params}")

    catboost_params = {
        'iterations': cb_study.best_params['iterations'],
        'depth': cb_study.best_params['depth'],
        'learning_rate': cb_study.best_params['learning_rate'],
        'l2_leaf_reg': cb_study.best_params['l2_leaf_reg'],
        'min_data_in_leaf': cb_study.best_params['min_data_in_leaf'],
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 50
    }

    # =========================================================================
    # 2. Tune LightGBM independently
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
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'verbose': -1
        }

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        probs = model.predict_proba(X_val)
        return log_loss(y_val, probs)

    lgb_study = optuna.create_study(direction='minimize')
    lgb_study.optimize(lightgbm_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"  Best LightGBM log loss: {lgb_study.best_value:.4f}")
    logger.info(f"  Best params: {lgb_study.best_params}")

    lightgbm_params = {
        'n_estimators': lgb_study.best_params['n_estimators'],
        'max_depth': lgb_study.best_params['max_depth'],
        'learning_rate': lgb_study.best_params['learning_rate'],
        'reg_lambda': lgb_study.best_params['reg_lambda'],
        'reg_alpha': lgb_study.best_params['reg_alpha'],
        'num_leaves': lgb_study.best_params['num_leaves'],
        'min_child_samples': lgb_study.best_params['min_child_samples'],
        'class_weight': 'balanced'
    }

    # =========================================================================
    # 3. Summary
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("TUNING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"CatBoost best log loss:  {cb_study.best_value:.4f}")
    logger.info(f"LightGBM best log loss:  {lgb_study.best_value:.4f}")

    return catboost_params, lightgbm_params


def get_next_version(model_dir: Path) -> str:
    """Get next semantic version number."""
    existing = list(model_dir.glob("model_v*.joblib"))
    if not existing:
        return "1.0.0"

    versions = []
    for f in existing:
        import re
        match = re.search(r'model_v(\d+)\.(\d+)\.(\d+)\.joblib', f.name)
        if match:
            versions.append(tuple(map(int, match.groups())))

    if versions:
        latest = max(versions)
        return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"

    return "1.0.0"


def save_model(catboost, lightgbm, feature_cols, metrics, model_dir: Path, version: str = None):
    """Save ensemble model and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = get_next_version(model_dir)

    # Save as a dict containing both models
    model_data = {
        'catboost': catboost,
        'lightgbm': lightgbm,
        'feature_cols': feature_cols,
        'model_type': 'ensemble'
    }

    model_path = model_dir / f"model_v{version}.joblib"
    joblib.dump(model_data, model_path)
    logger.info(f"\nModel saved to: {model_path}")

    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'CatBoost+LightGBM Ensemble',
        'features': len(feature_cols),
        'class_weights': 'balanced',
        'test_metrics': metrics
    }

    metadata_path = model_dir / f"model_v{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")

    # Update LATEST
    latest_path = model_dir / "LATEST"
    with open(latest_path, 'w') as f:
        f.write(f"model_v{version}.joblib")

    return version


def main():
    parser = argparse.ArgumentParser(description='Train V5 production model')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=50, help='Number of tuning trials')
    parser.add_argument('--version', help='Model version (default: auto-increment)')
    parser.add_argument('--output-dir', default='models/production', help='Model output directory')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("V5 MODEL TRAINING")
    logger.info("=" * 80)

    # Load data
    train_df, val_df, test_df, feature_cols = load_data(args.data)

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    y_result = test_df['result'].values

    # Optionally tune hyperparameters
    catboost_params = None
    lightgbm_params = None

    if args.tune:
        catboost_params, lightgbm_params = tune_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=args.trials
        )

    # Train ensemble
    catboost, lightgbm = train_ensemble(
        X_train, y_train, X_val, y_val,
        catboost_params, lightgbm_params
    )

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION (All Predictions)")
    logger.info("=" * 80)

    metrics, probs = evaluate_model(catboost, lightgbm, X_test, y_test, y_result)

    logger.info(f"Accuracy: {metrics['accuracy']*100:.1f}%")
    logger.info(f"Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"\nPrediction distribution:")
    logger.info(f"  Home: {metrics['predictions']['home']} ({metrics['predictions']['home']/len(y_test)*100:.1f}%)")
    logger.info(f"  Away: {metrics['predictions']['away']} ({metrics['predictions']['away']/len(y_test)*100:.1f}%)")
    logger.info(f"  Draw: {metrics['predictions']['draw']} ({metrics['predictions']['draw']/len(y_test)*100:.1f}%)")
    logger.info(f"\nWin rates:")
    logger.info(f"  Home: {metrics['home_win_rate']*100:.1f}%")
    logger.info(f"  Away: {metrics['away_win_rate']*100:.1f}%")
    logger.info(f"  Draw: {metrics['draw_win_rate']*100:.1f}%")

    # Evaluate with V5 thresholds (H=0.45, A=0.45, D=0.35)
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION (With V5 Thresholds: H>=45%, A>=45%, D>=35%)")
    logger.info("=" * 80)

    test_weeks = (test_df['match_date'].max() - test_df['match_date'].min()).days / 7
    thresholds = {'home': 0.45, 'away': 0.45, 'draw': 0.35}
    results = evaluate_with_thresholds(probs, y_result, thresholds)

    total_preds = sum(r['preds'] for r in results.values())
    total_correct = sum(r['correct'] for r in results.values())
    overall_wr = total_correct / total_preds * 100 if total_preds > 0 else 0

    logger.info(f"Total bets: {total_preds}")
    logger.info(f"Bets per week: {total_preds/test_weeks:.1f}")
    logger.info(f"Overall Win Rate: {overall_wr:.1f}%")
    logger.info("")

    for outcome in ['home', 'away', 'draw']:
        p = results[outcome]['preds']
        c = results[outcome]['correct']
        wr = c/p*100 if p > 0 else 0
        logger.info(f"  {outcome.upper()}: {p} bets, {c} correct, WR: {wr:.1f}%")

    # Add threshold results to metrics
    metrics['threshold_results'] = {
        'thresholds': thresholds,
        'total_bets': total_preds,
        'bets_per_week': round(total_preds/test_weeks, 1),
        'overall_win_rate': round(overall_wr, 1),
        'per_outcome': {
            outcome: {
                'bets': results[outcome]['preds'],
                'correct': results[outcome]['correct'],
                'win_rate': results[outcome]['correct']/results[outcome]['preds']*100 if results[outcome]['preds'] > 0 else 0
            }
            for outcome in ['home', 'away', 'draw']
        }
    }

    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)

    model_dir = Path(args.output_dir)
    version = save_model(catboost, lightgbm, feature_cols, metrics, model_dir, args.version)

    logger.info("\n" + "=" * 80)
    logger.info("USAGE EXAMPLE")
    logger.info("=" * 80)
    logger.info(f"""
To load and use this model:

```python
import joblib
import numpy as np

# Load model
model_data = joblib.load('{model_dir}/model_v{version}.joblib')
catboost = model_data['catboost']
lightgbm = model_data['lightgbm']
feature_cols = model_data['feature_cols']

# Get predictions for a match
X = match_features[feature_cols].values  # Shape: (1, {len(feature_cols)})
probs_cat = catboost.predict_proba(X)
probs_lgb = lightgbm.predict_proba(X)
probs = (probs_cat + probs_lgb) / 2  # [P(Away), P(Draw), P(Home)]

# Apply V5 thresholds
p_away, p_draw, p_home = probs[0]
if p_home >= 0.45:
    prediction = 'Home'
elif p_away >= 0.45:
    prediction = 'Away'
elif p_draw >= 0.35:
    prediction = 'Draw'
else:
    prediction = 'No Bet'
```
""")


if __name__ == '__main__':
    main()
