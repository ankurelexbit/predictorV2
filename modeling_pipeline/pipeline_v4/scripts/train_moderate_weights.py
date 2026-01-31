#!/usr/bin/env python3
"""
Train Model with Moderate Class Weights
========================================

Class weights:
- Away: 1.3x
- Draw: 1.5x
- Home: 1.0x

Uses 100 Optuna trials with CatBoost.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import catboost as cb
import optuna
import joblib
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load and preprocess data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Convert date
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Sort by date
    df = df.sort_values('match_date').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} samples")
    return df

def get_splits(df):
    """70/15/15 chronological split."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(f"Train: {len(train)} ({len(train)/n:.1%})")
    logger.info(f"Val:   {len(val)} ({len(val)/n:.1%})")
    logger.info(f"Test:  {len(test)} ({len(test)/n:.1%})")

    return train, val, test

def optimize_catboost(X_train, y_train, X_val, y_val, class_weights, n_trials=100):
    """Optimize CatBoost with Optuna."""
    logger.info(f"\nStarting Optuna optimization with {n_trials} trials...")
    logger.info(f"Class weights: {class_weights}")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'loss_function': 'MultiClass',
            'class_weights': list(class_weights.values()),
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }

        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred_proba = model.predict_proba(X_val)
        return log_loss(y_val, y_pred_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\nBest validation log loss: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study.best_params

def train_model(X_train, y_train, params, class_weights):
    """Train final model with best params."""
    params['class_weights'] = list(class_weights.values())
    params['loss_function'] = 'MultiClass'
    params['random_seed'] = 42
    params['verbose'] = False
    params['thread_count'] = -1

    model = cb.CatBoostClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    return model

def evaluate_model(model, X, y, name=""):
    """Evaluate model and return metrics."""
    y_pred_proba = model.predict_proba(X)
    y_pred = np.argmax(y_pred_proba, axis=1)

    loss = log_loss(y, y_pred_proba)
    acc = accuracy_score(y, y_pred)

    # Draw analysis
    draw_mask = (y == 1)
    draw_pred_mask = (y_pred == 1)

    draw_actual = draw_mask.sum()
    draw_predicted = draw_pred_mask.sum()
    draw_correct = (draw_mask & draw_pred_mask).sum()
    draw_accuracy = (draw_correct / draw_actual * 100) if draw_actual > 0 else 0

    # Prediction distribution
    away_pct = (y_pred == 0).sum() / len(y_pred) * 100
    draw_pct = (y_pred == 1).sum() / len(y_pred) * 100
    home_pct = (y_pred == 2).sum() / len(y_pred) * 100

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    metrics = {
        'log_loss': loss,
        'accuracy': acc,
        'draw_accuracy': draw_accuracy,
        'draw_actual': int(draw_actual),
        'draw_predicted': int(draw_predicted),
        'draw_correct': int(draw_correct),
        'away_pct': away_pct,
        'draw_pct': draw_pct,
        'home_pct': home_pct,
        'confusion_matrix': cm.tolist()
    }

    logger.info(f"\n{name} Results:")
    logger.info(f"  Log Loss: {loss:.4f}")
    logger.info(f"  Accuracy: {acc:.1%}")
    logger.info(f"  Draw Accuracy: {draw_accuracy:.2f}%")
    logger.info(f"  Predictions: {away_pct:.1f}% Away, {draw_pct:.1f}% Draw, {home_pct:.1f}% Home")

    return metrics

def apply_calibration(model, X_val, y_val, X_test, y_test):
    """Apply isotonic calibration."""
    from sklearn.calibration import CalibratedClassifierCV

    logger.info("\nApplying isotonic calibration...")

    # Create calibrated model
    calibrated = CalibratedClassifierCV(
        model,
        method='isotonic',
        cv='prefit'
    )
    calibrated.fit(X_val, y_val)

    return calibrated

def main():
    # Configuration
    data_path = 'data/training_data.csv'
    output_dir = Path('models/moderate_weights')
    output_dir.mkdir(parents=True, exist_ok=True)

    class_weights = {0: 1.3, 1: 1.5, 2: 1.0}  # Away/Draw/Home

    logger.info("="*80)
    logger.info("MODERATE WEIGHTS MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Class weights: Away={class_weights[0]}x, Draw={class_weights[1]}x, Home={class_weights[2]}x")

    # Load data
    df = load_data(data_path)

    # Get features
    features_to_exclude = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result', 'target',
        'home_team_name', 'away_team_name', 'state_id'
    ]
    feature_cols = [c for c in df.columns if c not in features_to_exclude]
    logger.info(f"Using {len(feature_cols)} features")

    # Split data
    train_df, val_df, test_df = get_splits(df)

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # Optimize hyperparameters
    best_params = optimize_catboost(X_train, y_train, X_val, y_val, class_weights, n_trials=100)

    # Train final model
    logger.info("\n" + "="*80)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*80)

    model = train_model(X_train, y_train, best_params, class_weights)

    # Evaluate uncalibrated
    val_metrics_uncal = evaluate_model(model, X_val, y_val, "Validation (Uncalibrated)")
    test_metrics_uncal = evaluate_model(model, X_test, y_test, "Test (Uncalibrated)")

    # Apply calibration
    calibrated_model = apply_calibration(model, X_val, y_val, X_test, y_test)

    # Evaluate calibrated
    val_metrics_cal = evaluate_model(calibrated_model, X_val, y_val, "Validation (Calibrated)")
    test_metrics_cal = evaluate_model(calibrated_model, X_test, y_test, "Test (Calibrated)")

    # Save models
    logger.info("\n" + "="*80)
    logger.info("SAVING MODELS")
    logger.info("="*80)

    uncal_path = output_dir / 'model_moderate_uncalibrated.joblib'
    cal_path = output_dir / 'model_moderate_calibrated.joblib'

    joblib.dump(model, uncal_path)
    joblib.dump(calibrated_model, cal_path)

    logger.info(f"Uncalibrated model saved to: {uncal_path}")
    logger.info(f"Calibrated model saved to: {cal_path}")

    # Save results
    results = {
        'config': {
            'class_weights': class_weights,
            'n_trials': 100,
            'best_params': best_params,
            'timestamp': datetime.now().isoformat()
        },
        'validation': {
            'uncalibrated': val_metrics_uncal,
            'calibrated': val_metrics_cal
        },
        'test': {
            'uncalibrated': test_metrics_uncal,
            'calibrated': test_metrics_cal
        }
    }

    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_path}")

    # Comparison summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"\nTest Set Performance:")
    logger.info(f"{'Model':<25} {'Log Loss':<12} {'Draw Acc':<12}")
    logger.info("-"*50)
    logger.info(f"{'Uncalibrated':<25} {test_metrics_uncal['log_loss']:<12.4f} {test_metrics_uncal['draw_accuracy']:<12.2f}%")
    logger.info(f"{'Calibrated':<25} {test_metrics_cal['log_loss']:<12.4f} {test_metrics_cal['draw_accuracy']:<12.2f}%")

    # Compare to previous models
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH PREVIOUS MODELS")
    logger.info("="*80)
    logger.info("\nPrevious Results (from models/final/):")
    logger.info("  No Weights Uncalibrated:     0.9851 log loss, 0.30% draw accuracy")
    logger.info("  Conservative Uncalibrated:   0.9983 log loss, 22.05% draw accuracy")
    logger.info(f"\nNew Moderate Weights Uncalibrated: {test_metrics_uncal['log_loss']:.4f} log loss, {test_metrics_uncal['draw_accuracy']:.2f}% draw accuracy")

    logger.info("\nTraining completed!")

if __name__ == '__main__':
    main()
