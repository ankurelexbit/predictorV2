#!/usr/bin/env python3
"""
Test Higher Draw Weights to Improve Draw Accuracy
=================================================

Tests three configurations with higher draw emphasis:
- Moderate-High: 1.3/2.0/1.0
- High: 1.3/2.5/1.0
- Very High: 1.3/3.0/1.0
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
    df['match_date'] = pd.to_datetime(df['match_date'])
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
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
    return train, val, test

def optimize_catboost(X_train, y_train, X_val, y_val, class_weights, n_trials=50):
    """Optimize CatBoost with Optuna (reduced trials for speed)."""
    logger.info(f"\nOptimizing with {n_trials} trials...")
    logger.info(f"Class weights: {class_weights}")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 600),
            'depth': trial.suggest_int('depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
            'border_count': trial.suggest_int('border_count', 50, 150),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 5),
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
    logger.info(f"Best validation log loss: {study.best_value:.4f}")
    return study.best_params

def train_and_evaluate(X_train, y_train, X_test, y_test, params, class_weights, name):
    """Train model and return evaluation metrics."""
    params['class_weights'] = list(class_weights.values())
    params['loss_function'] = 'MultiClass'
    params['random_seed'] = 42
    params['verbose'] = False
    params['thread_count'] = -1

    model = cb.CatBoostClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    # Per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    away_acc = cm[0, 0] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0
    draw_acc = cm[1, 1] / cm[1].sum() * 100 if cm[1].sum() > 0 else 0
    home_acc = cm[2, 2] / cm[2].sum() * 100 if cm[2].sum() > 0 else 0

    # Prediction distribution
    away_pct = (y_pred == 0).sum() / len(y_pred) * 100
    draw_pct = (y_pred == 1).sum() / len(y_pred) * 100
    home_pct = (y_pred == 2).sum() / len(y_pred) * 100

    logger.info(f"\n{name} Results:")
    logger.info(f"  Log Loss: {loss:.4f}")
    logger.info(f"  Overall Accuracy: {acc:.1%}")
    logger.info(f"  Draw Accuracy: {draw_acc:.2f}%")
    logger.info(f"  Predictions: {away_pct:.1f}% Away, {draw_pct:.1f}% Draw, {home_pct:.1f}% Home")

    return {
        'name': name,
        'class_weights': class_weights,
        'log_loss': loss,
        'accuracy': acc,
        'away_accuracy': away_acc,
        'draw_accuracy': draw_acc,
        'home_accuracy': home_acc,
        'away_pct': away_pct,
        'draw_pct': draw_pct,
        'home_pct': home_pct,
        'confusion_matrix': cm.tolist(),
        'model': model
    }

def main():
    # Load data
    data_path = 'data/training_data.csv'
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

    # Test different draw weight configurations
    configs = [
        {'name': 'Moderate-High Draw', 'weights': {0: 1.3, 1: 2.0, 2: 1.0}},
        {'name': 'High Draw', 'weights': {0: 1.3, 1: 2.5, 2: 1.0}},
        {'name': 'Very High Draw', 'weights': {0: 1.3, 1: 3.0, 2: 1.0}},
    ]

    results = []

    for config in configs:
        logger.info("\n" + "="*80)
        logger.info(f"TESTING: {config['name']}")
        logger.info(f"Weights: Away={config['weights'][0]}x, Draw={config['weights'][1]}x, Home={config['weights'][2]}x")
        logger.info("="*80)

        # Optimize hyperparameters
        best_params = optimize_catboost(X_train, y_train, X_val, y_val, config['weights'], n_trials=50)

        # Train and evaluate
        result = train_and_evaluate(X_train, y_train, X_test, y_test, best_params, config['weights'], config['name'])
        results.append(result)

    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{'Configuration':<25} {'Draw Weight':<12} {'Log Loss':<12} {'Draw Acc':<12} {'Draw Pred %':<12}")
    logger.info("-"*80)

    for r in results:
        logger.info(f"{r['name']:<25} {r['class_weights'][1]:<12.1f} {r['log_loss']:<12.4f} "
                   f"{r['draw_accuracy']:<12.2f} {r['draw_pct']:<12.1f}")

    # Save best model
    best_draw_model = max(results, key=lambda x: x['draw_accuracy'])

    output_dir = Path('models/high_draw_weights')
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'model_best_draw.joblib'
    joblib.dump(best_draw_model['model'], model_path)

    # Save results
    results_to_save = []
    for r in results:
        r_copy = r.copy()
        r_copy.pop('model')  # Remove model object for JSON serialization
        results_to_save.append(r_copy)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    logger.info(f"\n✅ Best draw model saved: {model_path}")
    logger.info(f"   Configuration: {best_draw_model['name']}")
    logger.info(f"   Draw Weight: {best_draw_model['class_weights'][1]}x")
    logger.info(f"   Draw Accuracy: {best_draw_model['draw_accuracy']:.2f}%")
    logger.info(f"   Log Loss: {best_draw_model['log_loss']:.4f}")

    # Compare to previous best
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH PREVIOUS MODELS")
    logger.info("="*80)
    logger.info("\nPrevious Best (Conservative 1.2/1.5/1.0):")
    logger.info("  Log Loss: 0.9983")
    logger.info("  Draw Accuracy: 22.05%")
    logger.info("  Draw Predictions: 18.8%")
    logger.info(f"\nNew Best ({best_draw_model['name']}):")
    logger.info(f"  Log Loss: {best_draw_model['log_loss']:.4f}")
    logger.info(f"  Draw Accuracy: {best_draw_model['draw_accuracy']:.2f}%")
    logger.info(f"  Draw Predictions: {best_draw_model['draw_pct']:.1f}%")

    improvement = best_draw_model['draw_accuracy'] - 22.05
    logger.info(f"\n{'✅' if improvement > 0 else '⚠️'} Draw Accuracy Change: {improvement:+.2f}%")

if __name__ == '__main__':
    main()
