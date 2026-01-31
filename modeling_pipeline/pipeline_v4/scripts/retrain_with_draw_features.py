#!/usr/bin/env python3
"""
Retrain Model with New Draw Features
=====================================

1. Regenerate training data with draw features
2. Train Conservative model (1.2/1.5/1.0)
3. Compare draw accuracy before vs after
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
from src.features.feature_orchestrator import FeatureOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def regenerate_training_data():
    """Regenerate training data with draw features."""
    logger.info("="*80)
    logger.info("STEP 1: REGENERATING TRAINING DATA WITH DRAW FEATURES")
    logger.info("="*80)

    orchestrator = FeatureOrchestrator(data_dir='data/historical')

    # Check feature count
    feature_count = orchestrator.get_feature_count()
    logger.info(f"\nFeature count: {feature_count['feature_columns']}")
    if feature_count['feature_columns'] < 162:
        logger.error(f"❌ Expected 162 features, got {feature_count['feature_columns']}")
        return None

    # Generate full training data
    df = orchestrator.generate_training_dataset(
        output_file='data/training_data_with_draw_features.csv'
    )

    logger.info(f"\n✅ Generated training data: {len(df)} rows, {feature_count['feature_columns']} features")
    return df

def train_model(df):
    """Train Conservative model with draw features."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAINING MODEL WITH CONSERVATIVE WEIGHTS (1.2/1.5/1.0)")
    logger.info("="*80)

    # Get features
    features_to_exclude = [
        'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
        'match_date', 'home_score', 'away_score', 'result', 'target',
        'home_team_name', 'away_team_name', 'state_id'
    ]
    feature_cols = [c for c in df.columns if c not in features_to_exclude]

    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Sort by date
    df = df.sort_values('match_date').reset_index(drop=True)

    # 70/15/15 split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Features: {len(feature_cols)}")

    # Optimize with Conservative weights
    class_weights = {0: 1.2, 1: 1.5, 2: 1.0}
    logger.info(f"Class weights: {class_weights}")

    logger.info("\nOptimizing hyperparameters (100 trials)...")

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
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    logger.info(f"\nBest validation log loss: {study.best_value:.4f}")

    # Train final model
    best_params = study.best_params
    best_params['class_weights'] = list(class_weights.values())
    best_params['loss_function'] = 'MultiClass'
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    best_params['thread_count'] = -1

    logger.info("\nTraining final model...")
    model = cb.CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compare with old results."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EVALUATION & COMPARISON")
    logger.info("="*80)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    # Draw analysis
    draw_mask = (y_test == 1)
    draw_pred_mask = (y_pred == 1)

    draw_actual = draw_mask.sum()
    draw_predicted = draw_pred_mask.sum()
    draw_correct = (draw_mask & draw_pred_mask).sum()
    draw_accuracy = (draw_correct / draw_actual * 100) if draw_actual > 0 else 0

    # Prediction distribution
    away_pct = (y_pred == 0).sum() / len(y_pred) * 100
    draw_pct = (y_pred == 1).sum() / len(y_pred) * 100
    home_pct = (y_pred == 2).sum() / len(y_pred) * 100

    # Actual distribution
    away_actual_pct = (y_test == 0).sum() / len(y_test) * 100
    draw_actual_pct = (y_test == 1).sum() / len(y_test) * 100
    home_actual_pct = (y_test == 2).sum() / len(y_test) * 100

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n📊 NEW MODEL RESULTS (With Draw Features):")
    logger.info(f"   Log Loss: {loss:.4f}")
    logger.info(f"   Overall Accuracy: {acc:.1%}")
    logger.info(f"   Draw Accuracy: {draw_accuracy:.2f}%")
    logger.info(f"   Predictions: {away_pct:.1f}% Away | {draw_pct:.1f}% Draw | {home_pct:.1f}% Home")
    logger.info(f"   Actual: {away_actual_pct:.1f}% Away | {draw_actual_pct:.1f}% Draw | {home_actual_pct:.1f}% Home")

    logger.info("\n📈 COMPARISON WITH OLD MODEL (Without Draw Features):")
    logger.info("   OLD Conservative Model:")
    logger.info("     Log Loss: 0.9983")
    logger.info("     Draw Accuracy: 22.05%")
    logger.info("     Predictions: 34.0% Away | 18.8% Draw | 47.1% Home")

    logger.info("\n   NEW Conservative Model (with draw features):")
    logger.info(f"     Log Loss: {loss:.4f} ({loss - 0.9983:+.4f})")
    logger.info(f"     Draw Accuracy: {draw_accuracy:.2f}% ({draw_accuracy - 22.05:+.2f}%)")
    logger.info(f"     Predictions: {away_pct:.1f}% Away | {draw_pct:.1f}% Draw | {home_pct:.1f}% Home")

    improvement = draw_accuracy - 22.05
    if improvement > 5:
        logger.info(f"\n✅ SIGNIFICANT IMPROVEMENT: {improvement:+.2f}% draw accuracy gain!")
    elif improvement > 2:
        logger.info(f"\n✅ MODERATE IMPROVEMENT: {improvement:+.2f}% draw accuracy gain")
    else:
        logger.info(f"\n⚠️ LIMITED IMPROVEMENT: {improvement:+.2f}% draw accuracy change")

    # Confusion matrix
    logger.info("\n📋 Confusion Matrix:")
    logger.info("                Predicted")
    logger.info("             Away   Draw   Home")
    logger.info(f"Actual Away:  {cm[0][0]:4d}   {cm[0][1]:4d}   {cm[0][2]:4d}")
    logger.info(f"Actual Draw:  {cm[1][0]:4d}   {cm[1][1]:4d}   {cm[1][2]:4d}")
    logger.info(f"Actual Home:  {cm[2][0]:4d}   {cm[2][1]:4d}   {cm[2][2]:4d}")

    # Save model
    output_dir = Path('models/with_draw_features')
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'conservative_with_draw_features.joblib'
    joblib.dump(model, model_path)
    logger.info(f"\n💾 Model saved: {model_path}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'features': 162,
        'class_weights': {'away': 1.2, 'draw': 1.5, 'home': 1.0},
        'test_metrics': {
            'log_loss': float(loss),
            'accuracy': float(acc),
            'draw_accuracy': float(draw_accuracy),
            'away_pct': float(away_pct),
            'draw_pct': float(draw_pct),
            'home_pct': float(home_pct),
            'confusion_matrix': cm.tolist()
        },
        'comparison': {
            'old_log_loss': 0.9983,
            'old_draw_accuracy': 22.05,
            'improvement_log_loss': float(loss - 0.9983),
            'improvement_draw_accuracy': float(draw_accuracy - 22.05)
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved: {output_dir / 'results.json'}")

    return results

def main():
    logger.info("="*80)
    logger.info("RETRAIN MODEL WITH DRAW-SPECIFIC FEATURES")
    logger.info("="*80)

    # Step 1: Regenerate training data
    df = regenerate_training_data()
    if df is None:
        return

    # Step 2: Train model
    model, X_test, y_test = train_model(df)

    # Step 3: Evaluate
    results = evaluate_model(model, X_test, y_test)

    logger.info("\n" + "="*80)
    logger.info("✅ RETRAINING COMPLETE!")
    logger.info("="*80)

if __name__ == '__main__':
    main()
