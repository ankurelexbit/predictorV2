#!/usr/bin/env python3
"""
Train Unbiased Model + Calibration
===================================

Trains a CatBoost model WITHOUT class weights (equal weights),
then applies isotonic regression calibration.

Hypothesis: Class weights were preventing successful calibration.
Without class weights, calibration might work properly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
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

# Configuration - NO CLASS WEIGHTS
CONFIG = {
    'model_name': 'Unbiased (Equal Weights)',
    'version': 'v4.3',
    'class_weights': None,  # Equal weights
    'n_trials': 50,  # Fewer trials for speed
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    'description': 'Model with equal class weights, then calibrated'
}

FEATURES_TO_EXCLUDE = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]


def load_and_prepare_data(data_path: Path):
    """Load training data."""
    logger.info("="*80)
    logger.info("LOADING TRAINING DATA")
    logger.info("="*80)

    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])

    feature_cols = [c for c in df.columns if c not in FEATURES_TO_EXCLUDE]

    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    df = df.sort_values('match_date').reset_index(drop=True)

    logger.info(f"✅ Loaded {len(df)} samples")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Date range: {df['match_date'].min()} to {df['match_date'].max()}")

    return df, feature_cols


def split_data(df, feature_cols):
    """Split data chronologically."""
    n = len(df)
    train_end = int(n * CONFIG['train_ratio'])
    val_end = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_base_model(X_train, y_train, X_val, y_val):
    """Train base CatBoost model WITHOUT class weights."""
    logger.info("\n" + "="*80)
    logger.info("TRAINING UNBIASED BASE MODEL (NO CLASS WEIGHTS)")
    logger.info("="*80)

    logger.info(f"\nClass Weights: NONE (equal weights for all classes)")
    logger.info(f"Hyperparameter trials: {CONFIG['n_trials']}")

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
            # NO class_weights parameter
            'random_seed': CONFIG['random_seed'],
            'verbose': False,
            'thread_count': -1
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)
        return log_loss(y_val, y_pred_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=CONFIG['n_trials'], show_progress_bar=True)

    logger.info(f"\n✅ Best validation log loss: {study.best_value:.4f}")

    # Train final model
    best_params = study.best_params
    best_params['loss_function'] = 'MultiClass'
    best_params['random_seed'] = CONFIG['random_seed']
    best_params['verbose'] = False
    best_params['thread_count'] = -1

    logger.info("\nTraining final unbiased model...")
    model = cb.CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    return model, best_params


def calibrate_model(base_model, X_val, y_val):
    """Apply isotonic regression calibration."""
    logger.info("\n" + "="*80)
    logger.info("CALIBRATING MODEL")
    logger.info("="*80)

    logger.info("\nApplying isotonic regression calibration...")

    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv='prefit',
        n_jobs=-1
    )

    calibrated_model.fit(X_val, y_val)

    logger.info("✅ Calibration complete")

    return calibrated_model


def evaluate_calibration(base_model, calibrated_model, X_test, y_test):
    """Evaluate both models."""
    logger.info("\n" + "="*80)
    logger.info("EVALUATION")
    logger.info("="*80)

    base_proba = base_model.predict_proba(X_test)
    calib_proba = calibrated_model.predict_proba(X_test)

    base_loss = log_loss(y_test, base_proba)
    calib_loss = log_loss(y_test, calib_proba)

    logger.info(f"\nBase Model (Unbiased):")
    logger.info(f"  Log Loss: {base_loss:.4f}")

    logger.info(f"\nCalibrated Model:")
    logger.info(f"  Log Loss: {calib_loss:.4f}")

    logger.info(f"\nCalibration Impact:")
    logger.info(f"  Change: {calib_loss - base_loss:+.4f}")
    if calib_loss < base_loss:
        logger.info(f"  ✅ Calibration IMPROVED log loss")
    else:
        logger.info(f"  ❌ Calibration WORSENED log loss")

    # Brier scores
    logger.info(f"\nBrier Scores by Class:")
    for i, class_name in enumerate(['Away', 'Draw', 'Home']):
        y_binary = (y_test == i).astype(int)
        base_brier = brier_score_loss(y_binary, base_proba[:, i])
        calib_brier = brier_score_loss(y_binary, calib_proba[:, i])

        logger.info(f"  {class_name}:")
        logger.info(f"    Base:       {base_brier:.4f}")
        logger.info(f"    Calibrated: {calib_brier:.4f}")
        logger.info(f"    Change:     {calib_brier - base_brier:+.4f}")

    return {
        'base_log_loss': base_loss,
        'calibrated_log_loss': calib_loss,
        'improvement': base_loss - calib_loss
    }


def save_models(base_model, calibrated_model, output_dir: Path,
                best_params: dict, metrics: dict, feature_count: int):
    """Save both models."""
    logger.info("\n" + "="*80)
    logger.info("SAVING MODELS")
    logger.info("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save base model
    base_path = output_dir / 'unbiased_base.joblib'
    joblib.dump(base_model, base_path)
    logger.info(f"✅ Base model saved: {base_path}")

    # Save calibrated model
    calib_path = output_dir / 'unbiased_calibrated.joblib'
    joblib.dump(calibrated_model, calib_path)
    logger.info(f"✅ Calibrated model saved: {calib_path}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'CatBoost (No Class Weights) + Isotonic Calibration',
        'features': feature_count,
        'class_weights': None,
        'best_params': best_params,
        'calibration_metrics': metrics,
        'config': CONFIG
    }

    metadata_path = output_dir / 'unbiased_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Metadata saved: {metadata_path}")

    return base_path, calib_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train unbiased + calibrated model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save models')

    args = parser.parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    logger.info("="*80)
    logger.info("UNBIASED MODEL + CALIBRATION TRAINING")
    logger.info("="*80)
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")

    try:
        # Load data
        df, feature_cols = load_and_prepare_data(data_path)

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, feature_cols)

        # Train unbiased model
        base_model, best_params = train_base_model(X_train, y_train, X_val, y_val)

        # Calibrate
        calibrated_model = calibrate_model(base_model, X_val, y_val)

        # Evaluate
        metrics = evaluate_calibration(base_model, calibrated_model, X_test, y_test)

        # Save
        base_path, calib_path = save_models(
            base_model, calibrated_model, output_dir,
            best_params, metrics, len(feature_cols)
        )

        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Base model: {base_path}")
        logger.info(f"Calibrated model: {calib_path}")
        logger.info(f"Calibration improvement: {metrics['improvement']:.4f}")

        return 0

    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
