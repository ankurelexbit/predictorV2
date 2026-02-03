#!/usr/bin/env python3
"""
Train Calibrated Model for EV Strategy
=======================================

Trains a CatBoost model with emphasis on calibration for better EV-based betting.

Approach:
1. Train base model with Option 3 weights
2. Apply isotonic regression calibration
3. Evaluate calibration quality
4. Test with EV strategy
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import catboost as cb
import optuna
import joblib
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Production config - Option 3 Balanced
PRODUCTION_CONFIG = {
    'model_name': 'Option 3: Calibrated for EV',
    'version': 'v4.2',
    'class_weights': {
        0: 1.1,  # Away
        1: 1.4,  # Draw
        2: 1.2   # Home
    },
    'n_trials': 100,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    'description': 'Calibrated model optimized for EV-based betting'
}

FEATURES_TO_EXCLUDE = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]


def load_and_prepare_data(data_path: Path):
    """Load training data and prepare for training."""
    logger.info("="*80)
    logger.info("LOADING TRAINING DATA")
    logger.info("="*80)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in FEATURES_TO_EXCLUDE]

    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Sort chronologically
    df = df.sort_values('match_date').reset_index(drop=True)

    logger.info(f"✅ Loaded {len(df)} samples")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Date range: {df['match_date'].min()} to {df['match_date'].max()}")

    return df, feature_cols


def split_data(df, feature_cols):
    """Split data chronologically."""
    n = len(df)
    train_end = int(n * PRODUCTION_CONFIG['train_ratio'])
    val_end = int(n * (PRODUCTION_CONFIG['train_ratio'] + PRODUCTION_CONFIG['val_ratio']))

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
    """Train base CatBoost model."""
    logger.info("\n" + "="*80)
    logger.info("TRAINING BASE MODEL")
    logger.info("="*80)

    class_weights = PRODUCTION_CONFIG['class_weights']
    logger.info(f"\nClass Weights: Away={class_weights[0]}, Draw={class_weights[1]}, Home={class_weights[2]}")

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
            'random_seed': PRODUCTION_CONFIG['random_seed'],
            'verbose': False,
            'thread_count': -1
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)
        return log_loss(y_val, y_pred_proba)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=PRODUCTION_CONFIG['n_trials'], show_progress_bar=True)

    logger.info(f"\n✅ Best validation log loss: {study.best_value:.4f}")

    # Train final model
    best_params = study.best_params
    best_params['class_weights'] = list(class_weights.values())
    best_params['loss_function'] = 'MultiClass'
    best_params['random_seed'] = PRODUCTION_CONFIG['random_seed']
    best_params['verbose'] = False
    best_params['thread_count'] = -1

    logger.info("\nTraining final base model...")
    model = cb.CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    return model, best_params


def calibrate_model(base_model, X_val, y_val):
    """Apply isotonic regression calibration."""
    logger.info("\n" + "="*80)
    logger.info("CALIBRATING MODEL")
    logger.info("="*80)

    logger.info("\nApplying isotonic regression calibration...")

    # CatBoost needs to be wrapped for CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',  # Isotonic regression (non-parametric)
        cv='prefit',  # Use pre-trained model
        n_jobs=-1
    )

    calibrated_model.fit(X_val, y_val)

    logger.info("✅ Calibration complete")

    return calibrated_model


def evaluate_calibration(model, X, y, name="Model"):
    """Evaluate model calibration quality."""
    y_pred_proba = model.predict_proba(X)

    # Overall metrics
    loss = log_loss(y, y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y, y_pred)

    logger.info(f"\n{name} Performance:")
    logger.info(f"  Log Loss: {loss:.4f}")
    logger.info(f"  Accuracy: {acc:.1%}")

    # Brier score (calibration metric) for each class
    logger.info(f"\n{name} Calibration (Brier Score):")
    for i, class_name in enumerate(['Away', 'Draw', 'Home']):
        y_binary = (y == i).astype(int)
        brier = brier_score_loss(y_binary, y_pred_proba[:, i])
        logger.info(f"  {class_name}: {brier:.4f} (lower is better)")

    return {
        'log_loss': loss,
        'accuracy': acc,
        'predictions': y_pred_proba
    }


def plot_calibration_curves(base_results, calibrated_results, y_test, output_dir):
    """Plot calibration curves for visual comparison."""
    logger.info("\nGenerating calibration plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ['Away', 'Draw', 'Home']

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        y_binary = (y_test == i).astype(int)

        # Base model
        prob_true_base, prob_pred_base = calibration_curve(
            y_binary,
            base_results['predictions'][:, i],
            n_bins=10,
            strategy='uniform'
        )

        # Calibrated model
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_binary,
            calibrated_results['predictions'][:, i],
            n_bins=10,
            strategy='uniform'
        )

        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(prob_pred_base, prob_true_base, 'o-', label='Base model')
        ax.plot(prob_pred_cal, prob_true_cal, 's-', label='Calibrated model')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.set_title(f'{class_name} Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'calibration_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Calibration plots saved: {plot_path}")
    plt.close()


def save_model(model, output_path: Path, metrics: dict, best_params: dict,
               feature_count: int, calibration_metrics: dict):
    """Save calibrated model and metadata."""
    logger.info("\n" + "="*80)
    logger.info("SAVING MODEL")
    logger.info("="*80)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, output_path)
    logger.info(f"✅ Model saved: {output_path}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'CatBoost + Isotonic Calibration',
        'features': feature_count,
        'class_weights': PRODUCTION_CONFIG['class_weights'],
        'best_params': best_params,
        'test_metrics': metrics,
        'calibration_metrics': calibration_metrics,
        'production_config': PRODUCTION_CONFIG
    }

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Metadata saved: {metadata_path}")

    return metadata_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train calibrated model for EV strategy')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save calibrated model')

    args = parser.parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)

    logger.info("="*80)
    logger.info("CALIBRATED MODEL TRAINING FOR EV STRATEGY")
    logger.info("="*80)
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_path}")

    try:
        # Load data
        df, feature_cols = load_and_prepare_data(data_path)

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, feature_cols)

        # Train base model
        base_model, best_params = train_base_model(X_train, y_train, X_val, y_val)

        # Evaluate base model calibration
        logger.info("\n" + "="*80)
        logger.info("BASE MODEL EVALUATION")
        logger.info("="*80)
        base_results = evaluate_calibration(base_model, X_test, y_test, "Base Model")

        # Calibrate model
        calibrated_model = calibrate_model(base_model, X_val, y_val)

        # Evaluate calibrated model
        logger.info("\n" + "="*80)
        logger.info("CALIBRATED MODEL EVALUATION")
        logger.info("="*80)
        calibrated_results = evaluate_calibration(calibrated_model, X_test, y_test, "Calibrated Model")

        # Compare calibration
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION IMPROVEMENT")
        logger.info("="*80)

        logger.info("\nLog Loss:")
        logger.info(f"  Base: {base_results['log_loss']:.4f}")
        logger.info(f"  Calibrated: {calibrated_results['log_loss']:.4f}")
        logger.info(f"  Improvement: {base_results['log_loss'] - calibrated_results['log_loss']:.4f}")

        # Plot calibration curves
        plot_calibration_curves(base_results, calibrated_results, y_test, output_path.parent)

        # Save calibrated model
        calibration_metrics = {
            'base_log_loss': base_results['log_loss'],
            'calibrated_log_loss': calibrated_results['log_loss'],
            'improvement': base_results['log_loss'] - calibrated_results['log_loss']
        }

        final_metrics = {
            'log_loss': calibrated_results['log_loss'],
            'accuracy': calibrated_results['accuracy']
        }

        metadata_path = save_model(
            calibrated_model,
            output_path,
            final_metrics,
            best_params,
            len(feature_cols),
            calibration_metrics
        )

        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Calibrated Model: {output_path}")
        logger.info(f"Metadata: {metadata_path}")
        logger.info(f"Calibration improvement: {calibration_metrics['improvement']:.4f}")

        return 0

    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
