#!/usr/bin/env python3
"""
Production Model Training Script
=================================

Trains CatBoost model with Option 3 (Balanced) weights and 162 features.
Configuration matches deployed production model for consistency.

**CURRENT PRODUCTION CONFIG:**
- Model: Option 3 (Balanced)
- Class Weights: Away=1.1, Draw=1.4, Home=1.2
- Version: v4.1

Usage:
    python3 scripts/train_production_model.py \\
        --data data/training_data_with_draw_features.csv \\
        --output models/weight_experiments/option3_balanced_retrain.joblib

For weekly retraining:
    python3 scripts/train_production_model.py \\
        --data data/training_data_latest.csv \\
        --output models/production/option3_$(date +%Y%m%d).joblib
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score
import catboost as cb
import optuna
import joblib
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import production config to stay in sync
try:
    from config import production_config
    USE_PRODUCTION_CONFIG = True
except ImportError:
    USE_PRODUCTION_CONFIG = False
    logger.warning("Could not import production_config, using defaults")

# Production configuration - OPTION 3 (BALANCED)
# These weights match the deployed production model
PRODUCTION_CONFIG = {
    'model_name': 'Option 3: Balanced',
    'version': 'v4.1',
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
    'description': 'Balanced weights for optimal draw performance'
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


def train_model(X_train, y_train, X_val, y_val):
    """Train CatBoost model with hyperparameter optimization."""
    logger.info("\n" + "="*80)
    logger.info("TRAINING MODEL")
    logger.info("="*80)

    logger.info(f"\n🤖 Model Configuration:")
    logger.info(f"   Name: {PRODUCTION_CONFIG['model_name']}")
    logger.info(f"   Version: {PRODUCTION_CONFIG['version']}")
    logger.info(f"   Description: {PRODUCTION_CONFIG['description']}")

    class_weights = PRODUCTION_CONFIG['class_weights']
    logger.info(f"\n⚖️  Class Weights:")
    logger.info(f"   Away (0): {class_weights[0]}")
    logger.info(f"   Draw (1): {class_weights[1]}")
    logger.info(f"   Home (2): {class_weights[2]}")

    logger.info(f"\n🔧 Hyperparameter trials: {PRODUCTION_CONFIG['n_trials']}")

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

    # Train final model with best parameters
    best_params = study.best_params
    best_params['class_weights'] = list(class_weights.values())
    best_params['loss_function'] = 'MultiClass'
    best_params['random_seed'] = PRODUCTION_CONFIG['random_seed']
    best_params['verbose'] = False
    best_params['thread_count'] = -1

    logger.info("\nTraining final model with best parameters...")
    model = cb.CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    return model, best_params


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    # Per-class metrics
    draw_mask = (y_test == 1)
    draw_pred_mask = (y_pred == 1)
    draw_accuracy = (draw_mask & draw_pred_mask).sum() / draw_mask.sum() * 100 if draw_mask.sum() > 0 else 0

    # Prediction distribution
    away_pct = (y_pred == 0).sum() / len(y_pred) * 100
    draw_pct = (y_pred == 1).sum() / len(y_pred) * 100
    home_pct = (y_pred == 2).sum() / len(y_pred) * 100

    logger.info(f"\n📊 Test Set Performance:")
    logger.info(f"   Log Loss: {loss:.4f}")
    logger.info(f"   Overall Accuracy: {acc:.1%}")
    logger.info(f"   Draw Accuracy: {draw_accuracy:.2f}%")
    logger.info(f"   Predictions: {away_pct:.1f}% Away | {draw_pct:.1f}% Draw | {home_pct:.1f}% Home")

    metrics = {
        'log_loss': float(loss),
        'accuracy': float(acc),
        'draw_accuracy': float(draw_accuracy),
        'prediction_distribution': {
            'away_pct': float(away_pct),
            'draw_pct': float(draw_pct),
            'home_pct': float(home_pct)
        }
    }

    return metrics


def save_model(model, output_path: Path, metrics: dict, best_params: dict, feature_count: int):
    """Save model and metadata."""
    logger.info("\n" + "="*80)
    logger.info("SAVING MODEL")
    logger.info("="*80)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, output_path)
    logger.info(f"✅ Model saved: {output_path}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'CatBoost',
        'features': feature_count,
        'class_weights': PRODUCTION_CONFIG['class_weights'],
        'best_params': best_params,
        'test_metrics': metrics,
        'production_config': PRODUCTION_CONFIG
    }

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Metadata saved: {metadata_path}")

    return metadata_path


def main():
    parser = argparse.ArgumentParser(description='Train production CatBoost model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save trained model')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna trials (default: 100)')
    parser.add_argument('--weight-home', type=float, default=1.2,
                       help='Class weight for home wins (default: 1.2 - Option 3)')
    parser.add_argument('--weight-draw', type=float, default=1.4,
                       help='Class weight for draws (default: 1.4 - Option 3)')
    parser.add_argument('--weight-away', type=float, default=1.1,
                       help='Class weight for away wins (default: 1.1 - Option 3)')

    args = parser.parse_args()

    # Override n_trials if specified
    if args.n_trials != 100:
        PRODUCTION_CONFIG['n_trials'] = args.n_trials

    # Override class weights if specified
    PRODUCTION_CONFIG['class_weights'] = {
        0: args.weight_away,  # Away
        1: args.weight_draw,  # Draw
        2: args.weight_home   # Home
    }

    data_path = Path(args.data)
    output_path = Path(args.output)

    logger.info("="*80)
    logger.info("PRODUCTION MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Load data
        df, feature_cols = load_and_prepare_data(data_path)

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, feature_cols)

        # Train model
        model, best_params = train_model(X_train, y_train, X_val, y_val)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Save
        metadata_path = save_model(model, output_path, metrics, best_params, len(feature_cols))

        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Model: {output_path}")
        logger.info(f"Metadata: {metadata_path}")
        logger.info(f"Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.1%}")

        return 0

    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
