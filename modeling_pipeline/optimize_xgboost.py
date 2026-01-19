"""
Simplified XGBoost Hyperparameter Optimization
===============================================

Focus on optimizing core XGBoost parameters that directly impact performance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR, MODELS_DIR, TRAIN_SEASONS, VALIDATION_SEASONS, TEST_SEASONS, RANDOM_SEED
from utils import setup_logger, set_random_seed, calculate_log_loss, season_based_split

logger = setup_logger("xgboost_optimization")
set_random_seed(RANDOM_SEED)


def objective(trial, train_df, val_df):
    """Objective function for XGBoost optimization."""

    # Define hyperparameter search space
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'gamma': trial.suggest_float('gamma', 0, 0.4),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0, log=True),
    }

    try:
        # Import XGBoost model
        import importlib.util
        spec = importlib.util.spec_from_file_location("xgb_module", "06_model_xgboost.py")
        xgb_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xgb_module)
        XGBoostFootballModel = xgb_module.XGBoostFootballModel

        # Train model
        model = XGBoostFootballModel(params=params)
        model.fit(train_df, val_df, early_stopping_rounds=30, verbose=False)

        # Calibrate and evaluate
        y_val = val_df['target'].values.astype(int)
        model.calibrate(val_df, y_val, method='isotonic')
        val_probs = model.predict_proba(val_df, calibrated=True)

        log_loss = calculate_log_loss(y_val, val_probs)

        return log_loss

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float('inf')


def main():
    print("="*80)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("="*80)

    # Load data
    logger.info("Loading data...")
    features_df = pd.read_csv(PROCESSED_DATA_DIR / "sportmonks_features.csv")
    features_df['date'] = pd.to_datetime(features_df['date'])

    # Filter to matches with results
    mask = features_df['target'].notna()
    df = features_df[mask].copy()

    # Split by season
    train_df, val_df, test_df = season_based_split(
        df, 'season_name',
        TRAIN_SEASONS, VALIDATION_SEASONS, TEST_SEASONS
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} matches")
    print(f"  Validation: {len(val_df)} matches")
    print(f"  Test: {len(test_df)} matches")

    # Create study
    print("\nStarting optimization (50 trials)...")
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_df, val_df),
        n_trials=50,
        show_progress_bar=True
    )

    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    print(f"\nBest validation log loss: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }

    output_file = MODELS_DIR / 'xgboost_optimized_params.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Test with best parameters
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*80)

    import importlib.util
    spec = importlib.util.spec_from_file_location("xgb_module", "06_model_xgboost.py")
    xgb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xgb_module)
    XGBoostFootballModel = xgb_module.XGBoostFootballModel

    best_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        **study.best_params
    }

    model = XGBoostFootballModel(params=best_params)
    model.fit(train_df, val_df, verbose=True)

    # Evaluate on test set
    y_test = test_df['target'].values.astype(int)
    model.calibrate(val_df, val_df['target'].values.astype(int), method='isotonic')
    test_probs = model.predict_proba(test_df, calibrated=True)
    test_loss = calculate_log_loss(y_test, test_probs)
    test_acc = (test_probs.argmax(axis=1) == y_test).mean()

    print(f"\nTest set performance:")
    print(f"  Log Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.1%}")

    # Save optimized model
    model.save(MODELS_DIR / "xgboost_optimized.joblib")
    print(f"\nOptimized model saved to: {MODELS_DIR / 'xgboost_optimized.joblib'}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review the optimized parameters above")
    print("2. Update config.py XGBOOST_PARAMS with these values")
    print("3. Retrain ensemble with: python 07_model_ensemble.py")

    return results


if __name__ == '__main__':
    main()
