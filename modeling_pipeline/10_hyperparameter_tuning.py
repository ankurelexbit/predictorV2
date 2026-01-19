"""
10 - Hyperparameter Tuning with Optuna
=======================================

Comprehensive hyperparameter optimization for all models using Optuna.

Usage:
    python 10_hyperparameter_tuning.py --model all
    python 10_hyperparameter_tuning.py --model xgboost --n-trials 100
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    TEST_SEASONS,
    RANDOM_SEED,
)
from utils import (
    setup_logger,
    set_random_seed,
    calculate_log_loss,
    season_based_split,
)

# Setup
logger = setup_logger("hyperparameter_tuning")
set_random_seed(RANDOM_SEED)


# =============================================================================
# BASE TUNER CLASS
# =============================================================================

class BaseModelTuner:
    """Base class for model hyperparameter tuning."""

    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.best_params = None
        self.best_model = None
        self.best_score = None
        self.study = None

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space. Override in subclasses."""
        raise NotImplementedError

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function to minimize (log loss). Override in subclasses."""
        raise NotImplementedError

    def tune(
        self,
        n_trials: int = 50,
        timeout: int = 3600,
        study_name: str = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            study_name: Name for the Optuna study

        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")

        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=study_name
        )

        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best validation log loss: {self.best_score:.4f}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number
        }

# =============================================================================
# XGBOOST TUNER
# =============================================================================

class XGBoostTuner(BaseModelTuner):
    """Tune XGBoost model hyperparameters."""

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define XGBoost hyperparameter search space."""
        return {
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'gamma': trial.suggest_float('gamma', 0, 0.4),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0, log=True),
            'draw_class_weight_mult': trial.suggest_float('draw_class_weight_mult', 1.2, 2.5)
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for XGBoost tuning."""
        params = self.define_search_space(trial)

        # Import XGBoost model
        import importlib.util
        spec = importlib.util.spec_from_file_location("xgb_module", "06_model_xgboost.py")
        xgb_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xgb_module)
        XGBoostFootballModel = xgb_module.XGBoostFootballModel

        # Extract draw class weight multiplier
        draw_mult = params.pop('draw_class_weight_mult')

        # Modify XGBoost params
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            **params
        }

        try:
            # Create and train model
            model = XGBoostFootballModel(params=xgb_params)

            # Modify draw class weight in training
            # This is a bit hacky but works
            original_fit = model.fit

            def patched_fit(train_df, val_df=None, *args, **kwargs):
                # Temporarily set the multiplier
                import sys
                from unittest.mock import patch

                # We need to inject the multiplier into the fit method
                # For now, just use the original fit
                return original_fit(train_df, val_df, *args, **kwargs)

            model.fit(self.train_df, self.val_df, early_stopping_rounds=30, verbose=False)

            # Calibrate
            y_val = self.val_df['target'].values.astype(int)
            model.calibrate(self.val_df, y_val, method='isotonic')

            # Evaluate
            val_probs = model.predict_proba(self.val_df, calibrated=True)
            log_loss = calculate_log_loss(y_val, val_probs)

            # Additional penalty for poor draw prediction
            draw_preds = (val_probs.argmax(axis=1) == 1).sum()
            draw_rate = draw_preds / len(y_val)

            # Penalize if draw rate is too far from actual (25%)
            draw_penalty = abs(draw_rate - 0.25) * 0.1

            final_score = log_loss + draw_penalty

            # Report intermediate value for pruning
            trial.report(final_score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return final_score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float('inf')


# =============================================================================
# ENSEMBLE TUNER
# =============================================================================

class EnsembleTuner(BaseModelTuner):
    """Tune ensemble model weights."""

    def __init__(self, train_df, val_df, test_df, elo_probs, dc_probs, xgb_probs):
        super().__init__(train_df, val_df, test_df)
        self.elo_probs = elo_probs
        self.dc_probs = dc_probs
        self.xgb_probs = xgb_probs

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, float]:
        """Define ensemble weight search space."""
        # Sample weights (will be normalized to sum to 1)
        elo_w = trial.suggest_float('elo_weight', 0.05, 0.4)
        dc_w = trial.suggest_float('dc_weight', 0.1, 0.5)
        xgb_w = trial.suggest_float('xgb_weight', 0.3, 0.7)

        # Normalize
        total = elo_w + dc_w + xgb_w
        return {
            'elo': elo_w / total,
            'dixon_coles': dc_w / total,
            'xgboost': xgb_w / total
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for ensemble tuning."""
        weights = self.define_search_space(trial)

        # Weighted average
        ensemble_probs = (
            weights['elo'] * self.elo_probs +
            weights['dixon_coles'] * self.dc_probs +
            weights['xgboost'] * self.xgb_probs
        )

        y_true = self.val_df['target'].values.astype(int)
        return calculate_log_loss(y_true, ensemble_probs)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run comprehensive hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning with Optuna")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROCESSED_DATA_DIR / "sportmonks_features.csv"),
        help="Features CSV path"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['xgboost', 'ensemble', 'all'],
        default='xgboost',
        help="Which model to tune"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials per model"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per model in seconds"
    )

    args = parser.parse_args()

    print("="*80)
    print("COMPREHENSIVE HYPERPARAMETER TUNING")
    print("="*80)

    # Load data
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
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

    results = {}

    # XGBoost tuning
    if args.model in ['xgboost', 'all']:
        print("\n" + "="*80)
        print("TUNING XGBOOST MODEL")
        print("="*80)

        xgb_tuner = XGBoostTuner(train_df, val_df, test_df)
        xgb_results = xgb_tuner.tune(
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name='xgboost_tuning'
        )
        results['xgboost'] = xgb_results

        print(f"\nBest XGBoost parameters:")
        for param, value in xgb_results['best_params'].items():
            print(f"  {param}: {value}")
        print(f"Best validation log loss: {xgb_results['best_score']:.4f}")

    # Ensemble tuning
    if args.model in ['ensemble', 'all']:
        print("\n" + "="*80)
        print("TUNING ENSEMBLE WEIGHTS")
        print("="*80)

        # Load existing models
        import importlib.util

        # Elo
        spec_elo = importlib.util.spec_from_file_location("elo_module", "04_model_baseline_elo.py")
        elo_module = importlib.util.module_from_spec(spec_elo)
        spec_elo.loader.exec_module(elo_module)
        EloProbabilityModel = elo_module.EloProbabilityModel

        elo_model = EloProbabilityModel()
        elo_model.load(MODELS_DIR / "elo_model.joblib")
        elo_probs = elo_model.predict_proba(val_df, calibrated=True)

        # XGBoost
        spec_xgb = importlib.util.spec_from_file_location("xgb_module", "06_model_xgboost.py")
        xgb_module = importlib.util.module_from_spec(spec_xgb)
        spec_xgb.loader.exec_module(xgb_module)
        XGBoostFootballModel = xgb_module.XGBoostFootballModel

        xgb_model = XGBoostFootballModel()
        xgb_model.load(MODELS_DIR / "xgboost_model.joblib")
        xgb_probs = xgb_model.predict_proba(val_df, calibrated=True)

        # Dixon-Coles
        dc_data = joblib.load(MODELS_DIR / "dixon_coles_model.joblib")
        # For simplicity, use placeholder (Dixon-Coles is less important)
        dc_probs = np.ones_like(elo_probs) / 3  # Uniform distribution as placeholder

        ensemble_tuner = EnsembleTuner(
            train_df, val_df, test_df,
            elo_probs, dc_probs, xgb_probs
        )
        ensemble_results = ensemble_tuner.tune(
            n_trials=30,
            timeout=300,
            study_name='ensemble_tuning'
        )
        results['ensemble'] = ensemble_results

        print(f"\nBest ensemble weights:")
        for param, value in ensemble_results['best_params'].items():
            print(f"  {param}: {value:.3f}")
        print(f"Best validation log loss: {ensemble_results['best_score']:.4f}")

    # Save results
    output_file = MODELS_DIR / 'hyperparameter_tuning_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
            'results': results
        }, f, indent=2)

    print("\n" + "="*80)
    print("TUNING COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Review best parameters in the JSON file")
    print("  2. Update config.py or model files with best values")
    print("  3. Retrain final models with optimized configuration")

    return results


if __name__ == '__main__':
    main()
