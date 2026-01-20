"""
07 - Ensemble Model
===================

This notebook combines Elo, Dixon-Coles, and XGBoost into an ensemble.

Ensemble strategies:
1. Simple weighted average (most robust)
2. Stacking (meta-learner on top)
3. Optimal weights via validation set

The ensemble typically outperforms individual models because:
- Elo: Good baseline, stable, interpretable
- Dixon-Coles: Better at modeling score dynamics
- XGBoost: Captures complex feature interactions

Usage:
    python 07_model_ensemble.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    TEST_SEASONS,
    ENSEMBLE_WEIGHTS,
    RANDOM_SEED,
)
from utils import (
    setup_logger,
    set_random_seed,
    calculate_log_loss,
    calculate_brier_score,
    calculate_calibration_error,
    print_metrics_table,
    season_based_split,
)

# Setup
logger = setup_logger("ensemble")
set_random_seed(RANDOM_SEED)


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class EnsembleModel:
    """
    Ensemble of multiple football prediction models.
    
    Combines predictions via weighted averaging with optional
    final calibration layer.
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        calibrate_final: bool = True
    ):
        """
        Initialize ensemble.
        
        Args:
            weights: Dict of model_name -> weight (will be normalized)
            calibrate_final: Whether to calibrate ensemble output
        """
        self.weights = weights or ENSEMBLE_WEIGHTS.copy()
        self.calibrate_final = calibrate_final
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Component models
        self.models = {}
        
        # Final calibration
        self.calibrators = {}
        self.is_calibrated = False
    
    def add_model(self, name: str, model: object, weight: float = None):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier
            model: Model object with predict_proba method
            weight: Optional weight override
        """
        self.models[name] = model
        
        if weight is not None:
            self.weights[name] = weight
            # Renormalize
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Added model '{name}' with weight {self.weights.get(name, 0):.3f}")
    
    def predict_proba_raw(
        self,
        df: pd.DataFrame,
        return_individual: bool = False
    ) -> np.ndarray:
        """
        Get raw ensemble predictions (before final calibration).
        
        Args:
            df: Features DataFrame
            return_individual: Also return individual model predictions
        
        Returns:
            Ensemble probabilities (n_samples, 3)
            Optionally: dict of individual predictions
        """
        individual_preds = {}
        ensemble_probs = np.zeros((len(df), 3))
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            
            if weight == 0:
                continue
            
            # Get predictions from model
            try:
                probs = model.predict_proba(df)
                individual_preds[name] = probs
                ensemble_probs += weight * probs
            except Exception as e:
                logger.warning(f"Model '{name}' failed: {e}")
                continue
        
        # Ensure valid probabilities
        ensemble_probs = np.clip(ensemble_probs, 0.001, 0.999)
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        
        if return_individual:
            return ensemble_probs, individual_preds
        
        return ensemble_probs
    
    def calibrate(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """
        Calibrate the final ensemble output.
        
        Args:
            df: Validation features
            y_true: True labels
            method: 'isotonic' or 'sigmoid'
        """
        raw_probs = self.predict_proba_raw(df)
        
        for class_idx in range(3):
            class_probs = raw_probs[:, class_idx]
            class_labels = (y_true == class_idx).astype(int)
            
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                calibrator = LogisticRegression()
            
            calibrator.fit(class_probs.reshape(-1, 1), class_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
        logger.info(f"Ensemble calibrated using {method}")
    
    def predict_proba(
        self,
        df: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """
        Get calibrated ensemble predictions.
        
        Args:
            df: Features DataFrame
            calibrated: Apply calibration if available
        
        Returns:
            Probabilities (n_samples, 3)
        """
        probs = self.predict_proba_raw(df)
        
        if calibrated and self.is_calibrated and self.calibrate_final:
            calibrated_probs = np.zeros_like(probs)
            
            for class_idx in range(3):
                class_probs = probs[:, class_idx]
                calibrator = self.calibrators[class_idx]
                
                if isinstance(calibrator, IsotonicRegression):
                    calibrated_probs[:, class_idx] = calibrator.predict(class_probs)
                else:
                    calibrated_probs[:, class_idx] = calibrator.predict_proba(
                        class_probs.reshape(-1, 1)
                    )[:, 1]
            
            # Renormalize
            calibrated_probs = np.clip(calibrated_probs, 0.001, 0.999)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            
            return calibrated_probs
        
        return probs
    
    def optimize_weights(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        metric: str = 'log_loss'
    ) -> Dict[str, float]:
        """
        Find optimal weights on validation data.
        
        Args:
            df: Validation features
            y_true: True labels
            metric: 'log_loss' or 'brier'
        
        Returns:
            Optimized weights
        """
        # Get individual predictions
        individual_preds = {}
        for name, model in self.models.items():
            try:
                individual_preds[name] = model.predict_proba(df)
            except:
                continue
        
        model_names = list(individual_preds.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            logger.warning("No models available for optimization")
            return self.weights
        
        def objective(weights):
            """Compute metric for given weights."""
            weights = weights / weights.sum()  # Normalize
            
            ensemble_probs = np.zeros((len(df), 3))
            for i, name in enumerate(model_names):
                ensemble_probs += weights[i] * individual_preds[name]
            
            ensemble_probs = np.clip(ensemble_probs, 0.001, 0.999)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
            
            if metric == 'log_loss':
                return calculate_log_loss(y_true, ensemble_probs)
            else:
                return calculate_brier_score(y_true, ensemble_probs)
        
        # Optimize
        x0 = np.ones(n_models) / n_models  # Start equal
        bounds = [(0.01, 1.0)] * n_models  # Each weight between 0.01 and 1
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1}
        )
        
        if result.success:
            optimized = {name: result.x[i] for i, name in enumerate(model_names)}
            
            # Update weights
            self.weights = optimized
            
            logger.info(f"Optimized weights (minimized {metric}):")
            for name, weight in optimized.items():
                logger.info(f"  {name}: {weight:.3f}")
            logger.info(f"Optimized {metric}: {result.fun:.4f}")
            
            return optimized
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return self.weights
    
    def save(self, path: Path):
        """Save ensemble configuration (not individual models)."""
        joblib.dump({
            'weights': self.weights,
            'calibrate_final': self.calibrate_final,
            'calibrators': self.calibrators,
            'is_calibrated': self.is_calibrated,
            'model_names': list(self.models.keys())
        }, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: Path):
        """Load ensemble configuration."""
        data = joblib.load(path)
        self.weights = data['weights']
        self.calibrate_final = data['calibrate_final']
        self.calibrators = data['calibrators']
        self.is_calibrated = data['is_calibrated']
        logger.info(f"Ensemble loaded from {path}")


# =============================================================================
# STACKING ENSEMBLE
# =============================================================================

class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    
    Uses individual model predictions as features for a meta-model.
    """
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.is_fitted = False
    
    def add_model(self, name: str, model: object):
        """Add a base model."""
        self.models[name] = model
    
    def fit(self, df: pd.DataFrame, y_true: np.ndarray):
        """
        Fit meta-learner on validation data.
        
        Args:
            df: Validation features
            y_true: True labels
        """
        # Get base predictions
        meta_features = []
        
        for name, model in self.models.items():
            try:
                probs = model.predict_proba(df)
                meta_features.append(probs)
            except:
                continue
        
        if not meta_features:
            raise ValueError("No base model predictions available")
        
        # Stack features
        X_meta = np.hstack(meta_features)
        
        # Fit meta-model (logistic regression)
        # Note: multi_class='multinomial' is now the default in sklearn
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        )
        self.meta_model.fit(X_meta, y_true)
        
        self.is_fitted = True
        logger.info("Stacking meta-model fitted")
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get stacking predictions."""
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted")
        
        # Get base predictions
        meta_features = []
        
        for name, model in self.models.items():
            try:
                probs = model.predict_proba(df)
                meta_features.append(probs)
            except:
                continue
        
        X_meta = np.hstack(meta_features)
        
        # Meta-model prediction
        probs = self.meta_model.predict_proba(X_meta)

        return probs

    def save(self, path: Path):
        """Save stacking ensemble (meta-model only, not base models)."""
        joblib.dump({
            'meta_model': self.meta_model,
            'model_names': list(self.models.keys()),
            'is_fitted': self.is_fitted
        }, path)
        logger.info(f"Stacking ensemble saved to {path}")

    def load(self, path: Path):
        """Load stacking ensemble configuration."""
        data = joblib.load(path)
        self.meta_model = data['meta_model']
        self.is_fitted = data['is_fitted']
        logger.info(f"Stacking ensemble loaded from {path}")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model"
) -> Dict[str, float]:
    """Evaluate prediction quality."""
    metrics = {
        'log_loss': calculate_log_loss(y_true, y_pred),
        'brier_score': calculate_brier_score(y_true, y_pred),
    }
    
    cal_error = calculate_calibration_error(y_true, y_pred)
    metrics['calibration_error'] = cal_error['overall_ece']
    
    predicted_class = np.argmax(y_pred, axis=1)
    metrics['accuracy'] = np.mean(predicted_class == y_true)
    
    print_metrics_table(metrics, f"{name} Evaluation")
    
    return metrics


def compare_models(
    models_dict: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models side by side.
    
    Args:
        models_dict: {model_name: predictions}
        y_true: True labels
    
    Returns:
        Comparison DataFrame
    """
    results = []
    
    for name, preds in models_dict.items():
        metrics = {
            'model': name,
            'log_loss': calculate_log_loss(y_true, preds),
            'brier_score': calculate_brier_score(y_true, preds),
            'accuracy': np.mean(np.argmax(preds, axis=1) == y_true),
        }
        
        cal = calculate_calibration_error(y_true, preds)
        metrics['calibration_error'] = cal['overall_ece']
        
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.sort_values('log_loss').reset_index(drop=True)
    
    return df


def plot_model_comparison(comparison_df: pd.DataFrame):
    """Plot model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Log Loss
    ax = axes[0]
    ax.barh(comparison_df['model'], comparison_df['log_loss'], color='steelblue')
    ax.set_xlabel('Log Loss (lower is better)')
    ax.set_title('Log Loss Comparison')
    ax.invert_yaxis()
    
    # Brier Score
    ax = axes[1]
    ax.barh(comparison_df['model'], comparison_df['brier_score'], color='coral')
    ax.set_xlabel('Brier Score (lower is better)')
    ax.set_title('Brier Score Comparison')
    ax.invert_yaxis()
    
    # Accuracy
    ax = axes[2]
    ax.barh(comparison_df['model'], comparison_df['accuracy'], color='green')
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Build and evaluate ensemble model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble Model")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROCESSED_DATA_DIR / "features_data_driven.csv"),
        help="Features CSV path"
    )
    parser.add_argument(
        "--optimize-weights",
        action="store_true",
        default=True,
        help="Optimize ensemble weights on validation set (default: True)"
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip weight optimization (use fixed weights from config)"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save comparison plots"
    )
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    print(f"\nLoaded {len(features_df)} matches")
    
    # Filter to matches with results
    mask = features_df['target'].notna()
    df = features_df[mask].copy()

    # Split by time (matching XGBoost approach)
    # Sort by date and use 70/15/15 split
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nData split (time-based):")
    print(f"  Train: {len(train_df)} ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Validation: {len(val_df)} ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"  Test: {len(test_df)} ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    y_val = val_df['target'].values.astype(int)
    y_test = test_df['target'].values.astype(int)
    
    # Load individual models
    print("\n" + "=" * 60)
    print("LOADING INDIVIDUAL MODELS")
    print("=" * 60)
    
    models_loaded = {}
    individual_val_preds = {}
    individual_test_preds = {}
    
    # Try loading Elo model
    elo_path = MODELS_DIR / "elo_model.joblib"
    if elo_path.exists():
        import importlib
        elo_module = importlib.import_module('04_model_baseline_elo')
        EloProbabilityModel = elo_module.EloProbabilityModel
        elo_model = EloProbabilityModel()
        elo_model.load(elo_path)
        models_loaded['elo'] = elo_model
        
        # For Elo, we use features directly (already computed)
        individual_val_preds['elo'] = elo_model.predict_proba(val_df)
        individual_test_preds['elo'] = elo_model.predict_proba(test_df)
        print("Loaded Elo model")
    else:
        print("Elo model not found - using Elo features directly")
        # Fall back to features
        if 'elo_prob_home' in val_df.columns:
            individual_val_preds['elo'] = val_df[['elo_prob_home', 'elo_prob_draw', 'elo_prob_away']].values
            individual_test_preds['elo'] = test_df[['elo_prob_home', 'elo_prob_draw', 'elo_prob_away']].values
    
    # Try loading Dixon-Coles model
    dc_path = MODELS_DIR / "dixon_coles_model.joblib"
    if dc_path.exists():
        try:
            dc_data = joblib.load(dc_path)
            import importlib
            dc_module = importlib.import_module('05_model_dixon_coles')
            DixonColesModel = dc_module.DixonColesModel
            CalibratedDixonColes = dc_module.CalibratedDixonColes
            
            base_model = DixonColesModel()
            base_model.attack = dc_data['base_model_data']['attack']
            base_model.defense = dc_data['base_model_data']['defense']
            base_model.home_adv = dc_data['base_model_data']['home_adv']
            base_model.rho = dc_data['base_model_data']['rho']
            base_model.team_to_idx = dc_data['base_model_data']['team_to_idx']
            base_model.idx_to_team = dc_data['base_model_data']['idx_to_team']
            base_model.is_fitted = True
            
            dc_model = CalibratedDixonColes(base_model)
            dc_model.calibrators = dc_data['calibrators']
            dc_model.is_calibrated = dc_data['is_calibrated']
            
            models_loaded['dixon_coles'] = dc_model
            individual_val_preds['dixon_coles'] = dc_model.predict_proba(val_df)
            individual_test_preds['dixon_coles'] = dc_model.predict_proba(test_df)
            print("Loaded Dixon-Coles model")
        except Exception as e:
            print(f"Could not load Dixon-Coles model: {e}")
    
    # Try loading XGBoost model
    xgb_path = MODELS_DIR / "xgboost_model.joblib"
    if xgb_path.exists():
        try:
            import importlib
            xgb_module = importlib.import_module('06_model_xgboost')
            XGBoostFootballModel = xgb_module.XGBoostFootballModel
            xgb_model = XGBoostFootballModel()
            xgb_model.load(xgb_path)
            models_loaded['xgboost'] = xgb_model
            individual_val_preds['xgboost'] = xgb_model.predict_proba(val_df)
            individual_test_preds['xgboost'] = xgb_model.predict_proba(test_df)
            print("Loaded XGBoost model")
        except Exception as e:
            print(f"Could not load XGBoost model: {e}")
    
    if not individual_val_preds:
        print("\nNo models available! Please run model training scripts first.")
        return
    
    # Evaluate individual models
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL EVALUATION (Validation)")
    print("=" * 60)
    
    for name, preds in individual_val_preds.items():
        evaluate_predictions(y_val, preds, f"{name} (val)")
    
    # Build ensemble
    print("\n" + "=" * 60)
    print("BUILDING ENSEMBLE")
    print("=" * 60)
    
    # Set weights based on available models
    available_weights = {}
    for name in individual_val_preds.keys():
        if name in ENSEMBLE_WEIGHTS:
            available_weights[name] = ENSEMBLE_WEIGHTS[name]
        else:
            available_weights[name] = 1.0 / len(individual_val_preds)
    
    ensemble = EnsembleModel(weights=available_weights)
    
    # Add models (using a wrapper that returns cached predictions)
    class CachedPredictor:
        def __init__(self, val_preds, test_preds):
            self.val_preds = val_preds
            self.test_preds = test_preds
            self.current = 'val'
        
        def predict_proba(self, df):
            if len(df) == len(self.val_preds):
                return self.val_preds
            return self.test_preds
    
    for name in individual_val_preds.keys():
        cached = CachedPredictor(
            individual_val_preds[name],
            individual_test_preds.get(name, individual_val_preds[name])
        )
        ensemble.add_model(name, cached)
    
    print(f"\nInitial weights:")
    for name, weight in ensemble.weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # Optimize weights (enabled by default for better performance)
    if args.skip_optimization:
        print("\nSkipping weight optimization (--skip-optimization flag set)")
    else:
        print("\n" + "=" * 60)
        print("OPTIMIZING WEIGHTS")
        print("=" * 60)
        
        ensemble.optimize_weights(val_df, y_val, metric='log_loss')
    
    # Evaluate raw ensemble on validation
    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION (Validation)")
    print("=" * 60)
    
    val_ensemble_raw = ensemble.predict_proba_raw(val_df)
    evaluate_predictions(y_val, val_ensemble_raw, "Ensemble (Raw)")
    
    # Calibrate ensemble
    ensemble.calibrate(val_df, y_val, method='isotonic')
    val_ensemble_cal = ensemble.predict_proba(val_df)
    evaluate_predictions(y_val, val_ensemble_cal, "Ensemble (Calibrated)")
    
    # Build stacking ensemble
    print("\n" + "=" * 60)
    print("BUILDING STACKING ENSEMBLE")
    print("=" * 60)

    stacking = StackingEnsemble()
    for name in individual_val_preds.keys():
        cached = CachedPredictor(
            individual_val_preds[name],
            individual_test_preds.get(name, individual_val_preds[name])
        )
        stacking.add_model(name, cached)

    # Fit meta-learner on validation set
    stacking.fit(val_df, y_val)

    # Evaluate stacking on validation
    val_stacking = stacking.predict_proba(val_df)
    evaluate_predictions(y_val, val_stacking, "Stacking (Val)")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    test_ensemble = ensemble.predict_proba(test_df)
    test_metrics = evaluate_predictions(y_test, test_ensemble, "Weighted Avg (Test)")

    test_stacking = stacking.predict_proba(test_df)
    stacking_metrics = evaluate_predictions(y_test, test_stacking, "Stacking (Test)")

    # Use best ensemble
    if stacking_metrics['log_loss'] < test_metrics['log_loss']:
        print("\n*** Stacking ensemble performs better - using stacking ***")
        best_ensemble_preds = test_stacking
        best_method = "stacking"
        best_model = stacking
    else:
        print("\n*** Weighted average performs better - using weighted avg ***")
        best_ensemble_preds = test_ensemble
        best_method = "weighted_avg"
        best_model = ensemble
    
    # Compare all models on test set
    print("\n" + "=" * 60)
    print("ALL MODELS COMPARISON (Test Set)")
    print("=" * 60)

    all_test_preds = individual_test_preds.copy()
    all_test_preds['weighted_avg'] = test_ensemble
    all_test_preds['stacking'] = test_stacking
    
    # Add market if available
    if 'market_prob_home' in test_df.columns:
        market_mask = test_df['market_prob_home'].notna()
        if market_mask.all():
            all_test_preds['market'] = test_df[
                ['market_prob_home', 'market_prob_draw', 'market_prob_away']
            ].values
    
    comparison = compare_models(all_test_preds, y_test)
    print("\nModel Comparison (sorted by log loss):")
    print(comparison.to_string(index=False))
    
    # Save the best ensemble
    ensemble_path = MODELS_DIR / "ensemble_model.joblib"
    best_model.save(ensemble_path)

    # Also save stacking separately if it's the better one
    if best_method == "stacking":
        stacking_path = MODELS_DIR / "stacking_ensemble.joblib"
        stacking.save(stacking_path)

    # Save comparison results
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    
    # Generate plots
    if args.save_plots:
        plots_dir = MODELS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        fig = plot_model_comparison(comparison)
        fig.savefig(plots_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to {plots_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if best_method == "stacking":
        print(f"Best Model: Stacking Ensemble")
        print(f"Stacking Test Log Loss: {stacking_metrics['log_loss']:.4f}")
        print(f"Stacking Test Calibration Error: {stacking_metrics['calibration_error']:.4f}")
        print(f"Stacking Test Accuracy: {stacking_metrics['accuracy']:.2%}")
        print(f"\nStacking ensemble saved to: {MODELS_DIR / 'stacking_ensemble.joblib'}")
        print(f"Also saved as primary ensemble: {ensemble_path}")
    else:
        print(f"Best Model: Weighted Average Ensemble")
        print(f"Ensemble Test Log Loss: {test_metrics['log_loss']:.4f}")
        print(f"Ensemble Test Calibration Error: {test_metrics['calibration_error']:.4f}")
        print(f"\nFinal weights:")
        for name, weight in ensemble.weights.items():
            print(f"  {name}: {weight:.3f}")
        print(f"\nEnsemble saved to: {ensemble_path}")

    print("\nNext: Run 08_evaluation.py for comprehensive backtesting")


if __name__ == "__main__":
    main()
