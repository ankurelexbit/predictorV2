"""
06 - XGBoost Model
==================

This notebook implements XGBoost for 1X2 prediction.

XGBoost advantages:
1. Handles many features automatically
2. Captures non-linear relationships
3. Built-in feature importance
4. Generally highest raw accuracy

Key considerations:
1. Must use proper time-based splits (no future leakage)
2. Calibration is essential for probability quality
3. Feature selection matters for generalization

Usage:
    python 06_model_xgboost.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
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
    XGBOOST_PARAMS,
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
logger = setup_logger("xgboost_model")
set_random_seed(RANDOM_SEED)


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Core features for XGBoost
FEATURE_COLUMNS = [
    # Elo features (most important)
    'home_elo',
    'away_elo', 
    'elo_diff',
    
    # Form features (5-match window)
    'home_form_5_ppg',
    'away_form_5_ppg',
    'home_form_5_gf',
    'away_form_5_gf',
    'home_form_5_ga',
    'away_form_5_ga',
    
    # Form features (3-match window for recency)
    'home_form_3_ppg',
    'away_form_3_ppg',
    
    # Rest days
    'home_rest_days',
    'away_rest_days',
    'rest_diff',
    
    # Head-to-head
    'h2h_home_wins',
    'h2h_draws',
    'h2h_away_wins',
    'h2h_home_win_rate',
    'h2h_total',
    
    # League position
    'home_position',
    'away_position',
    'position_diff',
    'home_league_points',
    'away_league_points',
]

# Features to potentially add if available
OPTIONAL_FEATURES = [
    # 10-match form
    'home_form_10_ppg',
    'away_form_10_ppg',
    
    # Elo probabilities as features (can help)
    'elo_prob_home',
    'elo_prob_draw', 
    'elo_prob_away',
]


# =============================================================================
# XGBOOST MODEL WRAPPER
# =============================================================================

class XGBoostFootballModel:
    """
    XGBoost model for 1X2 football prediction.
    
    Features:
    - Multiclass classification (softmax)
    - Isotonic calibration
    - Feature importance analysis
    """
    
    def __init__(
        self,
        params: Dict = None,
        feature_columns: List[str] = None
    ):
        self.params = params or XGBOOST_PARAMS.copy()
        self.feature_columns = feature_columns or FEATURE_COLUMNS.copy()
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.calibrators = {}
        
        # State
        self.is_fitted = False
        self.is_calibrated = False
        self.feature_importance_ = None
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Prepare feature matrix.
        
        Args:
            df: DataFrame with features
            fit_scaler: Whether to fit the scaler (True for training)
        
        Returns:
            Feature matrix
        """
        # Select available features
        available_features = [f for f in self.feature_columns if f in df.columns]
        
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        X = df[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isna().any():
                # Fill with median (computed on training data)
                if fit_scaler:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                else:
                    X[col] = X[col].fillna(X[col].median())
        
        # Scale features (optional, XGBoost doesn't strictly need it)
        # But can help with convergence
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, available_features
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Fit XGBoost model.
        
        Args:
            train_df: Training data with features and result_numeric
            val_df: Validation data for early stopping
            early_stopping_rounds: Early stopping patience
            verbose: Print progress
        """
        logger.info(f"Fitting XGBoost on {len(train_df)} training samples")
        
        # Prepare data
        X_train, features_used = self._prepare_features(train_df, fit_scaler=True)
        y_train = train_df['result_numeric'].values.astype(int)
        
        self.feature_columns = features_used  # Update to actually used features
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features_used)
        
        # Validation set
        evals = [(dtrain, 'train')]
        if val_df is not None:
            X_val, _ = self._prepare_features(val_df, fit_scaler=False)
            y_val = val_df['result_numeric'].values.astype(int)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=features_used)
            evals.append((dval, 'val'))
        
        # Train
        params = self.params.copy()
        n_estimators = params.pop('n_estimators', 500)
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if val_df is not None else None,
            verbose_eval=50 if verbose else False
        )
        
        # Store feature importance
        self.feature_importance_ = self.model.get_score(importance_type='gain')
        
        self.is_fitted = True
        logger.info(f"Model fitted with {self.model.best_iteration} rounds")
        
        return self
    
    def predict_proba_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Get raw (uncalibrated) probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X, _ = self._prepare_features(df, fit_scaler=False)
        dtest = xgb.DMatrix(X, feature_names=self.feature_columns)
        
        probs = self.model.predict(dtest)
        
        return probs
    
    def calibrate(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """
        Fit calibration on validation data.
        
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
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
            
            calibrator.fit(class_probs.reshape(-1, 1), class_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
        logger.info(f"Calibration fitted using {method}")
    
    def predict_proba(
        self,
        df: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """
        Get probabilities.
        
        Args:
            df: Features DataFrame
            calibrated: Whether to apply calibration
        
        Returns:
            Array of shape (n_samples, 3)
        """
        probs = self.predict_proba_raw(df)
        
        if calibrated and self.is_calibrated:
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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        
        importance = []
        for feat in self.feature_columns:
            importance.append({
                'feature': feat,
                'importance': self.feature_importance_.get(feat, 0)
            })
        
        df = pd.DataFrame(importance)
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def save(self, path: Path):
        """Save model to file."""
        # Save XGBoost model separately
        model_path = path.with_suffix('.xgb')
        self.model.save_model(str(model_path))
        
        # Save other components
        joblib.dump({
            'params': self.params,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'calibrators': self.calibrators,
            'is_fitted': self.is_fitted,
            'is_calibrated': self.is_calibrated,
            'feature_importance': self.feature_importance_,
            'model_path': str(model_path)
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        data = joblib.load(path)
        
        self.params = data['params']
        self.feature_columns = data['feature_columns']
        self.scaler = data['scaler']
        self.calibrators = data['calibrators']
        self.is_fitted = data['is_fitted']
        self.is_calibrated = data['is_calibrated']
        self.feature_importance_ = data['feature_importance']
        
        # Load XGBoost model
        model_path = data['model_path']
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        logger.info(f"Model loaded from {path}")


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    param_grid: Dict = None,
    n_trials: int = 20
) -> Dict:
    """
    Simple hyperparameter tuning via random search.
    
    Args:
        train_df: Training data
        val_df: Validation data
        param_grid: Parameter ranges to search
        n_trials: Number of random configurations to try
    
    Returns:
        Best parameters and results
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 2.0],
        }
    
    y_val = val_df['result_numeric'].values.astype(int)
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for trial in range(n_trials):
        # Random sample parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': 500,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
        }
        
        for param, values in param_grid.items():
            params[param] = np.random.choice(values)
        
        # Train and evaluate
        model = XGBoostFootballModel(params=params)
        model.fit(train_df, val_df, verbose=False)
        
        val_probs = model.predict_proba(val_df, calibrated=False)
        score = calculate_log_loss(y_val, val_probs)
        
        results.append({
            'trial': trial,
            'params': params.copy(),
            'log_loss': score
        })
        
        if score < best_score:
            best_score = score
            best_params = params.copy()
            logger.info(f"Trial {trial}: New best log_loss = {score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


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


def plot_feature_importance(
    model: XGBoostFootballModel,
    top_n: int = 15
):
    """Plot feature importance."""
    importance_df = model.get_feature_importance()
    
    if importance_df.empty:
        return None
    
    # Take top N
    plot_df = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(
        range(len(plot_df)),
        plot_df['importance'],
        color='steelblue'
    )
    
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)')
    ax.set_title('XGBoost Feature Importance')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_learning_curves(model: XGBoostFootballModel):
    """Plot training curves if available."""
    # This requires access to training history
    # For now, skip if not available
    pass


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Train and evaluate XGBoost model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="XGBoost Model")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROCESSED_DATA_DIR / "features_data_driven.csv"),
        help="Features CSV path"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of tuning trials"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots"
    )
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    features_df = pd.read_csv(args.features)
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    print(f"\nLoaded {len(features_df)} matches")
    
    # Filter to matches with results
    mask = features_df['result_numeric'].notna()
    df = features_df[mask].copy()
    print(f"Matches with results: {len(df)}")
    
    # Split by season
    train_df, val_df, test_df = season_based_split(
        df, 'season',
        TRAIN_SEASONS, VALIDATION_SEASONS, TEST_SEASONS
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} ({TRAIN_SEASONS})")
    print(f"  Validation: {len(val_df)} ({VALIDATION_SEASONS})")
    print(f"  Test: {len(test_df)} ({TEST_SEASONS})")
    
    # Hyperparameter tuning (optional)
    if args.tune:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        
        tune_results = tune_xgboost(train_df, val_df, n_trials=args.n_trials)
        
        print(f"\nBest parameters:")
        for param, value in tune_results['best_params'].items():
            print(f"  {param}: {value}")
        print(f"Best log loss: {tune_results['best_score']:.4f}")
        
        params = tune_results['best_params']
    else:
        params = XGBOOST_PARAMS.copy()
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)
    
    model = XGBoostFootballModel(params=params)
    model.fit(train_df, val_df, verbose=True)
    
    # Feature importance
    importance_df = model.get_feature_importance()
    print("\nTop 10 features by importance:")
    print(importance_df.head(10).to_string(index=False))
    
    # Get predictions
    y_train = train_df['result_numeric'].values.astype(int)
    y_val = val_df['result_numeric'].values.astype(int)
    y_test = test_df['result_numeric'].values.astype(int)
    
    # Evaluate raw model
    print("\n" + "=" * 60)
    print("UNCALIBRATED PREDICTIONS")
    print("=" * 60)
    
    val_probs_raw = model.predict_proba(val_df, calibrated=False)
    evaluate_predictions(y_val, val_probs_raw, "Validation (Raw)")
    
    # Calibrate
    print("\n" + "=" * 60)
    print("CALIBRATING MODEL")
    print("=" * 60)
    
    model.calibrate(val_df, y_val, method='isotonic')
    
    val_probs_cal = model.predict_proba(val_df, calibrated=True)
    evaluate_predictions(y_val, val_probs_cal, "Validation (Calibrated)")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_probs = model.predict_proba(test_df, calibrated=True)
    test_metrics = evaluate_predictions(y_test, test_probs, "Test Set")
    
    # Compare to market
    if 'market_prob_home' in test_df.columns:
        market_mask = test_df['market_prob_home'].notna()
        if market_mask.sum() > 0:
            market_probs = test_df.loc[market_mask, 
                ['market_prob_home', 'market_prob_draw', 'market_prob_away']].values
            test_probs_subset = test_probs[market_mask]
            y_test_subset = y_test[market_mask]
            
            print("\n" + "=" * 60)
            print("COMPARISON TO MARKET")
            print("=" * 60)
            
            model_metrics = evaluate_predictions(y_test_subset, test_probs_subset, "XGBoost")
            market_metrics = evaluate_predictions(y_test_subset, market_probs, "Market")
            
            edge = model_metrics['log_loss'] - market_metrics['log_loss']
            print(f"\nLog Loss Edge: {edge:+.4f}")
    
    # Save model
    model_path = MODELS_DIR / "xgboost_model.joblib"
    model.save(model_path)
    
    # Generate plots
    if args.save_plots:
        plots_dir = MODELS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Feature importance
        fig = plot_feature_importance(model)
        if fig:
            fig.savefig(plots_dir / "xgb_feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nPlots saved to {plots_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"XGBoost Test Log Loss: {test_metrics['log_loss']:.4f}")
    print(f"XGBoost Test Calibration Error: {test_metrics['calibration_error']:.4f}")
    print(f"XGBoost Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"\nModel saved to: {model_path}")
    print("\nNext: Run 07_model_ensemble.py to combine all models")


if __name__ == "__main__":
    main()
