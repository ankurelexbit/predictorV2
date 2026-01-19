"""
04 - Baseline Model: Elo
========================

This notebook implements Elo-based predictions as a baseline.

The Elo system from feature engineering already produces probabilities.
Here we:
1. Evaluate Elo predictions on test data
2. Calibrate the probabilities
3. Compare against market baseline

This is your MINIMUM VIABLE MODEL - ship this first, then improve.

Usage:
    python 04_model_baseline_elo.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
    calculate_brier_score,
    calculate_calibration_error,
    print_metrics_table,
    season_based_split,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Setup
logger = setup_logger("elo_model")
set_random_seed(RANDOM_SEED)


# =============================================================================
# ELO PROBABILITY MODEL
# =============================================================================

class EloProbabilityModel:
    """
    Wrapper for Elo-based probability predictions.
    
    This is essentially already computed in feature engineering.
    This class provides a sklearn-like interface for consistency.
    """
    
    def __init__(self, home_advantage=25):
        self.calibrators = {}  # One per class
        self.is_calibrated = False
        self.home_advantage = home_advantage
    
    def predict_proba(
        self,
        features_df: pd.DataFrame,
        calibrated: bool = True
    ) -> np.ndarray:
        """
        Get 1X2 probabilities from Elo ratings.

        Args:
            features_df: DataFrame with home_elo, away_elo
            calibrated: Whether to apply calibration

        Returns:
            Array of shape (n_samples, 3) with probabilities [away, draw, home]
        """
        # Calculate probabilities from raw Elo ratings
        home_rating = features_df['home_elo'].values + self.home_advantage
        away_rating = features_df['away_elo'].values

        elo_diff = home_rating - away_rating

        # Base expected score using Elo formula
        exp_home = 1 / (1 + 10 ** (-elo_diff / 400))

        # Draw probability estimation
        # Higher when teams are closely matched, lower when big gap
        base_draw_rate = 0.32
        elo_draw_factor = 1 - (np.abs(elo_diff) / 600)
        elo_draw_factor = np.clip(elo_draw_factor, 0.5, 1.5)

        p_draw = base_draw_rate * elo_draw_factor

        # Allocate remaining probability
        remaining = 1 - p_draw
        p_home = remaining * exp_home
        p_away = remaining * (1 - exp_home)

        # Stack into (n_samples, 3) array with order [away, draw, home]
        probs = np.column_stack([p_away, p_draw, p_home])

        if calibrated and self.is_calibrated:
            probs = self._calibrate(probs)

        # Ensure valid probabilities
        probs = np.clip(probs, 0.001, 0.999)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs
    
    def calibrate(
        self,
        features_df: pd.DataFrame,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ):
        """
        Fit calibration on validation data.
        
        Args:
            features_df: Features with Elo probabilities
            y_true: True labels (0, 1, 2)
            method: 'isotonic' or 'sigmoid'
        """
        raw_probs = self.predict_proba(features_df, calibrated=False)
        
        # Calibrate each class separately
        for class_idx in range(3):
            class_probs = raw_probs[:, class_idx]
            class_labels = (y_true == class_idx).astype(int)
            
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                # Platt scaling (logistic regression)
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
            
            calibrator.fit(class_probs.reshape(-1, 1), class_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_calibrated = True
        logger.info(f"Calibration fitted using {method} method")
    
    def _calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        calibrated = np.zeros_like(probs)
        
        for class_idx in range(3):
            class_probs = probs[:, class_idx]
            calibrator = self.calibrators[class_idx]
            
            if isinstance(calibrator, IsotonicRegression):
                calibrated[:, class_idx] = calibrator.predict(class_probs)
            else:
                calibrated[:, class_idx] = calibrator.predict_proba(
                    class_probs.reshape(-1, 1)
                )[:, 1]
        
        # Renormalize
        calibrated = np.clip(calibrated, 0.001, 0.999)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated
    
    def save(self, path: Path):
        """Save model to file."""
        joblib.dump({
            'calibrators': self.calibrators,
            'is_calibrated': self.is_calibrated
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        data = joblib.load(path)
        self.calibrators = data['calibrators']
        self.is_calibrated = data['is_calibrated']
        logger.info(f"Model loaded from {path}")


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate prediction quality.
    
    Args:
        y_true: True labels (0, 1, 2)
        y_pred: Predicted probabilities (n_samples, 3)
        name: Model name for logging
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'log_loss': calculate_log_loss(y_true, y_pred),
        'brier_score': calculate_brier_score(y_true, y_pred),
    }
    
    # Calibration error
    cal_error = calculate_calibration_error(y_true, y_pred)
    metrics['calibration_error'] = cal_error['overall_ece']
    
    # Accuracy (for reference, not primary metric)
    predicted_class = np.argmax(y_pred, axis=1)
    metrics['accuracy'] = np.mean(predicted_class == y_true)
    
    print_metrics_table(metrics, f"{name} Evaluation")
    
    return metrics


def compare_to_market(
    features_df: pd.DataFrame,
    model_probs: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, Dict]:
    """
    Compare model predictions to market implied probabilities.
    
    Args:
        features_df: DataFrame with market probabilities
        model_probs: Model predicted probabilities
        y_true: True labels
    
    Returns:
        Comparison metrics
    """
    # Check if market probs available
    if 'market_prob_home' not in features_df.columns:
        logger.warning("No market probabilities available for comparison")
        return {}
    
    # Filter to rows with market probs
    mask = features_df['market_prob_home'].notna()
    market_probs = features_df.loc[mask, ['market_prob_home', 'market_prob_draw', 'market_prob_away']].values
    model_subset = model_probs[mask]
    y_subset = y_true[mask]
    
    if len(y_subset) == 0:
        return {}
    
    print(f"\nComparing on {len(y_subset)} matches with market odds")
    
    # Evaluate both
    print("\n" + "=" * 60)
    print("MODEL vs MARKET COMPARISON")
    print("=" * 60)
    
    model_metrics = evaluate_predictions(y_subset, model_subset, "Elo Model")
    market_metrics = evaluate_predictions(y_subset, market_probs, "Market")
    
    # Edge calculation
    edge = model_metrics['log_loss'] - market_metrics['log_loss']
    print(f"\nLog Loss Edge: {edge:+.4f} ({'better' if edge < 0 else 'worse'} than market)")
    
    return {
        'model': model_metrics,
        'market': market_metrics,
        'edge': edge
    }


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Plot"
):
    """Plot calibration curve for each class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    for class_idx, (ax, name) in enumerate(zip(axes, class_names)):
        probs = y_pred[:, class_idx]
        actual = (y_true == class_idx).astype(int)
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges[1:-1])
        
        bin_means = []
        bin_trues = []
        bin_counts = []
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 10:  # Minimum samples for stability
                bin_means.append(probs[mask].mean())
                bin_trues.append(actual[mask].mean())
                bin_counts.append(mask.sum())
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.scatter(bin_means, bin_trues, s=100, alpha=0.7, label='Model')
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Frequency')
        ax.set_title(f'{name} Calibration')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_probability_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Probability Distributions"
):
    """Plot distribution of predicted probabilities by outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    for class_idx, (ax, name) in enumerate(zip(axes, class_names)):
        # Split by actual outcome
        for actual_class, color, label in [(class_idx, 'green', 'Correct'), 
                                           (None, 'red', 'Incorrect')]:
            if actual_class is not None:
                mask = y_true == actual_class
            else:
                mask = y_true != class_idx
            
            probs = y_pred[mask, class_idx]
            
            ax.hist(probs, bins=20, alpha=0.5, color=color, label=label, density=True)
        
        ax.set_xlabel(f'P({name})')
        ax.set_ylabel('Density')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def simulate_betting_roi(
    features_df: pd.DataFrame,
    model_probs: np.ndarray,
    y_true: np.ndarray,
    min_edge: float = 0.03,
    stake: float = 1.0
) -> Dict:
    """
    Simulate betting ROI using model predictions.
    
    Args:
        features_df: DataFrame with odds
        model_probs: Model predicted probabilities
        y_true: True labels
        min_edge: Minimum edge to place bet
        stake: Stake per bet
    
    Returns:
        Betting simulation results
    """
    if 'avg_home_odds' not in features_df.columns:
        logger.warning("No odds available for ROI simulation")
        return {}
    
    results = {
        'total_bets': 0,
        'winning_bets': 0,
        'total_stake': 0,
        'total_returns': 0,
        'bets': []
    }
    
    for idx in range(len(y_true)):
        row = features_df.iloc[idx]
        
        # Skip if no odds
        if pd.isna(row.get('avg_home_odds')):
            continue
        
        odds = [row['avg_home_odds'], row['avg_draw_odds'], row['avg_away_odds']]
        probs = model_probs[idx]
        
        # Calculate edge for each outcome
        for outcome_idx in range(3):
            if pd.isna(odds[outcome_idx]):
                continue
            
            implied_prob = 1 / odds[outcome_idx]
            edge = probs[outcome_idx] - implied_prob
            
            if edge >= min_edge:
                # Place bet
                results['total_bets'] += 1
                results['total_stake'] += stake
                
                # Check if won
                if y_true[idx] == outcome_idx:
                    results['winning_bets'] += 1
                    results['total_returns'] += stake * odds[outcome_idx]
                
                results['bets'].append({
                    'outcome': outcome_idx,
                    'prob': probs[outcome_idx],
                    'odds': odds[outcome_idx],
                    'edge': edge,
                    'won': y_true[idx] == outcome_idx
                })
    
    if results['total_bets'] > 0:
        results['win_rate'] = results['winning_bets'] / results['total_bets']
        results['profit'] = results['total_returns'] - results['total_stake']
        results['roi'] = results['profit'] / results['total_stake']
        results['yield'] = results['profit'] / results['total_bets']
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Train and evaluate Elo baseline model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Elo Baseline Model")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROCESSED_DATA_DIR / "features_data_driven.csv"),
        help="Features CSV path"
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
    
    # Filter to matches with results and Elo ratings
    mask = (
        features_df['target'].notna() &
        features_df['home_elo'].notna() &
        features_df['away_elo'].notna()
    )
    df = features_df[mask].copy()
    print(f"Matches with Elo predictions: {len(df)}")
    
    # Split by season
    train_df, val_df, test_df = season_based_split(
        df, 'season_name',
        TRAIN_SEASONS, VALIDATION_SEASONS, TEST_SEASONS
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} ({TRAIN_SEASONS})")
    print(f"  Validation: {len(val_df)} ({VALIDATION_SEASONS})")
    print(f"  Test: {len(test_df)} ({TEST_SEASONS})")
    
    # Initialize model
    model = EloProbabilityModel()
    
    # Get raw predictions
    y_train = train_df['target'].values.astype(int)
    y_val = val_df['target'].values.astype(int)
    y_test = test_df['target'].values.astype(int)
    
    print("\n" + "=" * 60)
    print("UNCALIBRATED ELO PREDICTIONS")
    print("=" * 60)
    
    # Evaluate on validation set (uncalibrated)
    val_probs_raw = model.predict_proba(val_df, calibrated=False)
    evaluate_predictions(y_val, val_probs_raw, "Validation (Raw)")
    
    # Calibrate on validation set
    print("\n" + "=" * 60)
    print("CALIBRATING ON VALIDATION SET")
    print("=" * 60)
    
    model.calibrate(val_df, y_val, method='isotonic')
    
    # Evaluate calibrated predictions on validation
    val_probs_cal = model.predict_proba(val_df, calibrated=True)
    evaluate_predictions(y_val, val_probs_cal, "Validation (Calibrated)")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_probs = model.predict_proba(test_df, calibrated=True)
    test_metrics = evaluate_predictions(y_test, test_probs, "Test Set")
    
    # Compare to market
    comparison = compare_to_market(test_df, test_probs, y_test)
    
    # Simulate betting
    print("\n" + "=" * 60)
    print("BETTING SIMULATION (3% min edge)")
    print("=" * 60)
    
    betting_results = simulate_betting_roi(test_df, test_probs, y_test, min_edge=0.03)
    
    if betting_results.get('total_bets', 0) > 0:
        print(f"Total bets: {betting_results['total_bets']}")
        print(f"Win rate: {betting_results['win_rate']:.1%}")
        print(f"Total stake: {betting_results['total_stake']:.0f} units")
        print(f"Total returns: {betting_results['total_returns']:.1f} units")
        print(f"Profit: {betting_results['profit']:+.1f} units")
        print(f"ROI: {betting_results['roi']:+.1%}")
        print(f"Yield: {betting_results['yield']:+.1%}")
    else:
        print("No bets placed (no odds data or no edges found)")
    
    # Save model
    model_path = MODELS_DIR / "elo_model.joblib"
    model.save(model_path)
    
    # Generate plots
    if args.save_plots:
        plots_dir = MODELS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Calibration plot
        fig = plot_calibration(y_test, test_probs, title="Elo Model Calibration (Test Set)")
        fig.savefig(plots_dir / "elo_calibration.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Probability distributions
        fig = plot_probability_distributions(y_test, test_probs)
        fig.savefig(plots_dir / "elo_prob_dist.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to {plots_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Elo Model Test Log Loss: {test_metrics['log_loss']:.4f}")
    print(f"Elo Model Test Calibration Error: {test_metrics['calibration_error']:.4f}")
    
    if comparison.get('market'):
        print(f"Market Log Loss: {comparison['market']['log_loss']:.4f}")
        print(f"Edge vs Market: {comparison['edge']:+.4f}")
    
    print(f"\nModel saved to: {model_path}")
    print("\nNext steps:")
    print("  1. Run 05_model_dixon_coles.py for Dixon-Coles model")
    print("  2. Run 06_model_xgboost.py for XGBoost model")
    print("  3. Run 07_model_ensemble.py to combine models")


if __name__ == "__main__":
    main()
