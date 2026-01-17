"""
08 - Comprehensive Evaluation
=============================

This notebook provides thorough evaluation of model performance:

1. Walk-forward validation (time-series proper)
2. Calibration analysis
3. ROI simulation
4. Edge analysis
5. Performance by league/team/time

This is CRITICAL for:
- Understanding model reliability
- Setting confidence thresholds
- Detecting model degradation
- Building user trust

Usage:
    python 08_evaluation.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    HISTORICAL_SEASONS,
    RANDOM_SEED,
)
from utils import (
    setup_logger,
    set_random_seed,
    calculate_log_loss,
    calculate_brier_score,
    calculate_calibration_error,
    print_metrics_table,
)

# Setup
logger = setup_logger("evaluation")
set_random_seed(RANDOM_SEED)


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

class WalkForwardValidator:
    """
    Time-series cross-validation for sports prediction.
    
    NEVER use random splits for sports data - future information
    would leak into training.
    """
    
    def __init__(
        self,
        train_window: int = 3,  # Number of seasons for training
        test_window: int = 1,   # Number of seasons for testing
        step_size: int = 1      # Seasons to move forward each fold
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
    
    def split(
        self,
        df: pd.DataFrame,
        season_column: str = 'season'
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits.
        
        Args:
            df: DataFrame with season column
            season_column: Name of season column
        
        Yields:
            (train_df, test_df) tuples
        """
        # Get unique seasons in order
        seasons = sorted(df[season_column].unique())
        n_seasons = len(seasons)
        
        splits = []
        
        start_idx = 0
        while start_idx + self.train_window + self.test_window <= n_seasons:
            train_seasons = seasons[start_idx:start_idx + self.train_window]
            test_seasons = seasons[start_idx + self.train_window:
                                   start_idx + self.train_window + self.test_window]
            
            train_df = df[df[season_column].isin(train_seasons)]
            test_df = df[df[season_column].isin(test_seasons)]
            
            splits.append({
                'train_seasons': train_seasons,
                'test_seasons': test_seasons,
                'train_df': train_df,
                'test_df': test_df
            })
            
            start_idx += self.step_size
        
        return splits
    
    def evaluate_model(
        self,
        df: pd.DataFrame,
        model_class,
        model_params: Dict = None,
        feature_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Run walk-forward evaluation.
        
        Args:
            df: Full dataset
            model_class: Model class with fit() and predict_proba() methods
            model_params: Parameters for model initialization
            feature_columns: Feature column names
        
        Returns:
            DataFrame with predictions and metrics for each fold
        """
        model_params = model_params or {}
        
        results = []
        all_predictions = []
        
        for i, split in enumerate(self.split(df)):
            logger.info(f"Fold {i+1}: Train on {split['train_seasons']}, Test on {split['test_seasons']}")
            
            train_df = split['train_df']
            test_df = split['test_df']
            
            # Initialize and train model
            model = model_class(**model_params)
            model.fit(train_df)
            
            # Get predictions
            y_test = test_df['result_numeric'].values.astype(int)
            test_probs = model.predict_proba(test_df)
            
            # Calculate metrics
            metrics = {
                'fold': i + 1,
                'train_seasons': str(split['train_seasons']),
                'test_seasons': str(split['test_seasons']),
                'n_train': len(train_df),
                'n_test': len(test_df),
                'log_loss': calculate_log_loss(y_test, test_probs),
                'brier_score': calculate_brier_score(y_test, test_probs),
                'accuracy': np.mean(np.argmax(test_probs, axis=1) == y_test),
            }
            
            cal = calculate_calibration_error(y_test, test_probs)
            metrics['calibration_error'] = cal['overall_ece']
            
            results.append(metrics)
            
            # Store predictions
            pred_df = test_df[['date', 'home_team', 'away_team', 'result_numeric']].copy()
            pred_df['p_home'] = test_probs[:, 0]
            pred_df['p_draw'] = test_probs[:, 1]
            pred_df['p_away'] = test_probs[:, 2]
            pred_df['fold'] = i + 1
            all_predictions.append(pred_df)
        
        results_df = pd.DataFrame(results)
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        return results_df, predictions_df


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def calibration_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Comprehensive calibration analysis.
    
    Returns:
        Dictionary with calibration metrics and data for plotting
    """
    n_samples = len(y_true)
    results = {
        'overall': {},
        'per_class': {},
        'bins': {}
    }
    
    class_names = ['Home', 'Draw', 'Away']
    
    for class_idx, class_name in enumerate(class_names):
        probs = y_pred[:, class_idx]
        actual = (y_true == class_idx).astype(int)
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges[1:-1])
        
        bin_data = []
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            count = mask.sum()
            
            if count > 0:
                mean_pred = probs[mask].mean()
                mean_actual = actual[mask].mean()
                bin_data.append({
                    'bin': bin_idx,
                    'bin_lower': bin_edges[bin_idx],
                    'bin_upper': bin_edges[bin_idx + 1],
                    'count': count,
                    'mean_predicted': mean_pred,
                    'actual_frequency': mean_actual,
                    'calibration_error': abs(mean_pred - mean_actual)
                })
        
        bin_df = pd.DataFrame(bin_data)
        results['bins'][class_name] = bin_df
        
        # ECE (Expected Calibration Error)
        if len(bin_df) > 0:
            ece = (bin_df['count'] / n_samples * bin_df['calibration_error']).sum()
        else:
            ece = 0
        
        # MCE (Maximum Calibration Error)
        mce = bin_df['calibration_error'].max() if len(bin_df) > 0 else 0
        
        results['per_class'][class_name] = {
            'ece': ece,
            'mce': mce,
            'n_bins_used': len(bin_df)
        }
    
    # Overall metrics
    results['overall']['mean_ece'] = np.mean([
        results['per_class'][c]['ece'] for c in class_names
    ])
    results['overall']['max_mce'] = max([
        results['per_class'][c]['mce'] for c in class_names
    ])
    
    return results


def plot_calibration_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Calibration Analysis"
) -> plt.Figure:
    """Create detailed calibration plots."""
    analysis = calibration_analysis(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    class_names = ['Home', 'Draw', 'Away']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Row 1: Calibration curves
    for idx, (class_name, color) in enumerate(zip(class_names, colors)):
        ax = axes[0, idx]
        bin_df = analysis['bins'][class_name]
        
        if len(bin_df) > 0:
            ax.scatter(
                bin_df['mean_predicted'],
                bin_df['actual_frequency'],
                s=bin_df['count'] / 10,
                c=color,
                alpha=0.7,
                label='Model'
            )
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Frequency')
        ax.set_title(f'{class_name} Win (ECE: {analysis["per_class"][class_name]["ece"]:.3f})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Row 2: Reliability diagrams with counts
    for idx, (class_name, color) in enumerate(zip(class_names, colors)):
        ax = axes[1, idx]
        bin_df = analysis['bins'][class_name]
        
        if len(bin_df) > 0:
            # Bar chart of counts per bin
            bin_centers = (bin_df['bin_lower'] + bin_df['bin_upper']) / 2
            ax.bar(bin_centers, bin_df['count'], width=0.08, color=color, alpha=0.7)
        
        ax.set_xlabel('Predicted Probability Bin')
        ax.set_ylabel('Sample Count')
        ax.set_title(f'{class_name} Win - Sample Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


# =============================================================================
# ROI SIMULATION
# =============================================================================

class BettingSimulator:
    """
    Simulate betting performance.
    
    Strategies:
    1. Flat staking on positive EV
    2. Kelly criterion
    3. Confidence-based staking
    """
    
    def __init__(
        self,
        bankroll: float = 1000.0,
        min_edge: float = 0.03,
        max_stake_pct: float = 0.05
    ):
        self.initial_bankroll = bankroll
        self.min_edge = min_edge
        self.max_stake_pct = max_stake_pct
    
    def flat_stake_simulation(
        self,
        predictions_df: pd.DataFrame,
        stake_per_bet: float = 10.0
    ) -> Dict[str, Any]:
        """
        Simulate flat staking strategy.
        
        Args:
            predictions_df: DataFrame with predictions and odds
            stake_per_bet: Fixed stake amount
        """
        results = {
            'bets': [],
            'cumulative_profit': [0],
            'bankroll_history': [self.initial_bankroll]
        }
        
        bankroll = self.initial_bankroll
        total_stake = 0
        total_returns = 0
        
        for _, row in predictions_df.iterrows():
            # Check if odds available
            if 'avg_home_odds' not in row or pd.isna(row.get('avg_home_odds')):
                continue
            
            probs = [row['p_home'], row['p_draw'], row['p_away']]
            odds = [row['avg_home_odds'], row['avg_draw_odds'], row['avg_away_odds']]
            
            # Find positive EV bets
            for outcome_idx in range(3):
                if pd.isna(odds[outcome_idx]):
                    continue
                
                implied_prob = 1 / odds[outcome_idx]
                edge = probs[outcome_idx] - implied_prob
                ev = probs[outcome_idx] * odds[outcome_idx] - 1
                
                if edge >= self.min_edge and ev > 0:
                    # Place bet
                    stake = min(stake_per_bet, bankroll * self.max_stake_pct)
                    
                    if stake <= 0:
                        continue
                    
                    actual_outcome = int(row['result_numeric'])
                    won = actual_outcome == outcome_idx
                    
                    if won:
                        returns = stake * odds[outcome_idx]
                        profit = returns - stake
                    else:
                        returns = 0
                        profit = -stake
                    
                    bankroll += profit
                    total_stake += stake
                    total_returns += returns
                    
                    results['bets'].append({
                        'date': row['date'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'outcome_bet': ['Home', 'Draw', 'Away'][outcome_idx],
                        'prob': probs[outcome_idx],
                        'odds': odds[outcome_idx],
                        'edge': edge,
                        'ev': ev,
                        'stake': stake,
                        'won': won,
                        'profit': profit,
                        'bankroll_after': bankroll
                    })
                    
                    results['cumulative_profit'].append(
                        results['cumulative_profit'][-1] + profit
                    )
                    results['bankroll_history'].append(bankroll)
        
        # Summary statistics
        bets_df = pd.DataFrame(results['bets'])
        
        if len(bets_df) > 0:
            results['summary'] = {
                'total_bets': len(bets_df),
                'winning_bets': bets_df['won'].sum(),
                'win_rate': bets_df['won'].mean(),
                'total_stake': total_stake,
                'total_returns': total_returns,
                'profit': total_returns - total_stake,
                'roi': (total_returns - total_stake) / total_stake if total_stake > 0 else 0,
                'yield': (total_returns - total_stake) / len(bets_df) if len(bets_df) > 0 else 0,
                'final_bankroll': bankroll,
                'bankroll_growth': (bankroll - self.initial_bankroll) / self.initial_bankroll,
                'max_drawdown': self._calculate_max_drawdown(results['bankroll_history']),
                'avg_edge': bets_df['edge'].mean(),
                'avg_odds': bets_df['odds'].mean(),
            }
        else:
            results['summary'] = {'total_bets': 0}
        
        results['bets_df'] = bets_df
        
        return results
    
    def kelly_simulation(
        self,
        predictions_df: pd.DataFrame,
        fraction: float = 0.25  # Fractional Kelly (more conservative)
    ) -> Dict[str, Any]:
        """
        Simulate Kelly criterion betting.
        
        Kelly stake = (p * odds - 1) / (odds - 1) * fraction
        """
        results = {
            'bets': [],
            'bankroll_history': [self.initial_bankroll]
        }
        
        bankroll = self.initial_bankroll
        
        for _, row in predictions_df.iterrows():
            if 'avg_home_odds' not in row or pd.isna(row.get('avg_home_odds')):
                continue
            
            probs = [row['p_home'], row['p_draw'], row['p_away']]
            odds = [row['avg_home_odds'], row['avg_draw_odds'], row['avg_away_odds']]
            
            for outcome_idx in range(3):
                if pd.isna(odds[outcome_idx]):
                    continue
                
                p = probs[outcome_idx]
                o = odds[outcome_idx]
                
                # Kelly formula
                kelly_fraction = (p * o - 1) / (o - 1)
                
                if kelly_fraction > 0:
                    # Apply fractional Kelly and max stake limit
                    stake_pct = min(kelly_fraction * fraction, self.max_stake_pct)
                    stake = bankroll * stake_pct
                    
                    if stake < 1:  # Minimum bet
                        continue
                    
                    actual_outcome = int(row['result_numeric'])
                    won = actual_outcome == outcome_idx
                    
                    if won:
                        profit = stake * (o - 1)
                    else:
                        profit = -stake
                    
                    bankroll += profit
                    
                    results['bets'].append({
                        'date': row['date'],
                        'kelly_fraction': kelly_fraction,
                        'stake_pct': stake_pct,
                        'stake': stake,
                        'won': won,
                        'profit': profit,
                        'bankroll_after': bankroll
                    })
                    
                    results['bankroll_history'].append(bankroll)
        
        results['final_bankroll'] = bankroll
        results['bets_df'] = pd.DataFrame(results['bets'])
        
        return results
    
    def _calculate_max_drawdown(self, bankroll_history: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(bankroll_history) < 2:
            return 0
        
        peak = bankroll_history[0]
        max_dd = 0
        
        for value in bankroll_history[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd


def plot_betting_performance(results: Dict) -> plt.Figure:
    """Plot betting simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cumulative profit
    ax = axes[0, 0]
    ax.plot(results['cumulative_profit'], color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bet Number')
    ax.set_ylabel('Cumulative Profit (units)')
    ax.set_title('Cumulative Profit Over Time')
    ax.grid(True, alpha=0.3)
    
    # Bankroll history
    ax = axes[0, 1]
    ax.plot(results['bankroll_history'], color='green')
    ax.axhline(y=results['bankroll_history'][0], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bet Number')
    ax.set_ylabel('Bankroll')
    ax.set_title('Bankroll Evolution')
    ax.grid(True, alpha=0.3)
    
    if 'bets_df' in results and len(results['bets_df']) > 0:
        bets_df = results['bets_df']
        
        # Profit distribution
        ax = axes[1, 0]
        ax.hist(bets_df['profit'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Profit per Bet')
        ax.set_ylabel('Frequency')
        ax.set_title('Profit Distribution')
        ax.grid(True, alpha=0.3)
        
        # Win rate by edge bucket
        ax = axes[1, 1]
        bets_df['edge_bucket'] = pd.cut(bets_df['edge'], bins=5)
        win_by_edge = bets_df.groupby('edge_bucket')['won'].mean()
        win_by_edge.plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel('Edge Bucket')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate by Edge')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# PERFORMANCE BY SEGMENT
# =============================================================================

def analyze_by_segment(
    predictions_df: pd.DataFrame,
    segment_column: str
) -> pd.DataFrame:
    """
    Analyze performance by segment (league, team, etc.).
    
    Args:
        predictions_df: DataFrame with predictions and results
        segment_column: Column to segment by
    
    Returns:
        Performance metrics by segment
    """
    results = []
    
    for segment, group in predictions_df.groupby(segment_column):
        y_true = group['result_numeric'].values.astype(int)
        y_pred = group[['p_home', 'p_draw', 'p_away']].values
        
        metrics = {
            segment_column: segment,
            'n_matches': len(group),
            'log_loss': calculate_log_loss(y_true, y_pred),
            'accuracy': np.mean(np.argmax(y_pred, axis=1) == y_true),
        }
        
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.sort_values('log_loss').reset_index(drop=True)
    
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run comprehensive evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "--features",
        type=str,
        default=str(PROCESSED_DATA_DIR / "features.csv"),
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
    
    # Filter to completed matches with predictions
    mask = (
        features_df['result_numeric'].notna() &
        features_df['elo_prob_home'].notna()
    )
    df = features_df[mask].copy()
    print(f"Matches with results and predictions: {len(df)}")
    
    # For this evaluation, we'll use Elo predictions as proxy
    # (In practice, load your ensemble predictions)
    df['p_home'] = df['elo_prob_home']
    df['p_draw'] = df['elo_prob_draw']
    df['p_away'] = df['elo_prob_away']
    
    y_true = df['result_numeric'].values.astype(int)
    y_pred = df[['p_home', 'p_draw', 'p_away']].values
    
    # Overall metrics
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)
    
    metrics = {
        'log_loss': calculate_log_loss(y_true, y_pred),
        'brier_score': calculate_brier_score(y_true, y_pred),
        'accuracy': np.mean(np.argmax(y_pred, axis=1) == y_true),
    }
    cal = calculate_calibration_error(y_true, y_pred)
    metrics['calibration_error'] = cal['overall_ece']
    
    print_metrics_table(metrics, "Overall Metrics")
    
    # Calibration analysis
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    
    cal_analysis = calibration_analysis(y_true, y_pred)
    
    for class_name in ['Home', 'Draw', 'Away']:
        class_metrics = cal_analysis['per_class'][class_name]
        print(f"\n{class_name}:")
        print(f"  ECE: {class_metrics['ece']:.4f}")
        print(f"  MCE: {class_metrics['mce']:.4f}")
    
    # Performance by season
    print("\n" + "=" * 60)
    print("PERFORMANCE BY SEASON")
    print("=" * 60)
    
    season_perf = analyze_by_segment(df, 'season')
    print(season_perf.to_string(index=False))
    
    # Performance by league
    if 'league_code' in df.columns:
        print("\n" + "=" * 60)
        print("PERFORMANCE BY LEAGUE")
        print("=" * 60)
        
        league_perf = analyze_by_segment(df, 'league_code')
        print(league_perf.to_string(index=False))
    
    # Betting simulation
    print("\n" + "=" * 60)
    print("BETTING SIMULATION")
    print("=" * 60)
    
    simulator = BettingSimulator(bankroll=1000, min_edge=0.03)
    betting_results = simulator.flat_stake_simulation(df)
    
    if betting_results['summary'].get('total_bets', 0) > 0:
        summary = betting_results['summary']
        print(f"\nFlat Stake Strategy (3% min edge):")
        print(f"  Total bets: {summary['total_bets']}")
        print(f"  Win rate: {summary['win_rate']:.1%}")
        print(f"  Total stake: {summary['total_stake']:.0f} units")
        print(f"  Profit: {summary['profit']:+.1f} units")
        print(f"  ROI: {summary['roi']:+.1%}")
        print(f"  Yield: {summary['yield']:+.2f} units/bet")
        print(f"  Max drawdown: {summary['max_drawdown']:.1%}")
        print(f"  Final bankroll: {summary['final_bankroll']:.0f}")
    else:
        print("\nNo bets placed (check if odds data is available)")
    
    # Save results
    results_dir = MODELS_DIR / "evaluation"
    results_dir.mkdir(exist_ok=True)
    
    # Save plots
    if args.save_plots:
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Calibration plots
        fig = plot_calibration_detailed(y_true, y_pred, "Model Calibration Analysis")
        fig.savefig(plots_dir / "calibration_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Betting performance
        if betting_results['summary'].get('total_bets', 0) > 0:
            fig = plot_betting_performance(betting_results)
            fig.savefig(plots_dir / "betting_simulation.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nPlots saved to {plots_dir}")
    
    # Save metrics
    season_perf.to_csv(results_dir / "performance_by_season.csv", index=False)
    
    if 'league_code' in df.columns:
        league_perf.to_csv(results_dir / "performance_by_league.csv", index=False)
    
    print(f"\nResults saved to {results_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total matches evaluated: {len(df)}")
    print(f"Overall Log Loss: {metrics['log_loss']:.4f}")
    print(f"Overall Calibration Error: {metrics['calibration_error']:.4f}")
    print(f"Overall Accuracy: {metrics['accuracy']:.1%}")
    
    if betting_results['summary'].get('total_bets', 0) > 0:
        print(f"Betting ROI: {betting_results['summary']['roi']:+.1%}")


if __name__ == "__main__":
    main()
