"""
Optimize Betting Strategy Thresholds
=====================================

Use historical data to find optimal thresholds for the betting strategy.

This script:
1. Loads historical predictions with actual results
2. Tests different threshold combinations
3. Finds the combination that maximizes ROI
4. Generates recommendations for optimal thresholds

Usage:
    python optimize_betting_thresholds.py --data recent_predictions_20260119_164742.csv
    python optimize_betting_thresholds.py --n-trials 100
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple
import json
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logger

logger = setup_logger("optimize_thresholds")


def load_historical_predictions(file_path: str) -> pd.DataFrame:
    """Load historical predictions with actual results."""
    df = pd.read_csv(file_path)

    # Map target to outcome names
    if 'target' in df.columns:
        outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        df['actual_outcome'] = df['target'].map(outcome_map)

    required_cols = ['home_prob', 'draw_prob', 'away_prob', 'actual_outcome']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def evaluate_thresholds(
    df: pd.DataFrame,
    away_min: float,
    draw_threshold: float,
    home_min: float,
    kelly_fraction: float = 0.25,
    initial_bankroll: float = 1000.0
) -> Dict:
    """
    Evaluate betting strategy with given thresholds.

    Args:
        df: DataFrame with predictions and actual outcomes
        away_min: Minimum away win probability to bet
        draw_threshold: Maximum difference between home/away to bet draw
        home_min: Minimum home win probability to bet
        kelly_fraction: Kelly Criterion fraction
        initial_bankroll: Starting bankroll

    Returns:
        Dictionary with performance metrics
    """
    bankroll = initial_bankroll
    bets = []

    for _, row in df.iterrows():
        home_prob = row['home_prob']
        draw_prob = row['draw_prob']
        away_prob = row['away_prob']
        actual = row['actual_outcome']

        # Calculate fair odds
        home_odds = 1 / home_prob if home_prob > 0 else 999
        draw_odds = 1 / draw_prob if draw_prob > 0 else 999
        away_odds = 1 / away_prob if away_prob > 0 else 999

        # Apply betting rules
        placed_bets = []

        # Rule 1: Bet away wins
        if away_prob >= away_min:
            kelly_pct = (away_odds * away_prob - 1) / (away_odds - 1)
            kelly_pct = kelly_pct * kelly_fraction
            kelly_pct = max(0, min(kelly_pct, 0.05))
            stake = max(1.0, kelly_pct * bankroll)

            if stake >= 1.0:
                placed_bets.append({
                    'outcome': 'Away Win',
                    'prob': away_prob,
                    'odds': away_odds,
                    'stake': stake
                })

        # Rule 2: Bet draws when close
        prob_diff = abs(home_prob - away_prob)
        if prob_diff < draw_threshold:
            kelly_pct = (draw_odds * draw_prob - 1) / (draw_odds - 1)
            kelly_pct = kelly_pct * kelly_fraction
            kelly_pct = max(0, min(kelly_pct, 0.05))
            stake = max(1.0, kelly_pct * bankroll)

            if stake >= 1.0:
                placed_bets.append({
                    'outcome': 'Draw',
                    'prob': draw_prob,
                    'odds': draw_odds,
                    'stake': stake
                })

        # Rule 3: Bet high confidence home wins
        if home_prob >= home_min:
            kelly_pct = (home_odds * home_prob - 1) / (home_odds - 1)
            kelly_pct = kelly_pct * kelly_fraction
            kelly_pct = max(0, min(kelly_pct, 0.05))
            stake = max(1.0, kelly_pct * bankroll)

            if stake >= 1.0:
                placed_bets.append({
                    'outcome': 'Home Win',
                    'prob': home_prob,
                    'odds': home_odds,
                    'stake': stake
                })

        # Process bets
        for bet in placed_bets:
            profit = bet['stake'] * (bet['odds'] - 1) if bet['outcome'] == actual else -bet['stake']

            bets.append({
                'outcome': bet['outcome'],
                'actual': actual,
                'stake': bet['stake'],
                'odds': bet['odds'],
                'profit': profit,
                'won': bet['outcome'] == actual
            })

            bankroll += profit if bet['outcome'] == actual else -bet['stake']

    if len(bets) == 0:
        return {
            'total_bets': 0,
            'roi': 0,
            'net_profit': 0,
            'win_rate': 0,
            'total_staked': 0,
            'final_bankroll': initial_bankroll
        }

    bets_df = pd.DataFrame(bets)
    total_staked = bets_df['stake'].sum()
    net_profit = bets_df['profit'].sum()
    wins = bets_df['won'].sum()

    # Calculate per-outcome metrics
    outcome_metrics = {}
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        outcome_bets = bets_df[bets_df['outcome'] == outcome]
        if len(outcome_bets) > 0:
            outcome_metrics[outcome] = {
                'bets': len(outcome_bets),
                'wins': outcome_bets['won'].sum(),
                'win_rate': outcome_bets['won'].mean(),
                'profit': outcome_bets['profit'].sum(),
                'roi': outcome_bets['profit'].sum() / outcome_bets['stake'].sum() if outcome_bets['stake'].sum() > 0 else 0
            }
        else:
            outcome_metrics[outcome] = {
                'bets': 0,
                'wins': 0,
                'win_rate': 0,
                'profit': 0,
                'roi': 0
            }

    return {
        'total_bets': len(bets),
        'wins': wins,
        'win_rate': wins / len(bets),
        'total_staked': total_staked,
        'net_profit': net_profit,
        'roi': net_profit / total_staked if total_staked > 0 else 0,
        'final_bankroll': bankroll,
        'bankroll_change': (bankroll - initial_bankroll) / initial_bankroll,
        'outcome_metrics': outcome_metrics
    }


def grid_search_optimization(
    df: pd.DataFrame,
    away_min_range: Tuple[float, float, float] = (0.25, 0.50, 0.05),
    draw_threshold_range: Tuple[float, float, float] = (0.05, 0.25, 0.05),
    home_min_range: Tuple[float, float, float] = (0.45, 0.70, 0.05)
) -> List[Dict]:
    """
    Grid search over threshold combinations.

    Args:
        df: Historical predictions
        away_min_range: (min, max, step) for away threshold
        draw_threshold_range: (min, max, step) for draw threshold
        home_min_range: (min, max, step) for home threshold

    Returns:
        List of results sorted by ROI
    """
    logger.info("Starting grid search optimization...")

    # Generate parameter grid
    away_values = np.arange(*away_min_range)
    draw_values = np.arange(*draw_threshold_range)
    home_values = np.arange(*home_min_range)

    total_combinations = len(away_values) * len(draw_values) * len(home_values)
    logger.info(f"Testing {total_combinations} threshold combinations...")

    results = []

    for i, (away_min, draw_threshold, home_min) in enumerate(product(away_values, draw_values, home_values)):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{total_combinations}")

        metrics = evaluate_thresholds(df, away_min, draw_threshold, home_min)

        # Only consider combinations with reasonable number of bets
        if metrics['total_bets'] >= 10:  # At least 10 bets
            results.append({
                'away_min': float(away_min),
                'draw_threshold': float(draw_threshold),
                'home_min': float(home_min),
                **metrics
            })

    # Sort by ROI
    results.sort(key=lambda x: x['roi'], reverse=True)

    logger.info(f"Grid search complete. Found {len(results)} valid combinations.")
    return results


def random_search_optimization(
    df: pd.DataFrame,
    n_trials: int = 200,
    away_min_bounds: Tuple[float, float] = (0.25, 0.50),
    draw_threshold_bounds: Tuple[float, float] = (0.05, 0.25),
    home_min_bounds: Tuple[float, float] = (0.45, 0.70)
) -> List[Dict]:
    """
    Random search optimization (faster for large spaces).

    Args:
        df: Historical predictions
        n_trials: Number of random trials
        away_min_bounds: (min, max) for away threshold
        draw_threshold_bounds: (min, max) for draw threshold
        home_min_bounds: (min, max) for home threshold

    Returns:
        List of results sorted by ROI
    """
    logger.info(f"Starting random search with {n_trials} trials...")

    results = []

    for i in range(n_trials):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{n_trials}")

        # Random sample
        away_min = np.random.uniform(*away_min_bounds)
        draw_threshold = np.random.uniform(*draw_threshold_bounds)
        home_min = np.random.uniform(*home_min_bounds)

        metrics = evaluate_thresholds(df, away_min, draw_threshold, home_min)

        if metrics['total_bets'] >= 10:
            results.append({
                'away_min': float(away_min),
                'draw_threshold': float(draw_threshold),
                'home_min': float(home_min),
                **metrics
            })

    results.sort(key=lambda x: x['roi'], reverse=True)

    logger.info(f"Random search complete. Found {len(results)} valid combinations.")
    return results


def analyze_current_thresholds(df: pd.DataFrame) -> Dict:
    """Evaluate current default thresholds."""
    logger.info("Evaluating current thresholds...")

    current = {
        'away_min': 0.35,
        'draw_threshold': 0.15,
        'home_min': 0.55
    }

    metrics = evaluate_thresholds(df, **current)

    return {
        'thresholds': current,
        'metrics': metrics
    }


def generate_report(
    current_results: Dict,
    optimized_results: List[Dict],
    df: pd.DataFrame,
    output_file: Path
):
    """Generate optimization report."""

    best = optimized_results[0] if optimized_results else None

    lines = []
    lines.append("=" * 80)
    lines.append("BETTING STRATEGY THRESHOLD OPTIMIZATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Dataset: {len(df)} matches")
    lines.append(f"Optimization trials: {len(optimized_results)}")
    lines.append("")

    # Current thresholds
    lines.append("=" * 80)
    lines.append("CURRENT THRESHOLDS")
    lines.append("=" * 80)
    lines.append("")
    curr_thresh = current_results['thresholds']
    curr_metrics = current_results['metrics']

    lines.append(f"Thresholds:")
    lines.append(f"  Away Win Min:      {curr_thresh['away_min']:.2f}")
    lines.append(f"  Draw Close Diff:   {curr_thresh['draw_threshold']:.2f}")
    lines.append(f"  Home Win Min:      {curr_thresh['home_min']:.2f}")
    lines.append("")
    lines.append(f"Performance:")
    lines.append(f"  Total Bets:        {curr_metrics['total_bets']}")
    lines.append(f"  Win Rate:          {curr_metrics['win_rate']*100:.1f}%")
    lines.append(f"  ROI:               {curr_metrics['roi']*100:+.2f}%")
    lines.append(f"  Net Profit:        ${curr_metrics['net_profit']:+.2f}")
    lines.append("")

    # Performance by outcome
    lines.append("By Outcome:")
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        om = curr_metrics['outcome_metrics'][outcome]
        if om['bets'] > 0:
            lines.append(f"  {outcome}:")
            lines.append(f"    Bets: {om['bets']}, Win Rate: {om['win_rate']*100:.1f}%, ROI: {om['roi']*100:+.1f}%")
    lines.append("")

    if best:
        # Optimized thresholds
        lines.append("=" * 80)
        lines.append("OPTIMIZED THRESHOLDS (BEST)")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Thresholds:")
        lines.append(f"  Away Win Min:      {best['away_min']:.2f}")
        lines.append(f"  Draw Close Diff:   {best['draw_threshold']:.2f}")
        lines.append(f"  Home Win Min:      {best['home_min']:.2f}")
        lines.append("")
        lines.append(f"Performance:")
        lines.append(f"  Total Bets:        {best['total_bets']}")
        lines.append(f"  Win Rate:          {best['win_rate']*100:.1f}%")
        lines.append(f"  ROI:               {best['roi']*100:+.2f}%")
        lines.append(f"  Net Profit:        ${best['net_profit']:+.2f}")
        lines.append("")

        # Performance by outcome
        lines.append("By Outcome:")
        for outcome in ['Home Win', 'Draw', 'Away Win']:
            om = best['outcome_metrics'][outcome]
            if om['bets'] > 0:
                lines.append(f"  {outcome}:")
                lines.append(f"    Bets: {om['bets']}, Win Rate: {om['win_rate']*100:.1f}%, ROI: {om['roi']*100:+.1f}%")
        lines.append("")

        # Improvement
        lines.append("=" * 80)
        lines.append("IMPROVEMENT")
        lines.append("=" * 80)
        lines.append("")
        roi_improvement = (best['roi'] - curr_metrics['roi']) * 100
        profit_improvement = best['net_profit'] - curr_metrics['net_profit']

        lines.append(f"ROI Change:        {roi_improvement:+.2f}% (from {curr_metrics['roi']*100:.2f}% to {best['roi']*100:.2f}%)")
        lines.append(f"Profit Change:     ${profit_improvement:+.2f} (from ${curr_metrics['net_profit']:.2f} to ${best['net_profit']:.2f})")
        lines.append(f"Bet Count Change:  {best['total_bets'] - curr_metrics['total_bets']:+d} (from {curr_metrics['total_bets']} to {best['total_bets']})")
        lines.append("")

        # Top 10 configurations
        lines.append("=" * 80)
        lines.append("TOP 10 CONFIGURATIONS")
        lines.append("=" * 80)
        lines.append("")

        for i, result in enumerate(optimized_results[:10], 1):
            lines.append(f"{i}. ROI: {result['roi']*100:+.2f}% | Away≥{result['away_min']:.2f}, Draw<{result['draw_threshold']:.2f}, Home≥{result['home_min']:.2f}")
            lines.append(f"   Bets: {result['total_bets']}, Win Rate: {result['win_rate']*100:.1f}%, Profit: ${result['net_profit']:+.2f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")

    if best and best['roi'] > curr_metrics['roi'] * 1.1:  # At least 10% better
        lines.append("✅ RECOMMENDED: Update thresholds to optimized values")
        lines.append("")
        lines.append("Update in 11_smart_betting_strategy.py:")
        lines.append(f"  away_win_min_prob: float = {best['away_min']:.2f}  # Was 0.35")
        lines.append(f"  draw_close_threshold: float = {best['draw_threshold']:.2f}  # Was 0.15")
        lines.append(f"  home_win_min_prob: float = {best['home_min']:.2f}  # Was 0.55")
    else:
        lines.append("✅ Current thresholds are already optimal or close to optimal")
        lines.append("   No significant improvement found through optimization")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Save report
    with open(output_file, 'w') as f:
        f.write(report)

    # Print to console
    print(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Optimize Betting Strategy Thresholds")
    parser.add_argument("--data", help="Path to historical predictions CSV")
    parser.add_argument("--method", choices=['grid', 'random'], default='random',
                        help="Optimization method (default: random)")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Number of trials for random search (default: 200)")

    args = parser.parse_args()

    print("=" * 80)
    print("BETTING STRATEGY THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print()

    # Find data file
    if args.data:
        data_file = Path(args.data)
    else:
        # Try to find recent predictions file
        predictions_files = sorted(Path('.').glob('recent_predictions_*.csv'), reverse=True)
        if predictions_files:
            data_file = predictions_files[0]
            print(f"Using most recent predictions file: {data_file}")
        else:
            print("❌ No predictions file found. Please specify with --data")
            return

    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return

    print(f"Loading data from: {data_file}")
    df = load_historical_predictions(str(data_file))
    print(f"✅ Loaded {len(df)} matches")
    print()

    # Evaluate current thresholds
    print("Step 1: Evaluating current thresholds...")
    current_results = analyze_current_thresholds(df)
    print(f"✅ Current ROI: {current_results['metrics']['roi']*100:+.2f}%")
    print()

    # Optimize
    print(f"Step 2: Running {args.method} search optimization...")

    if args.method == 'grid':
        optimized_results = grid_search_optimization(df)
    else:
        optimized_results = random_search_optimization(df, n_trials=args.n_trials)

    if not optimized_results:
        print("❌ No valid threshold combinations found")
        return

    print(f"✅ Found {len(optimized_results)} valid combinations")
    print(f"✅ Best ROI: {optimized_results[0]['roi']*100:+.2f}%")
    print()

    # Generate report
    print("Step 3: Generating report...")
    output_file = Path(f"threshold_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
    generate_report(current_results, optimized_results, df, output_file)

    # Save detailed results
    results_file = Path(f"threshold_optimization_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'current': current_results,
            'optimized': optimized_results[:50]  # Top 50
        }, f, indent=2)

    print()
    print("=" * 80)
    print(f"✅ Report saved to: {output_file}")
    print(f"✅ Detailed results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
