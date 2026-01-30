#!/usr/bin/env python3
"""
Optimize Betting Thresholds for Maximum PnL
============================================

Finds optimal confidence thresholds for each outcome (Home/Draw/Away)
that maximize net profit/loss using actual or typical betting odds.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_data():
    """Load trained model and validation data."""
    import joblib
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    df = pd.read_csv(data_path)
    
    # Rename columns
    column_map = {
        'home_goals': 'home_score',
        'away_goals': 'away_score',
        'starting_at': 'match_date'
    }
    df.rename(columns=column_map, inplace=True)
    
    # Filter to matches with results
    mask = df['home_score'].notna() & df['away_score'].notna()
    df = df[mask].copy()
    
    # Create target
    conditions = [
        (df['home_score'] < df['away_score']),
        (df['home_score'] == df['away_score']),
        (df['home_score'] > df['away_score'])
    ]
    df['target'] = np.select(conditions, [0, 1, 2], default=-1)
    
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    
    # Use December 2024 as validation set for threshold optimization
    val_mask = (df['match_date'] >= '2024-12-01') & (df['match_date'] < '2025-01-01')
    val_df = df[val_mask].copy()
    
    logger.info(f"Validation set: {len(val_df)} matches from December 2024")
    
    # Drop leakage columns
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 
                    'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in val_df.columns:
            val_df = val_df.drop(columns=[col])
    
    return model, val_df


def calculate_pnl(y_true, y_pred_proba, thresholds, stake=100):
    """
    Calculate PnL with separate thresholds for each outcome.
    
    Args:
        y_true: Actual outcomes (0=away, 1=draw, 2=home)
        y_pred_proba: Predicted probabilities [away, draw, home]
        thresholds: Dict with keys 'home', 'draw', 'away' for confidence thresholds
        stake: Bet amount per match
    
    Returns:
        dict with PnL metrics
    """
    # Typical market odds (conservative estimates)
    typical_odds = {
        0: 3.0,  # Away
        1: 3.5,  # Draw
        2: 2.0,  # Home
    }
    
    threshold_map = {
        0: thresholds['away'],
        1: thresholds['draw'],
        2: thresholds['home']
    }
    
    total_staked = 0
    total_return = 0
    bets_placed = 0
    wins = 0
    
    outcome_stats = {
        'home': {'bets': 0, 'wins': 0, 'staked': 0, 'return': 0},
        'draw': {'bets': 0, 'wins': 0, 'staked': 0, 'return': 0},
        'away': {'bets': 0, 'wins': 0, 'staked': 0, 'return': 0}
    }
    
    outcome_names = {0: 'away', 1: 'draw', 2: 'home'}
    
    for i in range(len(y_true)):
        # Get predicted outcome and confidence
        pred_outcome = np.argmax(y_pred_proba[i])
        confidence = y_pred_proba[i][pred_outcome]
        
        # Check if confidence meets threshold for this specific outcome
        if confidence >= threshold_map[pred_outcome]:
            bets_placed += 1
            total_staked += stake
            
            outcome_name = outcome_names[pred_outcome]
            outcome_stats[outcome_name]['bets'] += 1
            outcome_stats[outcome_name]['staked'] += stake
            
            # Check if won
            if y_true[i] == pred_outcome:
                wins += 1
                payout = stake * typical_odds[pred_outcome]
                total_return += payout
                outcome_stats[outcome_name]['wins'] += 1
                outcome_stats[outcome_name]['return'] += payout
    
    profit = total_return - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    # Calculate per-outcome metrics
    for outcome_name in outcome_stats:
        stats = outcome_stats[outcome_name]
        if stats['bets'] > 0:
            stats['profit'] = stats['return'] - stats['staked']
            stats['roi'] = (stats['profit'] / stats['staked'] * 100)
            stats['win_rate'] = (stats['wins'] / stats['bets'] * 100)
        else:
            stats['profit'] = 0
            stats['roi'] = 0
            stats['win_rate'] = 0
    
    return {
        'total_staked': total_staked,
        'total_return': total_return,
        'profit': profit,
        'roi': roi,
        'bets_placed': bets_placed,
        'wins': wins,
        'win_rate': win_rate,
        'outcome_stats': outcome_stats
    }


def optimize_thresholds(model, val_df):
    """Find optimal thresholds for each outcome."""
    logger.info("\nOptimizing thresholds for maximum PnL...")
    
    # Generate predictions
    y_true = val_df['target'].values
    y_pred_proba = model.predict_proba(val_df)
    
    # Test threshold combinations
    # Different ranges for each outcome based on their characteristics
    home_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    draw_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]  # Higher for draws (riskier)
    away_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    best_profit = -float('inf')
    best_thresholds = None
    best_metrics = None
    
    total_combinations = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds)
    logger.info(f"Testing {total_combinations} threshold combinations...")
    
    results = []
    
    for home_t, draw_t, away_t in product(home_thresholds, draw_thresholds, away_thresholds):
        thresholds = {'home': home_t, 'draw': draw_t, 'away': away_t}
        metrics = calculate_pnl(y_true, y_pred_proba, thresholds)
        
        results.append({
            'home_threshold': home_t,
            'draw_threshold': draw_t,
            'away_threshold': away_t,
            'profit': metrics['profit'],
            'roi': metrics['roi'],
            'bets': metrics['bets_placed'],
            'win_rate': metrics['win_rate'],
            'metrics': metrics
        })
        
        if metrics['profit'] > best_profit and metrics['bets_placed'] >= 20:  # Minimum 20 bets
            best_profit = metrics['profit']
            best_thresholds = thresholds
            best_metrics = metrics
    
    # Sort by profit
    results.sort(key=lambda x: x['profit'], reverse=True)
    
    # Show top 10
    logger.info("\nTop 10 Threshold Combinations:")
    logger.info(f"{'Home':<8} {'Draw':<8} {'Away':<8} {'Bets':<8} {'Win%':<8} {'ROI%':<10} {'Profit':<10}")
    logger.info("-" * 70)
    
    for r in results[:10]:
        logger.info(f"{r['home_threshold']:<8.2f} {r['draw_threshold']:<8.2f} {r['away_threshold']:<8.2f} "
                   f"{r['bets']:<8} {r['win_rate']:<8.1f} {r['roi']:<10.1f} ${r['profit']:<9.0f}")
    
    # Show best configuration details
    logger.info(f"\n{'='*70}")
    logger.info("OPTIMAL THRESHOLDS")
    logger.info(f"{'='*70}")
    logger.info(f"Home threshold:  {best_thresholds['home']:.2f}")
    logger.info(f"Draw threshold:  {best_thresholds['draw']:.2f}")
    logger.info(f"Away threshold:  {best_thresholds['away']:.2f}")
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Total bets:    {best_metrics['bets_placed']}")
    logger.info(f"  Wins:          {best_metrics['wins']} ({best_metrics['win_rate']:.1f}%)")
    logger.info(f"  Total staked:  ${best_metrics['total_staked']:,.0f}")
    logger.info(f"  Total return:  ${best_metrics['total_return']:,.0f}")
    logger.info(f"  Net profit:    ${best_metrics['profit']:,.0f}")
    logger.info(f"  ROI:           {best_metrics['roi']:.1f}%")
    
    logger.info(f"\nBreakdown by Outcome:")
    for outcome in ['home', 'draw', 'away']:
        stats = best_metrics['outcome_stats'][outcome]
        logger.info(f"\n{outcome.upper()}:")
        logger.info(f"  Bets:    {stats['bets']}")
        logger.info(f"  Wins:    {stats['wins']} ({stats['win_rate']:.1f}%)")
        logger.info(f"  Profit:  ${stats['profit']:,.0f}")
        logger.info(f"  ROI:     {stats['roi']:.1f}%")
    
    return best_thresholds, best_metrics


def main():
    """Main execution."""
    # Load model and data
    model, val_df = load_model_and_data()
    
    # Optimize thresholds
    best_thresholds, best_metrics = optimize_thresholds(model, val_df)
    
    # Save configuration
    config = {
        'thresholds': best_thresholds,
        'validation_performance': {
            'total_bets': best_metrics['bets_placed'],
            'win_rate': best_metrics['win_rate'],
            'roi': best_metrics['roi'],
            'profit': best_metrics['profit'],
            'outcome_stats': {
                outcome: {
                    'bets': stats['bets'],
                    'wins': stats['wins'],
                    'win_rate': stats['win_rate'],
                    'roi': stats['roi'],
                    'profit': stats['profit']
                }
                for outcome, stats in best_metrics['outcome_stats'].items()
            }
        },
        'typical_odds': {
            'home': 2.0,
            'draw': 3.5,
            'away': 3.0
        }
    }
    
    config_path = Path(__file__).parent.parent / 'models' / 'optimal_thresholds.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n✅ Configuration saved to: {config_path}")


if __name__ == '__main__':
    main()
