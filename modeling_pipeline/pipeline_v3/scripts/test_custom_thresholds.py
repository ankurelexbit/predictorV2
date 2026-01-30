#!/usr/bin/env python3
"""
Test Custom Thresholds
=======================

Tests user-specified thresholds:
- Home: 0.48
- Draw: 0.35
- Away: 0.45
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_pnl(y_true, y_pred_proba, thresholds, stake=100):
    """Calculate PnL with given thresholds."""
    typical_odds = {0: 3.0, 1: 3.5, 2: 2.0}
    threshold_map = {0: thresholds['away'], 1: thresholds['draw'], 2: thresholds['home']}
    
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
        pred_outcome = np.argmax(y_pred_proba[i])
        confidence = y_pred_proba[i][pred_outcome]
        
        if confidence >= threshold_map[pred_outcome]:
            bets_placed += 1
            total_staked += stake
            
            outcome_name = outcome_names[pred_outcome]
            outcome_stats[outcome_name]['bets'] += 1
            outcome_stats[outcome_name]['staked'] += stake
            
            if y_true[i] == pred_outcome:
                wins += 1
                payout = stake * typical_odds[pred_outcome]
                total_return += payout
                outcome_stats[outcome_name]['wins'] += 1
                outcome_stats[outcome_name]['return'] += payout
    
    profit = total_return - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
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


def main():
    import joblib
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    
    # Load January 2025 data
    data_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    df = pd.read_csv(data_path)
    
    column_map = {'home_goals': 'home_score', 'away_goals': 'away_score', 'starting_at': 'match_date'}
    df.rename(columns=column_map, inplace=True)
    
    mask = df['home_score'].notna() & df['away_score'].notna()
    df = df[mask].copy()
    
    conditions = [
        (df['home_score'] < df['away_score']),
        (df['home_score'] == df['away_score']),
        (df['home_score'] > df['away_score'])
    ]
    df['target'] = np.select(conditions, [0, 1, 2], default=-1)
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    
    # January 2025
    test_mask = (df['match_date'] >= '2025-01-01') & (df['match_date'] < '2025-02-01')
    test_df = df[test_mask].copy()
    
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
    # Generate predictions
    y_true = test_df['target'].values
    y_pred_proba = model.predict_proba(test_df)
    
    # Test both threshold sets
    logger.info("="*70)
    logger.info("THRESHOLD COMPARISON - JANUARY 2025")
    logger.info("="*70)
    
    # Original optimal
    optimal_thresholds = {'home': 0.40, 'draw': 0.50, 'away': 0.45}
    optimal_results = calculate_pnl(y_true, y_pred_proba, optimal_thresholds)
    
    # Custom thresholds
    custom_thresholds = {'home': 0.48, 'draw': 0.35, 'away': 0.45}
    custom_results = calculate_pnl(y_true, y_pred_proba, custom_thresholds)
    
    # Display comparison
    logger.info(f"\n{'Metric':<25} {'Optimal (0.40/0.50/0.45)':<30} {'Custom (0.48/0.35/0.45)':<30}")
    logger.info("-" * 85)
    logger.info(f"{'Total Bets':<25} {optimal_results['bets_placed']:<30} {custom_results['bets_placed']:<30}")
    logger.info(f"{'Wins':<25} {optimal_results['wins']:<30} {custom_results['wins']:<30}")
    logger.info(f"{'Win Rate':<25} {optimal_results['win_rate']:.1f}%{'':<26} {custom_results['win_rate']:.1f}%{'':<26}")
    logger.info(f"{'Total Staked':<25} ${optimal_results['total_staked']:,}{'':<24} ${custom_results['total_staked']:,}{'':<24}")
    logger.info(f"{'Total Return':<25} ${optimal_results['total_return']:,}{'':<24} ${custom_results['total_return']:,}{'':<24}")
    logger.info(f"{'Net Profit':<25} ${optimal_results['profit']:,}{'':<24} ${custom_results['profit']:,}{'':<24}")
    logger.info(f"{'ROI':<25} {optimal_results['roi']:.1f}%{'':<26} {custom_results['roi']:.1f}%{'':<26}")
    
    logger.info(f"\n{'Breakdown by Outcome':<25} {'Optimal':<30} {'Custom':<30}")
    logger.info("-" * 85)
    
    for outcome in ['home', 'draw', 'away']:
        opt_stats = optimal_results['outcome_stats'][outcome]
        cust_stats = custom_results['outcome_stats'][outcome]
        
        logger.info(f"\n{outcome.upper()}:")
        logger.info(f"  {'Bets':<23} {opt_stats['bets']:<30} {cust_stats['bets']:<30}")
        logger.info(f"  {'Wins':<23} {opt_stats['wins']:<30} {cust_stats['wins']:<30}")
        logger.info(f"  {'Win Rate':<23} {opt_stats['win_rate']:.1f}%{'':<26} {cust_stats['win_rate']:.1f}%{'':<26}")
        logger.info(f"  {'Profit':<23} ${opt_stats['profit']:,}{'':<24} ${cust_stats['profit']:,}{'':<24}")
        logger.info(f"  {'ROI':<23} {opt_stats['roi']:.1f}%{'':<26} {cust_stats['roi']:.1f}%{'':<26}")
    
    # Determine winner
    logger.info("\n" + "="*70)
    if custom_results['profit'] > optimal_results['profit']:
        diff = custom_results['profit'] - optimal_results['profit']
        logger.info(f"✅ CUSTOM THRESHOLDS WIN by ${diff:,.0f}")
    elif optimal_results['profit'] > custom_results['profit']:
        diff = optimal_results['profit'] - custom_results['profit']
        logger.info(f"✅ OPTIMAL THRESHOLDS WIN by ${diff:,.0f}")
    else:
        logger.info("🤝 TIE - Both configurations perform equally")
    logger.info("="*70)


if __name__ == '__main__':
    main()
