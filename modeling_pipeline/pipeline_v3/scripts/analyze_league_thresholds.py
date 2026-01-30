#!/usr/bin/env python3
"""
Analyze League-Specific Threshold Optimization
===============================================

Determines if different leagues need different thresholds.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_pnl(y_true, y_pred_proba, thresholds, stake=100):
    """Calculate PnL with given thresholds."""
    typical_odds = {0: 3.0, 1: 3.5, 2: 2.0}
    threshold_map = {0: thresholds['away'], 1: thresholds['draw'], 2: thresholds['home']}
    
    total_staked = total_return = bets_placed = wins = 0
    
    for i in range(len(y_true)):
        pred_outcome = np.argmax(y_pred_proba[i])
        confidence = y_pred_proba[i][pred_outcome]
        
        if confidence >= threshold_map[pred_outcome]:
            bets_placed += 1
            total_staked += stake
            
            if y_true[i] == pred_outcome:
                wins += 1
                total_return += stake * typical_odds[pred_outcome]
    
    profit = total_return - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    return {'profit': profit, 'roi': roi, 'bets': bets_placed, 'win_rate': win_rate}


def main():
    import joblib
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    
    # Load data
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
    
    # Check if league_id exists
    if 'league_id' not in test_df.columns:
        logger.warning("No league_id column found. Cannot analyze by league.")
        logger.info(f"Available columns: {test_df.columns.tolist()}")
        return
    
    # League mapping
    league_names = {
        8: 'Premier League',
        39: 'La Liga',
        140: 'Serie A',
        78: 'Bundesliga',
        135: 'Ligue 1',
        61: 'Eredivisie',
        62: 'Primeira Liga',
        564: 'Championship'
    }
    
    # Drop leakage
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
    # Custom thresholds
    thresholds = {'home': 0.48, 'draw': 0.35, 'away': 0.45}
    
    logger.info("="*70)
    logger.info("LEAGUE-SPECIFIC ANALYSIS - JANUARY 2025")
    logger.info("="*70)
    logger.info(f"\nUsing thresholds: Home={thresholds['home']}, Draw={thresholds['draw']}, Away={thresholds['away']}")
    
    # Analyze each league
    logger.info(f"\n{'League':<20} {'Matches':<10} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Profit':<12} {'ROI%':<10}")
    logger.info("-" * 80)
    
    total_profit = 0
    
    for league_id, league_name in league_names.items():
        league_df = test_df[test_df['league_id'] == league_id].copy()
        
        if len(league_df) == 0:
            continue
        
        y_true = league_df['target'].values
        y_pred_proba = model.predict_proba(league_df)
        
        results = calculate_pnl(y_true, y_pred_proba, thresholds)
        total_profit += results['profit']
        
        logger.info(f"{league_name:<20} {len(league_df):<10} {results['bets']:<8} "
                   f"{int(results['bets'] * results['win_rate'] / 100):<8} {results['win_rate']:<10.1f} "
                   f"${results['profit']:<11,.0f} {results['roi']:<10.1f}")
    
    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<20} {len(test_df):<10} {'':<8} {'':<8} {'':<10} ${total_profit:<11,.0f}")
    
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATION")
    logger.info("="*70)
    logger.info("League-specific thresholds could help if:")
    logger.info("  1. Some leagues show significantly different ROI")
    logger.info("  2. Draw rates vary significantly by league")
    logger.info("  3. You have enough data per league (>50 matches)")
    logger.info("\nFor now, use global thresholds unless you see major differences above.")


if __name__ == '__main__':
    main()
