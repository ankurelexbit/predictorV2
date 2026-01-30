#!/usr/bin/env python3
"""
Backtest on Q1 2023 with Real Odds
===================================

Uses actual bookmaker odds from the most recent period where we have data.
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


def calculate_pnl_real_odds(y_true, y_pred_proba, df, thresholds, stake=100):
    """Calculate PnL using real bookmaker odds."""
    threshold_map = {0: thresholds['away'], 1: thresholds['draw'], 2: thresholds['home']}
    
    total_staked = total_return = bets_placed = wins = 0
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
            if pred_outcome == 2:
                odds = df.iloc[i].get('odds_home')
            elif pred_outcome == 1:
                odds = df.iloc[i].get('odds_draw')
            else:
                odds = df.iloc[i].get('odds_away')
            
            if pd.isna(odds) or odds <= 1.0:
                continue
            
            bets_placed += 1
            total_staked += stake
            outcome_name = outcome_names[pred_outcome]
            outcome_stats[outcome_name]['bets'] += 1
            outcome_stats[outcome_name]['staked'] += stake
            
            if y_true[i] == pred_outcome:
                wins += 1
                payout = stake * odds
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
            stats['profit'] = stats['roi'] = stats['win_rate'] = 0
    
    return {'profit': profit, 'roi': roi, 'bets': bets_placed, 'wins': wins, 'win_rate': win_rate, 'outcome_stats': outcome_stats}


def main():
    """Main execution."""
    import joblib
    
    logger.info("="*70)
    logger.info("BACKTEST ON Q1 2023 WITH REAL BOOKMAKER ODDS")
    logger.info("="*70)
    
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    
    train_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    train_df = pd.read_csv(train_path)
    
    fixtures_path = Path(__file__).parent.parent / 'data' / 'csv' / 'fixtures.csv'
    fixtures_df = pd.read_csv(fixtures_path)[['fixture_id', 'odds_home', 'odds_draw', 'odds_away', 'league_id']]
    
    df = train_df.merge(fixtures_df, on='fixture_id', how='left', suffixes=('', '_fix'))
    if 'league_id_fix' in df.columns:
        df['league_id'] = df['league_id_fix'].fillna(df.get('league_id', 0))
        df = df.drop(columns=['league_id_fix'])
    
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
    
    # Q1 2023 (most recent with odds)
    test_mask = (df['match_date'] >= '2023-01-01') & (df['match_date'] < '2023-04-01')
    test_df = df[test_mask].copy()
    
    logger.info(f"\nQ1 2023: {len(test_df)} matches")
    logger.info(f"With odds: {test_df['odds_home'].notna().sum()}")
    
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
    # Use custom thresholds
    thresholds = {'home': 0.48, 'draw': 0.35, 'away': 0.45}
    
    y_true = test_df['target'].values
    y_pred_proba = model.predict_proba(test_df)
    
    results = calculate_pnl_real_odds(y_true, y_pred_proba, test_df, thresholds)
    
    logger.info("\n" + "="*70)
    logger.info("RESULTS - Q1 2023 WITH REAL ODDS")
    logger.info("="*70)
    logger.info(f"\nThresholds: Home={thresholds['home']}, Draw={thresholds['draw']}, Away={thresholds['away']}")
    logger.info(f"\nTotal bets: {results['bets']}")
    logger.info(f"Wins: {results['wins']} ({results['win_rate']:.1f}%)")
    logger.info(f"Total staked: ${results['bets'] * 100:,.0f}")
    logger.info(f"Net profit: ${results['profit']:,.0f}")
    logger.info(f"ROI: {results['roi']:.1f}%")
    
    logger.info(f"\nBreakdown by Outcome:")
    for outcome in ['home', 'draw', 'away']:
        stats = results['outcome_stats'][outcome]
        logger.info(f"\n{outcome.upper()}:")
        logger.info(f"  Bets: {stats['bets']}, Wins: {stats['wins']} ({stats['win_rate']:.1f}%)")
        logger.info(f"  Profit: ${stats['profit']:,.0f}, ROI: {stats['roi']:.1f}%")
    
    logger.info("\n" + "="*70)
    logger.info(f"NET PNL (Q1 2023 WITH REAL ODDS): ${results['profit']:,.0f}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
