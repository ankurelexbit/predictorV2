#!/usr/bin/env python3
"""
Backtest with League-Specific Thresholds and REAL Odds
=======================================================

Now using actual bookmaker odds extracted from historical JSON files.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_pnl_real_odds(y_true, y_pred_proba, df, thresholds, stake=100):
    """Calculate PnL using real bookmaker odds from fixtures."""
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
            # Get real odds
            if pred_outcome == 2:  # Home
                odds = df.iloc[i].get('odds_home')
            elif pred_outcome == 1:  # Draw
                odds = df.iloc[i].get('odds_draw')
            else:  # Away
                odds = df.iloc[i].get('odds_away')
            
            # Skip if odds not available
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
            stats['profit'] = 0
            stats['roi'] = 0
            stats['win_rate'] = 0
    
    return {
        'profit': profit,
        'roi': roi,
        'bets': bets_placed,
        'wins': wins,
        'win_rate': win_rate,
        'outcome_stats': outcome_stats
    }


def optimize_league(model, df, league_id, league_name):
    """Optimize thresholds for a specific league using real odds."""
    league_df = df[df['league_id'] == league_id].copy()
    
    if len(league_df) < 20:
        return None
    
    # December 2024 for optimization
    league_df['match_date'] = pd.to_datetime(league_df['match_date'], errors='coerce')
    val_mask = (league_df['match_date'] >= '2024-12-01') & (league_df['match_date'] < '2025-01-01')
    val_df = league_df[val_mask].copy()
    
    if len(val_df) < 5:
        return None
    
    # Prepare data
    column_map = {'home_goals': 'home_score', 'away_goals': 'away_score'}
    val_df.rename(columns=column_map, inplace=True)
    
    mask = val_df['home_score'].notna() & val_df['away_score'].notna()
    val_df = val_df[mask].copy()
    
    conditions = [
        (val_df['home_score'] < val_df['away_score']),
        (val_df['home_score'] == val_df['away_score']),
        (val_df['home_score'] > val_df['away_score'])
    ]
    val_df['target'] = np.select(conditions, [0, 1, 2], default=-1)
    
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in val_df.columns:
            val_df = val_df.drop(columns=[col])
    
    y_true = val_df['target'].values
    y_pred_proba = model.predict_proba(val_df)
    
    # Test thresholds
    home_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    draw_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    away_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    
    best_profit = -float('inf')
    best_thresholds = None
    best_metrics = None
    
    for home_t, draw_t, away_t in product(home_thresholds, draw_thresholds, away_thresholds):
        thresholds = {'home': home_t, 'draw': draw_t, 'away': away_t}
        metrics = calculate_pnl_real_odds(y_true, y_pred_proba, val_df, thresholds)
        
        if metrics['profit'] > best_profit and metrics['bets'] >= 3:
            best_profit = metrics['profit']
            best_thresholds = thresholds
            best_metrics = metrics
    
    if best_thresholds is None:
        return None
    
    return {
        'league_id': league_id,
        'league_name': league_name,
        'thresholds': best_thresholds,
        'val_metrics': best_metrics
    }


def main():
    """Main execution."""
    import joblib
    
    logger.info("="*70)
    logger.info("LEAGUE-SPECIFIC BACKTEST WITH REAL BOOKMAKER ODDS")
    logger.info("="*70)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    
    # Load training data
    train_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    train_df = pd.read_csv(train_path)
    
    # Load fixtures with real odds
    fixtures_path = Path(__file__).parent.parent / 'data' / 'csv' / 'fixtures.csv'
    fixtures_df = pd.read_csv(fixtures_path)
    fixtures_df = fixtures_df[['fixture_id', 'odds_home', 'odds_draw', 'odds_away', 'league_id']].copy()
    
    # Merge
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
    
    logger.info(f"\nTotal fixtures: {len(df)}")
    logger.info(f"Fixtures with odds: {df['odds_home'].notna().sum()} ({df['odds_home'].notna().sum()/len(df)*100:.1f}%)")
    
    # Leagues
    leagues = {8: 'Premier League', 564: 'Championship'}
    
    # Optimize each league
    logger.info("\nOptimizing thresholds per league (Dec 2024)...")
    league_configs = {}
    
    for league_id, league_name in leagues.items():
        result = optimize_league(model, df, league_id, league_name)
        if result:
            league_configs[league_id] = result
            logger.info(f"\n{league_name}:")
            logger.info(f"  Thresholds: H={result['thresholds']['home']:.2f}, "
                       f"D={result['thresholds']['draw']:.2f}, A={result['thresholds']['away']:.2f}")
            logger.info(f"  Val Profit: ${result['val_metrics']['profit']:,.0f}, "
                       f"ROI: {result['val_metrics']['roi']:.1f}%")
    
    # Backtest on January 2025
    logger.info("\n" + "="*70)
    logger.info("BACKTESTING ON JANUARY 2025 WITH REAL ODDS")
    logger.info("="*70)
    
    test_mask = (df['match_date'] >= '2025-01-01') & (df['match_date'] < '2025-02-01')
    test_df = df[test_mask].copy()
    
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
    logger.info(f"\nJanuary 2025: {len(test_df)} matches, {test_df['odds_home'].notna().sum()} with odds")
    
    # Test each league
    logger.info(f"\n{'League':<20} {'Matches':<10} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Profit':<12} {'ROI%':<10}")
    logger.info("-" * 85)
    
    total_profit = 0
    total_bets = 0
    total_wins = 0
    
    for league_id, config in league_configs.items():
        league_test = test_df[test_df['league_id'] == league_id].copy()
        
        if len(league_test) == 0:
            continue
        
        y_true = league_test['target'].values
        y_pred_proba = model.predict_proba(league_test)
        
        results = calculate_pnl_real_odds(y_true, y_pred_proba, league_test, config['thresholds'])
        
        total_profit += results['profit']
        total_bets += results['bets']
        total_wins += results['wins']
        
        logger.info(f"{config['league_name']:<20} {len(league_test):<10} {results['bets']:<8} "
                   f"{results['wins']:<8} {results['win_rate']:<10.1f} "
                   f"${results['profit']:<11,.0f} {results['roi']:<10.1f}")
    
    # Overall
    logger.info("-" * 85)
    overall_roi = (total_profit / (total_bets * 100) * 100) if total_bets > 0 else 0
    overall_win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
    logger.info(f"{'TOTAL':<20} {len(test_df):<10} {total_bets:<8} {total_wins:<8} "
               f"{overall_win_rate:<10.1f} ${total_profit:<11,.0f} {overall_roi:<10.1f}")
    
    logger.info("\n" + "="*70)
    logger.info(f"NET PNL (JANUARY 2025 WITH REAL ODDS): ${total_profit:,.0f}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
