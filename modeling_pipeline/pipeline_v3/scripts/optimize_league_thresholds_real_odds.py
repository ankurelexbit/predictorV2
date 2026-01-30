#!/usr/bin/env python3
"""
League-Specific Threshold Optimization with Real Odds
======================================================

Optimizes betting thresholds for each league separately using actual
bookmaker odds for accurate PnL calculation.
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


def load_data_with_odds():
    """Load training data merged with real odds."""
    import joblib
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    
    # Load training data
    train_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    train_df = pd.read_csv(train_path)
    
    # Load fixtures with odds
    fixtures_path = Path(__file__).parent.parent / 'data' / 'csv' / 'fixtures.csv'
    fixtures_df = pd.read_csv(fixtures_path)
    
    # Keep only necessary columns from fixtures
    fixtures_df = fixtures_df[['fixture_id', 'odds_home', 'odds_draw', 'odds_away', 'league_id']].copy()
    
    # Merge
    merged = train_df.merge(fixtures_df, on='fixture_id', how='left', suffixes=('', '_odds'))
    
    # Use league_id from fixtures if available
    if 'league_id_odds' in merged.columns:
        merged['league_id'] = merged['league_id_odds'].fillna(merged.get('league_id', 0))
        merged = merged.drop(columns=['league_id_odds'])
    
    logger.info(f"Loaded {len(merged)} matches")
    logger.info(f"Matches with odds: {merged['odds_home'].notna().sum()}")
    
    return model, merged


def calculate_pnl_real_odds(y_true, y_pred_proba, odds_df, thresholds, stake=100):
    """
    Calculate PnL using real bookmaker odds.
    
    Args:
        y_true: Actual outcomes (0=away, 1=draw, 2=home)
        y_pred_proba: Predicted probabilities
        odds_df: DataFrame with odds_home, odds_draw, odds_away
        thresholds: Dict with 'home', 'draw', 'away' thresholds
        stake: Bet amount
    """
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
        
        # Check if meets threshold
        if confidence >= threshold_map[pred_outcome]:
            # Get real odds for this match
            if pred_outcome == 2:  # Home
                odds = odds_df.iloc[i]['odds_home']
            elif pred_outcome == 1:  # Draw
                odds = odds_df.iloc[i]['odds_draw']
            else:  # Away
                odds = odds_df.iloc[i]['odds_away']
            
            # Skip if odds not available
            if pd.isna(odds) or odds <= 1.0:
                continue
            
            bets_placed += 1
            total_staked += stake
            
            outcome_name = outcome_names[pred_outcome]
            outcome_stats[outcome_name]['bets'] += 1
            outcome_stats[outcome_name]['staked'] += stake
            
            # Check if won
            if y_true[i] == pred_outcome:
                wins += 1
                payout = stake * odds
                total_return += payout
                outcome_stats[outcome_name]['wins'] += 1
                outcome_stats[outcome_name]['return'] += payout
    
    profit = total_return - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    # Calculate per-outcome stats
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
        'win_rate': win_rate,
        'outcome_stats': outcome_stats
    }


def optimize_league_thresholds(model, df, league_id, league_name):
    """Optimize thresholds for a specific league."""
    # Filter to this league
    league_df = df[df['league_id'] == league_id].copy()
    
    if len(league_df) < 30:  # Need minimum data
        logger.warning(f"Skipping {league_name}: only {len(league_df)} matches")
        return None
    
    # Filter to validation period (Dec 2024)
    league_df['match_date'] = pd.to_datetime(league_df.get('match_date') or league_df.get('starting_at'), errors='coerce')
    val_mask = (league_df['match_date'] >= '2024-12-01') & (league_df['match_date'] < '2025-01-01')
    val_df = league_df[val_mask].copy()
    
    if len(val_df) < 10:
        logger.warning(f"Skipping {league_name}: only {len(val_df)} validation matches")
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
    
    # Drop leakage
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in val_df.columns:
            val_df = val_df.drop(columns=[col])
    
    # Generate predictions
    y_true = val_df['target'].values
    y_pred_proba = model.predict_proba(val_df)
    
    # Test threshold combinations
    home_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    draw_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    away_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    
    best_profit = -float('inf')
    best_thresholds = None
    best_metrics = None
    
    for home_t, draw_t, away_t in product(home_thresholds, draw_thresholds, away_thresholds):
        thresholds = {'home': home_t, 'draw': draw_t, 'away': away_t}
        metrics = calculate_pnl_real_odds(y_true, y_pred_proba, val_df, thresholds)
        
        if metrics['profit'] > best_profit and metrics['bets'] >= 5:  # Minimum 5 bets
            best_profit = metrics['profit']
            best_thresholds = thresholds
            best_metrics = metrics
    
    if best_thresholds is None:
        return None
    
    logger.info(f"\n{league_name}:")
    logger.info(f"  Thresholds: H={best_thresholds['home']:.2f}, D={best_thresholds['draw']:.2f}, A={best_thresholds['away']:.2f}")
    logger.info(f"  Profit: ${best_metrics['profit']:,.0f}, ROI: {best_metrics['roi']:.1f}%, Bets: {best_metrics['bets']}")
    
    return {
        'league_id': league_id,
        'league_name': league_name,
        'thresholds': best_thresholds,
        'metrics': best_metrics
    }


def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("LEAGUE-SPECIFIC THRESHOLD OPTIMIZATION WITH REAL ODDS")
    logger.info("="*70)
    
    # Load data
    model, df = load_data_with_odds()
    
    # League mapping
    leagues = {
        8: 'Premier League',
        39: 'La Liga',
        140: 'Serie A',
        78: 'Bundesliga',
        135: 'Ligue 1',
        61: 'Eredivisie',
        62: 'Primeira Liga',
        564: 'Championship'
    }
    
    # Optimize for each league
    logger.info("\nOptimizing thresholds for each league...")
    
    league_configs = {}
    
    for league_id, league_name in leagues.items():
        result = optimize_league_thresholds(model, df, league_id, league_name)
        if result:
            league_configs[league_id] = result
    
    # Save configuration
    config = {
        'league_thresholds': {
            str(lid): {
                'name': cfg['league_name'],
                'thresholds': cfg['thresholds'],
                'validation_metrics': {
                    'profit': cfg['metrics']['profit'],
                    'roi': cfg['metrics']['roi'],
                    'bets': cfg['metrics']['bets'],
                    'win_rate': cfg['metrics']['win_rate']
                }
            }
            for lid, cfg in league_configs.items()
        },
        'default_thresholds': {'home': 0.48, 'draw': 0.35, 'away': 0.45}
    }
    
    config_path = Path(__file__).parent.parent / 'models' / 'league_specific_thresholds.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n✅ Configuration saved to: {config_path}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"{'League':<20} {'Home':<8} {'Draw':<8} {'Away':<8} {'Profit':<12} {'ROI%':<10}")
    logger.info("-" * 70)
    
    for lid, cfg in league_configs.items():
        t = cfg['thresholds']
        m = cfg['metrics']
        logger.info(f"{cfg['league_name']:<20} {t['home']:<8.2f} {t['draw']:<8.2f} {t['away']:<8.2f} "
                   f"${m['profit']:<11,.0f} {m['roi']:<10.1f}")


if __name__ == '__main__':
    main()
