#!/usr/bin/env python3
"""
Backtest Betting Strategy on January 2025
==========================================

Tests the optimized betting strategy on the most recent month of data
to calculate actual PnL performance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_config():
    """Load trained model and optimal thresholds."""
    import joblib
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load optimal thresholds
    config_path = Path(__file__).parent.parent / 'models' / 'optimal_thresholds.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded thresholds: Home={config['thresholds']['home']}, "
               f"Draw={config['thresholds']['draw']}, Away={config['thresholds']['away']}")
    
    return model, config


def load_test_data():
    """Load January 2025 data for backtesting."""
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
    
    # Filter to January 2025
    test_mask = (df['match_date'] >= '2025-01-01') & (df['match_date'] < '2025-02-01')
    test_df = df[test_mask].copy()
    
    logger.info(f"Backtest period: January 2025")
    logger.info(f"Total matches: {len(test_df)}")
    
    # Drop leakage columns
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 
                    'result', 'home_score', 'away_score']
    for col in leakage_cols:
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
    return test_df


def backtest_strategy(model, test_df, config, stake=100):
    """
    Backtest the betting strategy.
    
    Returns detailed PnL breakdown.
    """
    thresholds = config['thresholds']
    typical_odds = config['typical_odds']
    
    # Convert odds dict keys to integers
    odds_map = {
        0: typical_odds['away'],
        1: typical_odds['draw'],
        2: typical_odds['home']
    }
    
    threshold_map = {
        0: thresholds['away'],
        1: thresholds['draw'],
        2: thresholds['home']
    }
    
    # Generate predictions
    y_true = test_df['target'].values
    y_pred_proba = model.predict_proba(test_df)
    
    # Track all bets
    bets = []
    
    for i in range(len(y_true)):
        pred_outcome = np.argmax(y_pred_proba[i])
        confidence = y_pred_proba[i][pred_outcome]
        
        # Check if confidence meets threshold
        if confidence >= threshold_map[pred_outcome]:
            actual_outcome = y_true[i]
            won = (actual_outcome == pred_outcome)
            
            outcome_names = {0: 'Away', 1: 'Draw', 2: 'Home'}
            
            payout = stake * odds_map[pred_outcome] if won else 0
            profit = payout - stake
            
            bets.append({
                'match_idx': i,
                'predicted_outcome': outcome_names[pred_outcome],
                'actual_outcome': outcome_names[actual_outcome],
                'confidence': confidence,
                'odds': odds_map[pred_outcome],
                'stake': stake,
                'payout': payout,
                'profit': profit,
                'won': won
            })
    
    # Calculate overall metrics
    total_bets = len(bets)
    total_staked = sum(b['stake'] for b in bets)
    total_return = sum(b['payout'] for b in bets)
    total_profit = total_return - total_staked
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    wins = sum(1 for b in bets if b['won'])
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    
    # Breakdown by outcome
    outcome_stats = {}
    for outcome in ['Home', 'Draw', 'Away']:
        outcome_bets = [b for b in bets if b['predicted_outcome'] == outcome]
        if outcome_bets:
            outcome_wins = sum(1 for b in outcome_bets if b['won'])
            outcome_staked = sum(b['stake'] for b in outcome_bets)
            outcome_return = sum(b['payout'] for b in outcome_bets)
            outcome_profit = outcome_return - outcome_staked
            
            outcome_stats[outcome] = {
                'bets': len(outcome_bets),
                'wins': outcome_wins,
                'win_rate': (outcome_wins / len(outcome_bets) * 100),
                'staked': outcome_staked,
                'return': outcome_return,
                'profit': outcome_profit,
                'roi': (outcome_profit / outcome_staked * 100) if outcome_staked > 0 else 0
            }
        else:
            outcome_stats[outcome] = {
                'bets': 0,
                'wins': 0,
                'win_rate': 0,
                'staked': 0,
                'return': 0,
                'profit': 0,
                'roi': 0
            }
    
    return {
        'total_bets': total_bets,
        'total_staked': total_staked,
        'total_return': total_return,
        'total_profit': total_profit,
        'roi': roi,
        'wins': wins,
        'win_rate': win_rate,
        'outcome_stats': outcome_stats,
        'bets': bets
    }


def main():
    """Main backtesting execution."""
    logger.info("="*70)
    logger.info("BACKTESTING V3 STRATEGY ON JANUARY 2025")
    logger.info("="*70)
    
    # Load model and config
    model, config = load_model_and_config()
    
    # Load test data
    test_df = load_test_data()
    
    # Run backtest
    logger.info("\nRunning backtest...")
    results = backtest_strategy(model, test_df, config, stake=100)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("BACKTEST RESULTS - JANUARY 2025")
    logger.info("="*70)
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Total matches:     {len(test_df)}")
    logger.info(f"  Bets placed:       {results['total_bets']} ({results['total_bets']/len(test_df)*100:.1f}% of matches)")
    logger.info(f"  Wins:              {results['wins']} ({results['win_rate']:.1f}%)")
    logger.info(f"  Total staked:      ${results['total_staked']:,.0f}")
    logger.info(f"  Total return:      ${results['total_return']:,.0f}")
    logger.info(f"  Net profit:        ${results['total_profit']:,.0f}")
    logger.info(f"  ROI:               {results['roi']:.1f}%")
    
    logger.info(f"\nBreakdown by Outcome:")
    logger.info(f"{'Outcome':<10} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Staked':<12} {'Return':<12} {'Profit':<12} {'ROI%':<10}")
    logger.info("-" * 90)
    
    for outcome in ['Home', 'Draw', 'Away']:
        stats = results['outcome_stats'][outcome]
        logger.info(f"{outcome:<10} {stats['bets']:<8} {stats['wins']:<8} {stats['win_rate']:<10.1f} "
                   f"${stats['staked']:<11,.0f} ${stats['return']:<11,.0f} ${stats['profit']:<11,.0f} {stats['roi']:<10.1f}")
    
    # Show sample bets
    logger.info(f"\nSample Bets (first 10):")
    logger.info(f"{'Predicted':<12} {'Actual':<12} {'Confidence':<12} {'Odds':<8} {'Result':<10} {'Profit':<10}")
    logger.info("-" * 70)
    
    for bet in results['bets'][:10]:
        result = "✅ WIN" if bet['won'] else "❌ LOSS"
        logger.info(f"{bet['predicted_outcome']:<12} {bet['actual_outcome']:<12} {bet['confidence']:<12.1%} "
                   f"{bet['odds']:<8.2f} {result:<10} ${bet['profit']:<9.0f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'backtest_results'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'january_2025_backtest.json'
    
    # Prepare JSON-serializable results
    json_results = {
        'period': 'January 2025',
        'total_matches': len(test_df),
        'total_bets': results['total_bets'],
        'wins': results['wins'],
        'win_rate': results['win_rate'],
        'total_staked': results['total_staked'],
        'total_return': results['total_return'],
        'total_profit': results['total_profit'],
        'roi': results['roi'],
        'outcome_stats': results['outcome_stats'],
        'thresholds_used': config['thresholds'],
        'typical_odds_used': config['typical_odds']
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    
    # Save detailed bets CSV
    bets_df = pd.DataFrame(results['bets'])
    bets_csv = output_dir / 'january_2025_bets.csv'
    bets_df.to_csv(bets_csv, index=False)
    logger.info(f"✅ Detailed bets saved to: {bets_csv}")
    
    logger.info("\n" + "="*70)
    logger.info(f"NET PNL FOR JANUARY 2025: ${results['total_profit']:,.0f}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
