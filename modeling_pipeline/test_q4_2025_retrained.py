#!/usr/bin/env python3
"""
Test Retrained Model on Oct-Dec 2025 (Top 5 Leagues)

Comprehensive validation of the balanced model on Q4 2025 data.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from predict_live import load_models
from production_thresholds import get_production_thresholds
from utils import setup_logger

logger = setup_logger("q4_2025_test")

def main():
    logger.info("=" * 80)
    logger.info("RETRAINED MODEL TEST: OCT-DEC 2025 (TOP 5 LEAGUES)")
    logger.info("=" * 80)
    
    # Load model and thresholds
    models = load_models('xgboost')
    if 'xgboost' not in models:
        logger.error("Failed to load model!")
        return
    
    model = models['xgboost']
    thresholds = get_production_thresholds()
    
    logger.info(f"Thresholds: H={thresholds['home']:.2f}, D={thresholds['draw']:.2f}, A={thresholds['away']:.2f}")
    
    # Load data
    logger.info("\nLoading Oct-Dec 2025 data...")
    df = pd.read_csv('data/processed/sportmonks_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for Oct-Dec 2025
    q4_2025 = df[(df['date'] >= '2025-10-01') & (df['date'] < '2026-01-01')]
    
    # Filter for matches with results and odds
    q4_2025 = q4_2025[q4_2025['target'].notna() & q4_2025['odds_home'].notna()]
    
    # Top 5 leagues (if league_id available)
    # League IDs: 8 (EPL), 384 (La Liga), 564 (Bundesliga), 462 (Ligue 1), 301 (Serie A)
    if 'league_id' in q4_2025.columns:
        top_5_leagues = [8, 384, 564, 462, 301]
        q4_2025 = q4_2025[q4_2025['league_id'].isin(top_5_leagues)]
        logger.info(f"Filtered to top 5 leagues")
    
    logger.info(f"Found {len(q4_2025)} matches in Oct-Dec 2025")
    
    if len(q4_2025) == 0:
        logger.warning("No matches found in Oct-Dec 2025!")
        return
    
    # Make predictions
    logger.info("\nGenerating predictions...")
    probs = model.predict_proba(q4_2025, calibrated=True)
    
    p_away = probs[:, 0]
    p_draw = probs[:, 1]
    p_home = probs[:, 2]
    
    # Apply thresholds and calculate PnL
    results = []
    
    for idx, row in q4_2025.iterrows():
        i = len(results)
        
        # Get probabilities
        probs_dict = {
            'home': p_home[i],
            'draw': p_draw[i],
            'away': p_away[i]
        }
        
        # Find best bet
        best_bet = None
        best_prob = 0
        
        for outcome in ['home', 'draw', 'away']:
            if probs_dict[outcome] > thresholds[outcome] and probs_dict[outcome] > best_prob:
                best_bet = outcome
                best_prob = probs_dict[outcome]
        
        if best_bet:
            # Get actual outcome
            actual_target = int(row['target'])
            outcome_map = {0: 'away', 1: 'draw', 2: 'home'}
            actual = outcome_map[actual_target]
            
            # Get odds (use best odds if available, otherwise use feature odds)
            odds = {
                'home': row['odds_home'],
                'draw': row['odds_draw'],
                'away': row['odds_away']
            }
            
            # Calculate PnL
            won = (best_bet == actual)
            stake = 100
            profit = (stake * odds[best_bet] - stake) if won else -stake
            
            results.append({
                'date': str(row['date']),
                'bet_on': best_bet,
                'confidence': float(best_prob),
                'actual': actual,
                'won': won,
                'odds': odds[best_bet],
                'profit': profit,
                'p_home': float(p_home[i]),
                'p_draw': float(p_draw[i]),
                'p_away': float(p_away[i])
            })
    
    # Calculate performance
    if results:
        total_bets = len(results)
        total_won = sum(1 for r in results if r['won'])
        total_profit = sum(r['profit'] for r in results)
        total_staked = total_bets * 100
        roi = (total_profit / total_staked) * 100
        win_rate = (total_won / total_bets) * 100
        
        # Breakdown by bet type
        by_type = {'home': [], 'draw': [], 'away': []}
        for r in results:
            by_type[r['bet_on']].append(r)
        
        # Monthly breakdown
        monthly = {}
        for r in results:
            month = r['date'][:7]  # YYYY-MM
            if month not in monthly:
                monthly[month] = []
            monthly[month].append(r)
        
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Total Bets: {total_bets}")
        logger.info(f"  Won: {total_won} ({win_rate:.1f}%)")
        logger.info(f"  Lost: {total_bets - total_won}")
        logger.info(f"  Total Staked: ${total_staked:,.0f}")
        logger.info(f"  Total Profit: ${total_profit:+,.0f}")
        logger.info(f"  ROI: {roi:+.1f}%")
        
        logger.info(f"\nBreakdown by Bet Type:")
        for bet_type in ['home', 'draw', 'away']:
            bets = by_type[bet_type]
            if bets:
                type_won = sum(1 for b in bets if b['won'])
                type_profit = sum(b['profit'] for b in bets)
                type_roi = (type_profit / (len(bets) * 100)) * 100
                type_pct = (len(bets) / total_bets) * 100
                logger.info(f"  {bet_type.upper()}: {len(bets)} bets ({type_pct:.1f}%), {type_won} won ({type_won/len(bets)*100:.1f}%), Profit: ${type_profit:+,.0f}, ROI: {type_roi:+.1f}%")
        
        logger.info(f"\nMonthly Breakdown:")
        for month in sorted(monthly.keys()):
            month_bets = monthly[month]
            month_won = sum(1 for b in month_bets if b['won'])
            month_profit = sum(b['profit'] for b in month_bets)
            month_roi = (month_profit / (len(month_bets) * 100)) * 100
            logger.info(f"  {month}: {len(month_bets)} bets, {month_won} won ({month_won/len(month_bets)*100:.1f}%), Profit: ${month_profit:+,.0f}, ROI: {month_roi:+.1f}%")
        
        # Average probabilities
        avg_p_home = np.mean([r['p_home'] for r in results])
        avg_p_draw = np.mean([r['p_draw'] for r in results])
        avg_p_away = np.mean([r['p_away'] for r in results])
        
        logger.info(f"\nAverage Predicted Probabilities (for bets placed):")
        logger.info(f"  Home: {avg_p_home*100:.1f}%")
        logger.info(f"  Draw: {avg_p_draw*100:.1f}%")
        logger.info(f"  Away: {avg_p_away*100:.1f}%")
        
        # Save results
        output = {
            'period': 'Oct-Dec 2025',
            'total_matches': len(q4_2025),
            'summary': {
                'total_bets': total_bets,
                'won': total_won,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'roi': roi,
                'bet_frequency': (total_bets / len(q4_2025)) * 100
            },
            'by_type': {
                bet_type: {
                    'bets': len(bets),
                    'won': sum(1 for b in bets if b['won']),
                    'profit': sum(b['profit'] for b in bets)
                }
                for bet_type, bets in by_type.items() if bets
            },
            'results': results
        }
        
        output_file = Path('models/q4_2025_retrained_test.json')
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to {output_file}")
    else:
        logger.warning("No bets placed!")

if __name__ == "__main__":
    main()
