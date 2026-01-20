#!/usr/bin/env python3
"""
True Live API Test - Oct-Dec 2025

Tests the complete live prediction pipeline with REAL API calls
for every match in Q4 2025. This simulates actual production conditions.

WARNING: This will make ~35,000 API calls and take ~1 hour to complete.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import time

sys.path.insert(0, str(Path(__file__).parent))

from predict_live import LiveFeatureCalculator, load_models
from utils import setup_logger

logger = setup_logger("true_live_q4_2025")

# Adjusted thresholds for retrained model
THRESHOLDS = {
    'home': 0.48,
    'draw': 0.35,
    'away': 0.45
}

def main():
    logger.info("=" * 80)
    logger.info("TRUE LIVE API TEST: OCT-DEC 2025")
    logger.info("=" * 80)
    logger.info("This will make REAL API calls for each match")
    logger.info(f"Thresholds: H={THRESHOLDS['home']:.2f}, D={THRESHOLDS['draw']:.2f}, A={THRESHOLDS['away']:.2f}")
    
    # Load model
    models = load_models('xgboost')
    if 'xgboost' not in models:
        logger.error("Failed to load model!")
        return
    
    model = models['xgboost']
    
    # Load Q4 2025 matches
    logger.info("\nLoading Oct-Dec 2025 matches...")
    df = pd.read_csv('data/processed/sportmonks_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    q4_2025 = df[(df['date'] >= '2025-10-01') & (df['date'] < '2026-01-01')]
    q4_2025 = q4_2025[q4_2025['target'].notna() & q4_2025['fixture_id'].notna()].reset_index(drop=True)
    
    logger.info(f"Found {len(q4_2025)} matches")
    logger.info(f"Estimated time: ~{len(q4_2025) * 4 / 60:.0f} minutes")
    
    # Initialize calculator
    calculator = LiveFeatureCalculator()
    
    # Process matches
    results = []
    errors = 0
    start_time = time.time()
    
    for idx, row in q4_2025.iterrows():
        try:
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (len(q4_2025) - idx) / rate if rate > 0 else 0
                logger.info(f"\n[{idx+1}/{len(q4_2025)}] Progress: {idx/len(q4_2025)*100:.1f}% | ETA: {remaining/60:.0f} min")
            
            # Build features with REAL API calls
            features = calculator.build_features_for_match(
                home_team_id=int(row['home_team_id']),
                away_team_id=int(row['away_team_id']),
                fixture_date=row['date'],
                fixture_id=int(row['fixture_id'])
            )
            
            if not features:
                logger.warning(f"  Match {idx+1}: Could not generate features")
                errors += 1
                continue
            
            # Make prediction
            features_df = pd.DataFrame([features])
            probs = model.predict_proba(features_df, calibrated=True)[0]
            p_away, p_draw, p_home = probs
            
            # Apply thresholds
            probs_dict = {'home': p_home, 'draw': p_draw, 'away': p_away}
            
            best_bet = None
            best_prob = 0
            for outcome in ['home', 'draw', 'away']:
                if probs_dict[outcome] > THRESHOLDS[outcome] and probs_dict[outcome] > best_prob:
                    best_bet = outcome
                    best_prob = probs_dict[outcome]
            
            if best_bet:
                # Get actual outcome
                actual_target = int(row['target'])
                outcome_map = {0: 'away', 1: 'draw', 2: 'home'}
                actual = outcome_map[actual_target]
                
                # Get best odds from features
                best_odds = {
                    'home': features.get('best_odds_home', features['odds_home']),
                    'draw': features.get('best_odds_draw', features['odds_draw']),
                    'away': features.get('best_odds_away', features['odds_away'])
                }
                
                # Calculate PnL
                won = (best_bet == actual)
                stake = 100
                profit = (stake * best_odds[best_bet] - stake) if won else -stake
                
                results.append({
                    'fixture_id': int(row['fixture_id']),
                    'date': str(row['date']),
                    'bet_on': best_bet,
                    'confidence': float(best_prob),
                    'actual': actual,
                    'won': won,
                    'odds': best_odds[best_bet],
                    'profit': profit,
                    'p_home': float(p_home),
                    'p_draw': float(p_draw),
                    'p_away': float(p_away)
                })
        
        except Exception as e:
            logger.error(f"  Match {idx+1}: Error - {e}")
            errors += 1
            continue
    
    # Calculate performance
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nProcessing Stats:")
    logger.info(f"  Total matches: {len(q4_2025)}")
    logger.info(f"  Successfully processed: {len(q4_2025) - errors}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"  Average: {total_time/len(q4_2025):.1f} seconds/match")
    
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
        
        logger.info(f"\nBetting Performance:")
        logger.info(f"  Total Bets: {total_bets} ({total_bets/len(q4_2025)*100:.1f}% of matches)")
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
                logger.info(f"  {bet_type.upper()}: {len(bets)} ({type_pct:.1f}%), {type_won} won ({type_won/len(bets)*100:.1f}%), ROI: {type_roi:+.1f}%")
        
        # Save results
        output = {
            'test_type': 'true_live_api',
            'period': 'Oct-Dec 2025',
            'total_matches': len(q4_2025),
            'thresholds': THRESHOLDS,
            'processing_stats': {
                'total_time_minutes': total_time / 60,
                'avg_seconds_per_match': total_time / len(q4_2025),
                'errors': errors
            },
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
                    'profit': sum(b['profit'] for b in bets),
                    'roi': (sum(b['profit'] for b in bets) / (len(bets) * 100)) * 100
                }
                for bet_type, bets in by_type.items() if bets
            },
            'results': results
        }
        
        output_file = Path('models/q4_2025_true_live_test.json')
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to {output_file}")
    else:
        logger.warning("No bets placed!")

if __name__ == "__main__":
    main()
