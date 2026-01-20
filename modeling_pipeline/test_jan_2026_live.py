#!/usr/bin/env python3
"""
True Live Prediction Test for January 2026

Tests the complete live prediction pipeline on January 2026 matches
with real API calls and actual odds.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))

from predict_live import LiveFeatureCalculator, load_models
from production_thresholds import get_production_thresholds
from utils import setup_logger

logger = setup_logger("jan_2026_test")

def main():
    logger.info("=" * 80)
    logger.info("JANUARY 2026 LIVE PREDICTION TEST")
    logger.info("=" * 80)
    
    # Load model and thresholds
    models = load_models('xgboost')
    if 'xgboost' not in models:
        logger.error("Failed to load model!")
        return
    
    model = models['xgboost']
    thresholds = get_production_thresholds()
    
    logger.info(f"Thresholds: H={thresholds['home']:.2f}, D={thresholds['draw']:.2f}, A={thresholds['away']:.2f}")
    
    # Load matches from January 2026
    logger.info("\nLoading January 2026 matches from training data...")
    df = pd.read_csv('data/processed/sportmonks_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for January 2026
    jan_2026 = df[(df['date'] >= '2026-01-01') & (df['date'] < '2026-02-01')]
    
    # Filter for matches with results and odds
    jan_2026 = jan_2026[jan_2026['target'].notna() & jan_2026['odds_home'].notna()]
    
    logger.info(f"Found {len(jan_2026)} matches in January 2026 with results and odds")
    
    if len(jan_2026) == 0:
        logger.warning("No matches found in January 2026!")
        return
    
    # Sample matches to test (limit to avoid too many API calls)
    sample_size = min(50, len(jan_2026))
    test_matches = jan_2026.sample(sample_size, random_state=42)
    
    logger.info(f"Testing on {len(test_matches)} sampled matches")
    
    # Run predictions
    calculator = LiveFeatureCalculator()
    results = []
    
    for idx, (_, match) in enumerate(test_matches.iterrows()):
        try:
            logger.info(f"\n[{idx+1}/{len(test_matches)}] Fixture {match['fixture_id']}")
            
            # Build features with live API
            features = calculator.build_features_for_match(
                home_team_id=int(match['home_team_id']),
                away_team_id=int(match['away_team_id']),
                fixture_date=match['date'],
                fixture_id=int(match['fixture_id'])
            )
            
            if not features:
                logger.warning("  Could not generate features")
                continue
            
            # Make prediction
            features_df = pd.DataFrame([features])
            probs = model.predict_proba(features_df, calibrated=True)[0]
            p_away, p_draw, p_home = probs
            
            # Apply thresholds
            model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
            
            best_bet = None
            best_prob = 0
            
            for outcome in ['away', 'draw', 'home']:
                if model_probs[outcome] > thresholds[outcome] and model_probs[outcome] > best_prob:
                    best_bet = outcome
                    best_prob = model_probs[outcome]
            
            if best_bet:
                # Get actual outcome
                outcome_map = {0: 'away', 1: 'draw', 2: 'home'}
                actual = outcome_map[int(match['target'])]
                
                # Get best odds from features (now includes best_odds_*)
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
                    'fixture_id': match['fixture_id'],
                    'date': str(match['date']),
                    'bet_on': best_bet,
                    'confidence': best_prob,
                    'actual': actual,
                    'won': won,
                    'odds': best_odds[best_bet],
                    'profit': profit,
                    'p_home': p_home,
                    'p_draw': p_draw,
                    'p_away': p_away
                })
                
                logger.info(f"  Bet: {best_bet.upper()} @ {best_prob*100:.1f}% - {'WON' if won else 'LOST'} (${profit:+.0f})")
        
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue
    
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
                logger.info(f"  {bet_type.upper()}: {len(bets)} bets, {type_won} won ({type_won/len(bets)*100:.1f}%), ROI: {type_roi:+.1f}%")
        
        # Save results
        output_file = Path('models/jan_2026_live_test.json')
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_bets': total_bets,
                    'won': total_won,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'roi': roi
                },
                'results': results
            }, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to {output_file}")
    else:
        logger.warning("No bets placed!")

if __name__ == "__main__":
    main()
