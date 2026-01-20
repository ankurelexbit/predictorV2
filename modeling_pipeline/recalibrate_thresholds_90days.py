#!/usr/bin/env python3
"""
90-Day Threshold Recalibration

Analyzes last 90 days of matches and finds optimal thresholds
based on actual recent performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger("threshold_recalibration")

def calculate_roi_for_thresholds(df, probs, thresholds, stake=100):
    """Calculate ROI for given thresholds."""
    outcome_map = {'away': 0, 'draw': 1, 'home': 2}
    outcome_names = {0: 'away', 1: 'draw', 2: 'home'}
    
    total_profit = 0
    total_staked = 0
    bets_placed = 0
    bets_won = 0
    
    for idx, (i, row) in enumerate(df.iterrows()):
        p_away, p_draw, p_home = probs[idx]
        model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
        
        odds = {
            'away': row.get('odds_away', 0),
            'draw': row.get('odds_draw', 0),
            'home': row.get('odds_home', 0)
        }
        
        # Find best bet
        best_bet = None
        best_prob = 0
        
        for outcome in ['away', 'draw', 'home']:
            prob = model_probs[outcome]
            threshold = thresholds[outcome]
            
            if prob > threshold and prob > best_prob and odds[outcome] > 1.0:
                best_bet = outcome
                best_prob = prob
        
        if best_bet:
            actual_outcome_idx = int(row['target'])
            actual_outcome = outcome_names[actual_outcome_idx]
            
            won = (best_bet == actual_outcome)
            payout = stake * odds[best_bet] if won else 0
            profit = payout - stake
            
            total_profit += profit
            total_staked += stake
            bets_placed += 1
            if won:
                bets_won += 1
    
    if total_staked == 0:
        return None
    
    roi = (total_profit / total_staked) * 100
    win_rate = (bets_won / bets_placed) * 100 if bets_placed > 0 else 0
    
    return {
        'roi': roi,
        'profit': total_profit,
        'staked': total_staked,
        'bets': bets_placed,
        'won': bets_won,
        'win_rate': win_rate
    }

def main():
    print("=" * 80)
    print("90-DAY THRESHOLD RECALIBRATION")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for last 90 days
    cutoff_date = datetime.now() - timedelta(days=90)
    df_90d = df[df['date'] >= cutoff_date].copy()
    
    print(f"\n📅 Date Range: {df_90d['date'].min().date()} to {df_90d['date'].max().date()}")
    print(f"📊 Total Matches: {len(df_90d)}")
    
    # Load model
    logger.info("Loading draw-tuned XGBoost model...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("xgboost_model", Path(__file__).parent / "06_model_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    model = mod.XGBoostFootballModel()
    model.load(MODELS_DIR / "xgboost_model_draw_tuned.joblib")
    
    # Get predictions
    print("\n🔮 Generating predictions...")
    probs = model.predict_proba(df_90d, calibrated=True)
    
    # Filter for matches with odds
    has_odds = (df_90d['odds_home'] > 0) & (~df_90d['odds_home'].isna())
    df_with_odds = df_90d[has_odds].copy()
    probs_with_odds = probs[has_odds]
    
    print(f"Matches with odds: {len(df_with_odds)}")
    
    # Grid search for optimal thresholds
    print("\n" + "=" * 80)
    print("SEARCHING FOR OPTIMAL THRESHOLDS (90-DAY DATA)")
    print("=" * 80)
    
    # Define threshold ranges
    home_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    draw_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    away_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    
    total_combinations = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds)
    print(f"\nTesting {total_combinations} threshold combinations...")
    
    best_result = None
    best_thresholds = None
    all_results = []
    
    for home_t, draw_t, away_t in product(home_thresholds, draw_thresholds, away_thresholds):
        thresholds = {'home': home_t, 'draw': draw_t, 'away': away_t}
        
        result = calculate_roi_for_thresholds(df_with_odds, probs_with_odds, thresholds)
        
        if result and result['bets'] >= 10:  # Minimum 10 bets
            all_results.append({
                'home': home_t,
                'draw': draw_t,
                'away': away_t,
                **result
            })
            
            if best_result is None or result['roi'] > best_result['roi']:
                best_result = result
                best_thresholds = thresholds
                print(f"New best ROI: {result['roi']:.1f}% | "
                      f"Thresholds: H={home_t:.2f} D={draw_t:.2f} A={away_t:.2f} | "
                      f"Bets: {result['bets']}")
    
    # Show results
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLDS (90-DAY CALIBRATION)")
    print("=" * 80)
    
    if best_thresholds:
        print(f"\n🎯 Best Thresholds:")
        print(f"  Home Win: {best_thresholds['home']:.2f} ({best_thresholds['home']*100:.0f}%)")
        print(f"  Draw:     {best_thresholds['draw']:.2f} ({best_thresholds['draw']*100:.0f}%)")
        print(f"  Away Win: {best_thresholds['away']:.2f} ({best_thresholds['away']*100:.0f}%)")
        
        print(f"\n📊 Performance:")
        print(f"  ROI:          {best_result['roi']:.1f}%")
        print(f"  Profit:       ${best_result['profit']:,.2f}")
        print(f"  Bets Placed:  {best_result['bets']}")
        print(f"  Bets Won:     {best_result['won']}")
        print(f"  Win Rate:     {best_result['win_rate']:.1f}%")
        print(f"  Total Staked: ${best_result['staked']:,.2f}")
        
        # Compare to original thresholds
        print("\n" + "=" * 80)
        print("COMPARISON TO ORIGINAL THRESHOLDS")
        print("=" * 80)
        
        original_thresholds = {'home': 0.70, 'draw': 0.40, 'away': 0.55}
        original_result = calculate_roi_for_thresholds(df_with_odds, probs_with_odds, original_thresholds)
        
        print(f"\nOriginal Thresholds (H=0.70, D=0.40, A=0.55):")
        if original_result:
            print(f"  ROI:      {original_result['roi']:.1f}%")
            print(f"  Bets:     {original_result['bets']}")
            print(f"  Win Rate: {original_result['win_rate']:.1f}%")
        
        print(f"\nNew Thresholds (H={best_thresholds['home']:.2f}, D={best_thresholds['draw']:.2f}, A={best_thresholds['away']:.2f}):")
        print(f"  ROI:      {best_result['roi']:.1f}%")
        print(f"  Bets:     {best_result['bets']}")
        print(f"  Win Rate: {best_result['win_rate']:.1f}%")
        
        if original_result:
            roi_improvement = best_result['roi'] - original_result['roi']
            print(f"\n📈 Improvement: {roi_improvement:+.1f}% ROI")
        
        # Save new thresholds
        output = {
            'thresholds': best_thresholds,
            'performance': best_result,
            'calibration_date': datetime.now().isoformat(),
            'calibration_period': '90 days',
            'matches_analyzed': len(df_with_odds),
            'original_thresholds': original_thresholds,
            'original_performance': original_result if original_result else None
        }
        
        output_file = MODELS_DIR / 'optimal_thresholds_90day.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 New thresholds saved to: {output_file}")
        
        # Show top 10 combinations
        print("\n" + "=" * 80)
        print("TOP 10 THRESHOLD COMBINATIONS")
        print("=" * 80)
        
        results_df = pd.DataFrame(all_results).sort_values('roi', ascending=False)
        
        print(f"\n{'Rank':<5} {'Home':<6} {'Draw':<6} {'Away':<6} {'ROI':<8} {'Bets':<6} {'Win%':<8} {'Profit'}")
        print("-" * 80)
        
        for i, row in results_df.head(10).iterrows():
            print(f"{i+1:<5} {row['home']:<6.2f} {row['draw']:<6.2f} {row['away']:<6.2f} "
                  f"{row['roi']:<8.1f} {int(row['bets']):<6} {row['win_rate']:<8.1f} "
                  f"${row['profit']:,.2f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
