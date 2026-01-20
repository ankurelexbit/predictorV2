#!/usr/bin/env python3
"""
Validate New Thresholds on Last 10 Days

Tests the new optimal thresholds (H=0.65, D=0.45, A=0.70) on recent data
and compares to actual results to see if further calibration is needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger("threshold_validation")

def calculate_performance(df, probs, thresholds, stake=100):
    """Calculate detailed performance metrics."""
    outcome_names = {0: 'away', 1: 'draw', 2: 'home'}
    
    results = []
    
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
            
            results.append({
                'date': row['date'],
                'home_team': row.get('home_team_name', 'Unknown'),
                'away_team': row.get('away_team_name', 'Unknown'),
                'bet_on': best_bet,
                'probability': best_prob,
                'odds': odds[best_bet],
                'actual_outcome': actual_outcome,
                'won': won,
                'stake': stake,
                'payout': payout,
                'profit': profit,
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away,
            })
    
    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("NEW THRESHOLD VALIDATION - LAST 10 DAYS")
    print("=" * 80)
    print(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load new thresholds
    threshold_file = MODELS_DIR / 'optimal_thresholds_true_live_90day.json'
    
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            data = json.load(f)
        new_thresholds = data['thresholds']
        print(f"\n🎯 New Thresholds (90-day calibrated):")
        print(f"   Home: {new_thresholds['home']:.2f}")
        print(f"   Draw: {new_thresholds['draw']:.2f}")
        print(f"   Away: {new_thresholds['away']:.2f}")
    else:
        print("❌ Threshold file not found!")
        return
    
    # Load features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for last 10 days
    cutoff_date = datetime.now() - timedelta(days=10)
    df_10d = df[df['date'] >= cutoff_date].copy()
    
    print(f"\n📅 Validation Period: {df_10d['date'].min().date()} to {df_10d['date'].max().date()}")
    print(f"📊 Total Matches: {len(df_10d)}")
    
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
    probs = model.predict_proba(df_10d, calibrated=True)
    
    # Filter for matches with odds
    has_odds = (df_10d['odds_home'] > 0) & (~df_10d['odds_home'].isna())
    df_with_odds = df_10d[has_odds].copy()
    probs_with_odds = probs[has_odds]
    
    print(f"Matches with odds: {len(df_with_odds)}")
    
    # Calculate performance with new thresholds
    print("\n" + "=" * 80)
    print("PERFORMANCE WITH NEW THRESHOLDS")
    print("=" * 80)
    
    results_df = calculate_performance(df_with_odds, probs_with_odds, new_thresholds)
    
    if len(results_df) == 0:
        print("\n⚠️  No bets placed with new thresholds")
        return
    
    # Overall metrics
    total_bets = len(results_df)
    total_won = results_df['won'].sum()
    total_staked = results_df['stake'].sum()
    total_profit = results_df['profit'].sum()
    roi = (total_profit / total_staked) * 100
    win_rate = (total_won / total_bets) * 100
    
    print(f"\n📊 Overall Performance:")
    print(f"   Bets Placed:    {total_bets}")
    print(f"   Bets Won:       {total_won}")
    print(f"   Win Rate:       {win_rate:.1f}%")
    print(f"   Total Staked:   ${total_staked:,.2f}")
    print(f"   Total Profit:   ${total_profit:+,.2f}")
    print(f"   ROI:            {roi:+.1f}%")
    
    # Breakdown by bet type
    print("\n📋 Breakdown by Bet Type:")
    for bet_type in ['home', 'draw', 'away']:
        type_bets = results_df[results_df['bet_on'] == bet_type]
        if len(type_bets) > 0:
            type_won = type_bets['won'].sum()
            type_profit = type_bets['profit'].sum()
            type_staked = type_bets['stake'].sum()
            type_roi = (type_profit / type_staked) * 100
            
            print(f"\n   {bet_type.upper()} Bets:")
            print(f"      Count:      {len(type_bets)}")
            print(f"      Won:        {type_won} ({type_won/len(type_bets)*100:.1f}%)")
            print(f"      Avg Odds:   {type_bets['odds'].mean():.2f}")
            print(f"      Profit:     ${type_profit:+,.2f}")
            print(f"      ROI:        {type_roi:+.1f}%")
    
    # Compare to expected (90-day calibration)
    print("\n" + "=" * 80)
    print("COMPARISON TO 90-DAY CALIBRATION")
    print("=" * 80)
    
    expected_roi = 21.9
    expected_win_rate = 82.3
    
    print(f"\n{'Metric':<20} {'Expected (90d)':<15} {'Actual (10d)':<15} {'Difference'}")
    print("-" * 70)
    print(f"{'ROI':<20} {expected_roi:<15.1f}% {roi:<15.1f}% {roi - expected_roi:+.1f}%")
    print(f"{'Win Rate':<20} {expected_win_rate:<15.1f}% {win_rate:<15.1f}% {win_rate - expected_win_rate:+.1f}%")
    
    # Detailed fixture list
    print("\n" + "=" * 80)
    print("DETAILED FIXTURE RESULTS")
    print("=" * 80)
    
    print(f"\n{'Date':<12} {'Match':<40} {'Bet':<8} {'Odds':<6} {'Result':<8} {'P/L':<10}")
    print("-" * 90)
    
    for _, bet in results_df.iterrows():
        match = f"{bet['home_team'][:18]} vs {bet['away_team'][:18]}"
        result = "✅ Won" if bet['won'] else "❌ Lost"
        print(f"{str(bet['date'].date()):<12} {match:<40} {bet['bet_on'].upper():<8} "
              f"{bet['odds']:<6.2f} {result:<8} ${bet['profit']:>8,.2f}")
    
    # Analysis and recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    # Check if performance is within acceptable range
    roi_diff = abs(roi - expected_roi)
    win_rate_diff = abs(win_rate - expected_win_rate)
    
    print(f"\n📊 Performance Assessment:")
    
    if total_bets < 20:
        print(f"   ⚠️  SMALL SAMPLE SIZE ({total_bets} bets)")
        print(f"      Need at least 50 bets for reliable assessment")
        print(f"      Current results may not be statistically significant")
    
    if roi_diff < 10 and win_rate_diff < 10:
        print(f"   ✅ PERFORMANCE WITHIN EXPECTED RANGE")
        print(f"      ROI difference: {roi_diff:.1f}% (acceptable)")
        print(f"      Win rate difference: {win_rate_diff:.1f}% (acceptable)")
    elif roi > expected_roi:
        print(f"   🎉 OUTPERFORMING EXPECTATIONS")
        print(f"      ROI: {roi:.1f}% vs expected {expected_roi:.1f}%")
    else:
        print(f"   ⚠️  UNDERPERFORMING EXPECTATIONS")
        print(f"      ROI: {roi:.1f}% vs expected {expected_roi:.1f}%")
        print(f"      This may be due to small sample size or variance")
    
    # Calibration recommendations
    print(f"\n🎯 Calibration Recommendations:")
    
    if total_bets < 50:
        print(f"   1. ✅ KEEP CURRENT THRESHOLDS")
        print(f"      Sample size too small for recalibration")
        print(f"      Continue monitoring for 2-4 more weeks")
    elif roi < 10 and total_bets >= 50:
        print(f"   1. ⚠️  CONSIDER RECALIBRATION")
        print(f"      ROI below 10% with sufficient sample")
        print(f"      Suggested actions:")
        
        # Analyze which bet types are underperforming
        for bet_type in ['home', 'draw', 'away']:
            type_bets = results_df[results_df['bet_on'] == bet_type]
            if len(type_bets) >= 5:
                type_roi = (type_bets['profit'].sum() / type_bets['stake'].sum()) * 100
                if type_roi < 0:
                    current_threshold = new_thresholds[bet_type]
                    suggested_threshold = min(current_threshold + 0.05, 0.80)
                    print(f"      - Raise {bet_type} threshold: {current_threshold:.2f} → {suggested_threshold:.2f}")
    else:
        print(f"   1. ✅ NO CALIBRATION NEEDED")
        print(f"      Performance is acceptable")
        print(f"      Continue with current thresholds")
    
    print(f"\n   2. 📅 MONITORING SCHEDULE:")
    print(f"      - Weekly review of performance")
    print(f"      - Monthly recalibration if ROI < 15%")
    print(f"      - Quarterly full recalibration")
    
    # Save results
    output_file = MODELS_DIR / 'threshold_validation_10days.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
