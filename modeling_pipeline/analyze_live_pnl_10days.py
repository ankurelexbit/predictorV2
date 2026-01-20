#!/usr/bin/env python3
"""
Live PnL Analysis - Last 10 Days with Improved Pipeline

Uses the improved predict_live.py (271 features) to analyze recent matches
and calculate detailed PnL with optimal thresholds.
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

logger = setup_logger("live_pnl_10days")

def load_optimal_thresholds():
    """Load optimal betting thresholds."""
    threshold_file = MODELS_DIR / 'optimal_thresholds.json'
    
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            data = json.load(f)
        return data['thresholds']
    else:
        return {'home': 0.70, 'draw': 0.40, 'away': 0.55}

def main():
    print("=" * 80)
    print("LIVE PnL ANALYSIS - LAST 10 DAYS (IMPROVED PIPELINE)")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using: Improved pipeline with 271 features (EMA + rest days)")
    
    # Load features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for last 10 days
    cutoff_date = datetime.now() - timedelta(days=10)
    df_10d = df[df['date'] >= cutoff_date].copy()
    
    print(f"\n📅 Date Range: {df_10d['date'].min().date()} to {df_10d['date'].max().date()}")
    print(f"📊 Total Matches: {len(df_10d)}")
    
    # Load draw-tuned model
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
    preds = np.argmax(probs, axis=1)
    
    # Load optimal thresholds
    thresholds = load_optimal_thresholds()
    print(f"\n🎯 Using Optimal Thresholds:")
    print(f"   Home: {thresholds['home']:.2f} ({thresholds['home']*100:.0f}%)")
    print(f"   Draw: {thresholds['draw']:.2f} ({thresholds['draw']*100:.0f}%)")
    print(f"   Away: {thresholds['away']:.2f} ({thresholds['away']*100:.0f}%)")
    
    # Filter for matches with odds
    has_odds = (df_10d['odds_home'] > 0) & (~df_10d['odds_home'].isna())
    df_with_odds = df_10d[has_odds].copy()
    probs_with_odds = probs[has_odds]
    
    print(f"\nMatches with odds: {len(df_with_odds)}")
    
    # Calculate PnL with thresholds
    print("\n💰 Calculating fixture-level PnL...")
    
    outcome_names = {0: 'Away', 1: 'Draw', 2: 'Home'}
    outcome_map = {'away': 0, 'draw': 1, 'home': 2}
    
    bets = []
    stake = 100
    
    for idx, (i, row) in enumerate(df_with_odds.iterrows()):
        # Get model probabilities
        p_away, p_draw, p_home = probs_with_odds[idx]
        model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
        
        # Get odds
        odds = {
            'away': row.get('odds_away', 0),
            'draw': row.get('odds_draw', 0),
            'home': row.get('odds_home', 0)
        }
        
        # Find best bet (highest probability above threshold)
        best_bet = None
        best_prob = 0
        
        for outcome in ['away', 'draw', 'home']:
            prob = model_probs[outcome]
            threshold = thresholds[outcome]
            
            if prob > threshold and prob > best_prob and odds[outcome] > 1.0:
                best_bet = outcome
                best_prob = prob
        
        # Get actual outcome
        actual_outcome_idx = int(row['target'])
        actual_outcome = outcome_names[actual_outcome_idx].lower()
        
        # Record bet if placed
        if best_bet:
            won = (best_bet == actual_outcome)
            payout = stake * odds[best_bet] if won else 0
            profit = payout - stake
            
            bet_info = {
                'date': row['date'],
                'fixture_id': row.get('fixture_id', 0),
                'league': row.get('league_id', 'Unknown'),
                'home_team': row.get('home_team_name', 'Unknown'),
                'away_team': row.get('away_team_name', 'Unknown'),
                'bet_on': best_bet.capitalize(),
                'probability': best_prob,
                'odds': odds[best_bet],
                'stake': stake,
                'actual_outcome': actual_outcome.capitalize(),
                'won': won,
                'payout': payout,
                'profit': profit,
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away,
                'odds_home': odds['home'],
                'odds_draw': odds['draw'],
                'odds_away': odds['away'],
            }
            bets.append(bet_info)
    
    bets_df = pd.DataFrame(bets)
    
    if len(bets_df) == 0:
        print("\n⚠️  No bets placed (no matches exceeded thresholds)")
        return
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    
    total_bets = len(bets_df)
    total_won = bets_df['won'].sum()
    total_staked = bets_df['stake'].sum()
    total_return = bets_df['payout'].sum()
    total_profit = bets_df['profit'].sum()
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    win_rate = (total_won / total_bets) * 100 if total_bets > 0 else 0
    
    print(f"\n📊 Summary:")
    print(f"   Total Matches:     {len(df_with_odds)}")
    print(f"   Bets Placed:       {total_bets} ({total_bets/len(df_with_odds)*100:.1f}%)")
    print(f"   Bets Won:          {total_won}")
    print(f"   Bets Lost:         {total_bets - total_won}")
    print(f"   Win Rate:          {win_rate:.1f}%")
    
    print(f"\n💵 Financial:")
    print(f"   Total Staked:      ${total_staked:,.2f}")
    print(f"   Total Return:      ${total_return:,.2f}")
    print(f"   Net Profit:        ${total_profit:,.2f}")
    print(f"   ROI:               {roi:+.1f}%")
    
    # Breakdown by outcome
    print("\n" + "=" * 80)
    print("BREAKDOWN BY BET TYPE")
    print("=" * 80)
    
    for bet_type in ['Home', 'Draw', 'Away']:
        type_bets = bets_df[bets_df['bet_on'] == bet_type]
        if len(type_bets) > 0:
            type_won = type_bets['won'].sum()
            type_staked = type_bets['stake'].sum()
            type_profit = type_bets['profit'].sum()
            type_roi = (type_profit / type_staked) * 100 if type_staked > 0 else 0
            avg_odds = type_bets['odds'].mean()
            
            print(f"\n{bet_type} Bets:")
            print(f"   Count:      {len(type_bets)}")
            print(f"   Won:        {type_won} ({type_won/len(type_bets)*100:.1f}%)")
            print(f"   Avg Odds:   {avg_odds:.2f}")
            print(f"   Profit:     ${type_profit:+,.2f}")
            print(f"   ROI:        {type_roi:+.1f}%")
    
    # Daily PnL
    print("\n" + "=" * 80)
    print("DAILY PnL")
    print("=" * 80)
    
    bets_df['date_only'] = pd.to_datetime(bets_df['date']).dt.date
    daily_pnl = bets_df.groupby('date_only').agg({
        'profit': 'sum',
        'stake': 'sum',
        'won': 'sum',
        'bet_on': 'count'
    }).rename(columns={'bet_on': 'bets'})
    daily_pnl['roi'] = (daily_pnl['profit'] / daily_pnl['stake'] * 100).round(1)
    
    print(f"\n{'Date':<12} {'Bets':<6} {'Won':<6} {'Staked':<10} {'Profit':<12} {'ROI'}")
    print("-" * 80)
    for date, row in daily_pnl.iterrows():
        print(f"{str(date):<12} {int(row['bets']):<6} {int(row['won']):<6} "
              f"${row['stake']:>8,.0f} ${row['profit']:>10,.2f} {row['roi']:>6.1f}%")
    
    # FIXTURE-LEVEL DETAILS
    print("\n" + "=" * 80)
    print("FIXTURE-LEVEL DETAILS (ALL BETS)")
    print("=" * 80)
    
    print(f"\n{'Date':<12} {'Match':<40} {'Bet':<8} {'Prob':<6} {'Odds':<6} {'Result':<8} {'P/L':<10}")
    print("-" * 100)
    
    for _, bet in bets_df.iterrows():
        match = f"{bet['home_team'][:18]} vs {bet['away_team'][:18]}"
        result = "✅ Won" if bet['won'] else "❌ Lost"
        prob_pct = f"{bet['probability']*100:.0f}%"
        print(f"{str(bet['date'].date()):<12} {match:<40} {bet['bet_on']:<8} "
              f"{prob_pct:<6} {bet['odds']:<6.2f} {result:<8} ${bet['profit']:>8,.2f}")
    
    # Save detailed report
    output_file = MODELS_DIR / 'live_pnl_10days_detailed.csv'
    bets_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed fixture-level report saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if roi > 0:
        print(f"\n✅ PROFITABLE: {roi:+.1f}% ROI over 10 days")
    else:
        print(f"\n❌ LOSS: {roi:+.1f}% ROI over 10 days")
    
    print(f"\n📊 Key Metrics:")
    print(f"   Win Rate:       {win_rate:.1f}%")
    print(f"   Bet Frequency:  {total_bets/len(df_with_odds)*100:.1f}% of matches")
    print(f"   Avg Profit/Bet: ${total_profit/total_bets:.2f}")
    print(f"   Total Profit:   ${total_profit:+,.2f}")
    
    # Compare to expected
    expected_roi = 19.0
    print(f"\n📈 vs Expected (from 2-year backtest):")
    print(f"   Expected ROI:   {expected_roi:.1f}%")
    print(f"   Actual ROI:     {roi:.1f}%")
    print(f"   Difference:     {roi - expected_roi:+.1f}%")
    
    if abs(roi - expected_roi) < 10:
        print(f"   ✅ Performance within reasonable range")
    elif roi > expected_roi:
        print(f"   🎉 Outperforming expectations!")
    else:
        print(f"   ⚠️  Underperforming (small sample size)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
