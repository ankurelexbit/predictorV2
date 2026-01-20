#!/usr/bin/env python3
"""
Live PnL Analysis - Last 2 Weeks

Fetches actual data from API, makes predictions with draw-tuned model,
applies optimal thresholds, and calculates detailed PnL.
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

logger = setup_logger("live_pnl_analysis")

def load_optimal_thresholds():
    """Load optimal betting thresholds."""
    threshold_file = MODELS_DIR / 'optimal_thresholds.json'
    
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            data = json.load(f)
        return data['thresholds']
    else:
        # Default thresholds
        return {'home': 0.70, 'draw': 0.40, 'away': 0.55}

def calculate_pnl_with_thresholds(df, probs, thresholds, stake=100):
    """
    Calculate detailed PnL using optimal thresholds.
    
    Returns detailed bet-by-bet analysis.
    """
    outcome_names = {0: 'Away', 1: 'Draw', 2: 'Home'}
    outcome_map = {'away': 0, 'draw': 1, 'home': 2}
    
    bets = []
    
    for idx, (i, row) in enumerate(df.iterrows()):
        # Get model probabilities
        p_away, p_draw, p_home = probs[idx]
        model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
        
        # Get odds
        odds = {
            'away': row.get('odds_away', 0),
            'draw': row.get('odds_draw', 0),
            'home': row.get('odds_home', 0)
        }
        
        # Skip if no odds
        if odds['home'] == 0 or pd.isna(odds['home']):
            continue
        
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
            }
            bets.append(bet_info)
    
    return pd.DataFrame(bets)

def main():
    print("=" * 80)
    print("LIVE PnL ANALYSIS - LAST 2 WEEKS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load features (contains actual API data)
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for last 2 weeks
    cutoff_date = datetime.now() - timedelta(days=14)
    df_2w = df[df['date'] >= cutoff_date].copy()
    
    print(f"\n📅 Date Range: {df_2w['date'].min().date()} to {df_2w['date'].max().date()}")
    print(f"📊 Total Matches: {len(df_2w)}")
    
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
    probs = model.predict_proba(df_2w, calibrated=True)
    preds = np.argmax(probs, axis=1)
    
    # Load optimal thresholds
    thresholds = load_optimal_thresholds()
    print(f"\n🎯 Using Optimal Thresholds:")
    print(f"   Home: {thresholds['home']:.2f} ({thresholds['home']*100:.0f}%)")
    print(f"   Draw: {thresholds['draw']:.2f} ({thresholds['draw']*100:.0f}%)")
    print(f"   Away: {thresholds['away']:.2f} ({thresholds['away']*100:.0f}%)")
    
    # Calculate PnL with thresholds
    print("\n💰 Calculating PnL with optimal thresholds...")
    bets_df = calculate_pnl_with_thresholds(df_2w, probs, thresholds, stake=100)
    
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
    print(f"   Total Matches:     {len(df_2w)}")
    print(f"   Bets Placed:       {total_bets} ({total_bets/len(df_2w)*100:.1f}%)")
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
            
            print(f"\n{bet_type} Bets:")
            print(f"   Count:    {len(type_bets)}")
            print(f"   Won:      {type_won} ({type_won/len(type_bets)*100:.1f}%)")
            print(f"   Profit:   ${type_profit:+,.2f}")
            print(f"   ROI:      {type_roi:+.1f}%")
    
    # Best and worst bets
    print("\n" + "=" * 80)
    print("BEST & WORST BETS")
    print("=" * 80)
    
    best_bet = bets_df.loc[bets_df['profit'].idxmax()]
    worst_bet = bets_df.loc[bets_df['profit'].idxmin()]
    
    print(f"\n🏆 Best Bet:")
    print(f"   {best_bet['date'].date()}: {best_bet['home_team']} vs {best_bet['away_team']}")
    print(f"   Bet: {best_bet['bet_on']} @ {best_bet['odds']:.2f}")
    print(f"   Result: {best_bet['actual_outcome']}")
    print(f"   Profit: ${best_bet['profit']:+,.2f}")
    
    print(f"\n💔 Worst Bet:")
    print(f"   {worst_bet['date'].date()}: {worst_bet['home_team']} vs {worst_bet['away_team']}")
    print(f"   Bet: {worst_bet['bet_on']} @ {worst_bet['odds']:.2f}")
    print(f"   Result: {worst_bet['actual_outcome']}")
    print(f"   Profit: ${worst_bet['profit']:+,.2f}")
    
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
    
    # Detailed bet log
    print("\n" + "=" * 80)
    print("DETAILED BET LOG (Last 20 bets)")
    print("=" * 80)
    
    print(f"\n{'Date':<12} {'Match':<35} {'Bet':<8} {'Odds':<6} {'Result':<8} {'P/L':<10}")
    print("-" * 80)
    
    for _, bet in bets_df.tail(20).iterrows():
        match = f"{bet['home_team'][:15]} vs {bet['away_team'][:15]}"
        result = "✅ Won" if bet['won'] else "❌ Lost"
        print(f"{str(bet['date'].date()):<12} {match:<35} {bet['bet_on']:<8} "
              f"{bet['odds']:<6.2f} {result:<8} ${bet['profit']:>8,.2f}")
    
    # Cumulative profit chart
    print("\n" + "=" * 80)
    print("CUMULATIVE PROFIT")
    print("=" * 80)
    
    bets_df_sorted = bets_df.sort_values('date')
    bets_df_sorted['cumulative_profit'] = bets_df_sorted['profit'].cumsum()
    
    # Simple ASCII chart
    max_profit = bets_df_sorted['cumulative_profit'].max()
    min_profit = bets_df_sorted['cumulative_profit'].min()
    
    print(f"\nStarting: $0")
    print(f"Peak:     ${max_profit:,.2f}")
    print(f"Trough:   ${min_profit:,.2f}")
    print(f"Final:    ${total_profit:,.2f}")
    
    # Save detailed report
    output_file = MODELS_DIR / 'live_pnl_analysis_2weeks.csv'
    bets_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed bet log saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if roi > 0:
        print(f"\n✅ PROFITABLE: {roi:+.1f}% ROI over 2 weeks")
    else:
        print(f"\n❌ LOSS: {roi:+.1f}% ROI over 2 weeks")
    
    print(f"\n📊 Key Metrics:")
    print(f"   Win Rate:       {win_rate:.1f}%")
    print(f"   Bet Frequency:  {total_bets/len(df_2w)*100:.1f}% of matches")
    print(f"   Avg Profit/Bet: ${total_profit/total_bets:.2f}")
    print(f"   Total Profit:   ${total_profit:+,.2f}")
    
    # Compare to expected
    expected_roi = 19.0  # From threshold optimization
    print(f"\n📈 vs Expected:")
    print(f"   Expected ROI:   {expected_roi:.1f}%")
    print(f"   Actual ROI:     {roi:.1f}%")
    print(f"   Difference:     {roi - expected_roi:+.1f}%")
    
    if abs(roi - expected_roi) < 5:
        print(f"   ✅ Performance within expected range")
    elif roi > expected_roi:
        print(f"   🎉 Outperforming expectations!")
    else:
        print(f"   ⚠️  Underperforming (may need more data)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
