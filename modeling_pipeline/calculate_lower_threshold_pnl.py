#!/usr/bin/env python3
"""
Calculate Win Rate and PnL for Lower Thresholds

Matches the 35 bets from lower threshold analysis to actual outcomes
and calculates comprehensive PnL.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from datetime import datetime, timedelta

print("=" * 80)
print("WIN RATE & PNL ANALYSIS - LOWER THRESHOLDS")
print("=" * 80)

# Load the full dataset to get actual outcomes
df_all = pd.read_csv('data/processed/sportmonks_features.csv')
df_all['date'] = pd.to_datetime(df_all['date'])

# Filter for 90 days
cutoff = datetime.now() - timedelta(days=90)
df_90d = df_all[df_all['date'] >= cutoff]

# Parse predictions from API replay log
log_file = 'true_api_replay_200matches.log'
predictions = []
current_match = {}

with open(log_file, 'r') as f:
    for line in f:
        if match := re.search(r'\[(\d+)/200\] (.+) vs (.+)', line):
            if current_match:
                predictions.append(current_match)
            current_match = {
                'home_team': match.group(2).strip(),
                'away_team': match.group(3).strip()
            }
        
        if 'Date:' in line and current_match:
            date = re.search(r'Date: ([\d-]+)', line)
            if date:
                current_match['date'] = date.group(1)
        
        if match := re.search(r'H=([\d.]+)% D=([\d.]+)% A=([\d.]+)%', line):
            current_match['p_home'] = float(match.group(1)) / 100
            current_match['p_draw'] = float(match.group(2)) / 100
            current_match['p_away'] = float(match.group(3)) / 100

if current_match:
    predictions.append(current_match)

pred_df = pd.DataFrame(predictions)
pred_df['date'] = pd.to_datetime(pred_df['date'])

print(f"\nLoaded {len(pred_df)} predictions")

# Apply lower thresholds
new_thresholds = {'home': 0.55, 'draw': 0.35, 'away': 0.60}

bets = []
for _, row in pred_df.iterrows():
    model_probs = {
        'home': row['p_home'],
        'draw': row['p_draw'],
        'away': row['p_away']
    }
    
    best_bet = None
    best_prob = 0
    
    for outcome in ['home', 'draw', 'away']:
        prob = model_probs[outcome]
        if prob > new_thresholds[outcome] and prob > best_prob:
            best_bet = outcome
            best_prob = prob
    
    if best_bet:
        # Find matching row in dataset
        match_row = df_90d[
            (df_90d['home_team_name'] == row['home_team']) &
            (df_90d['away_team_name'] == row['away_team']) &
            (df_90d['date'].dt.date == row['date'].date())
        ]
        
        if len(match_row) > 0:
            match_row = match_row.iloc[0]
            
            # Get actual outcome
            outcome_map = {0: 'away', 1: 'draw', 2: 'home'}
            actual_outcome = outcome_map[int(match_row['target'])]
            
            # Get odds
            odds = {
                'home': match_row.get('odds_home', 1.5),
                'draw': match_row.get('odds_draw', 3.0),
                'away': match_row.get('odds_away', 2.0)
            }
            
            won = (best_bet == actual_outcome)
            stake = 100
            payout = stake * odds[best_bet] if won else 0
            profit = payout - stake
            
            bets.append({
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet_on': best_bet,
                'probability': best_prob,
                'odds': odds[best_bet],
                'actual_outcome': actual_outcome,
                'won': won,
                'stake': stake,
                'payout': payout,
                'profit': profit,
                'p_home': row['p_home'],
                'p_draw': row['p_draw'],
                'p_away': row['p_away']
            })

bets_df = pd.DataFrame(bets)

print(f"Matched {len(bets_df)} bets to actual outcomes")

# Calculate performance
total_bets = len(bets_df)
total_won = bets_df['won'].sum()
total_staked = bets_df['stake'].sum()
total_profit = bets_df['profit'].sum()
roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
win_rate = (total_won / total_bets) * 100 if total_bets > 0 else 0

print("\n" + "=" * 80)
print("PERFORMANCE WITH LOWER THRESHOLDS")
print("=" * 80)

print(f"\n📊 Overall Performance:")
print(f"   Bets Placed:    {total_bets}")
print(f"   Bets Won:       {total_won}")
print(f"   Bets Lost:      {total_bets - total_won}")
print(f"   Win Rate:       {win_rate:.1f}%")
print(f"   Total Staked:   ${total_staked:,.2f}")
print(f"   Total Return:   ${bets_df['payout'].sum():,.2f}")
print(f"   Net Profit:     ${total_profit:+,.2f}")
print(f"   ROI:            {roi:+.1f}%")
print(f"   Avg Profit/Bet: ${total_profit/total_bets:+.2f}")

# Breakdown by bet type
print(f"\n📋 Performance by Bet Type:")

for bet_type in ['home', 'draw', 'away']:
    type_bets = bets_df[bets_df['bet_on'] == bet_type]
    if len(type_bets) > 0:
        type_won = type_bets['won'].sum()
        type_staked = type_bets['stake'].sum()
        type_profit = type_bets['profit'].sum()
        type_roi = (type_profit / type_staked) * 100
        
        print(f"\n   {bet_type.upper()} Bets:")
        print(f"      Count:      {len(type_bets)}")
        print(f"      Won:        {type_won} ({type_won/len(type_bets)*100:.1f}%)")
        print(f"      Avg Odds:   {type_bets['odds'].mean():.2f}")
        print(f"      Profit:     ${type_profit:+,.2f}")
        print(f"      ROI:        {type_roi:+.1f}%")

# Comparison to old thresholds
print(f"\n" + "=" * 80)
print("COMPARISON: OLD vs NEW THRESHOLDS")
print("=" * 80)

print(f"\n{'Metric':<20} {'Old (H=0.65)':<15} {'New (H=0.55)':<15} {'Change'}")
print("-" * 70)
print(f"{'Bets Placed':<20} {'4':<15} {total_bets:<15} {total_bets - 4:+}")
print(f"{'Win Rate':<20} {'100.0%':<15} {f'{win_rate:.1f}%':<15} {win_rate - 100.0:+.1f}%")
print(f"{'ROI':<20} {'31.5%':<15} {f'{roi:.1f}%':<15} {roi - 31.5:+.1f}%")

# Show all bets
print(f"\n" + "=" * 80)
print("ALL BETS WITH OUTCOMES")
print("=" * 80)

print(f"\n{'Date':<12} {'Match':<40} {'Bet':<8} {'Odds':<6} {'Result':<8} {'P/L':<10}")
print("-" * 90)

for _, bet in bets_df.iterrows():
    match = f"{bet['home_team'][:18]} vs {bet['away_team'][:18]}"
    result = "✅ Won" if bet['won'] else "❌ Lost"
    print(f"{str(bet['date'].date()):<12} {match:<40} {bet['bet_on'].upper():<8} "
          f"{bet['odds']:<6.2f} {result:<8} ${bet['profit']:>8,.2f}")

# Save results
output_file = Path('models/lower_threshold_pnl.csv')
bets_df.to_csv(output_file, index=False)

summary = {
    'analysis_date': datetime.now().isoformat(),
    'thresholds': new_thresholds,
    'total_bets': int(total_bets),
    'bets_won': int(total_won),
    'win_rate': float(win_rate),
    'total_staked': float(total_staked),
    'total_profit': float(total_profit),
    'roi': float(roi),
    'bet_distribution': {
        'home': int(len(bets_df[bets_df['bet_on'] == 'home'])),
        'draw': int(len(bets_df[bets_df['bet_on'] == 'draw'])),
        'away': int(len(bets_df[bets_df['bet_on'] == 'away']))
    }
}

summary_file = Path('models/lower_threshold_pnl_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Results saved to:")
print(f"   {output_file}")
print(f"   {summary_file}")

# Final assessment
print(f"\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

if roi > 15:
    print(f"\n✅ EXCELLENT: ROI {roi:.1f}% exceeds 15% target")
elif roi > 10:
    print(f"\n✅ GOOD: ROI {roi:.1f}% is above 10%")
elif roi > 5:
    print(f"\n⚠️  MODERATE: ROI {roi:.1f}% is positive but below target")
else:
    print(f"\n❌ POOR: ROI {roi:.1f}% is too low")

if win_rate > 70:
    print(f"✅ Win rate {win_rate:.1f}% is excellent")
elif win_rate > 60:
    print(f"✅ Win rate {win_rate:.1f}% is good")
else:
    print(f"⚠️  Win rate {win_rate:.1f}% could be better")

print(f"\n🎯 Recommendation:")
if roi > 15 and win_rate > 60:
    print(f"   ✅ DEPLOY with lower thresholds (H=0.55, D=0.35, A=0.60)")
    print(f"   Expected performance: {roi:.1f}% ROI, {win_rate:.1f}% win rate")
elif roi > 10:
    print(f"   ✅ ACCEPTABLE for deployment")
    print(f"   Monitor closely and recalibrate if needed")
else:
    print(f"   ⚠️  Consider further threshold adjustment")

print("\n" + "=" * 80)
