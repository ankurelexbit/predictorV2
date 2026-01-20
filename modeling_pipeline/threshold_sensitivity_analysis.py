#!/usr/bin/env python3
"""
Threshold Sensitivity Analysis

Tests multiple threshold combinations to find optimal balance between:
- Bet frequency (target: 2+ bets per day = ~14% of matches)
- ROI (target: maximize while maintaining frequency)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from datetime import datetime, timedelta
from itertools import product

print("=" * 80)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 80)
print(f"Goal: Find thresholds for 2+ bets/day (~14% frequency) with max ROI")

# Load dataset
df_all = pd.read_csv('data/processed/sportmonks_features.csv')
df_all['date'] = pd.to_datetime(df_all['date'])

cutoff = datetime.now() - timedelta(days=90)
df_90d = df_all[df_all['date'] >= cutoff]

# Parse predictions from API replay
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

# Match to actual outcomes
matched_data = []
for _, row in pred_df.iterrows():
    match_row = df_90d[
        (df_90d['home_team_name'] == row['home_team']) &
        (df_90d['away_team_name'] == row['away_team']) &
        (df_90d['date'].dt.date == row['date'].date())
    ]
    
    if len(match_row) > 0:
        match_row = match_row.iloc[0]
        outcome_map = {0: 'away', 1: 'draw', 2: 'home'}
        
        matched_data.append({
            'p_home': row['p_home'],
            'p_draw': row['p_draw'],
            'p_away': row['p_away'],
            'actual_outcome': outcome_map[int(match_row['target'])],
            'odds_home': match_row.get('odds_home', 1.5),
            'odds_draw': match_row.get('odds_draw', 3.0),
            'odds_away': match_row.get('odds_away', 2.0)
        })

data_df = pd.DataFrame(matched_data)
print(f"Matched {len(data_df)} to outcomes")

# Test threshold combinations
print("\n" + "=" * 80)
print("TESTING THRESHOLD COMBINATIONS")
print("=" * 80)

# Define ranges
home_range = [0.50, 0.55, 0.60, 0.65, 0.70]
draw_range = [0.35, 0.40, 0.45, 0.50]
away_range = [0.55, 0.60, 0.65, 0.70]

results = []

for h_thresh, d_thresh, a_thresh in product(home_range, draw_range, away_range):
    thresholds = {'home': h_thresh, 'draw': d_thresh, 'away': a_thresh}
    
    bets = []
    for _, row in data_df.iterrows():
        model_probs = {
            'home': row['p_home'],
            'draw': row['p_draw'],
            'away': row['p_away']
        }
        
        best_bet = None
        best_prob = 0
        
        for outcome in ['home', 'draw', 'away']:
            if model_probs[outcome] > thresholds[outcome] and model_probs[outcome] > best_prob:
                best_bet = outcome
                best_prob = model_probs[outcome]
        
        if best_bet:
            odds = {
                'home': row['odds_home'],
                'draw': row['odds_draw'],
                'away': row['odds_away']
            }
            
            won = (best_bet == row['actual_outcome'])
            stake = 100
            profit = (stake * odds[best_bet] - stake) if won else -stake
            
            bets.append({
                'bet_on': best_bet,
                'won': won,
                'profit': profit
            })
    
    if len(bets) > 0:
        bets_df = pd.DataFrame(bets)
        total_profit = bets_df['profit'].sum()
        total_staked = len(bets) * 100
        roi = (total_profit / total_staked) * 100
        win_rate = (bets_df['won'].sum() / len(bets)) * 100
        bet_freq = len(bets) / len(data_df) * 100
        
        # Calculate bets per day (200 matches over ~90 days)
        days = 90
        bets_per_day = (len(bets) / len(data_df)) * (len(data_df) / days) * (892 / 200)  # Scale to full dataset
        
        results.append({
            'home_thresh': h_thresh,
            'draw_thresh': d_thresh,
            'away_thresh': a_thresh,
            'bets': len(bets),
            'bet_freq': bet_freq,
            'bets_per_day': bets_per_day,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'home_bets': len(bets_df[bets_df['bet_on'] == 'home']),
            'draw_bets': len(bets_df[bets_df['bet_on'] == 'draw']),
            'away_bets': len(bets_df[bets_df['bet_on'] == 'away'])
        })

results_df = pd.DataFrame(results)

# Filter for 2+ bets per day
target_bets_per_day = 2
filtered = results_df[results_df['bets_per_day'] >= target_bets_per_day].copy()

print(f"\nTotal combinations tested: {len(results_df)}")
print(f"Combinations with 2+ bets/day: {len(filtered)}")

if len(filtered) > 0:
    # Sort by ROI
    filtered = filtered.sort_values('roi', ascending=False)
    
    print("\n" + "=" * 80)
    print(f"TOP 15 THRESHOLD COMBINATIONS (2+ bets/day, sorted by ROI)")
    print("=" * 80)
    
    print(f"\n{'H':<5} {'D':<5} {'A':<5} {'Bets':<6} {'B/Day':<7} {'Win%':<7} {'ROI':<8} {'Profit'}")
    print("-" * 75)
    
    for i, (_, row) in enumerate(filtered.head(15).iterrows(), 1):
        print(f"{row['home_thresh']:<5.2f} {row['draw_thresh']:<5.2f} {row['away_thresh']:<5.2f} "
              f"{int(row['bets']):<6} {row['bets_per_day']:<7.1f} {row['win_rate']:<7.1f} "
              f"{row['roi']:<8.1f} ${row['profit']:>7,.0f}")
    
    # Show best by different criteria
    print("\n" + "=" * 80)
    print("BEST THRESHOLDS BY DIFFERENT CRITERIA")
    print("=" * 80)
    
    best_roi = filtered.iloc[0]
    best_winrate = filtered.sort_values('win_rate', ascending=False).iloc[0]
    best_frequency = filtered.sort_values('bets_per_day', ascending=False).iloc[0]
    
    print(f"\n1. HIGHEST ROI ({best_roi['roi']:.1f}%):")
    print(f"   Thresholds: H={best_roi['home_thresh']:.2f}, D={best_roi['draw_thresh']:.2f}, A={best_roi['away_thresh']:.2f}")
    print(f"   Bets/Day: {best_roi['bets_per_day']:.1f}")
    print(f"   Win Rate: {best_roi['win_rate']:.1f}%")
    print(f"   Profit: ${best_roi['profit']:,.0f}")
    
    print(f"\n2. HIGHEST WIN RATE ({best_winrate['win_rate']:.1f}%):")
    print(f"   Thresholds: H={best_winrate['home_thresh']:.2f}, D={best_winrate['draw_thresh']:.2f}, A={best_winrate['away_thresh']:.2f}")
    print(f"   Bets/Day: {best_winrate['bets_per_day']:.1f}")
    print(f"   ROI: {best_winrate['roi']:.1f}%")
    print(f"   Profit: ${best_winrate['profit']:,.0f}")
    
    print(f"\n3. MOST BETS ({best_frequency['bets_per_day']:.1f} bets/day):")
    print(f"   Thresholds: H={best_frequency['home_thresh']:.2f}, D={best_frequency['draw_thresh']:.2f}, A={best_frequency['away_thresh']:.2f}")
    print(f"   Win Rate: {best_frequency['win_rate']:.1f}%")
    print(f"   ROI: {best_frequency['roi']:.1f}%")
    print(f"   Profit: ${best_frequency['profit']:,.0f}")
    
    # Recommended
    print("\n" + "=" * 80)
    print("⭐ RECOMMENDED THRESHOLDS")
    print("=" * 80)
    
    # Find best balance: ROI > 10% and bets/day 2-4
    balanced = filtered[
        (filtered['roi'] > 10) &
        (filtered['bets_per_day'] >= 2) &
        (filtered['bets_per_day'] <= 4)
    ].sort_values('roi', ascending=False)
    
    if len(balanced) > 0:
        rec = balanced.iloc[0]
        print(f"\nBalanced Option (ROI > 10%, 2-4 bets/day):")
        print(f"   Thresholds: H={rec['home_thresh']:.2f}, D={rec['draw_thresh']:.2f}, A={rec['away_thresh']:.2f}")
        print(f"   Bets/Day: {rec['bets_per_day']:.1f}")
        print(f"   Win Rate: {rec['win_rate']:.1f}%")
        print(f"   ROI: {rec['roi']:.1f}%")
        print(f"   Profit: ${rec['profit']:,.0f}")
        
        recommended = {
            'home': rec['home_thresh'],
            'draw': rec['draw_thresh'],
            'away': rec['away_thresh']
        }
    else:
        rec = filtered.iloc[0]
        print(f"\nBest Available:")
        print(f"   Thresholds: H={rec['home_thresh']:.2f}, D={rec['draw_thresh']:.2f}, A={rec['away_thresh']:.2f}")
        print(f"   Bets/Day: {rec['bets_per_day']:.1f}")
        print(f"   Win Rate: {rec['win_rate']:.1f}%")
        print(f"   ROI: {rec['roi']:.1f}%")
        
        recommended = {
            'home': rec['home_thresh'],
            'draw': rec['draw_thresh'],
            'away': rec['away_thresh']
        }
    
    # Save results
    output_file = Path('models/threshold_sensitivity_analysis.csv')
    results_df.to_csv(output_file, index=False)
    
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'target_bets_per_day': target_bets_per_day,
        'combinations_tested': len(results_df),
        'viable_combinations': len(filtered),
        'recommended_thresholds': recommended,
        'expected_performance': {
            'bets_per_day': float(rec['bets_per_day']),
            'win_rate': float(rec['win_rate']),
            'roi': float(rec['roi'])
        }
    }
    
    summary_file = Path('models/threshold_sensitivity_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   {output_file}")
    print(f"   {summary_file}")

else:
    print("\n⚠️  No threshold combinations meet the 2+ bets/day criteria")
    print("   Consider lowering thresholds further")

print("\n" + "=" * 80)
