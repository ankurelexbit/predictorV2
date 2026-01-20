#!/usr/bin/env python3
"""
Re-analyze True API Replay with Lower Thresholds

Uses the predictions from the true API replay but applies lower thresholds
optimized for the 271-feature live pipeline.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from datetime import datetime

# Load the log file
log_file = Path('true_api_replay_200matches.log')

print("=" * 80)
print("RE-ANALYSIS WITH LOWER THRESHOLDS (LIVE PIPELINE OPTIMIZED)")
print("=" * 80)

# Parse predictions from log
predictions = []
current_match = {}

with open(log_file, 'r') as f:
    for line in f:
        # Match info
        if match := re.search(r'\[(\d+)/200\] (.+) vs (.+)', line):
            if current_match:
                predictions.append(current_match)
            current_match = {
                'match_num': int(match.group(1)),
                'home_team': match.group(2),
                'away_team': match.group(3)
            }
        
        # Date
        if 'Date:' in line and current_match:
            date = re.search(r'Date: ([\d-]+)', line)
            if date:
                current_match['date'] = date.group(1)
        
        # Predictions
        if match := re.search(r'H=([\d.]+)% D=([\d.]+)% A=([\d.]+)%', line):
            current_match['p_home'] = float(match.group(1)) / 100
            current_match['p_draw'] = float(match.group(2)) / 100
            current_match['p_away'] = float(match.group(3)) / 100
        
        # Bet result (if any)
        if 'BET' in line and 'HOME' in line:
            odds_match = re.search(r'@ ([\d.]+)', line)
            if odds_match:
                current_match['odds_home'] = float(odds_match.group(1))
                current_match['actual_outcome'] = 'home' if '✅' in line else 'away'

if current_match:
    predictions.append(current_match)

df = pd.DataFrame(predictions)

print(f"\nLoaded {len(df)} match predictions from API replay")

# Apply NEW lower thresholds
new_thresholds = {
    'home': 0.55,
    'draw': 0.35,
    'away': 0.60
}

old_thresholds = {
    'home': 0.65,
    'draw': 0.45,
    'away': 0.70
}

print(f"\n🎯 Threshold Comparison:")
print(f"{'Outcome':<10} {'Old':<10} {'New':<10} {'Change'}")
print("-" * 45)
for outcome in ['home', 'draw', 'away']:
    old = old_thresholds[outcome]
    new = new_thresholds[outcome]
    change = new - old
    print(f"{outcome.upper():<10} {old:<10.2f} {new:<10.2f} {change:+.2f}")

# Calculate bets with new thresholds
bets = []
stake = 100

for _, row in df.iterrows():
    model_probs = {
        'home': row['p_home'],
        'draw': row['p_draw'],
        'away': row['p_away']
    }
    
    # Find best bet
    best_bet = None
    best_prob = 0
    
    for outcome in ['home', 'draw', 'away']:
        prob = model_probs[outcome]
        threshold = new_thresholds[outcome]
        
        if prob > threshold and prob > best_prob:
            best_bet = outcome
            best_prob = prob
    
    if best_bet:
        # For this analysis, we'll assume odds of 1.5 for all bets
        # (we don't have odds for all matches in the log)
        assumed_odds = 1.5
        
        # We don't have actual outcomes for all matches, so we'll estimate
        # based on the probability being correct
        # For a more accurate analysis, we'd need the full dataset
        
        bets.append({
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'bet_on': best_bet,
            'probability': best_prob,
            'p_home': row['p_home'],
            'p_draw': row['p_draw'],
            'p_away': row['p_away']
        })

bets_df = pd.DataFrame(bets)

print(f"\n" + "=" * 80)
print("RESULTS WITH NEW THRESHOLDS")
print("=" * 80)

print(f"\n📊 Bet Frequency:")
print(f"   Old thresholds: 4/200 (2.0%)")
print(f"   New thresholds: {len(bets_df)}/200 ({len(bets_df)/200*100:.1f}%)")
print(f"   Increase: +{len(bets_df) - 4} bets")

print(f"\n📋 Bet Distribution:")
bet_counts = bets_df['bet_on'].value_counts()
for bet_type in ['home', 'draw', 'away']:
    count = bet_counts.get(bet_type, 0)
    pct = count / len(bets_df) * 100 if len(bets_df) > 0 else 0
    print(f"   {bet_type.upper()}: {count} ({pct:.1f}%)")

print(f"\n📈 Probability Ranges:")
for bet_type in ['home', 'draw', 'away']:
    type_bets = bets_df[bets_df['bet_on'] == bet_type]
    if len(type_bets) > 0:
        probs = type_bets['probability']
        print(f"   {bet_type.upper()}: {probs.min():.1%} - {probs.max():.1%} (avg: {probs.mean():.1%})")

# Show sample bets
print(f"\n" + "=" * 80)
print("SAMPLE BETS (First 10)")
print("=" * 80)

print(f"\n{'Match':<45} {'Bet':<8} {'Prob':<8} {'Probs (H/D/A)'}")
print("-" * 85)

for i, (_, bet) in enumerate(bets_df.head(10).iterrows()):
    match = f"{bet['home_team'][:20]} vs {bet['away_team'][:20]}"
    probs = f"{bet['p_home']*100:.0f}/{bet['p_draw']*100:.0f}/{bet['p_away']*100:.0f}"
    print(f"{match:<45} {bet['bet_on'].upper():<8} {bet['probability']:<8.1%} {probs}")

if len(bets_df) > 10:
    print(f"\n... and {len(bets_df) - 10} more bets")

# Summary
print(f"\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✅ Lower thresholds restore bet frequency:")
print(f"   Old: 2.0% → New: {len(bets_df)/200*100:.1f}%")
print(f"   Expected: 15-20%")

if len(bets_df) / 200 >= 0.15:
    print(f"\n✅ SUCCESS: Bet frequency is now in expected range!")
elif len(bets_df) / 200 >= 0.10:
    print(f"\n⚠️  MODERATE: Bet frequency improved but still below target")
    print(f"   Consider lowering thresholds further")
else:
    print(f"\n❌ STILL LOW: Bet frequency still too low")
    print(f"   Need to lower thresholds more aggressively")

# Save results
output = {
    'analysis_date': datetime.now().isoformat(),
    'old_thresholds': old_thresholds,
    'new_thresholds': new_thresholds,
    'matches_analyzed': len(df),
    'old_bets': 4,
    'new_bets': len(bets_df),
    'old_bet_rate': 2.0,
    'new_bet_rate': len(bets_df) / 200 * 100,
    'bet_distribution': {
        'home': int(bet_counts.get('home', 0)),
        'draw': int(bet_counts.get('draw', 0)),
        'away': int(bet_counts.get('away', 0))
    }
}

output_file = Path('models/lower_threshold_analysis.json')
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Analysis saved to: {output_file}")
print("\n" + "=" * 80)
