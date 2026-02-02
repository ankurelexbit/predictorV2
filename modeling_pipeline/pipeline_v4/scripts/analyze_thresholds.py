#!/usr/bin/env python3
"""
Analyze PnL and Find Optimal Thresholds
========================================

Analyzes predictions with actual results to find optimal betting thresholds.
"""

import os
import sys
from pathlib import Path
import psycopg2
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Get database credentials
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

# Connect to database
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Get all predictions with actual results for January 2026
print("=" * 80)
print("FETCHING PREDICTIONS WITH RESULTS")
print("=" * 80)

cursor.execute("""
    SELECT
        fixture_id,
        model_version,
        pred_home_prob,
        pred_draw_prob,
        pred_away_prob,
        predicted_outcome,
        actual_result,
        best_home_odds,
        best_draw_odds,
        best_away_odds,
        should_bet,
        bet_outcome,
        bet_won,
        bet_profit
    FROM predictions
    WHERE match_date >= '2026-01-01'
      AND match_date <= '2026-01-31'
      AND actual_result IS NOT NULL
    ORDER BY fixture_id
""")

columns = [desc[0] for desc in cursor.description]
data = [dict(zip(columns, row)) for row in cursor.fetchall()]

print(f"Found {len(data)} predictions with results")
print()

# Convert to DataFrame
df = pd.DataFrame(data)

# Current PnL with existing thresholds
print("=" * 80)
print("CURRENT PERFORMANCE (Existing Thresholds)")
print("=" * 80)

current_bets = df[df['should_bet'] == True]
if len(current_bets) > 0:
    total_bets = len(current_bets)
    wins = current_bets['bet_won'].sum()
    total_profit = current_bets['bet_profit'].sum()
    win_rate = wins / total_bets if total_bets > 0 else 0
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

    print(f"Total Bets: {total_bets}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    print()

    # Breakdown by bet type
    print("Breakdown by bet type:")
    for bet_type in ['Home Win', 'Draw', 'Away Win']:
        bets = current_bets[current_bets['bet_outcome'] == bet_type]
        if len(bets) > 0:
            bet_wins = bets['bet_won'].sum()
            bet_profit = bets['bet_profit'].sum()
            bet_wr = bet_wins / len(bets)
            print(f"  {bet_type}: {len(bets)} bets, {bet_wr:.1%} win rate, ${bet_profit:.2f} profit")
else:
    print("No bets placed with current thresholds")

print()

# Find optimal thresholds
print("=" * 80)
print("FINDING OPTIMAL THRESHOLDS")
print("=" * 80)

# Test different threshold combinations
best_profit = -float('inf')
best_thresholds = None
best_metrics = None

# Test range: 0.30 to 0.60 in 0.05 increments
thresholds_to_test = np.arange(0.30, 0.61, 0.05)

print(f"Testing {len(thresholds_to_test)**3} threshold combinations...")
print()

results = []

for home_thresh in thresholds_to_test:
    for draw_thresh in thresholds_to_test:
        for away_thresh in thresholds_to_test:
            # Simulate betting with these thresholds
            profit = 0
            bets_placed = 0
            wins = 0

            for _, row in df.iterrows():
                home_prob = row['pred_home_prob']
                draw_prob = row['pred_draw_prob']
                away_prob = row['pred_away_prob']
                actual = row['actual_result']

                best_home_odds = row['best_home_odds']
                best_draw_odds = row['best_draw_odds']
                best_away_odds = row['best_away_odds']

                # Determine if we would bet
                bet = None
                odds = None

                if home_prob >= home_thresh and home_prob == max(home_prob, draw_prob, away_prob):
                    bet = 'H'
                    odds = best_home_odds
                elif draw_prob >= draw_thresh and draw_prob == max(home_prob, draw_prob, away_prob):
                    bet = 'D'
                    odds = best_draw_odds
                elif away_prob >= away_thresh and away_prob == max(home_prob, draw_prob, away_prob):
                    bet = 'A'
                    odds = best_away_odds

                if bet and odds:
                    bets_placed += 1
                    if bet == actual:
                        wins += 1
                        profit += (odds - 1) * 1.0  # $1 stake
                    else:
                        profit -= 1.0

            if bets_placed > 0:
                win_rate = wins / bets_placed
                roi = (profit / bets_placed * 100)

                results.append({
                    'home_thresh': home_thresh,
                    'draw_thresh': draw_thresh,
                    'away_thresh': away_thresh,
                    'bets': bets_placed,
                    'wins': wins,
                    'win_rate': win_rate,
                    'profit': profit,
                    'roi': roi
                })

                if profit > best_profit:
                    best_profit = profit
                    best_thresholds = (home_thresh, draw_thresh, away_thresh)
                    best_metrics = {
                        'bets': bets_placed,
                        'wins': wins,
                        'win_rate': win_rate,
                        'profit': profit,
                        'roi': roi
                    }

# Sort results by profit
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('profit', ascending=False)

print("Top 10 threshold combinations by profit:")
print("-" * 80)
for i, row in results_df.head(10).iterrows():
    print(f"{row['home_thresh']:.2f}/{row['draw_thresh']:.2f}/{row['away_thresh']:.2f}: "
          f"{row['bets']} bets, {row['win_rate']:.1%} WR, ${row['profit']:.2f} profit, {row['roi']:.1f}% ROI")

print()
print("=" * 80)
print("RECOMMENDED THRESHOLDS")
print("=" * 80)
if best_thresholds:
    print(f"Home: {best_thresholds[0]:.2f}")
    print(f"Draw: {best_thresholds[1]:.2f}")
    print(f"Away: {best_thresholds[2]:.2f}")
    print()
    print(f"Expected Performance:")
    print(f"  Bets: {best_metrics['bets']}")
    print(f"  Wins: {best_metrics['wins']}")
    print(f"  Win Rate: {best_metrics['win_rate']:.2%}")
    print(f"  Total Profit: ${best_metrics['profit']:.2f}")
    print(f"  ROI: {best_metrics['roi']:.2f}%")
else:
    print("Could not find profitable thresholds")

print()
print("=" * 80)

conn.close()
