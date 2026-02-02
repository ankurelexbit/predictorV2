#!/usr/bin/env python3
"""
Analyze Optimal Thresholds (Win Rate Based)
============================================

Without odds data, we optimize for win rate and bet volume.
"""

import os
import sys
from pathlib import Path
import psycopg2
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Get all predictions with actual results
cursor.execute("""
    SELECT
        pred_home_prob,
        pred_draw_prob,
        pred_away_prob,
        actual_result
    FROM predictions
    WHERE match_date >= '2026-01-01'
      AND match_date <= '2026-01-31'
      AND actual_result IS NOT NULL
""")

data = cursor.fetchall()
df = pd.DataFrame(data, columns=['pred_home_prob', 'pred_draw_prob', 'pred_away_prob', 'actual_result'])

print("=" * 80)
print("THRESHOLD OPTIMIZATION (WIN RATE BASED)")
print("=" * 80)
print(f"Total matches: {len(df)}")
print()

# Test different threshold combinations
best_win_rate = 0
best_thresholds = None
best_metrics = None

thresholds_to_test = np.arange(0.30, 0.61, 0.05)

results = []

for home_thresh in thresholds_to_test:
    for draw_thresh in thresholds_to_test:
        for away_thresh in thresholds_to_test:
            bets = 0
            wins = 0

            for _, row in df.iterrows():
                home_prob = row['pred_home_prob']
                draw_prob = row['pred_draw_prob']
                away_prob = row['pred_away_prob']
                actual = row['actual_result']

                # Determine if we would bet
                bet = None
                if home_prob >= home_thresh and home_prob == max(home_prob, draw_prob, away_prob):
                    bet = 'H'
                elif draw_prob >= draw_thresh and draw_prob == max(home_prob, draw_prob, away_prob):
                    bet = 'D'
                elif away_prob >= away_thresh and away_prob == max(home_prob, draw_prob, away_prob):
                    bet = 'A'

                if bet:
                    bets += 1
                    if bet == actual:
                        wins += 1

            if bets >= 50:  # Minimum 50 bets for statistical significance
                win_rate = wins / bets

                results.append({
                    'home_thresh': home_thresh,
                    'draw_thresh': draw_thresh,
                    'away_thresh': away_thresh,
                    'bets': bets,
                    'wins': wins,
                    'win_rate': win_rate
                })

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_thresholds = (home_thresh, draw_thresh, away_thresh)
                    best_metrics = {'bets': bets, 'wins': wins, 'win_rate': win_rate}

# Sort by win rate
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('win_rate', ascending=False)

print("Top 10 threshold combinations by win rate (min 50 bets):")
print("-" * 80)
for _, row in results_df.head(10).iterrows():
    print(f"H:{row['home_thresh']:.2f} D:{row['draw_thresh']:.2f} A:{row['away_thresh']:.2f} | "
          f"{row['bets']} bets | {row['wins']} wins | {row['win_rate']:.1%} WR")

print()
print("=" * 80)
print("RECOMMENDED THRESHOLDS (MAXIMUM WIN RATE)")
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
    print()
    print(f"Note: Profit calculation requires odds data.")
    print(f"To get odds, run predictions on UPCOMING fixtures, not historical ones.")

print()

# Also show most selective thresholds (high win rate, fewer bets)
print("=" * 80)
print("HIGH SELECTIVITY OPTIONS (70%+ Win Rate)")
print("=" * 80)
high_wr = results_df[results_df['win_rate'] >= 0.70]
if len(high_wr) > 0:
    print("Thresholds with 70%+ win rate:")
    print("-" * 80)
    for _, row in high_wr.head(5).iterrows():
        print(f"H:{row['home_thresh']:.2f} D:{row['draw_thresh']:.2f} A:{row['away_thresh']:.2f} | "
              f"{row['bets']} bets | {row['wins']} wins | {row['win_rate']:.1%} WR")
else:
    print("No threshold combinations achieved 70%+ win rate with 50+ bets")

print()
print("=" * 80)

conn.close()
