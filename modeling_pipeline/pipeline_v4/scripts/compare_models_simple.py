#!/usr/bin/env python3
"""
Simple Model Comparison using Database Predictions
==================================================

Uses existing predictions from database to compare models.
Much faster than regenerating predictions.

Usage:
    python3 scripts/compare_models_simple.py
"""

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')

# Top 5 leagues
TOP_5_LEAGUES = [8, 82, 384, 564]

def load_predictions_from_db(start_date, end_date):
    """Load existing predictions from database."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            fixture_id,
            match_date,
            league_id,
            league_name,
            home_team_name,
            away_team_name,
            pred_home_prob,
            pred_draw_prob,
            pred_away_prob,
            best_home_odds,
            best_draw_odds,
            best_away_odds,
            actual_result,
            actual_home_score,
            actual_away_score
        FROM predictions
        WHERE match_date >= %s
          AND match_date <= %s
          AND actual_result IS NOT NULL
          AND best_home_odds IS NOT NULL
        ORDER BY match_date
    """, (start_date, end_date))

    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[
        'fixture_id', 'match_date', 'league_id', 'league_name',
        'home_team_name', 'away_team_name',
        'pred_home_prob', 'pred_draw_prob', 'pred_away_prob',
        'best_home_odds', 'best_draw_odds', 'best_away_odds',
        'actual_result', 'actual_home_score', 'actual_away_score'
    ])

    conn.close()

    return df

def optimize_thresholds(df, model_name, top_5_only=True):
    """Find optimal thresholds for maximum profit."""
    print(f"\n{'='*80}")
    print(f"THRESHOLD OPTIMIZATION: {model_name}")
    print(f"{'='*80}")

    if top_5_only:
        df = df[df['league_id'].isin(TOP_5_LEAGUES)].copy()
        print(f"Filtering to top 5 leagues: {len(df)} matches")

    print(f"Optimizing for maximum net profit...")
    print()

    # Test threshold combinations
    home_thresholds = np.arange(0.30, 0.85, 0.05)
    draw_thresholds = np.arange(0.25, 0.55, 0.05)
    away_thresholds = np.arange(0.25, 0.55, 0.05)

    best_profit = -float('inf')
    best_thresholds = None
    best_metrics = None

    results = []
    total_combinations = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds)

    print(f"Testing {total_combinations} threshold combinations...")

    count = 0
    for home_thresh in home_thresholds:
        for draw_thresh in draw_thresholds:
            for away_thresh in away_thresholds:
                count += 1
                if count % 100 == 0:
                    print(f"   Progress: {count}/{total_combinations}...")

                home_bets = draw_bets = away_bets = 0
                home_wins = draw_wins = away_wins = 0
                home_profit = draw_profit = away_profit = 0

                for _, row in df.iterrows():
                    home_prob = row['pred_home_prob']
                    draw_prob = row['pred_draw_prob']
                    away_prob = row['pred_away_prob']
                    actual = row['actual_result']

                    home_odds = row['best_home_odds']
                    draw_odds = row['best_draw_odds']
                    away_odds = row['best_away_odds']

                    max_prob = max(home_prob, draw_prob, away_prob)

                    # Home bet
                    if home_prob >= home_thresh and home_prob == max_prob:
                        home_bets += 1
                        if actual == 'H':
                            home_wins += 1
                            home_profit += (home_odds - 1)
                        else:
                            home_profit -= 1

                    # Draw bet
                    if draw_prob >= draw_thresh and draw_prob == max_prob:
                        draw_bets += 1
                        if actual == 'D':
                            draw_wins += 1
                            draw_profit += (draw_odds - 1)
                        else:
                            draw_profit -= 1

                    # Away bet
                    if away_prob >= away_thresh and away_prob == max_prob:
                        away_bets += 1
                        if actual == 'A':
                            away_wins += 1
                            away_profit += (away_odds - 1)
                        else:
                            away_profit -= 1

                total_bets = home_bets + draw_bets + away_bets
                total_profit = home_profit + draw_profit + away_profit

                if total_bets >= 30:  # Minimum bet threshold
                    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0

                    results.append({
                        'home_thresh': home_thresh,
                        'draw_thresh': draw_thresh,
                        'away_thresh': away_thresh,
                        'total_bets': total_bets,
                        'total_profit': total_profit,
                        'roi': roi,
                        'home_bets': home_bets,
                        'draw_bets': draw_bets,
                        'away_bets': away_bets,
                        'home_wins': home_wins,
                        'draw_wins': draw_wins,
                        'away_wins': away_wins,
                        'home_profit': home_profit,
                        'draw_profit': draw_profit,
                        'away_profit': away_profit
                    })

                    if total_profit > best_profit:
                        best_profit = total_profit
                        best_thresholds = (home_thresh, draw_thresh, away_thresh)
                        best_metrics = {
                            'total_bets': total_bets,
                            'total_profit': total_profit,
                            'roi': roi,
                            'home_bets': home_bets,
                            'draw_bets': draw_bets,
                            'away_bets': away_bets,
                            'home_wins': home_wins,
                            'draw_wins': draw_wins,
                            'away_wins': away_wins,
                            'home_profit': home_profit,
                            'draw_profit': draw_profit,
                            'away_profit': away_profit
                        }

    print(f"✅ Optimization complete")
    print()

    if best_thresholds:
        print(f"🎯 OPTIMAL THRESHOLDS (Max Profit):")
        print(f"   Home: {best_thresholds[0]:.2f}")
        print(f"   Draw: {best_thresholds[1]:.2f}")
        print(f"   Away: {best_thresholds[2]:.2f}")
        print()
        print(f"📊 EXPECTED PERFORMANCE:")
        print(f"   Total Bets: {best_metrics['total_bets']}")
        print(f"   Total Profit: ${best_metrics['total_profit']:.2f}")
        print(f"   ROI: {best_metrics['roi']:.1f}%")
        print()
        print(f"   Home: {best_metrics['home_bets']} bets, {best_metrics['home_wins']} wins, ${best_metrics['home_profit']:.2f}")
        print(f"   Draw: {best_metrics['draw_bets']} bets, {best_metrics['draw_wins']} wins, ${best_metrics['draw_profit']:.2f}")
        print(f"   Away: {best_metrics['away_bets']} bets, {best_metrics['away_wins']} wins, ${best_metrics['away_profit']:.2f}")

    return best_thresholds, best_metrics, results

def main():
    print(f"{'='*80}")
    print(f"SIMPLE MODEL COMPARISON (Using Existing Predictions)")
    print(f"{'='*80}")
    print()

    # Load predictions
    print("Loading predictions from database...")
    df = load_predictions_from_db('2026-01-01', '2026-01-31')

    print(f"✅ Loaded {len(df)} predictions with results")
    print()

    # Since all predictions use the same model (current), just run optimization
    best_thresh, best_metrics, all_results = optimize_thresholds(df, "Current Production", top_5_only=True)

    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")

    # Export results
    if all_results:
        output_file = 'threshold_optimization_results.xlsx'
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values('total_profit', ascending=False)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_results.head(50).to_excel(writer, sheet_name='Top50', index=False)
            df_results.to_excel(writer, sheet_name='All Results', index=False)

        print(f"✅ Results exported to: {output_file}")

if __name__ == '__main__':
    main()
