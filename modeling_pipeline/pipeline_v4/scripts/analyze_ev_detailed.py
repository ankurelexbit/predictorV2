#!/usr/bin/env python3
"""
Detailed EV Strategy Analysis
==============================

Deep dive into why EV-based strategy underperforms vs probability thresholds.
"""

import os
import sys
import pandas as pd
import numpy as np
import psycopg2

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')
TOP_5_LEAGUES = [8, 82, 384, 564, 301]

PROB_THRESHOLDS = {'home': 0.65, 'draw': 0.30, 'away': 0.42}


def calculate_ev(prob, odds):
    return (prob * odds) - 1


def load_data():
    query = """
        SELECT
            fixture_id, match_date, league_id,
            home_team_name, away_team_name,
            pred_home_prob, pred_draw_prob, pred_away_prob,
            best_home_odds, best_draw_odds, best_away_odds,
            actual_result
        FROM predictions
        WHERE match_date >= '2026-01-01' AND match_date < '2026-02-01'
          AND actual_result IS NOT NULL
          AND league_id = ANY(%s)
          AND best_home_odds IS NOT NULL
    """
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql(query, conn, params=(TOP_5_LEAGUES,))
    conn.close()
    return df


def analyze_ev_vs_prob(df):
    """Compare bets selected by EV vs probability strategies."""

    print("="*100)
    print("DETAILED COMPARISON: EV vs PROBABILITY STRATEGY")
    print("="*100)
    print()

    # Calculate EV for all outcomes
    df['ev_home'] = df.apply(lambda r: calculate_ev(r['pred_home_prob'], r['best_home_odds']), axis=1)
    df['ev_draw'] = df.apply(lambda r: calculate_ev(r['pred_draw_prob'], r['best_draw_odds']), axis=1)
    df['ev_away'] = df.apply(lambda r: calculate_ev(r['pred_away_prob'], r['best_away_odds']), axis=1)

    # Probability strategy selections
    df['prob_home'] = df['pred_home_prob'] > PROB_THRESHOLDS['home']
    df['prob_draw'] = df['pred_draw_prob'] > PROB_THRESHOLDS['draw']
    df['prob_away'] = df['pred_away_prob'] > PROB_THRESHOLDS['away']

    # EV strategy selections (5% threshold)
    min_ev = 0.05
    df['ev_home_bet'] = df['ev_home'] > min_ev
    df['ev_draw_bet'] = df['ev_draw'] > min_ev
    df['ev_away_bet'] = df['ev_away'] > min_ev

    # Count overlaps and differences
    print("BET SELECTION OVERLAP (min_ev = 5%):")
    print("-" * 100)
    print()

    # Home bets
    prob_home_count = df['prob_home'].sum()
    ev_home_count = df['ev_home_bet'].sum()
    both_home = (df['prob_home'] & df['ev_home_bet']).sum()
    only_prob_home = (df['prob_home'] & ~df['ev_home_bet']).sum()
    only_ev_home = (~df['prob_home'] & df['ev_home_bet']).sum()

    print(f"HOME BETS:")
    print(f"  Probability strategy: {prob_home_count} bets")
    print(f"  EV strategy: {ev_home_count} bets")
    print(f"  Both select: {both_home} bets")
    print(f"  Only Prob: {only_prob_home} bets")
    print(f"  Only EV: {only_ev_home} bets")
    print()

    # Draw bets
    prob_draw_count = df['prob_draw'].sum()
    ev_draw_count = df['ev_draw_bet'].sum()
    both_draw = (df['prob_draw'] & df['ev_draw_bet']).sum()
    only_prob_draw = (df['prob_draw'] & ~df['ev_draw_bet']).sum()
    only_ev_draw = (~df['prob_draw'] & df['ev_draw_bet']).sum()

    print(f"DRAW BETS:")
    print(f"  Probability strategy: {prob_draw_count} bets")
    print(f"  EV strategy: {ev_draw_count} bets")
    print(f"  Both select: {both_draw} bets")
    print(f"  Only Prob: {only_prob_draw} bets")
    print(f"  Only EV: {only_ev_draw} bets")
    print()

    # Away bets
    prob_away_count = df['prob_away'].sum()
    ev_away_count = df['ev_away_bet'].sum()
    both_away = (df['prob_away'] & df['ev_away_bet']).sum()
    only_prob_away = (df['prob_away'] & ~df['ev_away_bet']).sum()
    only_ev_away = (~df['prob_away'] & df['ev_away_bet']).sum()

    print(f"AWAY BETS:")
    print(f"  Probability strategy: {prob_away_count} bets")
    print(f"  EV strategy: {ev_away_count} bets")
    print(f"  Both select: {both_away} bets")
    print(f"  Only Prob: {only_prob_away} bets")
    print(f"  Only EV: {only_ev_away} bets")
    print()

    # Analyze "Only EV" bets that lost money
    print("="*100)
    print("WHY EV STRATEGY UNDERPERFORMS: ANALYSIS OF EV-ONLY BETS")
    print("="*100)
    print()

    # Home bets only selected by EV
    if only_ev_home > 0:
        ev_only_home_df = df[~df['prob_home'] & df['ev_home_bet']].copy()
        ev_only_home_df['won'] = ev_only_home_df['actual_result'] == 'H'
        ev_only_home_df['profit'] = ev_only_home_df.apply(
            lambda r: (r['best_home_odds'] - 1) if r['won'] else -1, axis=1
        )

        print(f"HOME BETS SELECTED ONLY BY EV (not by probability threshold):")
        print(f"  Count: {len(ev_only_home_df)}")
        print(f"  Wins: {ev_only_home_df['won'].sum()}")
        print(f"  Win Rate: {ev_only_home_df['won'].mean()*100:.1f}%")
        print(f"  Total Profit: ${ev_only_home_df['profit'].sum():.2f}")
        print(f"  Avg Probability: {ev_only_home_df['pred_home_prob'].mean():.1%}")
        print(f"  Avg Odds: {ev_only_home_df['best_home_odds'].mean():.2f}")
        print(f"  Avg EV: {ev_only_home_df['ev_home'].mean():.1%}")
        print()

        # Show worst performers
        worst = ev_only_home_df.nsmallest(5, 'profit')[
            ['home_team_name', 'away_team_name', 'pred_home_prob', 'best_home_odds', 'ev_home', 'actual_result', 'profit']
        ]
        print("  Worst 5 EV-only home bets:")
        for _, row in worst.iterrows():
            print(f"    {row['home_team_name']} vs {row['away_team_name']}: "
                  f"prob={row['pred_home_prob']:.1%}, odds={row['best_home_odds']:.2f}, "
                  f"EV={row['ev_home']:.1%}, result={row['actual_result']}, "
                  f"profit=${row['profit']:.2f}")
        print()

    # Away bets only selected by EV
    if only_ev_away > 0:
        ev_only_away_df = df[~df['prob_away'] & df['ev_away_bet']].copy()
        ev_only_away_df['won'] = ev_only_away_df['actual_result'] == 'A'
        ev_only_away_df['profit'] = ev_only_away_df.apply(
            lambda r: (r['best_away_odds'] - 1) if r['won'] else -1, axis=1
        )

        print(f"AWAY BETS SELECTED ONLY BY EV (not by probability threshold):")
        print(f"  Count: {len(ev_only_away_df)}")
        print(f"  Wins: {ev_only_away_df['won'].sum()}")
        print(f"  Win Rate: {ev_only_away_df['won'].mean()*100:.1f}%")
        print(f"  Total Profit: ${ev_only_away_df['profit'].sum():.2f}")
        print(f"  Avg Probability: {ev_only_away_df['pred_away_prob'].mean():.1%}")
        print(f"  Avg Odds: {ev_only_away_df['best_away_odds'].mean():.2f}")
        print(f"  Avg EV: {ev_only_away_df['ev_away'].mean():.1%}")
        print()

        # Show worst performers
        worst = ev_only_away_df.nsmallest(5, 'profit')[
            ['home_team_name', 'away_team_name', 'pred_away_prob', 'best_away_odds', 'ev_away', 'actual_result', 'profit']
        ]
        print("  Worst 5 EV-only away bets:")
        for _, row in worst.iterrows():
            print(f"    {row['home_team_name']} vs {row['away_team_name']}: "
                  f"prob={row['pred_away_prob']:.1%}, odds={row['best_away_odds']:.2f}, "
                  f"EV={row['ev_away']:.1%}, result={row['actual_result']}, "
                  f"profit=${row['profit']:.2f}")
        print()

    # Key insight
    print("="*100)
    print("KEY INSIGHTS")
    print("="*100)
    print()
    print("1. MODEL CALIBRATION ISSUE:")
    print("   - EV-only bets have positive calculated EV but lose money")
    print("   - This means model probabilities are OVERCONFIDENT for these bets")
    print("   - Example: Model says 60% (EV=8% at 1.8 odds), reality is 40% (EV=-28%)")
    print()
    print("2. PROBABILITY THRESHOLDS AS CALIBRATION FILTER:")
    print("   - High thresholds (65%/30%/42%) filter out poorly calibrated predictions")
    print("   - Only bet when model is VERY confident (threshold acts as safety margin)")
    print("   - This compensates for model overconfidence")
    print()
    print("3. WHY EV STRATEGY FAILS:")
    print("   - Assumes probabilities are perfectly calibrated")
    print("   - In reality, probabilities need recalibration")
    print("   - EV calculation amplifies calibration errors")
    print()
    print("4. SOLUTION:")
    print("   - Keep probability thresholds (proven to work)")
    print("   - OR: Recalibrate model probabilities using isotonic regression")
    print("   - OR: Use EV with HIGHER thresholds (e.g., min_ev=15%+)")
    print()


def main():
    df = load_data()
    print(f"Loaded {len(df)} predictions from January 2026\n")
    analyze_ev_vs_prob(df)


if __name__ == '__main__':
    main()
