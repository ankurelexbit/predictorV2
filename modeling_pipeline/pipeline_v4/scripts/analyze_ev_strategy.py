#!/usr/bin/env python3
"""
EV-Based Betting Strategy Analysis
===================================

Analyzes Expected Value (EV) based betting strategy using real predictions
from database and compares against current probability threshold strategy.

EV = (Probability × Odds) - 1

Only bet when EV > threshold (e.g., 5%, 10%)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("❌ DATABASE_URL environment variable not set")
    sys.exit(1)

# Top 5 European Leagues
TOP_5_LEAGUES = [8, 82, 384, 564, 301]

# Current production probability thresholds
PROB_THRESHOLDS = {
    'home': 0.65,
    'draw': 0.30,
    'away': 0.42
}


def calculate_ev(probability: float, odds: float) -> float:
    """Calculate Expected Value for a bet."""
    return (probability * odds) - 1


def get_predictions_from_db() -> pd.DataFrame:
    """Load predictions with features and odds from database."""
    print("="*80)
    print("LOADING PREDICTIONS FROM DATABASE")
    print("="*80)

    query = """
        SELECT
            fixture_id,
            match_date,
            league_id,
            home_team_name,
            away_team_name,
            features,
            best_home_odds,
            best_draw_odds,
            best_away_odds,
            actual_result
        FROM predictions
        WHERE match_date >= '2026-01-01'
          AND match_date < '2026-02-01'
          AND actual_result IS NOT NULL
          AND league_id = ANY(%s)
          AND best_home_odds IS NOT NULL
          AND best_draw_odds IS NOT NULL
          AND best_away_odds IS NOT NULL
    """

    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql(query, conn, params=(TOP_5_LEAGUES,))
    conn.close()

    print(f"✅ Loaded {len(df)} predictions from January 2026 (Top 5 leagues)")
    print(f"   Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    print()

    return df


def extract_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Extract model probabilities from features JSONB."""
    # Features contain the raw feature vector, we need to load model and predict
    # For now, we'll use stored probabilities if available, or calculate from prediction
    # Actually, let me check if we have probabilities stored directly

    # The predictions table should have predicted probabilities
    # Let me modify the query to get them
    print("Extracting model probabilities...")

    query = """
        SELECT
            fixture_id,
            match_date,
            league_id,
            home_team_name,
            away_team_name,
            pred_home_prob as home_prob,
            pred_draw_prob as draw_prob,
            pred_away_prob as away_prob,
            best_home_odds,
            best_draw_odds,
            best_away_odds,
            actual_result,
            predicted_outcome
        FROM predictions
        WHERE match_date >= '2026-01-01'
          AND match_date < '2026-02-01'
          AND actual_result IS NOT NULL
          AND league_id = ANY(%s)
          AND best_home_odds IS NOT NULL
          AND best_draw_odds IS NOT NULL
          AND best_away_odds IS NOT NULL
    """

    conn = psycopg2.connect(DATABASE_URL)
    df_new = pd.read_sql(query, conn, params=(TOP_5_LEAGUES,))
    conn.close()

    print(f"✅ Extracted probabilities for {len(df_new)} predictions")
    print()

    return df_new


def apply_ev_strategy(df: pd.DataFrame, min_ev: float) -> pd.DataFrame:
    """Apply EV-based betting strategy."""
    results = []

    for _, row in df.iterrows():
        # Calculate EV for each outcome
        ev_home = calculate_ev(row['home_prob'], row['best_home_odds'])
        ev_draw = calculate_ev(row['draw_prob'], row['best_draw_odds'])
        ev_away = calculate_ev(row['away_prob'], row['best_away_odds'])

        # Find all +EV opportunities above threshold
        opportunities = []
        if ev_home > min_ev:
            opportunities.append(('H', ev_home, row['home_prob'], row['best_home_odds']))
        if ev_draw > min_ev:
            opportunities.append(('D', ev_draw, row['draw_prob'], row['best_draw_odds']))
        if ev_away > min_ev:
            opportunities.append(('A', ev_away, row['away_prob'], row['best_away_odds']))

        # Bet on highest EV opportunity (if any)
        if opportunities:
            # Sort by EV descending
            opportunities.sort(key=lambda x: x[1], reverse=True)
            best = opportunities[0]

            bet_outcome = best[0]
            bet_ev = best[1]
            bet_prob = best[2]
            bet_odds = best[3]

            # Calculate profit
            won = (row['actual_result'] == bet_outcome)
            profit = (bet_odds - 1) if won else -1

            results.append({
                'fixture_id': row['fixture_id'],
                'match_date': row['match_date'],
                'home_team': row['home_team_name'],
                'away_team': row['away_team_name'],
                'bet_outcome': bet_outcome,
                'bet_ev': bet_ev,
                'bet_prob': bet_prob,
                'bet_odds': bet_odds,
                'actual_result': row['actual_result'],
                'won': won,
                'profit': profit,
                'ev_home': ev_home,
                'ev_draw': ev_draw,
                'ev_away': ev_away
            })

    return pd.DataFrame(results)


def apply_probability_threshold_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply current probability threshold strategy."""
    results = []

    for _, row in df.iterrows():
        # Check which outcomes cross thresholds
        candidates = []
        if row['home_prob'] > PROB_THRESHOLDS['home']:
            candidates.append(('H', row['home_prob'], row['best_home_odds']))
        if row['draw_prob'] > PROB_THRESHOLDS['draw']:
            candidates.append(('D', row['draw_prob'], row['best_draw_odds']))
        if row['away_prob'] > PROB_THRESHOLDS['away']:
            candidates.append(('A', row['away_prob'], row['best_away_odds']))

        # If multiple cross, pick highest probability
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]

            bet_outcome = best[0]
            bet_prob = best[1]
            bet_odds = best[2]

            # Calculate profit
            won = (row['actual_result'] == bet_outcome)
            profit = (bet_odds - 1) if won else -1

            results.append({
                'fixture_id': row['fixture_id'],
                'match_date': row['match_date'],
                'home_team': row['home_team_name'],
                'away_team': row['away_team_name'],
                'bet_outcome': bet_outcome,
                'bet_prob': bet_prob,
                'bet_odds': bet_odds,
                'actual_result': row['actual_result'],
                'won': won,
                'profit': profit
            })

    return pd.DataFrame(results)


def calculate_metrics(bets_df: pd.DataFrame) -> Dict:
    """Calculate performance metrics."""
    if len(bets_df) == 0:
        return {
            'total_bets': 0,
            'total_profit': 0,
            'roi': 0,
            'win_rate': 0,
            'avg_odds': 0,
            'by_outcome': {}
        }

    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    total_profit = bets_df['profit'].sum()
    roi = (total_profit / total_bets) * 100
    win_rate = (wins / total_bets) * 100
    avg_odds = bets_df['bet_odds'].mean()

    # By outcome
    by_outcome = {}
    for outcome in ['H', 'D', 'A']:
        outcome_bets = bets_df[bets_df['bet_outcome'] == outcome]
        if len(outcome_bets) > 0:
            by_outcome[outcome] = {
                'bets': len(outcome_bets),
                'wins': outcome_bets['won'].sum(),
                'win_rate': (outcome_bets['won'].sum() / len(outcome_bets)) * 100,
                'profit': outcome_bets['profit'].sum(),
                'roi': (outcome_bets['profit'].sum() / len(outcome_bets)) * 100,
                'avg_odds': outcome_bets['bet_odds'].mean()
            }
        else:
            by_outcome[outcome] = {
                'bets': 0, 'wins': 0, 'win_rate': 0,
                'profit': 0, 'roi': 0, 'avg_odds': 0
            }

    return {
        'total_bets': total_bets,
        'total_wins': wins,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate,
        'avg_odds': avg_odds,
        'by_outcome': by_outcome
    }


def print_strategy_results(name: str, metrics: Dict):
    """Print formatted strategy results."""
    print(f"\n{name}")
    print("-" * 80)

    if metrics['total_bets'] == 0:
        print("No bets made")
        return

    print(f"Total Bets:    {metrics['total_bets']}")
    print(f"Wins:          {metrics['total_wins']}")
    print(f"Win Rate:      {metrics['win_rate']:.1f}%")
    print(f"Total Profit:  ${metrics['total_profit']:.2f}")
    print(f"ROI:           {metrics['roi']:.1f}%")
    print(f"Avg Odds:      {metrics['avg_odds']:.2f}")

    print(f"\nBy Outcome:")
    for outcome_name, outcome_code in [('Away', 'A'), ('Draw', 'D'), ('Home', 'H')]:
        o = metrics['by_outcome'][outcome_code]
        if o['bets'] > 0:
            print(f"  {outcome_name:6s}: {o['bets']:3d} bets, {o['wins']:3d} wins ({o['win_rate']:5.1f}%), "
                  f"${o['profit']:7.2f} ({o['roi']:6.1f}% ROI), avg odds {o['avg_odds']:.2f}")
        else:
            print(f"  {outcome_name:6s}: 0 bets")


def main():
    print("="*80)
    print("EV-BASED BETTING STRATEGY ANALYSIS")
    print("="*80)
    print()
    print("Comparing Expected Value strategy vs Probability Threshold strategy")
    print("Dataset: January 2026, Top 5 Leagues")
    print()

    # Load predictions
    df = extract_probabilities(None)

    print("="*80)
    print("CURRENT STRATEGY: PROBABILITY THRESHOLDS")
    print("="*80)
    print(f"Thresholds: Home={PROB_THRESHOLDS['home']}, Draw={PROB_THRESHOLDS['draw']}, Away={PROB_THRESHOLDS['away']}")

    prob_bets = apply_probability_threshold_strategy(df)
    prob_metrics = calculate_metrics(prob_bets)
    print_strategy_results("PROBABILITY THRESHOLD STRATEGY", prob_metrics)

    print("\n")
    print("="*80)
    print("EV-BASED STRATEGIES")
    print("="*80)
    print()

    # Test different EV thresholds
    ev_thresholds = [0.02, 0.05, 0.08, 0.10, 0.15]

    ev_results = []

    for min_ev in ev_thresholds:
        ev_bets = apply_ev_strategy(df, min_ev)
        ev_metrics = calculate_metrics(ev_bets)
        ev_results.append({
            'min_ev': min_ev,
            'metrics': ev_metrics,
            'bets_df': ev_bets
        })

    # Print summary table
    print("EV Threshold Comparison:")
    print("-" * 100)
    print(f"{'Min EV':<10} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'Profit':<12} {'ROI':<10} {'Avg Odds':<10}")
    print("-" * 100)

    for result in ev_results:
        m = result['metrics']
        print(f"{result['min_ev']*100:>6.0f}%    {m['total_bets']:<8} {m['total_wins']:<8} "
              f"{m['win_rate']:>6.1f}%      ${m['total_profit']:>8.2f}   {m['roi']:>6.1f}%    "
              f"{m['avg_odds']:>6.2f}")

    # Current strategy for comparison
    print("-" * 100)
    print(f"{'CURRENT':<10} {prob_metrics['total_bets']:<8} {prob_metrics['total_wins']:<8} "
          f"{prob_metrics['win_rate']:>6.1f}%      ${prob_metrics['total_profit']:>8.2f}   "
          f"{prob_metrics['roi']:>6.1f}%    {prob_metrics['avg_odds']:>6.2f}")
    print("-" * 100)
    print()

    # Detailed breakdown for top 3 EV strategies
    print("\n")
    print("="*80)
    print("DETAILED BREAKDOWN - TOP EV STRATEGIES")
    print("="*80)

    for result in ev_results[:3]:
        print_strategy_results(f"EV STRATEGY (min_ev = {result['min_ev']*100:.0f}%)", result['metrics'])
        print()

    # Find best EV strategy
    best_profit_ev = max(ev_results, key=lambda x: x['metrics']['total_profit'])
    best_roi_ev = max(ev_results, key=lambda x: x['metrics']['roi'])
    best_winrate_ev = max(ev_results, key=lambda x: x['metrics']['win_rate'])

    print("\n")
    print("="*80)
    print("BEST PERFORMERS")
    print("="*80)
    print()

    print(f"🏆 HIGHEST PROFIT:")
    print(f"   EV Strategy (min_ev = {best_profit_ev['min_ev']*100:.0f}%): "
          f"${best_profit_ev['metrics']['total_profit']:.2f}")
    print(f"   vs Current: ${prob_metrics['total_profit']:.2f}")
    print(f"   Improvement: ${best_profit_ev['metrics']['total_profit'] - prob_metrics['total_profit']:.2f} "
          f"({(best_profit_ev['metrics']['total_profit'] - prob_metrics['total_profit']) / prob_metrics['total_profit'] * 100:+.1f}%)")
    print()

    print(f"🎯 HIGHEST ROI:")
    print(f"   EV Strategy (min_ev = {best_roi_ev['min_ev']*100:.0f}%): "
          f"{best_roi_ev['metrics']['roi']:.1f}%")
    print(f"   vs Current: {prob_metrics['roi']:.1f}%")
    print(f"   Improvement: {best_roi_ev['metrics']['roi'] - prob_metrics['roi']:+.1f} percentage points")
    print()

    print(f"✅ HIGHEST WIN RATE:")
    print(f"   EV Strategy (min_ev = {best_winrate_ev['min_ev']*100:.0f}%): "
          f"{best_winrate_ev['metrics']['win_rate']:.1f}%")
    print(f"   vs Current: {prob_metrics['win_rate']:.1f}%")
    print(f"   Improvement: {best_winrate_ev['metrics']['win_rate'] - prob_metrics['win_rate']:+.1f} percentage points")
    print()

    # Recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    # Find balanced option (good profit + reasonable volume)
    balanced = [r for r in ev_results if r['metrics']['total_bets'] >= 100][0] if any(r['metrics']['total_bets'] >= 100 for r in ev_results) else ev_results[0]

    profit_improvement = balanced['metrics']['total_profit'] - prob_metrics['total_profit']
    profit_improvement_pct = (profit_improvement / prob_metrics['total_profit']) * 100

    if profit_improvement > 0:
        print(f"✅ SWITCH TO EV-BASED STRATEGY")
        print(f"   Recommended: min_ev = {balanced['min_ev']*100:.0f}%")
        print(f"   Expected Performance:")
        print(f"   - Profit: ${balanced['metrics']['total_profit']:.2f} ({profit_improvement_pct:+.1f}% vs current)")
        print(f"   - ROI: {balanced['metrics']['roi']:.1f}% ({balanced['metrics']['roi'] - prob_metrics['roi']:+.1f}pp)")
        print(f"   - Win Rate: {balanced['metrics']['win_rate']:.1f}% ({balanced['metrics']['win_rate'] - prob_metrics['win_rate']:+.1f}pp)")
        print(f"   - Volume: {balanced['metrics']['total_bets']} bets")
    else:
        print(f"⚠️  KEEP CURRENT PROBABILITY STRATEGY")
        print(f"   EV strategy does not show significant improvement on this dataset")
        print(f"   Current strategy performs better or comparably")

    print()
    print("="*80)
    print()


if __name__ == '__main__':
    main()
