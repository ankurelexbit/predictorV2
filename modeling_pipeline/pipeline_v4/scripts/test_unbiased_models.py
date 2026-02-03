#!/usr/bin/env python3
"""
Test Unbiased Models
====================

Tests unbiased base model and calibrated version with both
probability and EV strategies.

Compares:
1. Unbiased Base + Probability Thresholds
2. Unbiased Base + EV Strategy
3. Unbiased Calibrated + Probability Thresholds
4. Unbiased Calibrated + EV Strategy
5. Option 3 (Biased) + Probability Thresholds (CURRENT PRODUCTION)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')
TOP_5_LEAGUES = [8, 82, 384, 564, 301]

PROB_THRESHOLDS = {'home': 0.65, 'draw': 0.30, 'away': 0.42}


def calculate_ev(prob, odds):
    return (prob * odds) - 1


def load_data_from_db():
    """Load features, odds, and results from database."""
    query = """
        SELECT fixture_id, home_team_name, away_team_name,
               features, best_home_odds, best_draw_odds, best_away_odds,
               actual_result
        FROM predictions
        WHERE match_date >= '2026-01-01' AND match_date < '2026-02-01'
          AND actual_result IS NOT NULL
          AND league_id = ANY(%s)
          AND best_home_odds IS NOT NULL
          AND features IS NOT NULL
    """

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query, (TOP_5_LEAGUES,))
    rows = cursor.fetchall()
    conn.close()

    feature_dicts = []
    metadata = []

    for row in rows:
        feature_dicts.append(row['features'])
        metadata.append({
            'fixture_id': row['fixture_id'],
            'home_team': row['home_team_name'],
            'away_team': row['away_team_name'],
            'best_home_odds': row['best_home_odds'],
            'best_draw_odds': row['best_draw_odds'],
            'best_away_odds': row['best_away_odds'],
            'actual_result': row['actual_result']
        })

    X = pd.DataFrame(feature_dicts)

    return X, metadata


def apply_probability_strategy(probas, metadata):
    """Apply probability threshold strategy."""
    results = []

    for proba, meta in zip(probas, metadata):
        home_prob, draw_prob, away_prob = proba[2], proba[1], proba[0]

        candidates = []
        if home_prob > PROB_THRESHOLDS['home']:
            candidates.append(('H', home_prob, meta['best_home_odds']))
        if draw_prob > PROB_THRESHOLDS['draw']:
            candidates.append(('D', draw_prob, meta['best_draw_odds']))
        if away_prob > PROB_THRESHOLDS['away']:
            candidates.append(('A', away_prob, meta['best_away_odds']))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]

            won = (meta['actual_result'] == best[0])
            profit = (best[2] - 1) if won else -1

            results.append({
                **meta,
                'bet_outcome': best[0],
                'bet_prob': best[1],
                'bet_odds': best[2],
                'won': won,
                'profit': profit
            })

    return pd.DataFrame(results)


def apply_ev_strategy(probas, metadata, min_ev):
    """Apply EV-based betting strategy."""
    results = []

    for proba, meta in zip(probas, metadata):
        home_prob, draw_prob, away_prob = proba[2], proba[1], proba[0]

        ev_home = calculate_ev(home_prob, meta['best_home_odds'])
        ev_draw = calculate_ev(draw_prob, meta['best_draw_odds'])
        ev_away = calculate_ev(away_prob, meta['best_away_odds'])

        opportunities = []
        if ev_home > min_ev:
            opportunities.append(('H', ev_home, home_prob, meta['best_home_odds']))
        if ev_draw > min_ev:
            opportunities.append(('D', ev_draw, draw_prob, meta['best_draw_odds']))
        if ev_away > min_ev:
            opportunities.append(('A', ev_away, away_prob, meta['best_away_odds']))

        if opportunities:
            opportunities.sort(key=lambda x: x[1], reverse=True)
            best = opportunities[0]

            won = (meta['actual_result'] == best[0])
            profit = (best[3] - 1) if won else -1

            results.append({
                **meta,
                'bet_outcome': best[0],
                'bet_ev': best[1],
                'bet_prob': best[2],
                'bet_odds': best[3],
                'won': won,
                'profit': profit
            })

    return pd.DataFrame(results)


def calculate_metrics(bets_df):
    """Calculate performance metrics."""
    if len(bets_df) == 0:
        return {'total_bets': 0, 'total_wins': 0, 'total_profit': 0,
                'roi': 0, 'win_rate': 0, 'avg_odds': 0}

    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    total_profit = bets_df['profit'].sum()
    roi = (total_profit / total_bets) * 100
    win_rate = (wins / total_bets) * 100
    avg_odds = bets_df['bet_odds'].mean()

    return {
        'total_bets': total_bets,
        'total_wins': wins,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate,
        'avg_odds': avg_odds
    }


def print_results(name, metrics):
    """Print formatted results."""
    print(f"\n{name}")
    print("-" * 100)

    if metrics['total_bets'] == 0:
        print("No bets made")
        return

    print(f"Total Bets:   {metrics['total_bets']}")
    print(f"Wins:         {metrics['total_wins']}")
    print(f"Win Rate:     {metrics['win_rate']:.1f}%")
    print(f"Total Profit: ${metrics['total_profit']:.2f}")
    print(f"ROI:          {metrics['roi']:.1f}%")
    print(f"Avg Odds:     {metrics['avg_odds']:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test unbiased models')
    parser.add_argument('--unbiased-base', type=str, required=True,
                       help='Path to unbiased base model')
    parser.add_argument('--unbiased-calibrated', type=str, required=True,
                       help='Path to unbiased calibrated model')
    parser.add_argument('--option3-model', type=str,
                       default='models/weight_experiments/option3_balanced.joblib',
                       help='Path to Option 3 model (for comparison)')
    parser.add_argument('--min-ev', type=float, default=0.05,
                       help='Minimum EV threshold')

    args = parser.parse_args()

    print("="*100)
    print("UNBIASED MODELS TEST")
    print("="*100)
    print(f"Unbiased base: {args.unbiased_base}")
    print(f"Unbiased calibrated: {args.unbiased_calibrated}")
    print(f"Option 3 (current): {args.option3_model}")
    print(f"Min EV: {args.min_ev*100:.0f}%")
    print()

    # Load data
    print("Loading data from database...")
    X, metadata = load_data_from_db()
    print(f"✅ Loaded {len(X)} samples\n")

    # Load models
    print("Loading models...")
    unbiased_base = joblib.load(args.unbiased_base)
    unbiased_calib = joblib.load(args.unbiased_calibrated)
    option3 = joblib.load(args.option3_model)
    print("✅ Models loaded\n")

    # Get predictions
    print("Generating predictions...")
    unbiased_base_proba = unbiased_base.predict_proba(X)
    unbiased_calib_proba = unbiased_calib.predict_proba(X)
    option3_proba = option3.predict_proba(X)
    print("✅ Predictions generated\n")

    # Test all strategies
    results = []

    # 1. Unbiased Base + Probability
    print("="*100)
    print("TEST 1: UNBIASED BASE + PROBABILITY THRESHOLDS")
    print("="*100)
    bets = apply_probability_strategy(unbiased_base_proba, metadata)
    metrics = calculate_metrics(bets)
    print_results("Unbiased Base + Probability", metrics)
    results.append(("Unbiased Base + Probability", metrics))

    # 2. Unbiased Base + EV
    print("\n")
    print("="*100)
    print("TEST 2: UNBIASED BASE + EV STRATEGY")
    print("="*100)
    bets = apply_ev_strategy(unbiased_base_proba, metadata, args.min_ev)
    metrics = calculate_metrics(bets)
    print_results(f"Unbiased Base + EV ({args.min_ev*100:.0f}%)", metrics)
    results.append((f"Unbiased Base + EV ({args.min_ev*100:.0f}%)", metrics))

    # 3. Unbiased Calibrated + Probability
    print("\n")
    print("="*100)
    print("TEST 3: UNBIASED CALIBRATED + PROBABILITY THRESHOLDS")
    print("="*100)
    bets = apply_probability_strategy(unbiased_calib_proba, metadata)
    metrics = calculate_metrics(bets)
    print_results("Unbiased Calibrated + Probability", metrics)
    results.append(("Unbiased Calibrated + Probability", metrics))

    # 4. Unbiased Calibrated + EV
    print("\n")
    print("="*100)
    print("TEST 4: UNBIASED CALIBRATED + EV STRATEGY")
    print("="*100)
    bets = apply_ev_strategy(unbiased_calib_proba, metadata, args.min_ev)
    metrics = calculate_metrics(bets)
    print_results(f"Unbiased Calibrated + EV ({args.min_ev*100:.0f}%)", metrics)
    results.append((f"Unbiased Calibrated + EV ({args.min_ev*100:.0f}%)", metrics))

    # 5. Option 3 + Probability (CURRENT)
    print("\n")
    print("="*100)
    print("TEST 5: OPTION 3 (BIASED) + PROBABILITY (CURRENT PRODUCTION)")
    print("="*100)
    bets = apply_probability_strategy(option3_proba, metadata)
    metrics = calculate_metrics(bets)
    print_results("Option 3 + Probability (CURRENT)", metrics)
    results.append(("Option 3 + Probability (CURRENT)", metrics))

    # Summary
    print("\n")
    print("="*100)
    print("SUMMARY COMPARISON")
    print("="*100)
    print()

    print(f"{'Strategy':<50} {'Bets':<8} {'Win Rate':<12} {'Profit':<12} {'ROI':<10}")
    print("-" * 100)

    for name, m in results:
        print(f"{name:<50} {m['total_bets']:<8} {m['win_rate']:>6.1f}%      "
              f"${m['total_profit']:>8.2f}   {m['roi']:>6.1f}%")

    print("-" * 100)

    # Find best
    best_profit = max(results, key=lambda x: x[1]['total_profit'])
    best_roi = max(results, key=lambda x: x[1]['roi'])
    best_wr = max(results, key=lambda x: x[1]['win_rate'])

    print()
    print("🏆 BEST PROFIT:")
    print(f"   {best_profit[0]}: ${best_profit[1]['total_profit']:.2f}")
    print()
    print("🎯 BEST ROI:")
    print(f"   {best_roi[0]}: {best_roi[1]['roi']:.1f}%")
    print()
    print("✅ BEST WIN RATE:")
    print(f"   {best_wr[0]}: {best_wr[1]['win_rate']:.1f}%")
    print()

    # Key finding
    print("="*100)
    print("KEY FINDINGS")
    print("="*100)
    print()

    # Compare unbiased vs biased
    unbiased_best = max([r for r in results if 'Unbiased' in r[0]],
                        key=lambda x: x[1]['total_profit'])
    option3_result = [r for r in results if 'Option 3' in r[0]][0]

    print(f"Best Unbiased Strategy: {unbiased_best[0]}")
    print(f"  Profit: ${unbiased_best[1]['total_profit']:.2f}")
    print(f"  ROI: {unbiased_best[1]['roi']:.1f}%")
    print()
    print(f"Current Option 3 (Biased): {option3_result[0]}")
    print(f"  Profit: ${option3_result[1]['total_profit']:.2f}")
    print(f"  ROI: {option3_result[1]['roi']:.1f}%")
    print()

    if unbiased_best[1]['total_profit'] > option3_result[1]['total_profit']:
        diff = unbiased_best[1]['total_profit'] - option3_result[1]['total_profit']
        pct = (diff / option3_result[1]['total_profit']) * 100
        print(f"✅ UNBIASED STRATEGY IS BETTER!")
        print(f"   Improvement: ${diff:.2f} ({pct:+.1f}%)")
        print(f"   RECOMMENDATION: Switch to unbiased model")
    else:
        diff = option3_result[1]['total_profit'] - unbiased_best[1]['total_profit']
        pct = (diff / option3_result[1]['total_profit']) * 100
        print(f"⚠️  OPTION 3 (BIASED) IS STILL BETTER")
        print(f"   Advantage: ${diff:.2f} ({pct:.1f}%)")
        print(f"   RECOMMENDATION: Keep current Option 3 strategy")

    print()
    print("="*100)


if __name__ == '__main__':
    main()
