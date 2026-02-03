#!/usr/bin/env python3
"""
Test Calibrated Model with EV Strategy
=======================================

Tests the calibrated model using EV-based betting strategy on January 2026 data.
Compares:
1. Uncalibrated model + Probability thresholds (current production)
2. Uncalibrated model + EV strategy
3. Calibrated model + EV strategy
4. Calibrated model + Probability thresholds
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://ankurgupta@localhost/football_predictions')
TOP_5_LEAGUES = [8, 82, 384, 564, 301]

PROB_THRESHOLDS = {'home': 0.65, 'draw': 0.30, 'away': 0.42}


def calculate_ev(prob, odds):
    return (prob * odds) - 1


def load_data_from_db():
    """Load features, odds, and results from database."""
    print("="*100)
    print("LOADING DATA FROM DATABASE")
    print("="*100)

    query = """
        SELECT
            fixture_id, match_date, league_id,
            home_team_name, away_team_name,
            features,
            best_home_odds, best_draw_odds, best_away_odds,
            actual_result,
            pred_home_prob as original_home_prob,
            pred_draw_prob as original_draw_prob,
            pred_away_prob as original_away_prob
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

    print(f"✅ Loaded {len(rows)} predictions from January 2026")
    print()

    return rows


def extract_features(rows):
    """Extract feature vectors from database."""
    feature_dicts = []
    metadata = []

    for row in rows:
        features = row['features']
        feature_dicts.append(features)
        metadata.append({
            'fixture_id': row['fixture_id'],
            'match_date': row['match_date'],
            'home_team': row['home_team_name'],
            'away_team': row['away_team_name'],
            'best_home_odds': row['best_home_odds'],
            'best_draw_odds': row['best_draw_odds'],
            'best_away_odds': row['best_away_odds'],
            'actual_result': row['actual_result'],
            'original_home_prob': row['original_home_prob'],
            'original_draw_prob': row['original_draw_prob'],
            'original_away_prob': row['original_away_prob']
        })

    # Convert to DataFrame
    X = pd.DataFrame(feature_dicts)

    print(f"✅ Extracted {len(X)} feature vectors with {len(X.columns)} features")
    print()

    return X, metadata


def predict_with_calibrated_model(model_path, X):
    """Generate predictions using calibrated model."""
    print(f"Loading calibrated model: {model_path}")

    model = joblib.load(model_path)

    print("Generating predictions with calibrated model...")
    proba = model.predict_proba(X)

    print(f"✅ Generated {len(proba)} predictions")
    print()

    return proba


def apply_ev_strategy(probas, metadata, min_ev):
    """Apply EV-based betting strategy."""
    results = []

    for i, (proba, meta) in enumerate(zip(probas, metadata)):
        home_prob, draw_prob, away_prob = proba[2], proba[1], proba[0]  # H, D, A

        # Calculate EV for each outcome
        ev_home = calculate_ev(home_prob, meta['best_home_odds'])
        ev_draw = calculate_ev(draw_prob, meta['best_draw_odds'])
        ev_away = calculate_ev(away_prob, meta['best_away_odds'])

        # Find best EV opportunity
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
                'profit': profit,
                'home_prob': home_prob,
                'draw_prob': draw_prob,
                'away_prob': away_prob
            })

    return pd.DataFrame(results)


def apply_probability_strategy(probas, metadata):
    """Apply probability threshold strategy."""
    results = []

    for proba, meta in zip(probas, metadata):
        home_prob, draw_prob, away_prob = proba[2], proba[1], proba[0]

        # Check thresholds
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
                'profit': profit,
                'home_prob': home_prob,
                'draw_prob': draw_prob,
                'away_prob': away_prob
            })

    return pd.DataFrame(results)


def calculate_metrics(bets_df):
    """Calculate performance metrics."""
    if len(bets_df) == 0:
        return {
            'total_bets': 0, 'total_wins': 0, 'total_profit': 0,
            'roi': 0, 'win_rate': 0, 'avg_odds': 0,
            'by_outcome': {'H': {}, 'D': {}, 'A': {}}
        }

    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    total_profit = bets_df['profit'].sum()
    roi = (total_profit / total_bets) * 100
    win_rate = (wins / total_bets) * 100
    avg_odds = bets_df['bet_odds'].mean()

    by_outcome = {}
    for outcome in ['H', 'D', 'A']:
        outcome_bets = bets_df[bets_df['bet_outcome'] == outcome]
        if len(outcome_bets) > 0:
            by_outcome[outcome] = {
                'bets': len(outcome_bets),
                'wins': outcome_bets['won'].sum(),
                'win_rate': (outcome_bets['won'].sum() / len(outcome_bets)) * 100,
                'profit': outcome_bets['profit'].sum(),
                'roi': (outcome_bets['profit'].sum() / len(outcome_bets)) * 100
            }
        else:
            by_outcome[outcome] = {
                'bets': 0, 'wins': 0, 'win_rate': 0, 'profit': 0, 'roi': 0
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

    print(f"\nBy Outcome:")
    for outcome_name, outcome_code in [('Home', 'H'), ('Draw', 'D'), ('Away', 'A')]:
        o = metrics['by_outcome'][outcome_code]
        if o['bets'] > 0:
            print(f"  {outcome_name:6s}: {o['bets']:3d} bets, {o['wins']:3d} wins "
                  f"({o['win_rate']:5.1f}%), ${o['profit']:7.2f} ({o['roi']:6.1f}% ROI)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test calibrated model with EV strategy')
    parser.add_argument('--calibrated-model', type=str, required=True,
                       help='Path to calibrated model')
    parser.add_argument('--uncalibrated-model', type=str,
                       default='models/weight_experiments/option3_balanced.joblib',
                       help='Path to uncalibrated model (for comparison)')
    parser.add_argument('--min-ev', type=float, default=0.05,
                       help='Minimum EV threshold (default: 0.05 = 5%%)')

    args = parser.parse_args()

    calibrated_path = Path(args.calibrated_model)
    uncalibrated_path = Path(args.uncalibrated_model)

    print("="*100)
    print("CALIBRATED MODEL + EV STRATEGY TEST")
    print("="*100)
    print(f"Calibrated model: {calibrated_path}")
    print(f"Uncalibrated model: {uncalibrated_path}")
    print(f"Min EV threshold: {args.min_ev*100:.0f}%")
    print()

    # Load data
    rows = load_data_from_db()
    X, metadata = extract_features(rows)

    # Test 1: Uncalibrated + Probability (current production)
    print("="*100)
    print("TEST 1: UNCALIBRATED MODEL + PROBABILITY THRESHOLDS (CURRENT)")
    print("="*100)

    uncalib_model = joblib.load(uncalibrated_path)
    uncalib_proba = uncalib_model.predict_proba(X)
    uncalib_prob_bets = apply_probability_strategy(uncalib_proba, metadata)
    uncalib_prob_metrics = calculate_metrics(uncalib_prob_bets)
    print_results("Uncalibrated + Probability Thresholds", uncalib_prob_metrics)

    # Test 2: Uncalibrated + EV
    print("\n")
    print("="*100)
    print("TEST 2: UNCALIBRATED MODEL + EV STRATEGY")
    print("="*100)

    uncalib_ev_bets = apply_ev_strategy(uncalib_proba, metadata, args.min_ev)
    uncalib_ev_metrics = calculate_metrics(uncalib_ev_bets)
    print_results(f"Uncalibrated + EV (min_ev={args.min_ev*100:.0f}%)", uncalib_ev_metrics)

    # Test 3: Calibrated + EV
    print("\n")
    print("="*100)
    print("TEST 3: CALIBRATED MODEL + EV STRATEGY")
    print("="*100)

    calib_proba = predict_with_calibrated_model(calibrated_path, X)
    calib_ev_bets = apply_ev_strategy(calib_proba, metadata, args.min_ev)
    calib_ev_metrics = calculate_metrics(calib_ev_bets)
    print_results(f"Calibrated + EV (min_ev={args.min_ev*100:.0f}%)", calib_ev_metrics)

    # Test 4: Calibrated + Probability
    print("\n")
    print("="*100)
    print("TEST 4: CALIBRATED MODEL + PROBABILITY THRESHOLDS")
    print("="*100)

    calib_prob_bets = apply_probability_strategy(calib_proba, metadata)
    calib_prob_metrics = calculate_metrics(calib_prob_bets)
    print_results("Calibrated + Probability Thresholds", calib_prob_metrics)

    # Summary comparison
    print("\n")
    print("="*100)
    print("SUMMARY COMPARISON")
    print("="*100)
    print()

    results = [
        ("Uncalibrated + Probability (CURRENT)", uncalib_prob_metrics),
        (f"Uncalibrated + EV ({args.min_ev*100:.0f}%)", uncalib_ev_metrics),
        (f"Calibrated + EV ({args.min_ev*100:.0f}%)", calib_ev_metrics),
        ("Calibrated + Probability", calib_prob_metrics)
    ]

    print(f"{'Strategy':<45} {'Bets':<8} {'Win Rate':<12} {'Profit':<12} {'ROI':<10}")
    print("-" * 100)

    for name, metrics in results:
        print(f"{name:<45} {metrics['total_bets']:<8} {metrics['win_rate']:>6.1f}%      "
              f"${metrics['total_profit']:>8.2f}   {metrics['roi']:>6.1f}%")

    print("-" * 100)

    # Find best
    best_profit = max(results, key=lambda x: x[1]['total_profit'])
    best_roi = max(results, key=lambda x: x[1]['roi'])

    print()
    print("🏆 BEST PROFIT:")
    print(f"   {best_profit[0]}: ${best_profit[1]['total_profit']:.2f}")
    print()
    print("🎯 BEST ROI:")
    print(f"   {best_roi[0]}: {best_roi[1]['roi']:.1f}%")
    print()

    # Recommendation
    print("="*100)
    print("RECOMMENDATION")
    print("="*100)
    print()

    if calib_ev_metrics['total_profit'] > uncalib_prob_metrics['total_profit']:
        improvement = ((calib_ev_metrics['total_profit'] - uncalib_prob_metrics['total_profit']) /
                      uncalib_prob_metrics['total_profit'] * 100)
        print(f"✅ SWITCH TO CALIBRATED MODEL + EV STRATEGY")
        print(f"   Improvement: +${calib_ev_metrics['total_profit'] - uncalib_prob_metrics['total_profit']:.2f} "
              f"({improvement:+.1f}%)")
        print(f"   New expected profit: ${calib_ev_metrics['total_profit']:.2f}")
        print(f"   New expected ROI: {calib_ev_metrics['roi']:.1f}%")
    else:
        print(f"⚠️  KEEP CURRENT STRATEGY (Uncalibrated + Probability)")
        print(f"   Current strategy still performs best")
        print(f"   Calibration helped but not enough to justify EV strategy")

    print()
    print("="*100)


if __name__ == '__main__':
    main()
