#!/usr/bin/env python3
"""
Calculate Actual PnL Using Real Odds Data
==========================================

Loads odds from raw JSON fixtures and calculates real PnL.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"
FIXTURES_DIR = Path(__file__).parent.parent / "data" / "historical" / "fixtures"

META_COLS = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target'
]


def load_odds_from_json():
    """Load 1x2 odds from raw JSON files."""
    print("Loading odds from raw JSON files...")

    odds_data = {}  # fixture_id -> {home: avg_odds, away: avg_odds, draw: avg_odds}

    files = list(FIXTURES_DIR.glob('*.json'))
    print(f"Found {len(files)} JSON files")

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            if not isinstance(data, list):
                continue

            for fixture in data:
                fixture_id = fixture.get('id')
                if not fixture_id or 'odds' not in fixture or not fixture['odds']:
                    continue

                # Collect odds for each outcome
                home_odds = []
                away_odds = []
                draw_odds = []

                for odd in fixture['odds']:
                    market_desc = odd.get('market_description', '')
                    if market_desc in ['Fulltime Result', 'Full Time Result', 'Match Winner', '3Way Result', '1X2']:
                        label = odd.get('label', '')
                        try:
                            value = float(odd.get('value', 0))
                            if value > 1:  # Valid odds
                                if label == 'Home':
                                    home_odds.append(value)
                                elif label == 'Away':
                                    away_odds.append(value)
                                elif label == 'Draw':
                                    draw_odds.append(value)
                        except:
                            pass

                if home_odds and away_odds and draw_odds:
                    odds_data[fixture_id] = {
                        'home': np.mean(home_odds),
                        'away': np.mean(away_odds),
                        'draw': np.mean(draw_odds)
                    }
        except Exception as e:
            continue

    print(f"Loaded odds for {len(odds_data)} fixtures")
    return odds_data


def load_data_with_odds(odds_data):
    """Load training data and merge with odds."""
    print("\nLoading training data...")
    df = pd.read_csv(DATA_PATH)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)
    df['target'] = df['result'].map({'A': 0, 'D': 1, 'H': 2})

    # Get test set
    n = len(df)
    test_df = df.iloc[int(n*0.85):].copy()

    # Add odds
    test_df['odds_home'] = test_df['fixture_id'].map(lambda x: odds_data.get(x, {}).get('home', np.nan))
    test_df['odds_away'] = test_df['fixture_id'].map(lambda x: odds_data.get(x, {}).get('away', np.nan))
    test_df['odds_draw'] = test_df['fixture_id'].map(lambda x: odds_data.get(x, {}).get('draw', np.nan))

    # Filter to only fixtures with odds
    has_odds = ~test_df['odds_home'].isna()
    test_with_odds = test_df[has_odds].copy()

    print(f"Test set: {len(test_df)} matches")
    print(f"With odds: {len(test_with_odds)} matches ({len(test_with_odds)/len(test_df)*100:.1f}%)")

    return test_df, test_with_odds


def calculate_pnl(probs, test_df, thresholds, include_draws=True, stake=100):
    """Calculate PnL using actual odds."""
    result_map = {'H': 2, 'A': 0, 'D': 1}

    results = {
        'home': {'bets': 0, 'wins': 0, 'stake': 0, 'returns': 0, 'odds_list': []},
        'away': {'bets': 0, 'wins': 0, 'stake': 0, 'returns': 0, 'odds_list': []},
        'draw': {'bets': 0, 'wins': 0, 'stake': 0, 'returns': 0, 'odds_list': []}
    }

    for i in range(len(probs)):
        row = test_df.iloc[i]
        p_away, p_draw, p_home = probs[i]
        true_label = result_map[row['result']]

        odds_home = row['odds_home']
        odds_away = row['odds_away']
        odds_draw = row['odds_draw']

        if pd.isna(odds_home) or pd.isna(odds_away) or pd.isna(odds_draw):
            continue

        candidates = []
        if p_home >= thresholds['home']:
            candidates.append(('home', p_home, 2, odds_home))
        if p_away >= thresholds['away']:
            candidates.append(('away', p_away, 0, odds_away))
        if include_draws and p_draw >= thresholds['draw']:
            candidates.append(('draw', p_draw, 1, odds_draw))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            outcome, prob, label, odds = best

            results[outcome]['bets'] += 1
            results[outcome]['stake'] += stake
            results[outcome]['odds_list'].append(odds)

            if label == true_label:
                results[outcome]['wins'] += 1
                results[outcome]['returns'] += stake * odds

    return results


def print_pnl_results(results, title):
    """Print PnL results."""
    print(f"\n{title}")
    print("="*80)

    total_bets = sum(r['bets'] for r in results.values())
    total_wins = sum(r['wins'] for r in results.values())
    total_stake = sum(r['stake'] for r in results.values())
    total_returns = sum(r['returns'] for r in results.values())
    total_profit = total_returns - total_stake

    if total_bets == 0:
        print("No bets placed")
        return

    roi = total_profit / total_stake * 100 if total_stake > 0 else 0
    wr = total_wins / total_bets * 100

    print(f"Total bets: {total_bets}")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Total stake: {total_stake:,.0f} units")
    print(f"Total returns: {total_returns:,.0f} units")
    print(f"PROFIT/LOSS: {total_profit:+,.0f} units")
    print(f"ROI: {roi:+.1f}%")
    print()

    for outcome in ['home', 'away', 'draw']:
        r = results[outcome]
        if r['bets'] > 0:
            profit = r['returns'] - r['stake']
            roi = profit / r['stake'] * 100
            wr = r['wins'] / r['bets'] * 100
            avg_odds = np.mean(r['odds_list']) if r['odds_list'] else 0
            print(f"  {outcome.upper()}: {r['bets']} bets, {r['wins']} wins ({wr:.1f}% WR), "
                  f"Avg odds: {avg_odds:.2f}, P/L: {profit:+,.0f}, ROI: {roi:+.1f}%")
        else:
            print(f"  {outcome.upper()}: 0 bets")

    return total_profit, roi


def main():
    # Load odds
    odds_data = load_odds_from_json()

    # Load data
    test_df_all, test_df = load_data_with_odds(odds_data)

    feature_cols = [c for c in test_df.columns if c not in META_COLS + ['odds_home', 'odds_away', 'odds_draw']]
    X_test = test_df[feature_cols].values

    test_weeks = (test_df['match_date'].max() - test_df['match_date'].min()).days / 7
    print(f"Test period: {test_df['match_date'].min().date()} to {test_df['match_date'].max().date()} ({test_weeks:.1f} weeks)")

    # Load models
    print("\nLoading models...")
    model_v2 = joblib.load('models/production/model_v2.0.0.joblib')
    probs_v2 = model_v2.predict_proba(X_test)

    model_v3 = joblib.load('models/production/model_v3.0.0.joblib')
    probs_v3 = (model_v3['catboost'].predict_proba(X_test) + model_v3['lightgbm'].predict_proba(X_test)) / 2

    # =========================================================================
    # v2.0.0 Results
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL v2.0.0")
    print("="*80)

    # Best config with draws
    thresh_v2_best = {'home': 0.65, 'away': 0.60, 'draw': 0.40}
    results = calculate_pnl(probs_v2, test_df, thresh_v2_best, include_draws=True)
    print_pnl_results(results, "v2.0.0 Best (H=0.65, A=0.60, D=0.40) - WITH Draws")

    # H/A only
    results = calculate_pnl(probs_v2, test_df, thresh_v2_best, include_draws=False)
    print_pnl_results(results, "v2.0.0 Best (H=0.65, A=0.60) - H/A ONLY")

    # Production thresholds
    thresh_v2_prod = {'home': 0.36, 'draw': 0.28, 'away': 0.40}
    results = calculate_pnl(probs_v2, test_df, thresh_v2_prod, include_draws=True)
    print_pnl_results(results, "v2.0.0 Production (H=0.36, A=0.40, D=0.28) - WITH Draws")

    results = calculate_pnl(probs_v2, test_df, thresh_v2_prod, include_draws=False)
    print_pnl_results(results, "v2.0.0 Production (H=0.36, A=0.40) - H/A ONLY")

    # =========================================================================
    # v3.0.0 Results
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL v3.0.0")
    print("="*80)

    # Best config with draws
    thresh_v3_best = {'home': 0.45, 'away': 0.60, 'draw': 0.45}
    results = calculate_pnl(probs_v3, test_df, thresh_v3_best, include_draws=True)
    print_pnl_results(results, "v3.0.0 Best (H=0.45, A=0.60, D=0.45) - WITH Draws")

    # H/A only
    results = calculate_pnl(probs_v3, test_df, thresh_v3_best, include_draws=False)
    print_pnl_results(results, "v3.0.0 Best (H=0.45, A=0.60) - H/A ONLY")

    # With production thresholds
    results = calculate_pnl(probs_v3, test_df, thresh_v2_prod, include_draws=True)
    print_pnl_results(results, "v3.0.0 with v2 Production (H=0.36, A=0.40, D=0.28) - WITH Draws")

    results = calculate_pnl(probs_v3, test_df, thresh_v2_prod, include_draws=False)
    print_pnl_results(results, "v3.0.0 with v2 Production (H=0.36, A=0.40) - H/A ONLY")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: BEST H/A ONLY OPTIONS")
    print("="*80)

    configs = [
        ('v2.0.0', probs_v2, {'home': 0.65, 'away': 0.60, 'draw': 0.40}),
        ('v2.0.0', probs_v2, {'home': 0.36, 'away': 0.40, 'draw': 0.28}),
        ('v3.0.0', probs_v3, {'home': 0.45, 'away': 0.60, 'draw': 0.45}),
        ('v3.0.0', probs_v3, {'home': 0.36, 'away': 0.40, 'draw': 0.28}),
    ]

    print(f"\n{'Model':<10} {'Thresholds':<20} {'Bets':<8} {'WR':<8} {'Profit':<12} {'ROI':<10}")
    print("-"*70)

    for name, probs, thresh in configs:
        results = calculate_pnl(probs, test_df, thresh, include_draws=False)
        total_bets = sum(r['bets'] for r in results.values())
        total_wins = sum(r['wins'] for r in results.values())
        total_stake = sum(r['stake'] for r in results.values())
        total_returns = sum(r['returns'] for r in results.values())
        total_profit = total_returns - total_stake
        wr = total_wins / total_bets * 100 if total_bets > 0 else 0
        roi = total_profit / total_stake * 100 if total_stake > 0 else 0

        thresh_str = f"H={thresh['home']}/A={thresh['away']}"
        print(f"{name:<10} {thresh_str:<20} {total_bets:<8} {wr:<8.1f}% {total_profit:<+12,.0f} {roi:<+10.1f}%")


if __name__ == '__main__':
    main()
