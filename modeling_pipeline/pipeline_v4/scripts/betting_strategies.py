#!/usr/bin/env python3
"""
Betting Strategies to Improve Results
=====================================

Strategies tested:
1. Value Betting - Only bet when model_prob > implied_prob (edge > 0)
2. Edge Threshold - Only bet when edge > X%
3. Odds Range Filtering - Avoid very low/high odds
4. Kelly Criterion - Optimal bet sizing based on edge
5. Combined Strategies - Multiple filters together
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FIXTURES_DIR = Path(__file__).parent.parent / "data" / "historical" / "fixtures"
DATA_PATH = Path(__file__).parent.parent / "data" / "training_data.csv"

META_COLS = ['fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
             'match_date', 'home_score', 'away_score', 'result', 'target']


def load_data_with_odds():
    """Load test data with best odds."""
    # Load odds
    odds_data = {}
    for f in FIXTURES_DIR.glob('*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if not isinstance(data, list):
                continue
            for fixture in data:
                fixture_id = fixture.get('id')
                if not fixture_id or 'odds' not in fixture or not fixture['odds']:
                    continue
                home_odds, away_odds, draw_odds = [], [], []
                for odd in fixture['odds']:
                    market_desc = odd.get('market_description', '')
                    if market_desc in ['Fulltime Result', 'Full Time Result', 'Match Winner', '3Way Result', '1X2']:
                        label = odd.get('label', '')
                        try:
                            value = float(odd.get('value', 0))
                            if value > 1:
                                if label == 'Home': home_odds.append(value)
                                elif label == 'Away': away_odds.append(value)
                                elif label == 'Draw': draw_odds.append(value)
                        except: pass
                if home_odds and away_odds and draw_odds:
                    odds_data[fixture_id] = {
                        'home_best': max(home_odds),
                        'away_best': max(away_odds),
                        'draw_best': max(draw_odds)
                    }
        except: continue

    # Load data
    df = pd.read_csv(DATA_PATH)
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)
    df['target'] = df['result'].map({'A': 0, 'D': 1, 'H': 2})

    n = len(df)
    test_df = df.iloc[int(n*0.85):].copy()

    for col in ['home_best', 'away_best', 'draw_best']:
        test_df[col] = test_df['fixture_id'].map(lambda x: odds_data.get(x, {}).get(col, np.nan))

    test_df = test_df[~test_df['home_best'].isna()].copy()
    feature_cols = [c for c in test_df.columns if c not in META_COLS + ['home_best', 'away_best', 'draw_best']]

    return test_df, feature_cols


def calculate_edge(model_prob, odds):
    """Calculate edge: model_prob - implied_prob."""
    implied_prob = 1 / odds
    return model_prob - implied_prob


def strategy_baseline(probs, test_df, thresholds, include_draws=False):
    """Baseline: Just use confidence thresholds."""
    result_map = {'H': 2, 'A': 0, 'D': 1}
    bets = []

    for i in range(len(probs)):
        row = test_df.iloc[i]
        p_away, p_draw, p_home = probs[i]
        true_label = result_map[row['result']]

        candidates = []
        if p_home >= thresholds['home']:
            candidates.append(('home', p_home, 2, row['home_best']))
        if p_away >= thresholds['away']:
            candidates.append(('away', p_away, 0, row['away_best']))
        if include_draws and p_draw >= thresholds['draw']:
            candidates.append(('draw', p_draw, 1, row['draw_best']))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            outcome, prob, label, odds = best
            won = 1 if label == true_label else 0
            bets.append({'outcome': outcome, 'odds': odds, 'won': won, 'stake': 1})

    return bets


def strategy_value_betting(probs, test_df, thresholds, min_edge=0.0, include_draws=False):
    """Value Betting: Only bet when edge > min_edge."""
    result_map = {'H': 2, 'A': 0, 'D': 1}
    bets = []

    for i in range(len(probs)):
        row = test_df.iloc[i]
        p_away, p_draw, p_home = probs[i]
        true_label = result_map[row['result']]

        candidates = []

        # Home
        if p_home >= thresholds['home']:
            edge = calculate_edge(p_home, row['home_best'])
            if edge >= min_edge:
                candidates.append(('home', p_home, 2, row['home_best'], edge))

        # Away
        if p_away >= thresholds['away']:
            edge = calculate_edge(p_away, row['away_best'])
            if edge >= min_edge:
                candidates.append(('away', p_away, 0, row['away_best'], edge))

        # Draw
        if include_draws and p_draw >= thresholds['draw']:
            edge = calculate_edge(p_draw, row['draw_best'])
            if edge >= min_edge:
                candidates.append(('draw', p_draw, 1, row['draw_best'], edge))

        if candidates:
            # Pick highest probability (or could pick highest edge)
            best = max(candidates, key=lambda x: x[1])
            outcome, prob, label, odds, edge = best
            won = 1 if label == true_label else 0
            bets.append({'outcome': outcome, 'odds': odds, 'won': won, 'stake': 1, 'edge': edge})

    return bets


def strategy_odds_range(probs, test_df, thresholds, odds_min=1.3, odds_max=5.0, include_draws=False):
    """Only bet within certain odds range."""
    result_map = {'H': 2, 'A': 0, 'D': 1}
    bets = []

    for i in range(len(probs)):
        row = test_df.iloc[i]
        p_away, p_draw, p_home = probs[i]
        true_label = result_map[row['result']]

        candidates = []

        if p_home >= thresholds['home'] and odds_min <= row['home_best'] <= odds_max:
            candidates.append(('home', p_home, 2, row['home_best']))
        if p_away >= thresholds['away'] and odds_min <= row['away_best'] <= odds_max:
            candidates.append(('away', p_away, 0, row['away_best']))
        if include_draws and p_draw >= thresholds['draw'] and odds_min <= row['draw_best'] <= odds_max:
            candidates.append(('draw', p_draw, 1, row['draw_best']))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            outcome, prob, label, odds = best
            won = 1 if label == true_label else 0
            bets.append({'outcome': outcome, 'odds': odds, 'won': won, 'stake': 1})

    return bets


def strategy_kelly(probs, test_df, thresholds, kelly_fraction=0.25, include_draws=False):
    """Kelly Criterion: Bet size proportional to edge."""
    result_map = {'H': 2, 'A': 0, 'D': 1}
    bets = []

    for i in range(len(probs)):
        row = test_df.iloc[i]
        p_away, p_draw, p_home = probs[i]
        true_label = result_map[row['result']]

        candidates = []

        # Home
        if p_home >= thresholds['home']:
            odds = row['home_best']
            edge = calculate_edge(p_home, odds)
            if edge > 0:
                # Kelly formula: (p * odds - 1) / (odds - 1) = edge / (odds - 1)
                kelly_stake = (p_home * odds - 1) / (odds - 1)
                kelly_stake = max(0, kelly_stake) * kelly_fraction  # Fractional Kelly
                if kelly_stake > 0:
                    candidates.append(('home', p_home, 2, odds, kelly_stake, edge))

        # Away
        if p_away >= thresholds['away']:
            odds = row['away_best']
            edge = calculate_edge(p_away, odds)
            if edge > 0:
                kelly_stake = (p_away * odds - 1) / (odds - 1)
                kelly_stake = max(0, kelly_stake) * kelly_fraction
                if kelly_stake > 0:
                    candidates.append(('away', p_away, 0, odds, kelly_stake, edge))

        # Draw
        if include_draws and p_draw >= thresholds['draw']:
            odds = row['draw_best']
            edge = calculate_edge(p_draw, odds)
            if edge > 0:
                kelly_stake = (p_draw * odds - 1) / (odds - 1)
                kelly_stake = max(0, kelly_stake) * kelly_fraction
                if kelly_stake > 0:
                    candidates.append(('draw', p_draw, 1, odds, kelly_stake, edge))

        if candidates:
            # Pick highest expected value (prob * odds)
            best = max(candidates, key=lambda x: x[1] * x[3])
            outcome, prob, label, odds, stake, edge = best
            won = 1 if label == true_label else 0
            bets.append({'outcome': outcome, 'odds': odds, 'won': won, 'stake': stake, 'edge': edge})

    return bets


def strategy_combined(probs, test_df, thresholds, min_edge=0.05, odds_min=1.5, odds_max=4.0, include_draws=False):
    """Combined: Edge threshold + Odds range."""
    result_map = {'H': 2, 'A': 0, 'D': 1}
    bets = []

    for i in range(len(probs)):
        row = test_df.iloc[i]
        p_away, p_draw, p_home = probs[i]
        true_label = result_map[row['result']]

        candidates = []

        # Home
        odds = row['home_best']
        if p_home >= thresholds['home'] and odds_min <= odds <= odds_max:
            edge = calculate_edge(p_home, odds)
            if edge >= min_edge:
                candidates.append(('home', p_home, 2, odds, edge))

        # Away
        odds = row['away_best']
        if p_away >= thresholds['away'] and odds_min <= odds <= odds_max:
            edge = calculate_edge(p_away, odds)
            if edge >= min_edge:
                candidates.append(('away', p_away, 0, odds, edge))

        # Draw
        if include_draws:
            odds = row['draw_best']
            if p_draw >= thresholds['draw'] and odds_min <= odds <= odds_max:
                edge = calculate_edge(p_draw, odds)
                if edge >= min_edge:
                    candidates.append(('draw', p_draw, 1, odds, edge))

        if candidates:
            best = max(candidates, key=lambda x: x[4])  # Pick highest edge
            outcome, prob, label, odds, edge = best
            won = 1 if label == true_label else 0
            bets.append({'outcome': outcome, 'odds': odds, 'won': won, 'stake': 1, 'edge': edge})

    return bets


def evaluate_bets(bets, title):
    """Evaluate betting results."""
    if not bets:
        print(f"\n{title}: No bets")
        return None

    df = pd.DataFrame(bets)
    total_bets = len(df)
    total_stake = df['stake'].sum()
    total_wins = df['won'].sum()
    returns = (df['won'] * df['odds'] * df['stake']).sum()
    profit = returns - total_stake
    roi = profit / total_stake * 100
    wr = total_wins / total_bets * 100

    print(f"\n{title}")
    print("="*80)
    print(f"Bets: {total_bets} | Wins: {total_wins} | WR: {wr:.1f}%")
    print(f"Stake: {total_stake:.2f} | Returns: {returns:.2f} | Profit: {profit:+.2f} | ROI: {roi:+.2f}%")

    # Per outcome
    for outcome in ['home', 'away', 'draw']:
        outcome_df = df[df['outcome'] == outcome]
        if len(outcome_df) > 0:
            o_stake = outcome_df['stake'].sum()
            o_wins = outcome_df['won'].sum()
            o_returns = (outcome_df['won'] * outcome_df['odds'] * outcome_df['stake']).sum()
            o_profit = o_returns - o_stake
            o_roi = o_profit / o_stake * 100
            o_wr = o_wins / len(outcome_df) * 100
            avg_odds = outcome_df['odds'].mean()
            print(f"  {outcome.upper()}: {len(outcome_df)} bets | {o_wr:.1f}% WR | Avg odds: {avg_odds:.2f} | Profit: {o_profit:+.2f} | ROI: {o_roi:+.2f}%")

    return {'bets': total_bets, 'profit': profit, 'roi': roi, 'wr': wr}


def main():
    print("Loading data...")
    test_df, feature_cols = load_data_with_odds()
    X_test = test_df[feature_cols].values

    print(f"Test matches: {len(test_df)}")

    # Load models
    print("Loading models...")
    model_v3 = joblib.load('models/production/model_v3.0.0.joblib')
    probs = (model_v3['catboost'].predict_proba(X_test) + model_v3['lightgbm'].predict_proba(X_test)) / 2

    # Base thresholds
    thresholds = {'home': 0.36, 'away': 0.40, 'draw': 0.28}

    print("\n" + "="*80)
    print("BETTING STRATEGIES COMPARISON (v3.0.0, H/A Only)")
    print("="*80)

    # 1. Baseline
    bets = strategy_baseline(probs, test_df, thresholds, include_draws=False)
    baseline = evaluate_bets(bets, "1. BASELINE (Confidence thresholds only)")

    # 2. Value Betting with different edge thresholds
    for min_edge in [0.0, 0.05, 0.10, 0.15, 0.20]:
        bets = strategy_value_betting(probs, test_df, thresholds, min_edge=min_edge, include_draws=False)
        evaluate_bets(bets, f"2. VALUE BETTING (Edge >= {min_edge:.0%})")

    # 3. Odds Range filtering
    odds_ranges = [(1.3, 3.0), (1.5, 4.0), (1.8, 5.0), (2.0, 4.0)]
    for odds_min, odds_max in odds_ranges:
        bets = strategy_odds_range(probs, test_df, thresholds, odds_min=odds_min, odds_max=odds_max, include_draws=False)
        evaluate_bets(bets, f"3. ODDS RANGE ({odds_min} - {odds_max})")

    # 4. Kelly Criterion
    for kelly_frac in [0.1, 0.25, 0.5]:
        bets = strategy_kelly(probs, test_df, thresholds, kelly_fraction=kelly_frac, include_draws=False)
        evaluate_bets(bets, f"4. KELLY CRITERION (Fraction: {kelly_frac})")

    # 5. Combined strategies
    combined_configs = [
        (0.05, 1.5, 4.0),
        (0.10, 1.5, 4.0),
        (0.05, 1.8, 3.5),
        (0.10, 2.0, 4.0),
        (0.15, 1.5, 5.0),
    ]
    for min_edge, odds_min, odds_max in combined_configs:
        bets = strategy_combined(probs, test_df, thresholds, min_edge=min_edge,
                                odds_min=odds_min, odds_max=odds_max, include_draws=False)
        evaluate_bets(bets, f"5. COMBINED (Edge>={min_edge:.0%}, Odds {odds_min}-{odds_max})")

    # Best strategies summary
    print("\n" + "="*80)
    print("TESTING WITH DRAWS INCLUDED")
    print("="*80)

    # Best value betting with draws
    for min_edge in [0.0, 0.05, 0.10]:
        bets = strategy_value_betting(probs, test_df, thresholds, min_edge=min_edge, include_draws=True)
        evaluate_bets(bets, f"VALUE BETTING with Draws (Edge >= {min_edge:.0%})")

    # Combined with draws
    bets = strategy_combined(probs, test_df, thresholds, min_edge=0.05,
                            odds_min=1.5, odds_max=4.0, include_draws=True)
    evaluate_bets(bets, "COMBINED with Draws (Edge>=5%, Odds 1.5-4.0)")


if __name__ == '__main__':
    main()
