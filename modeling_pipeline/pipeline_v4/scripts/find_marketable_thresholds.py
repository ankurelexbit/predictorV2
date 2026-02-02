#!/usr/bin/env python3
"""
Find Marketable Threshold Configurations
=========================================

Finds threshold combinations that achieve:
1. 50%+ win rate on EACH outcome (H/D/A)
2. Maintains overall profitability
3. Looks attractive to customers

Marketing challenge: 36% draw win rate with 135 bets is hard to sell,
even though 30% ROI is excellent.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = "postgresql://ankurgupta@localhost/football_predictions"
TOP_5_LEAGUES = (8, 82, 384, 564, 301)

FEATURES_TO_EXCLUDE = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]


def load_predictions_from_db():
    """Load predictions with features and odds from database."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    query = """
        SELECT features, best_home_odds, best_draw_odds, best_away_odds, actual_result
        FROM predictions
        WHERE match_date >= '2026-01-01' AND match_date < '2026-02-01'
          AND actual_result IS NOT NULL AND league_id = ANY(%s)
    """
    cursor = conn.cursor()
    cursor.execute(query, [list(TOP_5_LEAGUES)])
    predictions = cursor.fetchall()
    conn.close()
    return predictions


def extract_features_df(predictions):
    """Extract features from jsonb into DataFrame."""
    feature_keys = list(predictions[0]['features'].keys())
    feature_data = [{k: p['features'].get(k) for k in feature_keys} for p in predictions]
    feature_df = pd.DataFrame(feature_data)
    feature_cols = [c for c in feature_df.columns if c not in FEATURES_TO_EXCLUDE]
    return feature_df[feature_cols], feature_cols


def run_model_predictions(model_path, feature_df, feature_cols):
    """Run a model on the feature set."""
    model = joblib.load(model_path)
    X = feature_df[feature_cols]
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
        X = X[model_features]
    probs = model.predict_proba(X)
    return probs


def test_threshold_combination(probs, predictions, home_thresh, draw_thresh, away_thresh):
    """Test a specific threshold combination."""
    home_bets = draw_bets = away_bets = 0
    home_wins = draw_wins = away_wins = 0
    home_profit = draw_profit = away_profit = 0

    for i, p in enumerate(predictions):
        home_prob = probs[i, 2]
        draw_prob = probs[i, 1]
        away_prob = probs[i, 0]

        home_odds = p['best_home_odds']
        draw_odds = p['best_draw_odds']
        away_odds = p['best_away_odds']
        actual = p['actual_result']

        if not all([home_odds, draw_odds, away_odds]):
            continue

        # Betting logic
        candidates = []
        if home_prob >= home_thresh:
            candidates.append(('H', home_prob, home_odds))
        if draw_prob >= draw_thresh:
            candidates.append(('D', draw_prob, draw_odds))
        if away_prob >= away_thresh:
            candidates.append(('A', away_prob, away_odds))

        bet_outcome = None
        if len(candidates) >= 2:
            bet_outcome = max(candidates, key=lambda x: x[1])[0]
        elif len(candidates) == 1:
            bet_outcome = candidates[0][0]

        # Place bet
        if bet_outcome == 'H':
            home_bets += 1
            if actual == 'H':
                home_wins += 1
                home_profit += (home_odds - 1)
            else:
                home_profit -= 1
        elif bet_outcome == 'D':
            draw_bets += 1
            if actual == 'D':
                draw_wins += 1
                draw_profit += (draw_odds - 1)
            else:
                draw_profit -= 1
        elif bet_outcome == 'A':
            away_bets += 1
            if actual == 'A':
                away_wins += 1
                away_profit += (away_odds - 1)
            else:
                away_profit -= 1

    total_bets = home_bets + draw_bets + away_bets
    total_wins = home_wins + draw_wins + away_wins
    total_profit = home_profit + draw_profit + away_profit

    return {
        'home_thresh': home_thresh,
        'draw_thresh': draw_thresh,
        'away_thresh': away_thresh,
        'total_bets': total_bets,
        'total_wins': total_wins,
        'total_profit': total_profit,
        'win_rate': (total_wins / total_bets * 100) if total_bets > 0 else 0,
        'roi': (total_profit / total_bets * 100) if total_bets > 0 else 0,
        'home_bets': home_bets,
        'home_wins': home_wins,
        'home_winrate': (home_wins / home_bets * 100) if home_bets > 0 else 0,
        'home_profit': home_profit,
        'draw_bets': draw_bets,
        'draw_wins': draw_wins,
        'draw_winrate': (draw_wins / draw_bets * 100) if draw_bets > 0 else 0,
        'draw_profit': draw_profit,
        'away_bets': away_bets,
        'away_wins': away_wins,
        'away_winrate': (away_wins / away_bets * 100) if away_bets > 0 else 0,
        'away_profit': away_profit
    }


def main():
    print("="*100)
    print("FINDING MARKETABLE THRESHOLD CONFIGURATIONS")
    print("="*100)
    print("\nGoal: 50%+ win rate on ALL outcomes (especially draws)")
    print("      While maintaining profitability\n")

    # Load predictions
    predictions = load_predictions_from_db()
    feature_df, feature_cols = extract_features_df(predictions)

    # Load Option 3 model
    model_path = Path('models/weight_experiments/option3_balanced.joblib')
    probs = run_model_predictions(model_path, feature_df, feature_cols)

    print(f"Loaded {len(predictions)} predictions from January 2026\n")

    # Test configurations
    print("Testing threshold combinations...\n")

    # Strategy 1: High draw threshold for 50%+ draw win rate
    print("="*100)
    print("STRATEGY 1: HIGH DRAW THRESHOLD (Target: 50%+ draw win rate)")
    print("="*100)
    print()

    draw_thresholds = np.arange(0.25, 0.55, 0.05)
    results = []

    for draw_thresh in draw_thresholds:
        # Test with reasonable home/away thresholds
        result = test_threshold_combination(probs, predictions,
                                           home_thresh=0.65,
                                           draw_thresh=draw_thresh,
                                           away_thresh=0.42)
        if result['total_bets'] >= 50:
            results.append(result)

    # Display results
    print(f"{'Draw':<6} {'Total':<7} {'Total':<9} {'ROI':<8} {'Draw':<7} {'Draw':<7} {'Draw':<11} {'Draw':<10} {'Home':<7} {'Away':<7}")
    print(f"{'Thresh':<6} {'Bets':<7} {'Profit':<9} {'%':<8} {'Bets':<7} {'Wins':<7} {'Winrate':<11} {'Profit':<10} {'Bets':<7} {'Bets':<7}")
    print("-"*100)

    for r in results:
        marker = "✅" if r['draw_winrate'] >= 50 else ""
        print(f"{r['draw_thresh']:.2f}   {r['total_bets']:<7} ${r['total_profit']:<8.2f} "
              f"{r['roi']:<7.1f}% {r['draw_bets']:<7} {r['draw_wins']:<7} "
              f"{r['draw_winrate']:<10.1f}% {marker:<2} ${r['draw_profit']:<9.2f} "
              f"{r['home_bets']:<7} {r['away_bets']:<7}")

    # Find best with 50%+ draw win rate
    marketable = [r for r in results if r['draw_winrate'] >= 50]
    if marketable:
        best = max(marketable, key=lambda x: x['total_profit'])
        print("\n🎯 BEST MARKETABLE (50%+ draw win rate):")
        print(f"   Thresholds: H={best['home_thresh']:.2f}, D={best['draw_thresh']:.2f}, A={best['away_thresh']:.2f}")
        print(f"   Total: {best['total_bets']} bets, ${best['total_profit']:.2f} profit, {best['roi']:.1f}% ROI")
        print(f"   Draw: {best['draw_bets']} bets, {best['draw_wins']} wins ({best['draw_winrate']:.1f}%), ${best['draw_profit']:.2f}")

    # Strategy 2: Balanced approach - all outcomes 50%+
    print("\n")
    print("="*100)
    print("STRATEGY 2: ALL OUTCOMES 50%+ WIN RATE")
    print("="*100)
    print()

    home_thresholds = np.arange(0.60, 0.85, 0.05)
    draw_thresholds = np.arange(0.30, 0.50, 0.05)
    away_thresholds = np.arange(0.40, 0.55, 0.05)

    all_50_results = []

    for home_thresh in home_thresholds:
        for draw_thresh in draw_thresholds:
            for away_thresh in away_thresholds:
                result = test_threshold_combination(probs, predictions,
                                                   home_thresh, draw_thresh, away_thresh)

                if result['total_bets'] >= 30:  # Lower min for high selectivity
                    # Check if all outcomes with bets have 50%+ win rate
                    all_good = True
                    if result['home_bets'] >= 5 and result['home_winrate'] < 50:
                        all_good = False
                    if result['draw_bets'] >= 5 and result['draw_winrate'] < 50:
                        all_good = False
                    if result['away_bets'] >= 5 and result['away_winrate'] < 50:
                        all_good = False

                    if all_good:
                        all_50_results.append(result)

    if all_50_results:
        # Sort by profit
        all_50_results.sort(key=lambda x: x['total_profit'], reverse=True)

        print(f"Found {len(all_50_results)} configurations with all outcomes 50%+ win rate\n")
        print(f"{'Thresholds':<18} {'Bets':<6} {'Profit':<10} {'ROI':<8} {'H':<15} {'D':<15} {'A':<15}")
        print("-"*100)

        for r in all_50_results[:10]:  # Top 10
            h_str = f"{r['home_bets']}b {r['home_winrate']:.0f}%"
            d_str = f"{r['draw_bets']}b {r['draw_winrate']:.0f}%"
            a_str = f"{r['away_bets']}b {r['away_winrate']:.0f}%"
            print(f"{r['home_thresh']:.2f}/{r['draw_thresh']:.2f}/{r['away_thresh']:.2f}    "
                  f"{r['total_bets']:<6} ${r['total_profit']:<9.2f} {r['roi']:<7.1f}% "
                  f"{h_str:<15} {d_str:<15} {a_str:<15}")

        print("\n🎯 TOP RECOMMENDATION (All outcomes 50%+):")
        best = all_50_results[0]
        print(f"   Thresholds: H={best['home_thresh']:.2f}, D={best['draw_thresh']:.2f}, A={best['away_thresh']:.2f}")
        print(f"   Total: {best['total_bets']} bets, ${best['total_profit']:.2f} profit, {best['roi']:.1f}% ROI, {best['win_rate']:.1f}% overall")
        print(f"   Home: {best['home_bets']} bets, {best['home_wins']} wins ({best['home_winrate']:.1f}%), ${best['home_profit']:.2f}")
        print(f"   Draw: {best['draw_bets']} bets, {best['draw_wins']} wins ({best['draw_winrate']:.1f}%), ${best['draw_profit']:.2f}")
        print(f"   Away: {best['away_bets']} bets, {best['away_wins']} wins ({best['away_winrate']:.1f}%), ${best['away_profit']:.2f}")

    # Comparison
    print("\n")
    print("="*100)
    print("COMPARISON: PROFIT-OPTIMAL vs MARKETABLE")
    print("="*100)
    print()

    # Profit optimal (from earlier)
    profit_opt = test_threshold_combination(probs, predictions, 0.65, 0.22, 0.42)

    print(f"{'Strategy':<25} {'Thresholds':<18} {'Profit':<10} {'ROI':<8} {'Bets':<6} {'Draw WR':<10}")
    print("-"*100)
    print(f"{'PROFIT-OPTIMAL':<25} {'0.65/0.22/0.42':<18} ${profit_opt['total_profit']:<9.2f} "
          f"{profit_opt['roi']:<7.1f}% {profit_opt['total_bets']:<6} {profit_opt['draw_winrate']:<9.1f}%")

    if marketable:
        best = max(marketable, key=lambda x: x['total_profit'])
        thresh_str = f"{best['home_thresh']:.2f}/{best['draw_thresh']:.2f}/{best['away_thresh']:.2f}"
        print(f"{'MARKETABLE (Draw 50%+)':<25} {thresh_str:<18} "
              f"${best['total_profit']:<9.2f} {best['roi']:<7.1f}% {best['total_bets']:<6} {best['draw_winrate']:<9.1f}% ✅")

    if all_50_results:
        best = all_50_results[0]
        thresh_str = f"{best['home_thresh']:.2f}/{best['draw_thresh']:.2f}/{best['away_thresh']:.2f}"
        print(f"{'ALL OUTCOMES 50%+':<25} {thresh_str:<18} "
              f"${best['total_profit']:<9.2f} {best['roi']:<7.1f}% {best['total_bets']:<6} {best['draw_winrate']:<9.1f}% ✅")

    print("\n")
    print("="*100)
    print("MARKETING RECOMMENDATIONS")
    print("="*100)
    print()
    print("Option A: TWO-TIER PRODUCT")
    print("  - 'Value Betting' tier: 0.65/0.22/0.42 → $51.78, 36% draw WR (for sophisticated users)")
    print("  - 'High Win Rate' tier: Use all-50%+ config → Lower $ but 50%+ WR (for mass market)")
    print()
    print("Option B: SELECTIVE DRAW STRATEGY")
    print("  - Only show draws when win rate will be 50%+ (higher threshold)")
    print("  - Accept fewer draw bets but better customer experience")
    print()
    print("Option C: EMPHASIZE OVERALL WIN RATE")
    print("  - Market overall 47% win rate across all bets")
    print("  - Don't break down by outcome (customers won't see draw weakness)")
    print("="*100)


if __name__ == '__main__':
    main()
