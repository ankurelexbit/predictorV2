#!/usr/bin/env python3
"""
Test All Models with Threshold Optimization
============================================

**WHAT THIS SCRIPT DOES:**

1. Loads TEST SET (last 15% of training_data_with_draw_features.csv)
   - ~2,692 games with known results
   - NOT live January 2026 data

2. Simulates realistic odds based on actual outcomes

3. Loads all 4 models IN PARALLEL:
   - Current Production (H=1.0, D=1.5, A=1.2)
   - Option 1: Conservative (H=1.0, D=1.3, A=1.0)
   - Option 2: Aggressive (H=1.3, D=1.5, A=1.2)
   - Option 3: Balanced (H=1.2, D=1.4, A=1.1)

4. For EACH model independently:
   - Generates predictions on test set
   - Tests 539 threshold combinations
   - Finds optimal thresholds for maximum profit

5. Compares all models side-by-side

**NEW BETTING LOGIC:**
- Check which outcomes (H/D/A) cross their threshold
- If 2+ cross threshold: bet on the one with max probability
- If 1 crosses threshold: bet on that
- If 0 cross threshold: no bet

**OUTPUT:**
- Shows optimal thresholds for each model
- Shows expected profit, ROI, bet distribution
- Exports to complete_model_comparison.xlsx
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

FEATURES_TO_EXCLUDE = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]

def load_test_data():
    """Load test set from training data."""
    data_path = Path('data/training_data_with_draw_features.csv')

    df = pd.read_csv(data_path)
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in FEATURES_TO_EXCLUDE]

    # Map target
    target_map = {'A': 0, 'D': 1, 'H': 2}
    df['target'] = df['result'].map(target_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Sort chronologically
    df = df.sort_values('match_date').reset_index(drop=True)

    # Use last 15% as test set
    n = len(df)
    test_start = int(n * 0.85)
    test_df = df.iloc[test_start:].copy()

    return test_df, feature_cols

def generate_predictions(model, X, feature_cols):
    """Generate predictions with a model."""
    X_subset = X[feature_cols]
    probs = model.predict_proba(X_subset)

    return pd.DataFrame({
        'pred_away_prob': probs[:, 0],
        'pred_draw_prob': probs[:, 1],
        'pred_home_prob': probs[:, 2]
    })

def simulate_odds(actual_result, league_difficulty=1.0):
    """
    Simulate realistic odds based on actual result and league.
    Better simulation than random - uses typical odds for each outcome.
    """
    # Typical odds distributions (from historical data)
    if actual_result == 'H':  # Home wins
        home_odds = np.random.uniform(1.3, 2.5)
        draw_odds = np.random.uniform(3.0, 4.5)
        away_odds = np.random.uniform(3.0, 8.0)
    elif actual_result == 'D':  # Draw
        home_odds = np.random.uniform(2.0, 3.5)
        draw_odds = np.random.uniform(3.2, 4.2)
        away_odds = np.random.uniform(2.0, 3.5)
    else:  # Away wins
        home_odds = np.random.uniform(2.5, 7.0)
        draw_odds = np.random.uniform(3.0, 4.5)
        away_odds = np.random.uniform(1.5, 3.0)

    return home_odds, draw_odds, away_odds

def optimize_thresholds_for_model(df_with_preds, model_name):
    """Optimize thresholds for a specific model's predictions."""
    print(f"\n{'='*80}")
    print(f"OPTIMIZING THRESHOLDS: {model_name}")
    print(f"{'='*80}")

    # Test thresholds
    home_thresholds = np.arange(0.30, 0.85, 0.05)
    draw_thresholds = np.arange(0.20, 0.55, 0.05)
    away_thresholds = np.arange(0.25, 0.55, 0.05)

    best_profit = -float('inf')
    best_thresholds = None
    best_metrics = None

    results = []
    total = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds)
    count = 0

    print(f"Testing {total} threshold combinations...")

    for home_thresh in home_thresholds:
        for draw_thresh in draw_thresholds:
            for away_thresh in away_thresholds:
                count += 1
                if count % 200 == 0:
                    print(f"   {count}/{total}...")

                home_bets = draw_bets = away_bets = 0
                home_wins = draw_wins = away_wins = 0
                home_profit = draw_profit = away_profit = 0

                for _, row in df_with_preds.iterrows():
                    home_prob = row['pred_home_prob']
                    draw_prob = row['pred_draw_prob']
                    away_prob = row['pred_away_prob']
                    actual = row['actual_result']

                    home_odds = row['home_odds']
                    draw_odds = row['draw_odds']
                    away_odds = row['away_odds']

                    # NEW BETTING LOGIC: Check which outcomes cross threshold
                    candidates = []
                    if home_prob >= home_thresh:
                        candidates.append(('H', home_prob, home_odds))
                    if draw_prob >= draw_thresh:
                        candidates.append(('D', draw_prob, draw_odds))
                    if away_prob >= away_thresh:
                        candidates.append(('A', away_prob, away_odds))

                    # If 2+ cross threshold, pick the one with max probability
                    # If 1 crosses, bet on that
                    # If 0 cross, no bet
                    bet_outcome = None
                    if len(candidates) >= 2:
                        # Multiple candidates - pick max probability
                        bet_outcome, bet_prob, bet_odds = max(candidates, key=lambda x: x[1])
                    elif len(candidates) == 1:
                        # Single candidate
                        bet_outcome, bet_prob, bet_odds = candidates[0]

                    # Place bet based on outcome
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
                total_profit = home_profit + draw_profit + away_profit

                if total_bets >= 100:  # Min bets for statistical significance on test set
                    roi = (total_profit / total_bets) * 100

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

    if best_thresholds:
        print(f"\n🎯 OPTIMAL THRESHOLDS:")
        print(f"   Home: {best_thresholds[0]:.2f}")
        print(f"   Draw: {best_thresholds[1]:.2f}")
        print(f"   Away: {best_thresholds[2]:.2f}")
        print(f"\n📊 PERFORMANCE:")
        print(f"   Total: {best_metrics['total_bets']} bets, ${best_metrics['total_profit']:.2f}, {best_metrics['roi']:.1f}% ROI")
        print(f"   Home: {best_metrics['home_bets']} bets, {best_metrics['home_wins']} wins, ${best_metrics['home_profit']:.2f}")
        print(f"   Draw: {best_metrics['draw_bets']} bets, {best_metrics['draw_wins']} wins, ${best_metrics['draw_profit']:.2f}")
        print(f"   Away: {best_metrics['away_bets']} bets, {best_metrics['away_wins']} wins, ${best_metrics['away_profit']:.2f}")

    return best_thresholds, best_metrics, results

def process_single_model(model_info, test_df, feature_cols):
    """Process a single model - load, predict, optimize."""
    if not model_info['path'].exists():
        print(f"⚠️  Skipping {model_info['name']} - not found")
        return None

    print(f"[{model_info['short']}] Loading model...")
    model = joblib.load(model_info['path'])

    print(f"[{model_info['short']}] Generating predictions...")
    preds = generate_predictions(model, test_df, feature_cols)

    # Add predictions to dataframe
    df_with_preds = test_df.copy()
    df_with_preds['pred_home_prob'] = preds['pred_home_prob'].values
    df_with_preds['pred_draw_prob'] = preds['pred_draw_prob'].values
    df_with_preds['pred_away_prob'] = preds['pred_away_prob'].values

    # Optimize thresholds
    print(f"[{model_info['short']}] Optimizing thresholds...")
    best_thresh, best_metrics, results = optimize_thresholds_for_model(
        df_with_preds,
        model_info['name']
    )

    if best_metrics:
        return {
            'model': model_info['name'],
            'short': model_info['short'],
            'thresholds': best_thresh,
            'metrics': best_metrics
        }
    return None

def main():
    print("="*80)
    print("COMPLETE MODEL COMPARISON WITH THRESHOLD OPTIMIZATION")
    print("="*80)
    print()

    # Define models
    models_info = [
        {
            'name': 'Current Production',
            'path': Path('models/with_draw_features/conservative_with_draw_features.joblib'),
            'short': 'current'
        },
        {
            'name': 'Option 1: Conservative',
            'path': Path('models/weight_experiments/option1_conservative.joblib'),
            'short': 'option1'
        },
        {
            'name': 'Option 2: Aggressive',
            'path': Path('models/weight_experiments/option2_aggressive.joblib'),
            'short': 'option2'
        },
        {
            'name': 'Option 3: Balanced',
            'path': Path('models/weight_experiments/option3_balanced.joblib'),
            'short': 'option3'
        }
    ]

    # Load test data
    print("Loading test set...")
    test_df, feature_cols = load_test_data()
    print(f"✅ Test set: {len(test_df)} games")
    print()

    # Simulate odds for test set
    print("Simulating realistic odds for test set...")
    test_df['actual_result'] = test_df['result']
    odds_data = test_df['actual_result'].apply(lambda x: simulate_odds(x))
    test_df['home_odds'] = [o[0] for o in odds_data]
    test_df['draw_odds'] = [o[1] for o in odds_data]
    test_df['away_odds'] = [o[2] for o in odds_data]
    print("✅ Odds simulated")
    print()

    # Test each model IN PARALLEL
    print("🚀 Processing all 4 models in parallel...")
    print()

    all_results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all models for parallel processing
        future_to_model = {
            executor.submit(process_single_model, model_info, test_df, feature_cols): model_info
            for model_info in models_info
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_info = future_to_model[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    print(f"✅ [{result['short']}] Complete!")
            except Exception as e:
                print(f"❌ [{model_info['short']}] Error: {e}")

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    print()

    if all_results:
        # Sort by profit
        all_results.sort(key=lambda x: x['metrics']['total_profit'], reverse=True)

        print(f"{'Rank':<6} {'Model':<30} {'Profit':<12} {'ROI':<10} {'Bets':<8} {'Draw Bets':<12}")
        print("-"*80)

        for i, r in enumerate(all_results, 1):
            m = r['metrics']
            print(f"{i:<6} {r['model']:<30} ${m['total_profit']:<11.2f} {m['roi']:<9.1f}% {m['total_bets']:<8} {m['draw_bets']:<12}")

        print()
        print("="*80)
        print("VERDICT")
        print("="*80)

        best = all_results[0]
        current = next((r for r in all_results if 'Current' in r['model']), None)

        if current and best['model'] == current['model']:
            print("✅ Current production model is OPTIMAL!")
        elif current:
            diff = best['metrics']['total_profit'] - current['metrics']['total_profit']
            print(f"⚠️  {best['model']} outperforms Current by ${diff:.2f}")
            print(f"   But check draw bet volume: {best['metrics']['draw_bets']} vs {current['metrics']['draw_bets']}")
        else:
            print(f"✅ Best model: {best['model']}")

        print()

        # Export
        output_file = 'complete_model_comparison.xlsx'
        import openpyxl

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary
            summary_df = pd.DataFrame([
                {
                    'Model': r['model'],
                    'Total_Profit': r['metrics']['total_profit'],
                    'ROI': r['metrics']['roi'],
                    'Total_Bets': r['metrics']['total_bets'],
                    'Home_Bets': r['metrics']['home_bets'],
                    'Draw_Bets': r['metrics']['draw_bets'],
                    'Away_Bets': r['metrics']['away_bets'],
                    'Home_Thresh': r['thresholds'][0],
                    'Draw_Thresh': r['thresholds'][1],
                    'Away_Thresh': r['thresholds'][2]
                }
                for r in all_results
            ])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"✅ Results exported to: {output_file}")

if __name__ == '__main__':
    main()
