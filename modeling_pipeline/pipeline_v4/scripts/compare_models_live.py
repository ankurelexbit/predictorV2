#!/usr/bin/env python3
"""
Live Model Comparison & Threshold Optimization
===============================================

Tests all models on real January 2026 data with actual results and odds.
Optimizes thresholds for maximum profit and compares performance.

Usage:
    python3 scripts/compare_models_live.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sportmonks_client import SportMonksClient
from scripts.predict_live_with_history import ProductionLivePipeline

# Top 5 leagues
TOP_5_LEAGUES = [8, 82, 384, 564]  # Premier League, Bundesliga, Serie A, Ligue 1

def load_model_info():
    """Define all models to test."""
    models = [
        {
            'name': 'Current Production',
            'path': Path('models/with_draw_features/conservative_with_draw_features.joblib'),
            'weights': 'H=1.0, D=1.5, A=1.2',
            'short_name': 'current'
        },
        {
            'name': 'Option 1: Conservative',
            'path': Path('models/weight_experiments/option1_conservative.joblib'),
            'weights': 'H=1.0, D=1.3, A=1.0',
            'short_name': 'option1'
        },
        {
            'name': 'Option 2: Aggressive',
            'path': Path('models/weight_experiments/option2_aggressive.joblib'),
            'weights': 'H=1.3, D=1.5, A=1.2',
            'short_name': 'option2'
        },
        {
            'name': 'Option 3: Balanced',
            'path': Path('models/weight_experiments/option3_balanced.joblib'),
            'weights': 'H=1.2, D=1.4, A=1.1',
            'short_name': 'option3'
        }
    ]

    # Filter to only existing models
    existing_models = []
    for m in models:
        if m['path'].exists():
            existing_models.append(m)
        else:
            print(f"⚠️  Skipping {m['name']} - model file not found")

    return existing_models


def generate_predictions_for_model(model_path, model_name, start_date, end_date):
    """Generate predictions for a specific model."""
    print(f"\n{'='*80}")
    print(f"GENERATING PREDICTIONS: {model_name}")
    print(f"{'='*80}")

    # Initialize pipeline with this specific model
    pipeline = ProductionLivePipeline(
        model_path=str(model_path)
    )

    print(f"Fetching fixtures from {start_date} to {end_date}...")

    # Get fixtures
    client = SportMonksClient()
    fixtures = client.get_fixtures_between(start_date, end_date)

    print(f"✅ Found {len(fixtures)} fixtures")

    # Generate predictions
    predictions = []

    for i, fixture in enumerate(fixtures, 1):
        if i % 50 == 0:
            print(f"   Processing {i}/{len(fixtures)}...")

        try:
            result = pipeline.predict_fixture(fixture)
            if result:
                predictions.append(result)
        except Exception as e:
            print(f"   ⚠️  Error on fixture {fixture.get('id')}: {e}")
            continue

    print(f"✅ Generated {len(predictions)} predictions")

    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    return df


def fetch_actual_results(df):
    """Fetch actual results for fixtures."""
    print(f"\n{'='*80}")
    print(f"FETCHING ACTUAL RESULTS")
    print(f"{'='*80}")

    client = SportMonksClient()

    # Get unique dates
    df['match_date'] = pd.to_datetime(df['match_date'])
    min_date = df['match_date'].min().strftime('%Y-%m-%d')
    max_date = df['match_date'].max().strftime('%Y-%m-%d')

    print(f"Fetching results from {min_date} to {max_date}...")

    fixtures = client.get_fixtures_between(min_date, max_date)

    # Create results lookup
    results_map = {}
    for fixture in fixtures:
        fixture_id = fixture.get('id')
        state = fixture.get('state', {})

        if state.get('state') == 'FT':  # Finished
            scores = fixture.get('scores', [])
            home_score = None
            away_score = None

            for score in scores:
                if score.get('description') == 'CURRENT':
                    home_score = score.get('score', {}).get('goals')
                    away_score = score.get('score', {}).get('goals')
                    break

            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    result = 'H'
                elif home_score < away_score:
                    result = 'A'
                else:
                    result = 'D'

                results_map[fixture_id] = {
                    'home_score': home_score,
                    'away_score': away_score,
                    'result': result
                }

    print(f"✅ Found results for {len(results_map)} fixtures")

    # Add results to dataframe
    df['actual_home_score'] = df['fixture_id'].map(lambda x: results_map.get(x, {}).get('home_score'))
    df['actual_away_score'] = df['fixture_id'].map(lambda x: results_map.get(x, {}).get('away_score'))
    df['actual_result'] = df['fixture_id'].map(lambda x: results_map.get(x, {}).get('result'))

    # Filter to only games with results
    df_with_results = df[df['actual_result'].notna()].copy()

    print(f"✅ {len(df_with_results)} predictions have actual results")

    return df_with_results


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

    # Test threshold combinations (coarser grid for speed)
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

                    home_odds = row.get('best_home_odds', 0)
                    draw_odds = row.get('best_draw_odds', 0)
                    away_odds = row.get('best_away_odds', 0)

                    if home_odds == 0 or draw_odds == 0 or away_odds == 0:
                        continue

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


def create_comparison_report(all_results):
    """Create comprehensive comparison report."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON REPORT")
    print(f"{'='*80}")
    print()

    # Sort by profit
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['total_profit'], reverse=True)

    print(f"📊 PERFORMANCE RANKING (by Net Profit):")
    print(f"{'='*80}")
    print()

    for i, result in enumerate(sorted_results, 1):
        model = result['model_name']
        thresh = result['thresholds']
        metrics = result['metrics']

        print(f"#{i}. {model}")
        print(f"    Weights: {result['weights']}")
        print(f"    Optimal Thresholds: H={thresh[0]:.2f}, D={thresh[1]:.2f}, A={thresh[2]:.2f}")
        print(f"    Profit: ${metrics['total_profit']:.2f} | ROI: {metrics['roi']:.1f}% | Bets: {metrics['total_bets']}")
        print(f"    Distribution: {metrics['home_bets']}H / {metrics['draw_bets']}D / {metrics['away_bets']}A")
        print(f"    Profits: ${metrics['home_profit']:.2f}H / ${metrics['draw_profit']:.2f}D / ${metrics['away_profit']:.2f}A")
        print()

    print(f"{'='*80}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    print()

    # Create comparison table
    df_comparison = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Total_Profit': r['metrics']['total_profit'],
            'ROI': r['metrics']['roi'],
            'Total_Bets': r['metrics']['total_bets'],
            'Home_Bets': r['metrics']['home_bets'],
            'Draw_Bets': r['metrics']['draw_bets'],
            'Away_Bets': r['metrics']['away_bets'],
            'Home_Profit': r['metrics']['home_profit'],
            'Draw_Profit': r['metrics']['draw_profit'],
            'Away_Profit': r['metrics']['away_profit'],
            'Home_Thresh': r['thresholds'][0],
            'Draw_Thresh': r['thresholds'][1],
            'Away_Thresh': r['thresholds'][2]
        }
        for r in sorted_results
    ])

    print(df_comparison.to_string(index=False))
    print()

    # Export to Excel
    output_file = 'model_comparison_live_results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary sheet
        df_comparison.to_excel(writer, sheet_name='Summary', index=False)

        # Individual model sheets
        for result in sorted_results:
            # Top 20 threshold combinations
            if result['all_results']:
                df_results = pd.DataFrame(result['all_results'])
                df_results = df_results.sort_values('total_profit', ascending=False).head(20)
                sheet_name = result['short_name'][:31]  # Excel sheet name limit
                df_results.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Detailed results exported to: {output_file}")
    print()

    # Recommendations
    print(f"{'='*80}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*80}")
    print()

    best = sorted_results[0]
    worst = sorted_results[-1]

    profit_diff = best['metrics']['total_profit'] - worst['metrics']['total_profit']

    print(f"✅ BEST MODEL: {best['model_name']}")
    print(f"   Profit: ${best['metrics']['total_profit']:.2f}/month")
    print(f"   ROI: {best['metrics']['roi']:.1f}%")
    print(f"   Advantage: ${profit_diff:.2f}/month more than worst model")
    print()

    # Check if current is best
    current = next((r for r in sorted_results if 'Current' in r['model_name']), None)
    if current:
        current_rank = sorted_results.index(current) + 1
        if current_rank == 1:
            print(f"✅ Current production model is already optimal!")
        else:
            improvement = best['metrics']['total_profit'] - current['metrics']['total_profit']
            print(f"⚠️  Current model ranks #{current_rank}")
            print(f"   Switching to {best['model_name']} would gain ${improvement:.2f}/month")

    print()

    # Check draw performance
    print(f"📊 DRAW BET ANALYSIS:")
    print(f"{'Model':<30} {'Draw Bets':<12} {'Draw Profit':<15} {'Draw ROI':<12}")
    print(f"{'-'*80}")
    for r in sorted_results:
        m = r['metrics']
        draw_roi = (m['draw_profit'] / m['draw_bets'] * 100) if m['draw_bets'] > 0 else 0
        print(f"{r['model_name']:<30} {m['draw_bets']:<12} ${m['draw_profit']:<14.2f} {draw_roi:<12.1f}%")

    print()

    # Warning if draw bets dropped significantly
    if current:
        best_draw_bets = best['metrics']['draw_bets']
        current_draw_bets = current['metrics']['draw_bets']

        if best_draw_bets < current_draw_bets * 0.7:  # 30% drop
            print(f"⚠️  WARNING: Best model has {best_draw_bets} draw bets vs {current_draw_bets} for current")
            print(f"   Draw bets are highly profitable - losing volume could hurt overall ROI")
            print()

    print(f"{'='*80}")

    return df_comparison


def main():
    print(f"{'='*80}")
    print(f"LIVE MODEL COMPARISON & THRESHOLD OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Parameters
    START_DATE = '2026-01-01'
    END_DATE = '2026-01-31'

    # Load model information
    models = load_model_info()

    if len(models) == 0:
        print("❌ No models found!")
        return 1

    print(f"✅ Found {len(models)} models to test")
    print()

    # Generate predictions for all models
    all_predictions = {}

    for model_info in models:
        try:
            df_preds = generate_predictions_for_model(
                model_info['path'],
                model_info['name'],
                START_DATE,
                END_DATE
            )
            all_predictions[model_info['short_name']] = {
                'df': df_preds,
                'info': model_info
            }
        except Exception as e:
            print(f"❌ Error generating predictions for {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_predictions) == 0:
        print("❌ No predictions generated!")
        return 1

    # Fetch actual results for each model's predictions
    for short_name, pred_data in all_predictions.items():
        df = pred_data['df']
        df_with_results = fetch_actual_results(df)
        pred_data['df'] = df_with_results

    # Optimize thresholds for each model
    all_results = []

    for short_name, pred_data in all_predictions.items():
        df = pred_data['df']
        model_info = pred_data['info']

        try:
            best_thresh, best_metrics, all_thresh_results = optimize_thresholds(
                df,
                model_info['name'],
                top_5_only=True
            )

            all_results.append({
                'model_name': model_info['name'],
                'short_name': short_name,
                'weights': model_info['weights'],
                'thresholds': best_thresh,
                'metrics': best_metrics,
                'all_results': all_thresh_results
            })
        except Exception as e:
            print(f"❌ Error optimizing {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison report
    if all_results:
        create_comparison_report(all_results)

    print()
    print(f"{'='*80}")
    print(f"✅ COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
