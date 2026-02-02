#!/usr/bin/env python3
"""
Compare All Models Using Existing Database Predictions
======================================================

Loads predictions from database (with features + odds + results),
runs all 4 models on the same feature set, and compares profitability.

Much faster than live comparison - no API calls needed!
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = "postgresql://ankurgupta@localhost/football_predictions"

# Top 5 leagues
TOP_5_LEAGUES = (8, 82, 384, 564, 301)  # PL, Bundesliga, Serie A, Ligue 1, La Liga

# Feature columns to exclude from model input
FEATURES_TO_EXCLUDE = [
    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
    'match_date', 'home_score', 'away_score', 'result', 'target',
    'home_team_name', 'away_team_name', 'state_id'
]


def load_predictions_from_db(start_date, end_date, league_filter=None):
    """Load predictions with features and odds from database."""
    logger.info(f"Loading predictions from {start_date} to {end_date}...")

    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

    query = """
        SELECT
            fixture_id,
            match_date,
            league_id,
            league_name,
            home_team_name,
            away_team_name,
            features,
            best_home_odds,
            best_draw_odds,
            best_away_odds,
            actual_result
        FROM predictions
        WHERE match_date >= %s
          AND match_date < %s
          AND actual_result IS NOT NULL
    """

    params = [start_date, end_date]

    if league_filter:
        query += " AND league_id = ANY(%s)"
        params.append(list(league_filter))

    query += " ORDER BY match_date"

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    logger.info(f"✅ Loaded {len(rows)} predictions with features")

    return rows


def extract_features_df(predictions):
    """Extract features from jsonb into DataFrame."""
    logger.info("Extracting features from jsonb...")

    # Get all feature keys from first prediction
    if not predictions:
        return pd.DataFrame()

    feature_keys = list(predictions[0]['features'].keys())

    # Build feature matrix
    feature_data = []
    for pred in predictions:
        features = pred['features']
        feature_data.append({k: features.get(k) for k in feature_keys})

    df = pd.DataFrame(feature_data)

    # Remove excluded columns
    feature_cols = [c for c in df.columns if c not in FEATURES_TO_EXCLUDE]
    df = df[feature_cols]

    logger.info(f"✅ Extracted {len(df)} rows × {len(feature_cols)} features")

    return df, feature_cols


def run_model_predictions(model_path, feature_df, feature_cols):
    """Run a model on the feature set."""
    logger.info(f"Loading model: {model_path.name}")
    model = joblib.load(model_path)

    # Ensure we only use features the model was trained on
    X = feature_df[feature_cols]

    # Handle missing columns or extra columns
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        # Add missing columns with 0
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
        # Only use model features
        X = X[model_features]

    probs = model.predict_proba(X)

    return pd.DataFrame({
        'pred_away_prob': probs[:, 0],
        'pred_draw_prob': probs[:, 1],
        'pred_home_prob': probs[:, 2]
    })


def optimize_thresholds(predictions_df, model_name):
    """Optimize thresholds for a model's predictions."""
    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZING THRESHOLDS: {model_name}")
    logger.info(f"{'='*80}")

    # Test thresholds
    home_thresholds = np.arange(0.30, 0.85, 0.01)
    draw_thresholds = np.arange(0.20, 0.55, 0.01)
    away_thresholds = np.arange(0.25, 0.55, 0.01)

    best_profit = -float('inf')
    best_thresholds = None
    best_metrics = None

    total = len(home_thresholds) * len(draw_thresholds) * len(away_thresholds)
    count = 0

    logger.info(f"Testing {total} threshold combinations...")

    for home_thresh in home_thresholds:
        for draw_thresh in draw_thresholds:
            for away_thresh in away_thresholds:
                count += 1
                if count % 5000 == 0:
                    logger.info(f"   {count}/{total}...")

                home_bets = draw_bets = away_bets = 0
                home_wins = draw_wins = away_wins = 0
                home_profit = draw_profit = away_profit = 0

                for _, row in predictions_df.iterrows():
                    home_prob = row['pred_home_prob']
                    draw_prob = row['pred_draw_prob']
                    away_prob = row['pred_away_prob']
                    actual = row['actual_result']

                    home_odds = row['best_home_odds']
                    draw_odds = row['best_draw_odds']
                    away_odds = row['best_away_odds']

                    # Skip if missing odds
                    if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
                        continue

                    # Betting logic: Check which outcomes cross threshold
                    candidates = []
                    if home_prob >= home_thresh:
                        candidates.append(('H', home_prob, home_odds))
                    if draw_prob >= draw_thresh:
                        candidates.append(('D', draw_prob, draw_odds))
                    if away_prob >= away_thresh:
                        candidates.append(('A', away_prob, away_odds))

                    # If 2+ cross threshold, pick max probability
                    # If 1 crosses, bet on that
                    # If 0 cross, no bet
                    bet_outcome = None
                    if len(candidates) >= 2:
                        bet_outcome, bet_prob, bet_odds = max(candidates, key=lambda x: x[1])
                    elif len(candidates) == 1:
                        bet_outcome, bet_prob, bet_odds = candidates[0]

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
                total_profit = home_profit + draw_profit + away_profit

                if total_bets >= 50:  # Min bets for significance
                    roi = (total_profit / total_bets) * 100

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
        logger.info(f"\n🎯 OPTIMAL THRESHOLDS:")
        logger.info(f"   Home: {best_thresholds[0]:.2f}")
        logger.info(f"   Draw: {best_thresholds[1]:.2f}")
        logger.info(f"   Away: {best_thresholds[2]:.2f}")
        logger.info(f"\n📊 PERFORMANCE:")
        logger.info(f"   Total: {best_metrics['total_bets']} bets, ${best_metrics['total_profit']:.2f}, {best_metrics['roi']:.1f}% ROI")
        logger.info(f"   Home: {best_metrics['home_bets']} bets, {best_metrics['home_wins']} wins, ${best_metrics['home_profit']:.2f}")
        logger.info(f"   Draw: {best_metrics['draw_bets']} bets, {best_metrics['draw_wins']} wins, ${best_metrics['draw_profit']:.2f}")
        logger.info(f"   Away: {best_metrics['away_bets']} bets, {best_metrics['away_wins']} wins, ${best_metrics['away_profit']:.2f}")

    return best_thresholds, best_metrics


def main():
    logger.info("="*80)
    logger.info("MODEL COMPARISON FROM DATABASE PREDICTIONS")
    logger.info("="*80)
    logger.info("")

    # Define models
    models_info = [
        {
            'name': 'Current Production',
            'path': Path('models/with_draw_features/conservative_with_draw_features.joblib'),
            'tag': 'current'
        },
        {
            'name': 'Option 1: Conservative',
            'path': Path('models/weight_experiments/option1_conservative.joblib'),
            'tag': 'option1'
        },
        {
            'name': 'Option 2: Aggressive',
            'path': Path('models/weight_experiments/option2_aggressive.joblib'),
            'tag': 'option2'
        },
        {
            'name': 'Option 3: Balanced',
            'path': Path('models/weight_experiments/option3_balanced.joblib'),
            'tag': 'option3'
        }
    ]

    # Load predictions from database
    predictions = load_predictions_from_db(
        start_date='2026-01-01',
        end_date='2026-02-01',
        league_filter=TOP_5_LEAGUES
    )

    if not predictions:
        logger.error("No predictions found in database!")
        return

    # Extract features
    feature_df, feature_cols = extract_features_df(predictions)

    # Build metadata DataFrame
    metadata_df = pd.DataFrame([{
        'fixture_id': p['fixture_id'],
        'match_date': p['match_date'],
        'league_id': p['league_id'],
        'league_name': p['league_name'],
        'home_team_name': p['home_team_name'],
        'away_team_name': p['away_team_name'],
        'best_home_odds': p['best_home_odds'],
        'best_draw_odds': p['best_draw_odds'],
        'best_away_odds': p['best_away_odds'],
        'actual_result': p['actual_result']
    } for p in predictions])

    logger.info(f"✅ Loaded {len(metadata_df)} predictions with results")
    logger.info("")

    # Run all models and optimize
    all_results = []

    for model_info in models_info:
        if not model_info['path'].exists():
            logger.warning(f"⚠️  Skipping {model_info['name']} - not found")
            continue

        try:
            # Run model predictions
            preds = run_model_predictions(model_info['path'], feature_df, feature_cols)

            # Combine with metadata
            combined_df = pd.concat([metadata_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

            # Optimize thresholds
            best_thresh, best_metrics = optimize_thresholds(combined_df, model_info['name'])

            if best_metrics:
                all_results.append({
                    'model': model_info['name'],
                    'tag': model_info['tag'],
                    'thresholds': best_thresh,
                    'metrics': best_metrics
                })
                logger.info(f"✅ [{model_info['tag']}] Complete!")

        except Exception as e:
            logger.error(f"❌ [{model_info['tag']}] Error: {e}")
            import traceback
            traceback.print_exc()

    # Final comparison
    logger.info(f"\n{'='*80}")
    logger.info("FINAL COMPARISON - JANUARY 2026 (TOP 5 LEAGUES)")
    logger.info(f"{'='*80}")
    logger.info("")

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
            print("✅ Current production model is OPTIMAL on real January 2026 data!")
        elif current:
            diff = best['metrics']['total_profit'] - current['metrics']['total_profit']
            print(f"⚠️  {best['model']} outperforms Current by ${diff:.2f}")
            print(f"   But check draw bet volume: {best['metrics']['draw_bets']} vs {current['metrics']['draw_bets']}")
        else:
            print(f"✅ Best model: {best['model']}")

        print()

        # Export to file
        import openpyxl
        output_file = 'model_comparison_january_2026.xlsx'

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary
            summary_df = pd.DataFrame([{
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
            } for r in all_results])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        logger.info(f"✅ Results exported to: {output_file}")

    else:
        logger.error("No models completed successfully!")


if __name__ == '__main__':
    main()
