#!/usr/bin/env python3
"""
Analyze Home Threshold Performance for Option 3
================================================

Shows how different home thresholds affect performance,
keeping draw and away thresholds at optimal values.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
        WHERE match_date >= '2026-01-01'
          AND match_date < '2026-02-01'
          AND actual_result IS NOT NULL
          AND league_id = ANY(%s)
        ORDER BY match_date
    """

    cursor = conn.cursor()
    cursor.execute(query, [list(TOP_5_LEAGUES)])
    rows = cursor.fetchall()
    conn.close()

    return rows


def extract_features_df(predictions):
    """Extract features from jsonb into DataFrame."""
    if not predictions:
        return pd.DataFrame(), []

    feature_keys = list(predictions[0]['features'].keys())
    feature_data = []
    for pred in predictions:
        features = pred['features']
        feature_data.append({k: features.get(k) for k in feature_keys})

    df = pd.DataFrame(feature_data)
    feature_cols = [c for c in df.columns if c not in FEATURES_TO_EXCLUDE]
    df = df[feature_cols]

    return df, feature_cols


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

    return pd.DataFrame({
        'pred_away_prob': probs[:, 0],
        'pred_draw_prob': probs[:, 1],
        'pred_home_prob': probs[:, 2]
    })


def test_home_threshold(df, home_thresh, draw_thresh=0.22, away_thresh=0.42):
    """Test a specific home threshold combination."""
    home_bets = draw_bets = away_bets = 0
    home_wins = draw_wins = away_wins = 0
    home_profit = draw_profit = away_profit = 0

    for _, row in df.iterrows():
        home_prob = row['pred_home_prob']
        draw_prob = row['pred_draw_prob']
        away_prob = row['pred_away_prob']
        actual = row['actual_result']

        home_odds = row['best_home_odds']
        draw_odds = row['best_draw_odds']
        away_odds = row['best_away_odds']

        if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
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
            bet_outcome, _, _ = max(candidates, key=lambda x: x[1])
        elif len(candidates) == 1:
            bet_outcome, _, _ = candidates[0]

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
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

    return {
        'home_thresh': home_thresh,
        'total_bets': total_bets,
        'total_profit': total_profit,
        'roi': roi,
        'home_bets': home_bets,
        'home_wins': home_wins,
        'home_profit': home_profit,
        'home_roi': (home_profit / home_bets * 100) if home_bets > 0 else 0,
        'draw_bets': draw_bets,
        'away_bets': away_bets,
        'draw_profit': draw_profit,
        'away_profit': away_profit
    }


def main():
    logger.info("="*80)
    logger.info("HOME THRESHOLD ANALYSIS - OPTION 3: BALANCED")
    logger.info("="*80)
    logger.info("")

    # Load predictions
    predictions = load_predictions_from_db()
    feature_df, feature_cols = extract_features_df(predictions)

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

    # Load Option 3 model
    model_path = Path('models/weight_experiments/option3_balanced.joblib')
    preds = run_model_predictions(model_path, feature_df, feature_cols)
    combined_df = pd.concat([metadata_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    logger.info(f"Loaded {len(combined_df)} predictions from January 2026")
    logger.info("")

    # Test different home thresholds
    logger.info("Testing Home Thresholds (Draw=0.22, Away=0.42 fixed):")
    logger.info("="*110)
    print(f"{'Home':<6} {'Total':<7} {'Total':<9} {'ROI':<7} {'Home':<7} {'Home':<7} {'Home':<9} {'Home':<8} {'Draw':<7} {'Away':<7}")
    print(f"{'Thresh':<6} {'Bets':<7} {'Profit':<9} {'%':<7} {'Bets':<7} {'Wins':<7} {'Profit':<9} {'ROI %':<8} {'Bets':<7} {'Bets':<7}")
    print("-"*110)

    results = []
    for home_thresh in np.arange(0.30, 0.85, 0.05):
        result = test_home_threshold(combined_df, home_thresh)
        results.append(result)

        print(f"{result['home_thresh']:.2f}   "
              f"{result['total_bets']:<7} "
              f"${result['total_profit']:<8.2f} "
              f"{result['roi']:<6.1f}% "
              f"{result['home_bets']:<7} "
              f"{result['home_wins']:<7} "
              f"${result['home_profit']:<8.2f} "
              f"{result['home_roi']:<7.1f}% "
              f"{result['draw_bets']:<7} "
              f"{result['away_bets']:<7}")

    print("="*110)
    logger.info("")

    # Find best overall
    best_profit = max(results, key=lambda x: x['total_profit'])
    logger.info(f"✅ Best Overall Profit: Home Threshold = {best_profit['home_thresh']:.2f}")
    logger.info(f"   Total: ${best_profit['total_profit']:.2f}, {best_profit['roi']:.1f}% ROI")
    logger.info(f"   Home: {best_profit['home_bets']} bets, ${best_profit['home_profit']:.2f}")
    logger.info("")

    # Find best home ROI
    home_results = [r for r in results if r['home_bets'] >= 5]
    if home_results:
        best_home = max(home_results, key=lambda x: x['home_roi'])
        logger.info(f"✅ Best Home ROI: Home Threshold = {best_home['home_thresh']:.2f}")
        logger.info(f"   Home: {best_home['home_bets']} bets, ${best_home['home_profit']:.2f}, {best_home['home_roi']:.1f}% ROI")
        logger.info("")

    # Export to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('home_threshold_analysis_option3.csv', index=False)
    logger.info("✅ Results exported to: home_threshold_analysis_option3.csv")


if __name__ == '__main__':
    main()
