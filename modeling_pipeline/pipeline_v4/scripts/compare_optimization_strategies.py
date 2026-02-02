#!/usr/bin/env python3
"""
Compare Optimization Strategies: Win Rate vs Profit vs ROI
===========================================================

Shows optimal thresholds for each model when optimizing for:
1. Win Rate (% of bets won)
2. Total Profit ($)
3. ROI (% return per bet)
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


def test_threshold_combination(df, home_thresh, draw_thresh, away_thresh):
    """Test a specific threshold combination."""
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
    total_wins = home_wins + draw_wins + away_wins
    total_profit = home_profit + draw_profit + away_profit

    win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
    roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

    return {
        'home_thresh': home_thresh,
        'draw_thresh': draw_thresh,
        'away_thresh': away_thresh,
        'total_bets': total_bets,
        'total_wins': total_wins,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'roi': roi,
        'home_bets': home_bets,
        'home_wins': home_wins,
        'draw_bets': draw_bets,
        'draw_wins': draw_wins,
        'away_bets': away_bets,
        'away_wins': away_wins
    }


def optimize_for_metric(df, metric='profit', min_bets=50):
    """
    Optimize thresholds for a specific metric.

    metric: 'profit', 'roi', or 'win_rate'
    """
    home_thresholds = np.arange(0.30, 0.85, 0.05)
    draw_thresholds = np.arange(0.20, 0.55, 0.05)
    away_thresholds = np.arange(0.25, 0.55, 0.05)

    best_value = -float('inf')
    best_result = None

    # Map metric names to result keys
    metric_key = 'total_profit' if metric == 'profit' else metric

    for home_thresh in home_thresholds:
        for draw_thresh in draw_thresholds:
            for away_thresh in away_thresholds:
                result = test_threshold_combination(df, home_thresh, draw_thresh, away_thresh)

                if result['total_bets'] >= min_bets:
                    value = result[metric_key]

                    if value > best_value:
                        best_value = value
                        best_result = result

    return best_result


def process_model(model_info, combined_df):
    """Process a single model with all optimization strategies."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{model_info['name']}")
    logger.info(f"{'='*80}")

    results = {}

    # Optimize for Win Rate
    logger.info("Optimizing for: WIN RATE...")
    win_rate_result = optimize_for_metric(combined_df, metric='win_rate')
    results['win_rate'] = win_rate_result

    # Optimize for Profit
    logger.info("Optimizing for: PROFIT...")
    profit_result = optimize_for_metric(combined_df, metric='profit')
    results['profit'] = profit_result

    # Optimize for ROI
    logger.info("Optimizing for: ROI...")
    roi_result = optimize_for_metric(combined_df, metric='roi')
    results['roi'] = roi_result

    # Display results
    logger.info("")
    logger.info("OPTIMIZATION COMPARISON:")
    logger.info("-" * 80)

    print(f"{'Optimize For':<15} {'Thresh (H/D/A)':<20} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Profit':<10} {'ROI%':<8}")
    print("-" * 80)

    for opt_type, result in results.items():
        if result:
            thresholds = f"{result['home_thresh']:.2f}/{result['draw_thresh']:.2f}/{result['away_thresh']:.2f}"
            print(f"{opt_type.upper():<15} {thresholds:<20} {result['total_bets']:<6} "
                  f"{result['total_wins']:<6} {result['win_rate']:<7.1f}% "
                  f"${result['total_profit']:<9.2f} {result['roi']:<7.1f}%")

    return results


def main():
    logger.info("="*80)
    logger.info("OPTIMIZATION STRATEGY COMPARISON")
    logger.info("Win Rate vs Profit vs ROI")
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

    logger.info(f"Loaded {len(metadata_df)} predictions from January 2026")

    # Process each model
    all_results = {}

    for model_info in models_info:
        if not model_info['path'].exists():
            logger.warning(f"⚠️  Skipping {model_info['name']} - not found")
            continue

        # Run model predictions
        preds = run_model_predictions(model_info['path'], feature_df, feature_cols)
        combined_df = pd.concat([metadata_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        # Process with all strategies
        results = process_model(model_info, combined_df)
        all_results[model_info['name']] = results

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY - BEST MODEL FOR EACH OPTIMIZATION STRATEGY")
    logger.info(f"{'='*80}")
    logger.info("")

    for metric in ['win_rate', 'profit', 'roi']:
        logger.info(f"\n{metric.upper().replace('_', ' ')} OPTIMIZATION:")
        logger.info("-" * 80)

        best_model = None
        best_value = -float('inf')

        # Map metric names to result keys
        metric_key = 'total_profit' if metric == 'profit' else metric

        for model_name, results in all_results.items():
            if metric in results and results[metric]:
                value = results[metric][metric_key]
                if value > best_value:
                    best_value = value
                    best_model = (model_name, results[metric])

        if best_model:
            name, result = best_model
            logger.info(f"🏆 Winner: {name}")
            logger.info(f"   Thresholds: H={result['home_thresh']:.2f}, D={result['draw_thresh']:.2f}, A={result['away_thresh']:.2f}")
            logger.info(f"   Performance: {result['total_bets']} bets, {result['total_wins']} wins ({result['win_rate']:.1f}%)")
            logger.info(f"   Financial: ${result['total_profit']:.2f} profit, {result['roi']:.1f}% ROI")

    logger.info("")


if __name__ == '__main__':
    main()
