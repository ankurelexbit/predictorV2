#!/usr/bin/env python3
"""
Quick Q4 2025 Analysis Using Cached Feature Building
=====================================================

The first validation run successfully built features for all 890 fixtures.
This script extracts fixture info from logs, rebuilds features (fast since 
player DB is already built), and generates predictions + performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import importlib
import pickle
from tqdm import tqdm

# Import model and calculator
xgb_module = importlib.import_module('06_model_xgboost')
XGBoostFootballModel = xgb_module.XGBoostFootballModel

from predict_live import LiveFeatureCalculator
from production_thresholds import get_production_thresholds

def load_q4_fixtures():
    """Load Q4 2025 fixtures with results."""
    fixtures = pd.read_csv('data/raw/sportmonks/fixtures.csv', parse_dates=['date'])
    
    q4_start = datetime(2025, 10, 1)
    q4_end = datetime(2025, 12, 31)
    
    q4 = fixtures[(fixtures['date'] >= q4_start) & (fixtures['date'] <= q4_end)]
    q4 = q4[q4['result'].notna()].copy()
    
    return q4

def build_features_batch(calculator, fixtures_df, cache_file='data/cache/q4_features.pkl'):
    """Build features for all fixtures, using cache if available."""
    cache_path = Path(cache_file)
    
    # Check cache
    if cache_path.exists():
        print(f"Loading cached features from {cache_file}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Building features for all fixtures...")
    features_list = []
    
    for idx, row in tqdm(fixtures_df.iterrows(), total=len(fixtures_df), desc="Building features"):
        try:
            features = calculator.build_features_for_match(
                home_team_id=int(row['home_team_id']),
                away_team_id=int(row['away_team_id']),
                fixture_date=row['date'],
                home_team_name=row.get('home_team_name'),
                away_team_name=row.get('away_team_name'),
                league_name=row.get('league_name'),
                fixture_id=int(row['fixture_id'])
            )
            
            if features:
                features['fixture_id'] = row['fixture_id']
                features['actual_result'] = row['result']
                features['home_team_name'] = row['home_team_name']
                features['away_team_name'] = row['away_team_name']
                features['date'] = row['date']
                features['odds_h'] = row.get('odds_home', 2.0)
                features['odds_d'] = row.get('odds_draw', 3.5)
                features['odds_a'] = row.get('odds_away', 3.0)
                features_list.append(features)
        except Exception as e:
            print(f"Error on fixture {row['fixture_id']}: {e}")
            continue
    
    # Cache for next time
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(features_list, f)
    
    return features_list

def make_predictions(model, features_list):
    """Generate predictions from features."""
    predictions = []
    
    print("Generating predictions...")
    for features in tqdm(features_list, desc="Predicting"):
        try:
            # Extract metadata
            fixture_id = features.pop('fixture_id')
            actual_result = features.pop('actual_result')
            home_team = features.pop('home_team_name')
            away_team = features.pop('away_team_name')
            date = features.pop('date')
            odds_h = features.pop('odds_h')
            odds_d = features.pop('odds_d')
            odds_a = features.pop('odds_a')
            
            # Check for lineup data
            has_lineup = features.get('home_player_rating_5', 0) > 0
            injuries_home = features.get('home_injuries', 0)
            injuries_away = features.get('away_injuries', 0)
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Get probabilities
            probs = model.predict_proba(features_df)[0]
            prob_away, prob_draw, prob_home = probs
            
            predictions.append({
                'fixture_id': fixture_id,
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'prob_home': prob_home,
                'prob_draw': prob_draw,
                'prob_away': prob_away,
                'actual_result': actual_result,
                'has_lineup': has_lineup,
                'injuries_home': injuries_home,
                'injuries_away': injuries_away,
                'odds_h': odds_h,
                'odds_d': odds_d,
                'odds_a': odds_a
            })
        except Exception as e:
            print(f"Prediction error: {e}")
            continue
    
    return pd.DataFrame(predictions)

def evaluate_predictions(predictions_df, thresholds):
    """Evaluate predictions and calculate metrics."""
    bets = []
    
    for _, row in predictions_df.iterrows():
        bet = None
        confidence = 0
        
        if row['prob_home'] >= thresholds['home']:
            bet = 'H'
            confidence = row['prob_home']
        elif row['prob_draw'] >= thresholds['draw']:
            bet = 'D'
            confidence = row['prob_draw']
        elif row['prob_away'] >= thresholds['away']:
            bet = 'A'
            confidence = row['prob_away']
        
        if bet:
            correct = (bet == row['actual_result'])
            
            if correct:
                odds = row[f'odds_{bet.lower()}']
                pnl = odds - 1
            else:
                pnl = -1
            
            bets.append({
                'fixture_id': row['fixture_id'],
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'bet': bet,
                'confidence': confidence,
                'actual': row['actual_result'],
                'correct': correct,
                'pnl': pnl,
                'has_lineup': row['has_lineup'],
                'injuries_home': row['injuries_home'],
                'injuries_away': row['injuries_away']
            })
    
    return pd.DataFrame(bets)

def print_results(bets_df, predictions_df, total_fixtures):
    """Print comprehensive results."""
    print("\n" + "="*80)
    print("Q4 2025 VALIDATION RESULTS")
    print("="*80 + "\n")
    
    print(f"Total Fixtures: {total_fixtures}")
    print(f"Predictions Generated: {len(predictions_df)}")
    print(f"Bets Placed: {len(bets_df)} ({len(bets_df)/total_fixtures*100:.1f}% of fixtures)")
    print()
    
    if len(bets_df) == 0:
        print("No bets placed (no predictions exceeded thresholds)")
        return
    
    # Overall metrics
    total_bets = len(bets_df)
    correct_bets = bets_df['correct'].sum()
    win_rate = correct_bets / total_bets
    total_pnl = bets_df['pnl'].sum()
    roi = (total_pnl / total_bets) * 100
    
    print(f"Correct Bets: {correct_bets}/{total_bets}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total P&L: {total_pnl:+.2f} units")
    print(f"ROI: {roi:+.1f}%")
    print()
    
    # Breakdown by bet type
    print("Breakdown by Bet Type:")
    for bet_type in ['H', 'D', 'A']:
        subset = bets_df[bets_df['bet'] == bet_type]
        if len(subset) > 0:
            wr = subset['correct'].mean()
            pnl = subset['pnl'].sum()
            roi_type = (pnl / len(subset)) * 100
            print(f"  {bet_type}: {len(subset)} bets, {wr:.1%} win rate, {pnl:+.2f} P&L, {roi_type:+.1f}% ROI")
    print()
    
    # Lineup impact
    with_lineup = bets_df[bets_df['has_lineup']]
    without_lineup = bets_df[~bets_df['has_lineup']]
    
    print("Lineup Data Impact:")
    print(f"Bets with lineup data: {len(with_lineup)} ({len(with_lineup)/total_bets*100:.1f}%)")
    if len(with_lineup) > 0:
        print(f"  Win rate: {with_lineup['correct'].mean():.1%}")
        print(f"  ROI: {(with_lineup['pnl'].sum()/len(with_lineup))*100:+.1f}%")
    
    print(f"Bets without lineup data: {len(without_lineup)} ({len(without_lineup)/total_bets*100:.1f}%)")
    if len(without_lineup) > 0:
        print(f"  Win rate: {without_lineup['correct'].mean():.1%}")
        print(f"  ROI: {(without_lineup['pnl'].sum()/len(without_lineup))*100:+.1f}%")
    print()
    
    # Injury stats
    print("Injury Data:")
    print(f"Average injuries: Home={bets_df['injuries_home'].mean():.1f}, Away={bets_df['injuries_away'].mean():.1f}")
    print()

def main():
    print("="*80)
    print("Q4 2025 QUICK VALIDATION")
    print("="*80)
    print()
    
    # Load fixtures
    print("Loading Q4 2025 fixtures...")
    fixtures = load_q4_fixtures()
    print(f"Found {len(fixtures)} completed fixtures\n")
    
    # Load model
    print("Loading model...")
    model = XGBoostFootballModel()
    model.load(Path('models/xgboost_model_draw_tuned.joblib'))
    print("Model loaded\n")
    
    # Load thresholds
    thresholds = get_production_thresholds()
    print(f"Thresholds: H={thresholds['home']:.2f}, D={thresholds['draw']:.2f}, A={thresholds['away']:.2f}\n")
    
    # Initialize calculator
    print("Initializing feature calculator...")
    calculator = LiveFeatureCalculator()
    print()
    
    # Build features (or load from cache)
    features_list = build_features_batch(calculator, fixtures)
    print(f"Features ready for {len(features_list)} fixtures\n")
    
    # Make predictions
    predictions_df = make_predictions(model, features_list)
    print(f"Generated {len(predictions_df)} predictions\n")
    
    # Evaluate
    bets_df = evaluate_predictions(predictions_df, thresholds)
    
    # Print results
    print_results(bets_df, predictions_df, len(fixtures))
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f"data/predictions/q4_2025_validation_{timestamp}.csv"
    bets_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return bets_df

if __name__ == '__main__':
    results = main()
