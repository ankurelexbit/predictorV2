#!/usr/bin/env python3
"""
True Live Validation for Q4 2025 with Improvements
===================================================

Tests the improved pipeline (with injury and lineup data) against Q4 2025 historical data.
Simulates how the system would have performed with the new features.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import joblib
from tqdm import tqdm
import json
import importlib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from predict_live import LiveFeatureCalculator
from production_thresholds import get_production_thresholds

# Import XGBoost model class
xgb_module = importlib.import_module('06_model_xgboost')
XGBoostFootballModel = xgb_module.XGBoostFootballModel

def load_q4_2025_fixtures():
    """Load Q4 2025 completed fixtures."""
    fixtures = pd.read_csv('data/raw/sportmonks/fixtures.csv', parse_dates=['date'])
    
    # Filter for Q4 2025
    q4_start = datetime(2025, 10, 1)
    q4_end = datetime(2025, 12, 31)
    
    q4 = fixtures[(fixtures['date'] >= q4_start) & (fixtures['date'] <= q4_end)]
    q4 = q4[q4['result'].notna()].copy()  # Only completed matches
    
    return q4

def simulate_live_prediction(calculator, model, fixture_row):
    """Simulate a live prediction for a single fixture."""
    try:
        # Build features (will use injury and lineup data if available)
        features = calculator.build_features_for_match(
            home_team_id=int(fixture_row['home_team_id']),
            away_team_id=int(fixture_row['away_team_id']),
            fixture_date=fixture_row['date'],
            home_team_name=fixture_row.get('home_team_name'),
            away_team_name=fixture_row.get('away_team_name'),
            league_name=fixture_row.get('league_name'),
            fixture_id=int(fixture_row['fixture_id'])
        )
        
        if not features:
            return None
        
        # Convert features dict to DataFrame for model prediction
        features_df = pd.DataFrame([features])
        
        # Get probabilities using model's predict_proba method
        probs = model.predict_proba(features_df)[0]
        
        # Map to home/draw/away (model returns [away, draw, home])
        prob_away, prob_draw, prob_home = probs
        
        return {
            'prob_home': prob_home,
            'prob_draw': prob_draw,
            'prob_away': prob_away,
            'features': features
        }
        
    except Exception as e:
        print(f"Error processing fixture {fixture_row.get('fixture_id')}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_predictions(predictions_df, thresholds):
    """Evaluate predictions with thresholds."""
    results = []
    
    for _, row in predictions_df.iterrows():
        # Apply thresholds
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
            # Check if correct
            correct = (bet == row['actual_result'])
            
            # Calculate P&L (assuming 1 unit bet)
            if correct:
                # Win: get back stake + profit based on odds
                odds = row.get(f'odds_{bet.lower()}', 2.0)
                pnl = odds - 1  # Profit
            else:
                pnl = -1  # Lost stake
            
            results.append({
                'fixture_id': row['fixture_id'],
                'date': row['date'],
                'home_team': row['home_team_name'],
                'away_team': row['away_team_name'],
                'bet': bet,
                'confidence': confidence,
                'actual': row['actual_result'],
                'correct': correct,
                'pnl': pnl,
                'has_lineup': row.get('has_lineup', False),
                'injuries_home': row.get('injuries_home', 0),
                'injuries_away': row.get('injuries_away', 0)
            })
    
    return pd.DataFrame(results)

def main():
    print("="*80)
    print("Q4 2025 TRUE LIVE VALIDATION - WITH IMPROVEMENTS")
    print("="*80)
    print()
    
    # Load fixtures
    print("Loading Q4 2025 fixtures...")
    fixtures = load_q4_2025_fixtures()
    print(f"Found {len(fixtures)} completed fixtures")
    print()
    
    # Load model
    print("Loading model...")
    model = XGBoostFootballModel()
    model.load(Path('models/xgboost_model_draw_tuned.joblib'))
    print("Model loaded")
    print()
    
    # Load thresholds
    thresholds = get_production_thresholds()
    print(f"Thresholds: H={thresholds['home']:.2f}, D={thresholds['draw']:.2f}, A={thresholds['away']:.2f}")
    print()
    
    # Initialize calculator
    print("Initializing feature calculator...")
    calculator = LiveFeatureCalculator()
    print()
    
    # Run predictions
    print("Running predictions...")
    predictions = []
    
    for idx, row in tqdm(fixtures.iterrows(), total=len(fixtures), desc="Processing"):
        pred = simulate_live_prediction(calculator, model, row)
        
        if pred:
            predictions.append({
                'fixture_id': row['fixture_id'],
                'date': row['date'],
                'home_team_name': row['home_team_name'],
                'away_team_name': row['away_team_name'],
                'prob_home': pred['prob_home'],
                'prob_draw': pred['prob_draw'],
                'prob_away': pred['prob_away'],
                'actual_result': row['result'],
                'has_lineup': pred['features'].get('home_player_rating_5', 0) > 0,
                'injuries_home': pred['features'].get('home_injuries', 0),
                'injuries_away': pred['features'].get('away_injuries', 0),
                'odds_h': row.get('odds_home', 2.0),
                'odds_d': row.get('odds_draw', 3.5),
                'odds_a': row.get('odds_away', 3.0)
            })
    
    predictions_df = pd.DataFrame(predictions)
    print(f"\nGenerated {len(predictions_df)} predictions")
    print()
    
    # Evaluate
    print("Evaluating predictions...")
    bets_df = evaluate_predictions(predictions_df, thresholds)
    
    if len(bets_df) == 0:
        print("No bets placed (no predictions exceeded thresholds)")
        return
    
    # Calculate metrics
    total_bets = len(bets_df)
    correct_bets = bets_df['correct'].sum()
    win_rate = correct_bets / total_bets
    total_pnl = bets_df['pnl'].sum()
    roi = (total_pnl / total_bets) * 100
    
    # Breakdown by bet type
    bet_breakdown = bets_df.groupby('bet').agg({
        'correct': ['count', 'sum', 'mean'],
        'pnl': 'sum'
    }).round(3)
    
    # Lineup availability
    with_lineup = bets_df[bets_df['has_lineup']].copy()
    without_lineup = bets_df[~bets_df['has_lineup']].copy()
    
    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    print(f"Total Fixtures: {len(fixtures)}")
    print(f"Predictions Generated: {len(predictions_df)}")
    print(f"Bets Placed: {total_bets} ({total_bets/len(fixtures)*100:.1f}% of fixtures)")
    print()
    print(f"Correct Bets: {correct_bets}/{total_bets}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total P&L: {total_pnl:+.2f} units")
    print(f"ROI: {roi:+.1f}%")
    print()
    
    print("Breakdown by Bet Type:")
    print(bet_breakdown)
    print()
    
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
    
    print("Injury Data:")
    print(f"Average injuries per match: H={bets_df['injuries_home'].mean():.1f}, A={bets_df['injuries_away'].mean():.1f}")
    print()
    
    # Save results
    output_file = f"data/predictions/q4_2025_improved_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    bets_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return {
        'total_bets': total_bets,
        'win_rate': win_rate,
        'roi': roi,
        'total_pnl': total_pnl,
        'with_lineup_count': len(with_lineup),
        'with_lineup_winrate': with_lineup['correct'].mean() if len(with_lineup) > 0 else 0,
        'without_lineup_count': len(without_lineup),
        'without_lineup_winrate': without_lineup['correct'].mean() if len(without_lineup) > 0 else 0
    }

if __name__ == '__main__':
    results = main()
