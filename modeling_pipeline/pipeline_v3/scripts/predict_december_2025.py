#!/usr/bin/env python3
"""
Live Predictions for December 2025
===================================

Generate betting recommendations for December 2025 matches.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generate predictions for December 2025."""
    logger.info("="*70)
    logger.info("LIVE PREDICTIONS - DECEMBER 2025")
    logger.info("="*70)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    logger.info(f"✅ Loaded model from {model_path}")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    df = pd.read_csv(data_path)
    
    # Filter to December 2025
    df['match_date'] = pd.to_datetime(df['starting_at'], errors='coerce')
    dec_mask = (df['match_date'] >= '2025-12-01') & (df['match_date'] < '2026-01-01')
    dec_df = df[dec_mask].copy()
    
    logger.info(f"\n📅 December 2025: {len(dec_df)} matches")
    
    # Remove leakage columns
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 'result', 
                    'home_score', 'away_score', 'home_goals', 'away_goals', 'target']
    for col in leakage_cols:
        if col in dec_df.columns:
            dec_df = dec_df.drop(columns=[col])
    
    # Generate predictions
    logger.info("\n🔮 Generating predictions...")
    y_pred_proba = model.predict_proba(dec_df)
    
    # Add predictions to dataframe
    dec_df['prob_away'] = y_pred_proba[:, 0]
    dec_df['prob_draw'] = y_pred_proba[:, 1]
    dec_df['prob_home'] = y_pred_proba[:, 2]
    
    # Determine predicted outcome
    dec_df['predicted_outcome'] = np.argmax(y_pred_proba, axis=1)
    dec_df['predicted_outcome_name'] = dec_df['predicted_outcome'].map({
        0: 'Away Win', 1: 'Draw', 2: 'Home Win'
    })
    dec_df['confidence'] = np.max(y_pred_proba, axis=1)
    
    # Apply thresholds
    thresholds = {'home': 0.48, 'draw': 0.35, 'away': 0.45}
    
    def should_bet(row):
        if row['predicted_outcome'] == 2:  # Home
            return row['prob_home'] >= thresholds['home']
        elif row['predicted_outcome'] == 1:  # Draw
            return row['prob_draw'] >= thresholds['draw']
        else:  # Away
            return row['prob_away'] >= thresholds['away']
    
    dec_df['bet_recommended'] = dec_df.apply(should_bet, axis=1)
    
    # Filter to recommended bets
    bets_df = dec_df[dec_df['bet_recommended']].copy()
    
    logger.info(f"\n✅ Betting recommendations: {len(bets_df)}/{len(dec_df)} matches")
    
    # Summary by outcome
    logger.info("\n📊 Breakdown by Outcome:")
    for outcome_id, outcome_name in [(2, 'Home Win'), (1, 'Draw'), (0, 'Away Win')]:
        outcome_bets = bets_df[bets_df['predicted_outcome'] == outcome_id]
        logger.info(f"  {outcome_name}: {len(outcome_bets)} bets")
    
    # Show top recommendations
    logger.info("\n🎯 TOP 20 BETTING RECOMMENDATIONS:")
    logger.info("-" * 100)
    logger.info(f"{'Date':<12} {'Home Team':<25} {'Away Team':<25} {'Prediction':<12} {'Confidence':<12}")
    logger.info("-" * 100)
    
    # Sort by confidence
    top_bets = bets_df.nlargest(20, 'confidence')
    
    for _, row in top_bets.iterrows():
        date_str = row['match_date'].strftime('%Y-%m-%d') if pd.notna(row['match_date']) else 'Unknown'
        home = str(row.get('home_team', 'Unknown'))[:24]
        away = str(row.get('away_team', 'Unknown'))[:24]
        pred = row['predicted_outcome_name']
        conf = f"{row['confidence']:.1%}"
        
        logger.info(f"{date_str:<12} {home:<25} {away:<25} {pred:<12} {conf:<12}")
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / 'predictions' / 'december_2025_predictions.csv'
    output_path.parent.mkdir(exist_ok=True)
    
    # Select relevant columns
    output_cols = ['fixture_id', 'starting_at', 'home_team', 'away_team', 'league_id',
                   'predicted_outcome_name', 'confidence', 'prob_home', 'prob_draw', 'prob_away',
                   'bet_recommended']
    
    available_cols = [col for col in output_cols if col in bets_df.columns]
    bets_df[available_cols].to_csv(output_path, index=False)
    
    logger.info(f"\n💾 Saved predictions to: {output_path}")
    
    logger.info("\n" + "="*70)
    logger.info(f"SUMMARY: {len(bets_df)} BETTING RECOMMENDATIONS FOR DECEMBER 2025")
    logger.info("="*70)
    
    # Thresholds used
    logger.info(f"\nThresholds used:")
    logger.info(f"  Home: {thresholds['home']:.2f}")
    logger.info(f"  Draw: {thresholds['draw']:.2f}")
    logger.info(f"  Away: {thresholds['away']:.2f}")


if __name__ == '__main__':
    main()
