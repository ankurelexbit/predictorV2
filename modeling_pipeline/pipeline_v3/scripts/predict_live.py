#!/usr/bin/env python3
"""
Live Prediction System for V3 Model
====================================

Generates predictions for upcoming matches with betting recommendations
based on optimized thresholds.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_config():
    """Load trained model and optimal thresholds."""
    import joblib
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load optimal thresholds
    config_path = Path(__file__).parent.parent / 'models' / 'optimal_thresholds.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded thresholds: Home={config['thresholds']['home']}, "
               f"Draw={config['thresholds']['draw']}, Away={config['thresholds']['away']}")
    
    return model, config


def generate_predictions(model, matches_df, config):
    """
    Generate predictions with betting recommendations.
    
    Args:
        model: Trained XGBoost model
        matches_df: DataFrame with match features
        config: Configuration with thresholds and odds
    
    Returns:
        DataFrame with predictions and recommendations
    """
    thresholds = config['thresholds']
    typical_odds = config['typical_odds']
    
    # Generate probabilities
    probs = model.predict_proba(matches_df)
    
    predictions = []
    
    for i in range(len(matches_df)):
        # Get probabilities
        away_prob, draw_prob, home_prob = probs[i]
        
        # Determine prediction
        pred_idx = np.argmax(probs[i])
        pred_outcome = ['Away', 'Draw', 'Home'][pred_idx]
        confidence = probs[i][pred_idx]
        
        # Check if meets threshold
        threshold_map = {
            0: thresholds['away'],
            1: thresholds['draw'],
            2: thresholds['home']
        }
        
        odds_map = {
            0: typical_odds['away'],
            1: typical_odds['draw'],
            2: typical_odds['home']
        }
        
        meets_threshold = confidence >= threshold_map[pred_idx]
        
        # Betting recommendation
        if meets_threshold:
            bet_recommendation = f"BET {pred_outcome}"
            expected_roi = config['validation_performance']['roi']
            bet_odds = odds_map[pred_idx]
        else:
            bet_recommendation = "SKIP"
            expected_roi = 0
            bet_odds = 0
        
        predictions.append({
            'home_team': matches_df.iloc[i].get('home_team', 'Unknown'),
            'away_team': matches_df.iloc[i].get('away_team', 'Unknown'),
            'match_date': matches_df.iloc[i].get('match_date', 'Unknown'),
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob,
            'prediction': pred_outcome,
            'confidence': confidence,
            'threshold': threshold_map[pred_idx],
            'meets_threshold': meets_threshold,
            'bet_recommendation': bet_recommendation,
            'bet_odds': bet_odds,
            'expected_roi': expected_roi if meets_threshold else 0
        })
    
    return pd.DataFrame(predictions)


def main():
    """Main execution for live predictions."""
    logger.info("="*70)
    logger.info("V3 LIVE PREDICTION SYSTEM")
    logger.info("="*70)
    
    # Load model and config
    model, config = load_model_and_config()
    
    # For demonstration, use a sample from test data
    # In production, this would fetch from SportMonks API
    logger.info("\nLoading sample matches for demonstration...")
    
    data_path = Path(__file__).parent.parent / 'data' / 'csv' / 'training_data_complete_v2.csv'
    df = pd.read_csv(data_path)
    
    # Rename columns
    column_map = {
        'home_goals': 'home_score',
        'away_goals': 'away_score',
        'starting_at': 'match_date'
    }
    df.rename(columns=column_map, inplace=True)
    
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    
    # Get recent matches (last 10 from test set as example)
    test_mask = df['match_date'] >= '2025-01-01'
    sample_df = df[test_mask].tail(10).copy()
    
    # Drop leakage columns
    leakage_cols = ['target_home_win', 'target_draw', 'target_away_win', 
                    'result', 'home_score', 'away_score', 'target']
    for col in leakage_cols:
        if col in sample_df.columns:
            sample_df = sample_df.drop(columns=[col])
    
    logger.info(f"Generating predictions for {len(sample_df)} matches...")
    
    # Generate predictions
    predictions_df = generate_predictions(model, sample_df, config)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("PREDICTIONS")
    logger.info("="*70)
    
    for idx, row in predictions_df.iterrows():
        logger.info(f"\nMatch {idx+1}: {row['home_team']} vs {row['away_team']}")
        logger.info(f"  Date: {row['match_date']}")
        logger.info(f"  Probabilities: H={row['home_prob']:.1%}, D={row['draw_prob']:.1%}, A={row['away_prob']:.1%}")
        logger.info(f"  Prediction: {row['prediction']} ({row['confidence']:.1%} confidence)")
        logger.info(f"  Threshold: {row['threshold']:.1%}")
        
        if row['meets_threshold']:
            logger.info(f"  ✅ {row['bet_recommendation']} @ {row['bet_odds']:.2f}")
            logger.info(f"  Expected ROI: {row['expected_roi']:.1f}%")
        else:
            logger.info(f"  ❌ {row['bet_recommendation']} (confidence below threshold)")
    
    # Summary
    bets_to_place = predictions_df[predictions_df['meets_threshold']]
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Total matches analyzed: {len(predictions_df)}")
    logger.info(f"Bets recommended: {len(bets_to_place)} ({len(bets_to_place)/len(predictions_df)*100:.1f}%)")
    
    if len(bets_to_place) > 0:
        logger.info(f"\nRecommended Bets:")
        for outcome in ['Home', 'Draw', 'Away']:
            count = len(bets_to_place[bets_to_place['prediction'] == outcome])
            if count > 0:
                logger.info(f"  {outcome}: {count} bets")
    
    # Save predictions
    output_dir = Path(__file__).parent.parent / 'predictions'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'live_predictions_{timestamp}.csv'
    predictions_df.to_csv(output_file, index=False)
    
    logger.info(f"\n✅ Predictions saved to: {output_file}")


if __name__ == '__main__':
    main()
