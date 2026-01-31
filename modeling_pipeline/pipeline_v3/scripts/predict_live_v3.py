#!/usr/bin/env python3
"""
V3 Live Predictions - Using FeatureOrchestrator (CORRECT)
==========================================================

Uses the SAME FeatureOrchestrator that training uses.
This generates all 150-180 features from the 3 pillars.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.feature_orchestrator import FeatureOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run live predictions for January 2026."""
    logger.info("="*70)
    logger.info("V3 LIVE PREDICTIONS - USING FEATUREORCHESTRATOR")
    logger.info("="*70)
    
    # Load fixtures with fresh odds
    fixtures_path = Path(__file__).parent.parent / 'predictions' / 'january_2026_upcoming_fixtures.csv'
    jan_fixtures = pd.read_csv(fixtures_path)
    
    logger.info(f"\n📊 {len(jan_fixtures)} fixtures with FRESH API odds")
    
    # Initialize FeatureOrchestrator (SAME as training)
    logger.info("\n🔧 Initializing FeatureOrchestrator...")
    orchestrator = FeatureOrchestrator(data_dir='data/csv')
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    logger.info("✅ Loaded V3 model")
    
    # Generate predictions
    logger.info("\n🔮 Generating predictions...")
    
    predictions = []
    
    for idx, fixture in jan_fixtures.head(5).iterrows():  # Test on first 5
        try:
            logger.info(f"\n[{idx+1}] {fixture['home_team_name']} vs {fixture['away_team_name']}")
            
            # Generate features using FeatureOrchestrator (SAME as training)
            features = orchestrator.generate_features_for_fixture(
                fixture_id=fixture['fixture_id']
            )
            
            if not features:
                logger.warning("  ⚠️ Could not generate features")
                continue
            
            logger.info(f"  ✅ Generated {len(features)} features using FeatureOrchestrator")
            
            # Make prediction
            features_df = pd.DataFrame([features])
            probs = model.predict_proba(features_df)[0]
            
            logger.info(f"  🎯 Predictions: H={probs[2]:.1%} D={probs[1]:.1%} A={probs[0]:.1%}")
            
            # Apply thresholds
            thresholds = {'home': 0.48, 'draw': 0.35, 'away': 0.45}
            pred_outcome = np.argmax(probs)
            confidence = probs[pred_outcome]
            threshold_map = {0: thresholds['away'], 1: thresholds['draw'], 2: thresholds['home']}
            
            bet_recommended = confidence >= threshold_map[pred_outcome]
            
            if bet_recommended:
                outcome_name = ['Away', 'Draw', 'Home'][pred_outcome]
                logger.info(f"  💰 RECOMMENDATION: Bet {outcome_name} @ {confidence:.1%}")
            else:
                logger.info(f"  ⏭️ No bet (below threshold)")
            
            predictions.append({
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'prob_home': probs[2],
                'prob_draw': probs[1],
                'prob_away': probs[0],
                'predicted_outcome': ['Away', 'Draw', 'Home'][pred_outcome],
                'confidence': confidence,
                'bet_recommended': bet_recommended,
                'odds_home': fixture['odds_home'],
                'odds_draw': fixture['odds_draw'],
                'odds_away': fixture['odds_away']
            })
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Predictions generated: {len(predictions)}")
    bets = [p for p in predictions if p['bet_recommended']]
    logger.info(f"Betting recommendations: {len(bets)}")
    
    if bets:
        logger.info("\n📋 BETTING RECOMMENDATIONS:")
        for pred in bets:
            logger.info(f"\n  {pred['home_team']} vs {pred['away_team']}")
            logger.info(f"  → Bet: {pred['predicted_outcome']} @ {pred['confidence']:.1%}")
            logger.info(f"  → Odds: H={pred['odds_home']:.2f} D={pred['odds_draw']:.2f} A={pred['odds_away']:.2f}")
    
    logger.info("\n✅ SUCCESS - Features generated using FeatureOrchestrator (same as training)!")


if __name__ == '__main__':
    main()
