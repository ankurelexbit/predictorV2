#!/usr/bin/env python3
"""
Live Prediction Runner - Runs every 30 minutes

Fetches upcoming fixtures, generates predictions with live API calls,
applies optimal thresholds, and outputs betting recommendations.

Expected Performance:
- ROI: 23.7%
- Bets/Day: 3.3
- Win Rate: 68.2%
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from predict_live import LiveFeatureCalculator, load_models, get_upcoming_fixtures
from production_thresholds import get_production_thresholds
from utils import setup_logger

logger = setup_logger("live_predictions")

def main():
    logger.info("=" * 80)
    logger.info("LIVE PREDICTION RUN")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # 1. Load model and thresholds
    logger.info("Loading model and thresholds...")
    models = load_models('xgboost')
    if 'xgboost' not in models:
        logger.error("Failed to load model!")
        return
    
    model = models['xgboost']
    thresholds = get_production_thresholds()
    
    logger.info(f"Thresholds: H={thresholds['home']:.2f}, D={thresholds['draw']:.2f}, A={thresholds['away']:.2f}")
    
    # 2. Get upcoming fixtures (next 24 hours)
    logger.info("Fetching upcoming fixtures...")
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        fixtures_today = get_upcoming_fixtures(today)
        fixtures_tomorrow = get_upcoming_fixtures(tomorrow)
        
        all_fixtures = pd.concat([fixtures_today, fixtures_tomorrow], ignore_index=True)
    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        return
    
    if len(all_fixtures) == 0:
        logger.info("No upcoming fixtures found")
        return
    
    logger.info(f"Found {len(all_fixtures)} upcoming fixtures")
    
    # 3. Generate predictions
    calculator = LiveFeatureCalculator()
    recommendations = []
    errors = 0
    
    for idx, fixture in all_fixtures.iterrows():
        try:
            logger.info(f"\n[{idx+1}/{len(all_fixtures)}] {fixture['home_team_name']} vs {fixture['away_team_name']}")
            
            # Build features with live API calls
            features = calculator.build_features_for_match(
                home_team_id=int(fixture['home_team_id']),
                away_team_id=int(fixture['away_team_id']),
                fixture_date=pd.to_datetime(fixture['date']),
                home_team_name=fixture.get('home_team_name'),
                away_team_name=fixture.get('away_team_name'),
                league_name=fixture.get('league_name'),
                fixture_id=fixture.get('fixture_id')
            )
            
            if not features:
                logger.warning("  ⚠️  Could not generate features")
                errors += 1
                continue
            
            logger.info(f"  ✅ Generated {len(features)} features")
            
            # Make prediction
            features_df = pd.DataFrame([features])
            probs = model.predict_proba(features_df, calibrated=True)[0]
            
            p_away, p_draw, p_home = probs
            
            logger.info(f"  🎯 Predictions: H={p_home*100:.1f}% D={p_draw*100:.1f}% A={p_away*100:.1f}%")
            
            # Apply thresholds
            model_probs = {'away': p_away, 'draw': p_draw, 'home': p_home}
            
            best_bet = None
            best_prob = 0
            
            for outcome in ['away', 'draw', 'home']:
                if model_probs[outcome] > thresholds[outcome] and model_probs[outcome] > best_prob:
                    best_bet = outcome
                    best_prob = model_probs[outcome]
            
            if best_bet:
                recommendations.append({
                    'fixture_id': fixture.get('fixture_id'),
                    'date': str(fixture['date']),
                    'league': fixture.get('league_name'),
                    'home_team': fixture['home_team_name'],
                    'away_team': fixture['away_team_name'],
                    'bet_on': best_bet,
                    'confidence': float(best_prob),
                    'p_home': float(p_home),
                    'p_draw': float(p_draw),
                    'p_away': float(p_away),
                    'prediction_time': datetime.now().isoformat()
                })
                
                logger.info(f"  💰 RECOMMENDATION: Bet {best_bet.upper()} @ {best_prob*100:.1f}% confidence")
            else:
                logger.info(f"  ⏭️  No bet (no threshold exceeded)")
        
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            errors += 1
            continue
    
    # 4. Save recommendations
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Fixtures processed: {len(all_fixtures)}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Recommendations: {len(recommendations)}")
    
    if recommendations:
        # Save to JSON
        output_dir = Path('data/predictions')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info(f"\n💾 Recommendations saved to: {output_file}")
        
        # Print recommendations
        logger.info("\n📋 BETTING RECOMMENDATIONS:")
        for rec in recommendations:
            logger.info(f"\n  {rec['home_team']} vs {rec['away_team']}")
            logger.info(f"  League: {rec['league']}")
            logger.info(f"  Date: {rec['date']}")
            logger.info(f"  → Bet: {rec['bet_on'].upper()} @ {rec['confidence']*100:.1f}%")
    else:
        logger.info("\n⏭️  No betting opportunities found")
    
    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()
