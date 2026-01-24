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
from db_predictions import PredictionsDB

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
            
            # Build features with live API calls (now returns features and metadata)
            result = calculator.build_features_for_match(
                home_team_id=int(fixture['home_team_id']),
                away_team_id=int(fixture['away_team_id']),
                fixture_date=pd.to_datetime(fixture['date']),
                home_team_name=fixture.get('home_team_name'),
                away_team_name=fixture.get('away_team_name'),
                league_name=fixture.get('league_name'),
                fixture_id=fixture.get('fixture_id')
            )
            
            if not result:
                logger.warning("  ⚠️  Could not generate features")
                errors += 1
                continue
            
            # Unpack features and metadata
            features, metadata = result
            
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
            
            # Extract odds from features or metadata (needed for all predictions)
            odds_home = metadata.get('odds_home') or features.get('odds_home')
            odds_draw = metadata.get('odds_draw') or features.get('odds_draw')
            odds_away = metadata.get('odds_away') or features.get('odds_away')
            
            # Calculate our fair odds
            from prediction_metadata import calculate_fair_odds
            our_odds = calculate_fair_odds(p_home, p_draw, p_away)
            
            # Store ALL predictions (including NO_BET)
            recommendations.append({
                'fixture_id': fixture.get('fixture_id'),
                'date': str(fixture['date']),
                'league': fixture.get('league_name'),
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'bet_on': best_bet if best_bet else 'no_bet',
                'confidence': float(best_prob) if best_prob > 0 else None,
                'p_home': float(p_home),
                'p_draw': float(p_draw),
                'p_away': float(p_away),
                'odds_home': float(odds_home) if odds_home else None,
                'odds_draw': float(odds_draw) if odds_draw else None,
                'odds_away': float(odds_away) if odds_away else None,
                'features': features,  # Store all features
                'metadata': metadata,  # Store comprehensive metadata
                'our_odds': our_odds,  # Our calculated fair odds
                'prediction_time': datetime.now().isoformat()
            })
            
            if best_bet:
                logger.info(f"  💰 RECOMMENDATION: Bet {best_bet.upper()} @ {best_prob*100:.1f}% confidence")
            else:
                logger.info(f"  ⏭️  No bet (no threshold exceeded)")
        
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            errors += 1
            continue
    
    # 4. Save predictions (all predictions, including NO_BET)
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Fixtures processed: {len(all_fixtures)}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total predictions: {len(recommendations)}")
    
    # Count actual betting recommendations (excluding NO_BET)
    betting_recommendations = [r for r in recommendations if r.get('bet_on') != 'no_bet']
    logger.info(f"Betting recommendations: {len(betting_recommendations)}")
    logger.info(f"No-bet predictions: {len(recommendations) - len(betting_recommendations)}")
    
    if recommendations:
        # Save to JSON
        output_dir = Path('data/predictions')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info(f"\n💾 Recommendations saved to: {output_file}")
        
        # Save to Supabase database
        logger.info("\n💾 Saving predictions to Supabase...")
        try:
            with PredictionsDB() as db:
                saved_count = 0
                for rec in recommendations:
                    # Extract metadata
                    metadata = rec.get('metadata', {})
                    our_odds = rec.get('our_odds', {})
                    
                    # Extract odds
                    odds_home = rec.get('odds_home')
                    odds_draw = rec.get('odds_draw')
                    odds_away = rec.get('odds_away')
                    features_dict = rec.get('features', {})
                    
                    # Calculate best odds
                    best_odds = None
                    if odds_home and odds_draw and odds_away:
                        bet_type = rec.get('bet_on', '').upper()
                        if bet_type == 'HOME':
                            best_odds = odds_home
                        elif bet_type == 'DRAW':
                            best_odds = odds_draw
                        elif bet_type == 'AWAY':
                            best_odds = odds_away
                    
                    # Prepare prediction data for database with comprehensive metadata
                    prediction_data = {
                        # Basic prediction info
                        'fixture_id': rec.get('fixture_id'),
                        'match_date': rec.get('date'),
                        'home_team': rec.get('home_team'),
                        'away_team': rec.get('away_team'),
                        'league': rec.get('league'),
                        'league_id': None,
                        
                        # Probabilities (convert numpy types to Python float)
                        'prob_home': float(rec.get('p_home')),
                        'prob_draw': float(rec.get('p_draw')),
                        'prob_away': float(rec.get('p_away')),
                        
                        # Recommendation
                        'recommended_bet': rec.get('bet_on', 'NO_BET').upper(),
                        'confidence': float(rec.get('confidence')) if rec.get('confidence') else None,
                        
                        # Market odds
                        'odds_home': float(odds_home) if odds_home else None,
                        'odds_draw': float(odds_draw) if odds_draw else None,
                        'odds_away': float(odds_away) if odds_away else None,
                        'best_odds': float(best_odds) if best_odds else None,
                        
                        # Features
                        'features_count': len(features_dict) if features_dict else 271,
                        'model_version': 'xgboost_draw_tuned',
                        'thresholds': json.dumps(thresholds),
                        'features': json.dumps(features_dict, default=float) if features_dict else None,
                        
                        # NEW: Timing data
                        'kickoff_time': metadata.get('kickoff_time'),
                        'prediction_time': metadata.get('prediction_time'),
                        'hours_before_kickoff': float(metadata.get('hours_before_kickoff')) if metadata.get('hours_before_kickoff') else None,
                        
                        # NEW: Lineup data
                        'home_lineup': json.dumps(metadata.get('home_lineup'), default=float) if metadata.get('home_lineup') else None,
                        'away_lineup': json.dumps(metadata.get('away_lineup'), default=float) if metadata.get('away_lineup') else None,
                        'lineup_available': metadata.get('lineup_available', False),
                        'lineup_coverage_home': float(metadata.get('lineup_coverage_home', 0)),
                        'lineup_coverage_away': float(metadata.get('lineup_coverage_away', 0)),
                        
                        # NEW: Injury data
                        'home_injuries_count': int(metadata.get('home_injuries_count', 0)),
                        'away_injuries_count': int(metadata.get('away_injuries_count', 0)),
                        'home_injured_players': json.dumps(metadata.get('home_injured_players')) if metadata.get('home_injured_players') else None,
                        'away_injured_players': json.dumps(metadata.get('away_injured_players')) if metadata.get('away_injured_players') else None,
                        
                        # NEW: Bookmaker odds
                        'bookmaker_odds': json.dumps(metadata.get('bookmaker_odds'), default=float) if metadata.get('bookmaker_odds') else None,
                        'best_odds_home': float(metadata.get('best_odds_home') or odds_home) if (metadata.get('best_odds_home') or odds_home) else None,
                        'best_odds_draw': float(metadata.get('best_odds_draw') or odds_draw) if (metadata.get('best_odds_draw') or odds_draw) else None,
                        'best_odds_away': float(metadata.get('best_odds_away') or odds_away) if (metadata.get('best_odds_away') or odds_away) else None,
                        
                        # NEW: Our calculated fair odds
                        'our_odds_home': float(our_odds.get('our_odds_home')) if our_odds.get('our_odds_home') else None,
                        'our_odds_draw': float(our_odds.get('our_odds_draw')) if our_odds.get('our_odds_draw') else None,
                        'our_odds_away': float(our_odds.get('our_odds_away')) if our_odds.get('our_odds_away') else None,
                        
                        # NEW: Data quality flags
                        'used_lineup_data': metadata.get('used_lineup_data', False),
                        'used_injury_data': metadata.get('used_injury_data', False),
                        'data_quality_score': float(metadata.get('data_quality_score', 0))
                    }
                    
                    prediction_id = db.insert_prediction(prediction_data)
                    if prediction_id:
                        saved_count += 1
                
                logger.info(f"✅ Saved {saved_count}/{len(recommendations)} predictions to database")
        except Exception as e:
            logger.error(f"❌ Database save failed: {e}")
            logger.info("Predictions still saved to JSON file")
        
        # Print only actual betting recommendations (not NO_BET)
        if betting_recommendations:
            logger.info("\n📋 BETTING RECOMMENDATIONS:")
            for rec in betting_recommendations:
                logger.info(f"\n  {rec['home_team']} vs {rec['away_team']}")
                logger.info(f"  League: {rec['league']}")
                logger.info(f"  Date: {rec['date']}")
                logger.info(f"  → Bet: {rec['bet_on'].upper()} @ {rec['confidence']*100:.1f}%")
        else:
            logger.info("\n⏭️  No betting opportunities found (all predictions below threshold)")
    else:
        logger.info("\n⏭️  No predictions generated")
    
    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()
