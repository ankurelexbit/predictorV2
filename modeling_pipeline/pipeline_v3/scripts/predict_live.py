#!/usr/bin/env python3
"""
V3 Live Predictions - Using FeaturePipeline (CORRECT APPROACH)
===============================================================

Uses the SAME FeaturePipeline that training uses, but with fresh API data.
This ensures feature names match exactly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import joblib
import requests
from datetime import datetime
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V3LivePredictor:
    """Live predictions using FeaturePipeline with fresh API data."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.sportmonks.com/v3/football'
        self.cache = {}
        
        # Use the SAME feature pipeline as training
        self.feature_pipeline = FeaturePipeline()
        
        logger.info("✅ Initialized with FeaturePipeline (same as training)")
    
    def _api_call(self, endpoint: str, params: dict = None) -> dict:
        """Make API call with caching."""
        cache_key = f"{endpoint}_{str(params)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if params is None:
            params = {}
        params['api_token'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.cache[cache_key] = data
            time.sleep(0.1)  # Rate limiting
            return data
        except Exception as e:
            logger.error(f"API call failed: {endpoint} - {e}")
            return {}
    
    def get_team_recent_matches(self, team_id: int, limit: int = 15) -> list:
        """
        Fetch team's recent matches from API using correct endpoint.
        
        Returns matches in the format expected by FeaturePipeline:
        [
            {
                'fixture_id': int,
                'match_date': datetime,
                'is_home': bool,
                'opponent_id': int,
                'goals_scored': int,
                'goals_conceded': int,
                'result': 'W'/'D'/'L',
                'team_stats': {...},
                'opponent_stats': {...}
            },
            ...
        ]
        """
        # Use correct endpoint: /fixtures/between/{start}/{end}/{teamID}
        # Get COMPLETED matches from last 180 days (before today)
        from datetime import timedelta
        end_date = datetime.now() - timedelta(days=1)  # Yesterday (to avoid today's matches)
        start_date = end_date - timedelta(days=180)
        
        endpoint = f"fixtures/between/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}/{team_id}"
        params = {
            'include': 'participants;scores;statistics',
            'per_page': 50
        }
        
        logger.debug(f"  Fetching fixtures from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        data = self._api_call(endpoint, params)
        fixtures = data.get('data', [])
        
        matches = []
        for fixture in fixtures:
            try:
                participants = fixture.get('participants', [])
                if len(participants) < 2:
                    continue
                
                # Find home and away
                home = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                away = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
                
                if not home or not away:
                    continue
                
                is_home = home.get('id') == team_id
                
                # Get scores from scores array (not from meta)
                scores = fixture.get('scores', [])
                team_score = None
                opp_score = None
                
                for score in scores:
                    if score.get('description') == 'CURRENT':
                        participant_id = score.get('participant_id')
                        goals = score.get('score', {}).get('goals')
                        
                        if participant_id == home.get('id'):
                            home_score = goals
                        elif participant_id == away.get('id'):
                            away_score = goals
                
                # Assign based on perspective
                if is_home:
                    team_score = home_score if 'home_score' in locals() else None
                    opp_score = away_score if 'away_score' in locals() else None
                else:
                    team_score = away_score if 'away_score' in locals() else None
                    opp_score = home_score if 'home_score' in locals() else None
                
                if team_score is None or opp_score is None:
                    continue
                
                # Determine result
                if team_score > opp_score:
                    result = 'W'
                elif team_score == opp_score:
                    result = 'D'
                else:
                    result = 'L'
                
                # Parse statistics
                stats = fixture.get('statistics', [])
                team_stats = self._parse_statistics(stats, team_id)
                opp_id = away.get('id') if is_home else home.get('id')
                opp_stats = self._parse_statistics(stats, opp_id)
                
                matches.append({
                    'fixture_id': fixture.get('id'),
                    'match_date': datetime.fromisoformat(fixture.get('starting_at', '').replace('Z', '+00:00')),
                    'is_home': is_home,
                    'opponent_id': opp_id,
                    'goals_scored': team_score,
                    'goals_conceded': opp_score,
                    'result': result,
                    'team_stats': team_stats,
                    'opponent_stats': opp_stats
                })
            except Exception as e:
                logger.debug(f"Error parsing fixture: {e}")
                continue
        
        # Sort by date (oldest first for Elo calculation)
        matches.sort(key=lambda x: x['match_date'])
        
        logger.info(f"  Parsed {len(matches)} valid matches for team {team_id}")
        
        return matches
    
    def _parse_statistics(self, stats_list: list, team_id: int) -> dict:
        """Parse statistics for a team."""
        team_stats = {}
        
        for stat_group in stats_list:
            if stat_group.get('participant_id') == team_id:
                details = stat_group.get('details', [])
                
                for detail in details:
                    type_name = detail.get('type', {}).get('name', '')
                    value = detail.get('value', {}).get('total', 0)
                    
                    # Map to expected keys
                    if 'Shots Total' in type_name:
                        team_stats['shots_total'] = value
                    elif 'Shots On Target' in type_name:
                        team_stats['shots_on_target'] = value
                    elif 'Shots Insidebox' in type_name:
                        team_stats['shots_insidebox'] = value
                    elif 'Dangerous Attacks' in type_name:
                        team_stats['dangerous_attacks'] = value
                    elif 'Attacks' in type_name and 'Dangerous' not in type_name:
                        team_stats['attacks'] = value
                    elif 'Possession' in type_name:
                        team_stats['possession'] = value
                    elif 'Corners' in type_name:
                        team_stats['corners'] = value
        
        return team_stats
    
    def get_h2h_matches(self, team1_id: int, team2_id: int) -> list:
        """Get H2H matches between two teams."""
        endpoint = f"fixtures/head-to-head/{team1_id}/{team2_id}"
        params = {
            'per_page': 10,
            'include': 'scores'
        }
        
        data = self._api_call(endpoint, params)
        fixtures = data.get('data', [])
        
        h2h = []
        for fixture in fixtures:
            try:
                participants = fixture.get('participants', [])
                if len(participants) < 2:
                    continue
                
                home = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                away = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
                
                if not home or not away:
                    continue
                
                home_score = home.get('meta', {}).get('score')
                away_score = away.get('meta', {}).get('score')
                
                if home_score is None or away_score is None:
                    continue
                
                h2h.append({
                    'home_team_id': home.get('id'),
                    'away_team_id': away.get('id'),
                    'home_goals': home_score,
                    'away_goals': away_score
                })
            except Exception as e:
                logger.debug(f"Error parsing H2H fixture: {e}")
                continue
        
        return h2h
    
    def predict_match(self, home_team_id: int, away_team_id: int, fixture_id: int) -> dict:
        """
        Generate features and make prediction for a match.
        
        Uses the SAME FeaturePipeline as training.
        """
        logger.info(f"Predicting: {home_team_id} vs {away_team_id}")
        
        # Fetch recent matches from API
        home_matches = self.get_team_recent_matches(home_team_id, limit=15)
        away_matches = self.get_team_recent_matches(away_team_id, limit=15)
        h2h_matches = self.get_h2h_matches(home_team_id, away_team_id)
        
        if len(home_matches) < 3 or len(away_matches) < 3:
            logger.warning("  ⚠️ Insufficient match data")
            return None
        
        logger.info(f"  📊 Home: {len(home_matches)} matches, Away: {len(away_matches)} matches, H2H: {len(h2h_matches)}")
        
        # Generate features using FeaturePipeline (SAME as training)
        try:
            features = self.feature_pipeline.calculate_features_for_match(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                match_date=datetime.now(),
                home_matches=home_matches,
                away_matches=away_matches,
                h2h_matches=h2h_matches
            )
            
            logger.info(f"  ✅ Generated {len(features)} features using FeaturePipeline")
            
            return features
            
        except Exception as e:
            logger.error(f"  ❌ Error generating features: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Run live predictions for January 2026."""
    logger.info("="*70)
    logger.info("V3 LIVE PREDICTIONS - USING FEATUREPIPELINE")
    logger.info("="*70)
    
    # Load fixtures with fresh odds
    fixtures_path = Path(__file__).parent.parent / 'predictions' / 'january_2026_upcoming_fixtures.csv'
    jan_fixtures = pd.read_csv(fixtures_path)
    
    logger.info(f"\n📊 {len(jan_fixtures)} fixtures with FRESH API odds")
    
    # Initialize predictor
    import os
    api_key = os.getenv('SPORTMONKS_API_KEY', 'PeZeQDLtEN57cNh6q0e97drjLgCygFYV8BQRSuyg91fa8krbpmlX658H73r8')
    predictor = V3LivePredictor(api_key)
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_roi_optimized.joblib'
    model = joblib.load(model_path)
    logger.info("✅ Loaded V3 model")
    
    # Generate predictions
    logger.info("\n🔮 Generating predictions...")
    
    predictions = []
    
    for idx, fixture in jan_fixtures.head(3).iterrows():  # Test on first 3
        try:
            logger.info(f"\n[{idx+1}] {fixture['home_team_name']} vs {fixture['away_team_name']}")
            
            # Generate features using FeaturePipeline
            features = predictor.predict_match(
                home_team_id=fixture['home_team_id'],
                away_team_id=fixture['away_team_id'],
                fixture_id=fixture['fixture_id']
            )
            
            if not features:
                continue
            
            # Add required metadata
            features['fixture_id'] = fixture['fixture_id']
            features['home_team_id'] = fixture['home_team_id']
            features['away_team_id'] = fixture['away_team_id']
            features['starting_at'] = fixture['starting_at']
            features['league_id'] = fixture['league_id']
            
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
    
    logger.info("\n✅ SUCCESS - Features generated using FeaturePipeline (same as training)!")


if __name__ == '__main__':
    main()
