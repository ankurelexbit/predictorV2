#!/usr/bin/env python3
"""
Live API Feature Generator
===========================

Generates features fresh from SportMonks API for live predictions.
NO CSV data, NO training data - everything fetched from API.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveAPIFeatureGenerator:
    """Generate features from SportMonks API for live predictions."""
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.base_url = 'https://api.sportmonks.com/v3/football'
        self.cache = {}  # Cache API responses
        
        # Elo constants
        self.K_FACTOR = 32
        self.INITIAL_ELO = 1500
        self.HOME_ADVANTAGE = 35
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with caching."""
        cache_key = f"{endpoint}_{str(params)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if params is None:
            params = {}
        
        params['api_token'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            self.cache[cache_key] = data
            time.sleep(0.1)  # Rate limiting
            
            return data
        except Exception as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            return {}
    
    def fetch_team_recent_matches(
        self,
        team_id: int,
        before_date: str,
        n: int = 10
    ) -> List[Dict]:
        """
        Fetch team's recent matches before a date.
        
        Args:
            team_id: Team ID
            before_date: Date string (YYYY-MM-DD)
            n: Number of matches
        
        Returns:
            List of match dictionaries
        """
        logger.info(f"Fetching last {n} matches for team {team_id} before {before_date}")
        
        # Convert date
        target_date = datetime.strptime(before_date[:10], '%Y-%m-%d')
        
        # Fetch team fixtures using team-specific endpoint
        endpoint = f"teams/{team_id}/schedules"
        params = {
            'include': 'participants',
            'per_page': 50
        }
        
        data = self._make_request(endpoint, params)
        fixtures = data.get('data', [])
        
        # Filter and sort
        team_matches = []
        for fixture in fixtures:
            try:
                fixture_date = datetime.strptime(fixture['starting_at'][:10], '%Y-%m-%d')
                if fixture_date < target_date:
                    team_matches.append(fixture)
            except:
                continue
        
        # Sort by date descending
        team_matches.sort(key=lambda x: x['starting_at'], reverse=True)
        
        return team_matches[:n]
    
    def fetch_h2h_matches(
        self,
        team1_id: int,
        team2_id: int,
        before_date: str,
        n: int = 5
    ) -> List[Dict]:
        """Fetch head-to-head matches."""
        logger.info(f"Fetching H2H for {team1_id} vs {team2_id}")
        
        target_date = datetime.strptime(before_date[:10], '%Y-%m-%d')
        
        endpoint = f"fixtures/head-to-head/{team1_id}/{team2_id}"
        params = {
            'per_page': 20
        }
        
        data = self._make_request(endpoint, params)
        fixtures = data.get('data', [])
        
        # Filter by date
        h2h_matches = []
        for fixture in fixtures:
            fixture_date = datetime.strptime(fixture['starting_at'][:10], '%Y-%m-%d')
            if fixture_date < target_date:
                h2h_matches.append(fixture)
        
        h2h_matches.sort(key=lambda x: x['starting_at'], reverse=True)
        
        return h2h_matches[:n]
    
    def calculate_form_features(self, matches: List[Dict], team_id: int) -> Dict:
        """Calculate form features from recent matches."""
        if not matches:
            return {
                'form_points_l5': 0,
                'form_goals_scored_l5': 0,
                'form_goals_conceded_l5': 0,
                'form_wins_l5': 0,
                'form_draws_l5': 0,
                'form_losses_l5': 0
            }
        
        points = 0
        goals_scored = 0
        goals_conceded = 0
        wins = draws = losses = 0
        
        for match in matches[:5]:
            participants = match.get('participants', [])
            home = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), {})
            away = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), {})
            
            is_home = home.get('id') == team_id
            team_score = home.get('meta', {}).get('score') if is_home else away.get('meta', {}).get('score')
            opp_score = away.get('meta', {}).get('score') if is_home else home.get('meta', {}).get('score')
            
            if team_score is None or opp_score is None:
                continue
            
            goals_scored += team_score
            goals_conceded += opp_score
            
            if team_score > opp_score:
                points += 3
                wins += 1
            elif team_score == opp_score:
                points += 1
                draws += 1
            else:
                losses += 1
        
        return {
            'form_points_l5': points,
            'form_goals_scored_l5': goals_scored,
            'form_goals_conceded_l5': goals_conceded,
            'form_wins_l5': wins,
            'form_draws_l5': draws,
            'form_losses_l5': losses
        }
    
    def calculate_elo_rating(self, matches: List[Dict], team_id: int) -> float:
        """Calculate current Elo rating from match history."""
        elo = self.INITIAL_ELO
        
        for match in reversed(matches):  # Process oldest first
            participants = match.get('participants', [])
            home = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), {})
            away = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), {})
            
            is_home = home.get('id') == team_id
            team_score = home.get('meta', {}).get('score') if is_home else away.get('meta', {}).get('score')
            opp_score = away.get('meta', {}).get('score') if is_home else home.get('meta', {}).get('score')
            
            if team_score is None or opp_score is None:
                continue
            
            # Actual result
            if team_score > opp_score:
                actual = 1.0
            elif team_score == opp_score:
                actual = 0.5
            else:
                actual = 0.0
            
            # Expected (simplified - assume opponent is average)
            expected = 0.5
            
            # Update Elo
            elo += self.K_FACTOR * (actual - expected)
        
        return elo
    
    def calculate_attack_defense_metrics(self, matches: List[Dict], team_id: int) -> Dict:
        """Calculate attack and defense strength."""
        if not matches:
            return {
                'goals_scored_avg': 0,
                'goals_conceded_avg': 0,
                'shots_avg': 0,
                'shots_on_target_avg': 0
            }
        
        total_scored = 0
        total_conceded = 0
        total_shots = 0
        total_sot = 0
        count = 0
        
        for match in matches[:10]:
            participants = match.get('participants', [])
            home = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), {})
            away = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), {})
            
            is_home = home.get('id') == team_id
            team_score = home.get('meta', {}).get('score') if is_home else away.get('meta', {}).get('score')
            opp_score = away.get('meta', {}).get('score') if is_home else home.get('meta', {}).get('score')
            
            if team_score is not None and opp_score is not None:
                total_scored += team_score
                total_conceded += opp_score
                count += 1
            
            # Get statistics if available
            stats = match.get('statistics', [])
            for stat in stats:
                if stat.get('participant_id') == team_id:
                    total_shots += stat.get('shots_total', 0) or 0
                    total_sot += stat.get('shots_on_target', 0) or 0
        
        return {
            'goals_scored_avg': total_scored / count if count > 0 else 0,
            'goals_conceded_avg': total_conceded / count if count > 0 else 0,
            'shots_avg': total_shots / count if count > 0 else 0,
            'shots_on_target_avg': total_sot / count if count > 0 else 0
        }
    
    def generate_features_for_fixture(
        self,
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
        fixture_date: str,
        league_id: int
    ) -> Dict:
        """
        Generate all features for a fixture from API data.
        
        Args:
            fixture_id: Fixture ID
            home_team_id: Home team ID
            away_team_id: Away team ID
            fixture_date: Fixture date (YYYY-MM-DD)
            league_id: League ID
        
        Returns:
            Dictionary with all features
        """
        logger.info(f"Generating features for fixture {fixture_id}")
        
        # Fetch data from API
        home_matches = self.fetch_team_recent_matches(home_team_id, fixture_date, n=10)
        away_matches = self.fetch_team_recent_matches(away_team_id, fixture_date, n=10)
        h2h_matches = self.fetch_h2h_matches(home_team_id, away_team_id, fixture_date, n=5)
        
        # Calculate features
        home_form = self.calculate_form_features(home_matches, home_team_id)
        away_form = self.calculate_form_features(away_matches, away_team_id)
        
        home_elo = self.calculate_elo_rating(home_matches, home_team_id)
        away_elo = self.calculate_elo_rating(away_matches, away_team_id)
        
        home_attack_defense = self.calculate_attack_defense_metrics(home_matches, home_team_id)
        away_attack_defense = self.calculate_attack_defense_metrics(away_matches, away_team_id)
        
        # Build feature dictionary
        features = {
            'fixture_id': fixture_id,
            'league_id': league_id,
            
            # Elo features
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            
            # Form features
            'home_form_points_l5': home_form['form_points_l5'],
            'away_form_points_l5': away_form['form_points_l5'],
            'home_form_goals_scored_l5': home_form['form_goals_scored_l5'],
            'away_form_goals_scored_l5': away_form['form_goals_scored_l5'],
            'home_form_goals_conceded_l5': home_form['form_goals_conceded_l5'],
            'away_form_goals_conceded_l5': away_form['form_goals_conceded_l5'],
            
            # Attack/Defense
            'home_goals_scored_avg': home_attack_defense['goals_scored_avg'],
            'away_goals_scored_avg': away_attack_defense['goals_scored_avg'],
            'home_goals_conceded_avg': home_attack_defense['goals_conceded_avg'],
            'away_goals_conceded_avg': away_attack_defense['goals_conceded_avg'],
            
            # H2H
            'h2h_matches': len(h2h_matches),
        }
        
        # Add remaining features with defaults (simplified for now)
        # In production, calculate all 181 features
        for i in range(181 - len(features)):
            features[f'feature_{i}'] = 0
        
        return features


if __name__ == '__main__':
    # Test
    import os
    api_key = os.getenv('SPORTMONKS_API_KEY')
    
    generator = LiveAPIFeatureGenerator(api_key)
    
    # Test on a sample fixture
    features = generator.generate_features_for_fixture(
        fixture_id=19432035,
        home_team_id=2,  # Blackburn
        away_team_id=283,  # Wrexham
        fixture_date='2026-01-01',
        league_id=9
    )
    
    print(f"Generated {len(features)} features")
    print(f"Sample: {list(features.items())[:10]}")
