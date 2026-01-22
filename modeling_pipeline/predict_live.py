"""
Predict Live Matches - Using Real-Time API Data
================================================

This script fetches live data from Sportmonks API and calculates features
in real-time for upcoming matches.

Key differences from predict_upcoming.py:
- Fetches team's recent matches from API (not from stale CSV)
- Calculates features on-the-fly
- Uses current form, latest stats, recent injuries
- More accurate predictions with up-to-date data

Usage:
    python predict_live.py --date today
    python predict_live.py --date tomorrow
    python predict_live.py --date 2026-01-19
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib
import argparse
import requests
import urllib3
from collections import defaultdict
import importlib.util

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR
from fetch_standings import get_team_standings
from utils import setup_logger
from player_stats_manager import PlayerStatsManager
from odds_fetcher import OddsFetcher

# Import betting strategy (dynamic import to handle numeric prefix in filename)
spec = importlib.util.spec_from_file_location(
    "betting_strategy",
    Path(__file__).parent / "11_smart_betting_strategy.py"
)
betting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(betting_module)
SmartMultiOutcomeStrategy = betting_module.SmartMultiOutcomeStrategy

# Setup
logger = setup_logger("predict_live")

# Sportmonks API configuration
API_KEY = "LmEYRdsf8CSmiblNVTn6JfV0y0s8tc4aQsEkJ6JuoBhOWa3Hd1FEanGcrijo"
BASE_URL = "https://api.sportmonks.com/v3/football"

# Default Elo rating for new teams
DEFAULT_ELO = 1500
ELO_K_FACTOR = 32


class LiveFeatureCalculator:
    """Calculate features from live API data."""

    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
        # Load pre-computed Elo ratings from training data
        elo_file = Path(__file__).parent / 'data' / 'processed' / 'team_elo_ratings.json'
        if elo_file.exists():
            import json
            with open(elo_file, 'r') as f:
                self.elo_ratings = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"Loaded Elo ratings for {len(self.elo_ratings)} teams")
        else:
            logger.warning(f"Elo ratings file not found: {elo_file}")
            self.elo_ratings = {}

        # Load position and points lookup (actual data from training)
        position_points_file = Path(__file__).parent / 'data' / 'processed' / 'team_position_points.json'
        if position_points_file.exists():
            import json
            with open(position_points_file, 'r') as f:
                self.position_points = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"Loaded position/points data for {len(self.position_points)} teams")
        else:
            logger.warning(f"Position/points file not found: {position_points_file}")
            self.position_points = {}

        # Initialize player stats manager
        self.player_manager = PlayerStatsManager()
        if self.player_manager.is_loaded:
            logger.info("✅ Player database loaded - will use real lineup data when available")
        else:
            logger.warning("⚠️ Player database not loaded - will use approximations for player stats")
        
        # Initialize odds fetcher
        self.odds_fetcher = OddsFetcher()
        logger.info("✅ Odds fetcher initialized")

    def _api_call(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API call with error handling."""
        url = f"{BASE_URL}/{endpoint}"
        params = params or {}
        params['api_token'] = self.api_key

        try:
            response = requests.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    def get_team_recent_matches(self, team_id: int, limit: int = 20) -> List[Dict]:
        """
        Fetch team's recent matches from API.

        Args:
            team_id: Sportmonks team ID
            limit: Number of recent matches to fetch

        Returns:
            List of match dictionaries with statistics
        """
        logger.info(f"Fetching last {limit} matches for team {team_id}")

        # Use teams endpoint with latest matches included
        endpoint = f"teams/{team_id}"
        params = {
            'include': 'latest.statistics;latest.participants;latest.scores'
        }

        data = self._api_call(endpoint, params)
        if not data or 'data' not in data:
            return []

        # Extract latest matches from team data
        latest_fixtures = data['data'].get('latest', [])
        if not latest_fixtures:
            logger.warning(f"No latest matches found for team {team_id}")
            return []

        matches = []
        for fixture in latest_fixtures[:limit]:
            # Skip if not finished (state_id 5 = FT)
            if fixture.get('state_id') != 5:
                continue

            # Get participants
            participants = fixture.get('participants', [])

            # Find home and away teams
            home_participant = None
            away_participant = None

            for p in participants:
                if p.get('meta', {}).get('location') == 'home':
                    home_participant = p
                elif p.get('meta', {}).get('location') == 'away':
                    away_participant = p

            if not home_participant or not away_participant:
                # If participants not included, we need to parse from fixture meta
                # The fixture has a 'meta' field with location info
                fixture_meta = fixture.get('meta', {})
                location = fixture_meta.get('location', 'unknown')

                # Skip if we can't determine teams
                if not participants or len(participants) < 2:
                    continue

                # Use basic participant info
                if location == 'home':
                    home_team_id = team_id
                    away_team_id = participants[0].get('id', 0) if participants[0].get('id') != team_id else participants[1].get('id', 0)
                else:
                    away_team_id = team_id
                    home_team_id = participants[0].get('id', 0) if participants[0].get('id') != team_id else participants[1].get('id', 0)

                home_team_name = fixture.get('name', '').split(' vs ')[0] if ' vs ' in fixture.get('name', '') else 'Unknown'
                away_team_name = fixture.get('name', '').split(' vs ')[1] if ' vs ' in fixture.get('name', '') else 'Unknown'

                # Try to get scores from result_info or other fields
                result_info = fixture.get('result_info', '')
                home_score = 0
                away_score = 0
            else:
                home_team_id = home_participant.get('id')
                home_team_name = home_participant.get('name', 'Unknown')
                away_team_id = away_participant.get('id')
                away_team_name = away_participant.get('name', 'Unknown')

                # Extract scores from scores array (type_id 1525 = CURRENT/final score)
                scores = fixture.get('scores', [])
                home_score = 0
                away_score = 0

                for score in scores:
                    if score.get('type_id') == 1525:  # CURRENT score
                        score_data = score.get('score', {})
                        goals = score_data.get('goals', 0)
                        participant = score_data.get('participant', '')

                        if participant == 'home':
                            home_score = goals
                        elif participant == 'away':
                            away_score = goals

            # Extract statistics
            stats = fixture.get('statistics', [])
            match_data = {
                'fixture_id': fixture['id'],
                'date': fixture.get('starting_at'),
                'home_team_id': home_team_id,
                'home_team_name': home_team_name,
                'away_team_id': away_team_id,
                'away_team_name': away_team_name,
                'home_score': home_score,
                'away_score': away_score,
                'statistics': self._parse_statistics(stats)
            }

            matches.append(match_data)

        logger.info(f"Found {len(matches)} recent matches")
        return matches

    def search_fixture_by_teams(
        self,
        home_team_name: str,
        away_team_name: str,
        days_ahead: int = 7
    ) -> Optional[Dict]:
        """
        Search for a fixture by team names.

        Args:
            home_team_name: Home team name (e.g., "Man City", "Liverpool")
            away_team_name: Away team name
            days_ahead: Search within next N days (default: 7)

        Returns:
            Dictionary with fixture details including fixture_id, or None if not found
        """
        try:
            # Get upcoming fixtures
            today = datetime.now()
            end_date = today + timedelta(days=days_ahead)

            # Fetch fixtures in date range
            response = self._api_call(
                f"fixtures/between/{today.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
                params={'include': 'participants'}
            )

            if not response or 'data' not in response:
                logger.warning("No fixtures found in date range")
                return None

            # Normalize team names for matching
            home_normalized = home_team_name.lower().strip()
            away_normalized = away_team_name.lower().strip()

            # Search through fixtures
            for fixture in response['data']:
                participants = fixture.get('participants', [])
                if len(participants) < 2:
                    continue

                home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                away_team = next((p for p in participants if p.get('meta', {}).get('location') != 'home'), None)

                if not home_team or not away_team:
                    continue

                # Get team names and normalize
                fixture_home_name = home_team.get('name', '').lower().strip()
                fixture_away_name = away_team.get('name', '').lower().strip()

                # Check for match (exact or partial match)
                home_match = (
                    home_normalized in fixture_home_name or
                    fixture_home_name in home_normalized or
                    home_normalized.replace(' ', '') == fixture_home_name.replace(' ', '')
                )

                away_match = (
                    away_normalized in fixture_away_name or
                    fixture_away_name in away_normalized or
                    away_normalized.replace(' ', '') == fixture_away_name.replace(' ', '')
                )

                if home_match and away_match:
                    logger.info(f"✅ Found fixture: {home_team['name']} vs {away_team['name']}")
                    logger.info(f"   Fixture ID: {fixture['id']}")
                    logger.info(f"   Date: {fixture.get('starting_at')}")

                    return {
                        'fixture_id': fixture['id'],
                        'date': fixture.get('starting_at'),
                        'home_team_id': home_team['id'],
                        'home_team_name': home_team['name'],
                        'away_team_id': away_team['id'],
                        'away_team_name': away_team['name'],
                        'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                        'venue': fixture.get('venue', {}).get('name', 'Unknown') if fixture.get('venue') else 'Unknown'
                    }

            logger.warning(f"No fixture found matching: {home_team_name} vs {away_team_name}")
            logger.info("Searched fixtures:")
            for fixture in response['data'][:10]:  # Show first 10
                participants = fixture.get('participants', [])
                if len(participants) >= 2:
                    home = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), {})
                    away = next((p for p in participants if p.get('meta', {}).get('location') != 'home'), {})
                    logger.info(f"  - {home.get('name', 'Unknown')} vs {away.get('name', 'Unknown')}")

            return None

        except Exception as e:
            logger.error(f"Error searching for fixture: {e}")
            return None

    def get_fixture_lineups(self, fixture_id: int) -> Optional[Dict]:
        """
        Fetch lineups for a specific fixture.

        Args:
            fixture_id: Fixture ID

        Returns:
            Dictionary with home and away player IDs, or None if not available
        """
        try:
            response = self._api_call(f"fixtures/{fixture_id}", params={
                'include': 'lineups;participants'  # Need participants to get team IDs
            })

            if not response or 'data' not in response:
                return None

            fixture = response['data']
            lineups = fixture.get('lineups', [])

            if not lineups:
                logger.debug(f"No lineups available for fixture {fixture_id}")
                return None

            # Extract home and away team IDs
            participants = fixture.get('participants', [])
            home_team_id = None
            away_team_id = None

            for p in participants:
                if p.get('meta', {}).get('location') == 'home':
                    home_team_id = p.get('id')
                else:
                    away_team_id = p.get('id')

            # Group lineups by team
            home_player_ids = []
            away_player_ids = []

            for lineup in lineups:
                player_id = lineup.get('player_id')
                team_id = lineup.get('team_id')
                type_id = lineup.get('type_id')  # 11 = starter, 12 = substitute

                # Only include starters
                if type_id == 11 and player_id:
                    if team_id == home_team_id:
                        home_player_ids.append(player_id)
                    elif team_id == away_team_id:
                        away_player_ids.append(player_id)

            if not home_player_ids or not away_player_ids:
                logger.debug(f"Incomplete lineups for fixture {fixture_id}")
                return None

            logger.info(f"✅ Found lineups: {len(home_player_ids)} home, {len(away_player_ids)} away")

            return {
                'home_player_ids': home_player_ids,
                'away_player_ids': away_player_ids,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id
            }

        except Exception as e:
            logger.warning(f"Error fetching lineups for fixture {fixture_id}: {e}")
            return None

    def get_team_injuries(self, team_id: int) -> int:
        """
        Fetch current injury/suspension count for a team.
        
        Args:
            team_id: Team ID
            
        Returns:
            Number of currently unavailable players (injured/suspended)
        """
        try:
            response = self._api_call(f"teams/{team_id}", params={
                'include': 'sidelined'
            })
            
            if not response or 'data' not in response:
                return 0
            
            team_data = response['data']
            sidelined = team_data.get('sidelined', [])
            
            if not sidelined:
                return 0
            
            # Count currently unavailable players
            today = datetime.now().date()
            count = 0
            
            for injury in sidelined:
                # Check if injury/suspension is current
                completed = injury.get('completed', False)
                
                if completed:
                    continue  # Player recovered
                
                end_date_str = injury.get('end_date')
                
                if end_date_str:
                    try:
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                        if end_date >= today:
                            count += 1  # Still injured/suspended
                    except:
                        count += 1  # Can't parse date, assume unavailable
                else:
                    count += 1  # No end date, assume unavailable
            
            logger.info(f"Team {team_id}: {count} injured/suspended players")
            return count
            
        except Exception as e:
            logger.warning(f"Error fetching injuries for team {team_id}: {e}")
            return 0


    def _parse_statistics(self, stats: List[Dict]) -> Dict:
        """Parse statistics from API format to usable dict."""
        stat_dict = defaultdict(lambda: {'home': 0, 'away': 0})

        for stat in stats:
            stat_type_id = stat.get('type_id')
            location = stat.get('location', '').lower()
            # Value is nested in 'data' object
            value = stat.get('data', {}).get('value', 0)

            # Map stat type IDs to readable names (from STAT_TYPE_MAP in feature engineering)
            stat_map = {
                # Shooting
                34: 'corners',
                41: 'shots_off_target',
                42: 'shots_total',
                49: 'shots_insidebox',
                50: 'shots_outsidebox',
                52: 'stat_goals',
                54: 'goal_attempts',
                58: 'shots_blocked',
                64: 'hit_woodwork',
                86: 'shots_on_target',

                # Attacks
                43: 'attacks',
                44: 'dangerous_attacks',
                580: 'big_chances_created',
                581: 'big_chances_missed',

                # Passing
                45: 'possession_pct',
                46: 'ball_safe',
                62: 'long_passes',
                63: 'short_passes',
                80: 'passes',
                81: 'successful_passes',
                82: 'successful_passes_pct',
                116: 'accurate_passes',
                117: 'key_passes',
                122: 'long_balls',
                123: 'long_balls_won',
                124: 'through_balls',
                125: 'through_balls_won',

                # Crossing
                98: 'total_crosses',
                99: 'accurate_crosses',

                # Defense
                47: 'penalties',
                51: 'offsides',
                53: 'goal_kicks',
                55: 'free_kicks',
                56: 'fouls',
                57: 'saves',
                60: 'throwins',
                78: 'tackles',
                100: 'interceptions',
                101: 'clearances',
                102: 'clearances_won',
                103: 'punches',
                104: 'saves_insidebox',
                105: 'total_duels',
                106: 'duels_won',
                107: 'aerials_won',

                # Dribbling
                108: 'dribble_attempts',
                109: 'successful_dribbles',
                110: 'dribbled_past',

                # Headers
                65: 'successful_headers',
                70: 'headers',

                # Cards
                83: 'redcards',
                84: 'yellowcards',
                85: 'yellowred_cards',

                # Other
                59: 'substitutions',
                94: 'dispossessed',
                95: 'offsides_provoked',
                96: 'fouls_drawn',
                97: 'blocked_shots'
            }

            stat_name = stat_map.get(stat_type_id)
            if stat_name and location in ['home', 'away']:
                try:
                    stat_dict[stat_name][location] = float(value)
                except (ValueError, TypeError):
                    pass

        return dict(stat_dict)

    def calculate_elo(self, team_id: int, matches: List[Dict] = None) -> float:
        """
        Get Elo rating for team from pre-computed ratings.

        Note: We use pre-computed Elo ratings from training data instead of
        recalculating from recent matches, because:
        1. Training Elo is based on full historical sequence (not just recent matches)
        2. Ensures consistency between training and prediction features

        Args:
            team_id: Team ID
            matches: Unused, kept for API compatibility

        Returns:
            Elo rating (from training data or default)
        """
        # Return pre-loaded Elo rating
        return self.elo_ratings.get(team_id, DEFAULT_ELO)

    def calculate_rolling_stats(self, matches: List[Dict], team_id: int, window: int = 5) -> Dict:
        """
        Calculate rolling statistics over a window of matches.

        Args:
            matches: Team's recent matches
            team_id: Team ID
            window: Number of matches to average over

        Returns:
            Dictionary of rolling statistics
        """
        if not matches or len(matches) == 0:
            return {}

        # Take last N matches
        recent = matches[-window:] if len(matches) >= window else matches

        stats = {
            'goals': [],
            'goals_conceded': [],
            'shots_total': [],
            'shots_on_target': [],
            'dangerous_attacks': [],
            'possession_pct': [],
            'successful_passes': [],
            'passes': [],
            'tackles': [],
            'interceptions': [],
            'corners': []
        }

        # Also track what opponents did (for _conceded features)
        opp_stats = {
            'shots_total': [],
            'shots_on_target': [],
            'dangerous_attacks': [],
            'possession_pct': [],
            'successful_passes': [],
            'passes': [],
            'tackles': [],
            'interceptions': [],
            'corners': []
        }

        wins = 0
        draws = 0
        losses = 0

        for match in recent:
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']

            # Result
            if team_score > opp_score:
                wins += 1
            elif team_score == opp_score:
                draws += 1
            else:
                losses += 1

            # Goals
            stats['goals'].append(team_score)
            stats['goals_conceded'].append(opp_score)

            # Other stats from match statistics
            match_stats = match.get('statistics', {})
            side = 'home' if is_home else 'away'
            opp_side = 'away' if is_home else 'home'

            # Team's own stats
            for stat_name in stats.keys():
                if stat_name not in ['goals', 'goals_conceded']:
                    value = match_stats.get(stat_name, {}).get(side, 0)
                    stats[stat_name].append(value)

            # Opponent's stats (for _conceded features)
            for stat_name in opp_stats.keys():
                opp_value = match_stats.get(stat_name, {}).get(opp_side, 0)
                opp_stats[stat_name].append(opp_value)

        # Calculate averages for team's own stats
        result = {
            f'{stat}_avg': np.mean(values) if values else 0
            for stat, values in stats.items()
        }

        # Calculate averages for opponent's stats (_conceded suffix)
        for stat_name, values in opp_stats.items():
            result[f'{stat_name}_conceded_avg'] = np.mean(values) if values else 0

        # Add form (total points, not normalized - to match training data)
        total_games = len(recent)
        result['wins'] = wins
        result['draws'] = draws
        result['losses'] = losses
        result['form'] = wins * 3 + draws  # Points from wins (3) and draws (1)

        # Calculate xG (approximation from shots)
        if result['shots_total_avg'] > 0:
            result['xg_avg'] = result['shots_on_target_avg'] * 0.3 + result['shots_total_avg'] * 0.05
        else:
            result['xg_avg'] = 0

        # xG conceded (opponent's xG against this team)
        if result['shots_total_conceded_avg'] > 0:
            result['xg_conceded_avg'] = result['shots_on_target_conceded_avg'] * 0.3 + result['shots_total_conceded_avg'] * 0.05
        else:
            result['xg_conceded_avg'] = 0

        # Passing percentage (now using 'passes' and 'successful_passes')
        if result['passes_avg'] > 0:
            result['passes_pct'] = (result['successful_passes_avg'] / result['passes_avg']) * 100
        else:
            result['passes_pct'] = 0

        # Opponent passing percentage
        if result['passes_conceded_avg'] > 0:
            result['passes_pct_conceded'] = (result['successful_passes_conceded_avg'] / result['passes_conceded_avg']) * 100
        else:
            result['passes_pct_conceded'] = 0

        # Big chances created (approximation from shots on target)
        result['big_chances_created_avg'] = result['shots_on_target_avg'] * 0.3
        result['big_chances_created_conceded_avg'] = result['shots_on_target_conceded_avg'] * 0.3

        return result

    def calculate_h2h(self, home_team_id: int, away_team_id: int) -> Dict:
        """Calculate head-to-head statistics."""
        # For simplicity, we'll use default values
        # In production, you'd fetch H2H matches from API
        return {
            'h2h_home_wins': 0,
            'h2h_draws': 0,
            'h2h_away_wins': 0,
            'h2h_home_goals_avg': 0,
            'h2h_away_goals_avg': 0
        }

    def calculate_ema(self, values: List[float], alpha: float = 0.3) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            values: List of values (oldest first)
            alpha: Smoothing factor (0-1), higher = more weight on recent
        
        Returns:
            EMA value
        """
        if not values:
            return 0.0
        
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return ema

    def calculate_ema_features(self, matches: List[Dict], alpha: float = 0.3) -> Dict[str, float]:
        """
        Calculate EMA features for key statistics.
        
        Args:
            matches: List of recent matches (oldest first)
            alpha: Smoothing factor
        
        Returns:
            Dictionary of EMA features
        """
        ema_features = {}
        
        # Stats to calculate EMA for
        stats_to_ema = {
            'goals': 'goals_ema',
            'goals_conceded': 'goals_conceded_ema',
            'xg': 'xg_ema',
            'xg_conceded': 'xg_conceded_ema',
            'shots_total': 'shots_total_ema',
            'shots_total_conceded': 'shots_total_conceded_ema',
            'shots_on_target': 'shots_on_target_ema',
            'shots_on_target_conceded': 'shots_on_target_conceded_ema',
            'possession_pct': 'possession_pct_ema',
            'possession_pct_conceded': 'possession_pct_conceded_ema',
        }
        
        for stat_key, ema_key in stats_to_ema.items():
            values = []
            for match in matches:
                if stat_key in match:
                    values.append(match[stat_key])
                elif stat_key == 'xg' and 'shots_on_target' in match:
                    # Approximate xG from shots
                    values.append(match['shots_on_target'] * 0.3 + match.get('shots_total', 0) * 0.05)
                elif stat_key == 'xg_conceded' and 'shots_on_target_conceded' in match:
                    values.append(match['shots_on_target_conceded'] * 0.3 + match.get('shots_total_conceded', 0) * 0.05)
                elif stat_key == 'possession_pct_conceded':
                    # Opponent's possession
                    values.append(100 - match.get('possession_pct', 50))
                else:
                    values.append(0)
            
            ema_features[ema_key] = self.calculate_ema(values, alpha) if values else 0.0
        
        return ema_features

    def calculate_rest_days(self, matches: List[Dict], fixture_date: datetime) -> Dict[str, float]:
        """
        Calculate rest days features.
        
        Args:
            matches: List of recent matches (oldest first)
            fixture_date: Date of upcoming fixture
        
        Returns:
            Dictionary of rest days features
        """
        if not matches:
            return {
                'days_rest': 7,
                'short_rest': 0,
            }
        
        # Get last match date
        last_match = matches[-1]
        last_match_date = last_match['date']
        
        # Calculate days since last match
        if isinstance(last_match_date, str):
            last_match_date = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))
        
        days_rest = (fixture_date - last_match_date).days
        
        # Short rest indicator (less than 4 days)
        short_rest = 1 if days_rest < 4 else 0
        
        return {
            'days_rest': days_rest,
            'short_rest': short_rest,
        }

    def build_features_for_match(
        self,
        home_team_id: int,
        away_team_id: int,
        fixture_date: datetime,
        home_team_name: str = None,
        away_team_name: str = None,
        league_name: str = None,
        league_id: int = None,
        fixture_id: int = None
    ) -> Optional[Dict]:
        """
        Build complete feature vector for a match using live API data.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            fixture_date: Date of fixture
            home_team_name: Home team name (for standings lookup)
            away_team_name: Away team name (for standings lookup)
            league_name: League name (for standings lookup)
            fixture_id: Optional fixture ID (for fetching lineups)

        Returns:
            Dictionary of features or None if data unavailable
        """
        logger.info(f"Building features for match: {home_team_id} vs {away_team_id}")

        # Try to fetch lineups if fixture_id provided
        lineups_data = None
        use_real_player_data = False

        if fixture_id and self.player_manager.is_loaded:
            lineups_data = self.get_fixture_lineups(fixture_id)
            if lineups_data:
                use_real_player_data = True
                logger.info("✅ Using REAL player data from lineups")
            else:
                logger.info("⚠️ Lineups not available, using approximations")

        # Fetch recent matches for both teams
        home_matches = self.get_team_recent_matches(home_team_id, limit=15)
        away_matches = self.get_team_recent_matches(away_team_id, limit=15)

        if not home_matches or not away_matches:
            logger.warning("Insufficient match data")
            return None

        # Sort matches by date (oldest first for Elo calculation)
        home_matches.sort(key=lambda x: x['date'])
        away_matches.sort(key=lambda x: x['date'])

        # Calculate Elo ratings
        home_elo = self.calculate_elo(home_team_id, home_matches)
        away_elo = self.calculate_elo(away_team_id, away_matches)

        # Calculate rolling stats for different windows
        home_stats_3 = self.calculate_rolling_stats(home_matches, home_team_id, window=3)
        home_stats_5 = self.calculate_rolling_stats(home_matches, home_team_id, window=5)
        home_stats_10 = self.calculate_rolling_stats(home_matches, home_team_id, window=10)

        away_stats_3 = self.calculate_rolling_stats(away_matches, away_team_id, window=3)
        away_stats_5 = self.calculate_rolling_stats(away_matches, away_team_id, window=5)
        away_stats_10 = self.calculate_rolling_stats(away_matches, away_team_id, window=10)

        # Build feature vector
        features = {
            # Elo
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,

            # Form (3 games)
            'home_form_3': home_stats_3.get('form', 0),
            'away_form_3': away_stats_3.get('form', 0),
            'form_diff_3': home_stats_3.get('form', 0) - away_stats_3.get('form', 0),
            'home_wins_3': home_stats_3.get('wins', 0),
            'away_wins_3': away_stats_3.get('wins', 0),
            'home_draws_3': home_stats_3.get('draws', 0),
            'away_draws_3': away_stats_3.get('draws', 0),
            'home_losses_3': home_stats_3.get('losses', 0),
            'away_losses_3': away_stats_3.get('losses', 0),

            # Rolling stats (3 games)
            'home_goals_3': home_stats_3.get('goals_avg', 0),
            'away_goals_3': away_stats_3.get('goals_avg', 0),
            'home_goals_conceded_3': home_stats_3.get('goals_conceded_avg', 0),
            'away_goals_conceded_3': away_stats_3.get('goals_conceded_avg', 0),
            'home_xg_3': home_stats_3.get('xg_avg', 0),
            'away_xg_3': away_stats_3.get('xg_avg', 0),
            'home_xg_conceded_3': home_stats_3.get('xg_conceded_avg', 0),
            'away_xg_conceded_3': away_stats_3.get('xg_conceded_avg', 0),
            'home_shots_total_3': home_stats_3.get('shots_total_avg', 0),
            'away_shots_total_3': away_stats_3.get('shots_total_avg', 0),
            'home_shots_total_conceded_3': home_stats_3.get('shots_total_conceded_avg', 0),
            'away_shots_total_conceded_3': away_stats_3.get('shots_total_conceded_avg', 0),
            'home_shots_on_target_3': home_stats_3.get('shots_on_target_avg', 0),
            'away_shots_on_target_3': away_stats_3.get('shots_on_target_avg', 0),
            'home_shots_on_target_conceded_3': home_stats_3.get('shots_on_target_conceded_avg', 0),
            'away_shots_on_target_conceded_3': away_stats_3.get('shots_on_target_conceded_avg', 0),
            'home_possession_pct_3': home_stats_3.get('possession_pct_avg', 0),
            'away_possession_pct_3': away_stats_3.get('possession_pct_avg', 0),
            'home_possession_pct_conceded_3': home_stats_3.get('possession_pct_conceded_avg', 0),
            'away_possession_pct_conceded_3': away_stats_3.get('possession_pct_conceded_avg', 0),
            'home_dangerous_attacks_3': home_stats_3.get('dangerous_attacks_avg', 0),
            'away_dangerous_attacks_3': away_stats_3.get('dangerous_attacks_avg', 0),
            'home_dangerous_attacks_conceded_3': home_stats_3.get('dangerous_attacks_conceded_avg', 0),
            'away_dangerous_attacks_conceded_3': away_stats_3.get('dangerous_attacks_conceded_avg', 0),
            'home_corners_3': home_stats_3.get('corners_avg', 0),
            'away_corners_3': away_stats_3.get('corners_avg', 0),
            'home_corners_conceded_3': home_stats_3.get('corners_conceded_avg', 0),
            'away_corners_conceded_3': away_stats_3.get('corners_conceded_avg', 0),
            'home_passes_3': home_stats_3.get('passes_avg', 0),
            'away_passes_3': away_stats_3.get('passes_avg', 0),
            'home_passes_conceded_3': home_stats_3.get('passes_conceded_avg', 0),
            'away_passes_conceded_3': away_stats_3.get('passes_conceded_avg', 0),
            'home_successful_passes_pct_3': home_stats_3.get('passes_pct', 0),
            'away_successful_passes_pct_3': away_stats_3.get('passes_pct', 0),
            'home_tackles_3': home_stats_3.get('tackles_avg', 0),
            'away_tackles_3': away_stats_3.get('tackles_avg', 0),
            'home_tackles_conceded_3': home_stats_3.get('tackles_conceded_avg', 0),
            'away_tackles_conceded_3': away_stats_3.get('tackles_conceded_avg', 0),
            'home_interceptions_3': home_stats_3.get('interceptions_avg', 0),
            'away_interceptions_3': away_stats_3.get('interceptions_avg', 0),
            'home_interceptions_conceded_3': home_stats_3.get('interceptions_conceded_avg', 0),
            'away_interceptions_conceded_3': away_stats_3.get('interceptions_conceded_avg', 0),
            'home_successful_passes_3': home_stats_3.get('successful_passes_avg', 0),
            'away_successful_passes_3': away_stats_3.get('successful_passes_avg', 0),
            'home_successful_passes_conceded_3': home_stats_3.get('successful_passes_conceded_avg', 0),
            'away_successful_passes_conceded_3': away_stats_3.get('successful_passes_conceded_avg', 0),

            # Player stats (approximations for window 3)
            'home_successful_passes_pct_conceded_3': home_stats_3.get('passes_pct_conceded', 0),
            'away_successful_passes_pct_conceded_3': away_stats_3.get('passes_pct_conceded', 0),
            'home_big_chances_created_3': home_stats_3.get('big_chances_created_avg', 0),
            'away_big_chances_created_3': away_stats_3.get('big_chances_created_avg', 0),
            'home_big_chances_created_conceded_3': home_stats_3.get('big_chances_created_conceded_avg', 0),
            'away_big_chances_created_conceded_3': away_stats_3.get('big_chances_created_conceded_avg', 0),
            'home_player_clearances_3': home_stats_3.get('interceptions_avg', 0) * 1.5,
            'away_player_clearances_3': away_stats_3.get('interceptions_avg', 0) * 1.5,
            'home_player_rating_3': 6.5 + home_stats_3.get('form', 0) * 0.1,
            'away_player_rating_3': 6.5 + away_stats_3.get('form', 0) * 0.1,
            'home_player_touches_3': home_stats_3.get('possession_pct_avg', 0) * 6,
            'away_player_touches_3': away_stats_3.get('possession_pct_avg', 0) * 6,
            'home_player_duels_won_3': home_stats_3.get('tackles_avg', 0) * 1.2,
            'away_player_duels_won_3': away_stats_3.get('tackles_avg', 0) * 1.2,

            # Form (5 games)
            'home_form_5': home_stats_5.get('form', 0),
            'away_form_5': away_stats_5.get('form', 0),
            'form_diff_5': home_stats_5.get('form', 0) - away_stats_5.get('form', 0),
            'home_wins_5': home_stats_5.get('wins', 0),
            'away_wins_5': away_stats_5.get('wins', 0),
            'home_draws_5': home_stats_5.get('draws', 0),
            'away_draws_5': away_stats_5.get('draws', 0),
            'home_losses_5': home_stats_5.get('losses', 0),
            'away_losses_5': away_stats_5.get('losses', 0),

            # Rolling stats (5 games)
            'home_goals_5': home_stats_5.get('goals_avg', 0),
            'away_goals_5': away_stats_5.get('goals_avg', 0),
            'home_goals_conceded_5': home_stats_5.get('goals_conceded_avg', 0),
            'away_goals_conceded_5': away_stats_5.get('goals_conceded_avg', 0),
            'home_xg_5': home_stats_5.get('xg_avg', 0),
            'away_xg_5': away_stats_5.get('xg_avg', 0),
            'home_xg_conceded_5': home_stats_5.get('xg_conceded_avg', 0),
            'away_xg_conceded_5': away_stats_5.get('xg_conceded_avg', 0),
            'home_shots_total_5': home_stats_5.get('shots_total_avg', 0),
            'away_shots_total_5': away_stats_5.get('shots_total_avg', 0),
            'home_shots_on_target_5': home_stats_5.get('shots_on_target_avg', 0),
            'away_shots_on_target_5': away_stats_5.get('shots_on_target_avg', 0),
            'home_shots_total_conceded_5': home_stats_5.get('shots_total_conceded_avg', 0),
            'away_shots_total_conceded_5': away_stats_5.get('shots_total_conceded_avg', 0),
            'home_shots_on_target_conceded_5': home_stats_5.get('shots_on_target_conceded_avg', 0),
            'away_shots_on_target_conceded_5': away_stats_5.get('shots_on_target_conceded_avg', 0),
            'home_dangerous_attacks_5': home_stats_5.get('dangerous_attacks_avg', 0),
            'away_dangerous_attacks_5': away_stats_5.get('dangerous_attacks_avg', 0),
            'home_dangerous_attacks_conceded_5': home_stats_5.get('dangerous_attacks_conceded_avg', 0),
            'away_dangerous_attacks_conceded_5': away_stats_5.get('dangerous_attacks_conceded_avg', 0),
            'home_possession_pct_5': home_stats_5.get('possession_pct_avg', 0),
            'away_possession_pct_5': away_stats_5.get('possession_pct_avg', 0),
            'home_possession_pct_conceded_5': home_stats_5.get('possession_pct_conceded_avg', 0),
            'away_possession_pct_conceded_5': away_stats_5.get('possession_pct_conceded_avg', 0),
            'home_successful_passes_pct_5': home_stats_5.get('passes_pct', 0),
            'away_successful_passes_pct_5': away_stats_5.get('passes_pct', 0),
            'home_passes_5': home_stats_5.get('passes_avg', 0),
            'away_passes_5': away_stats_5.get('passes_avg', 0),
            'home_passes_conceded_5': home_stats_5.get('passes_conceded_avg', 0),
            'away_passes_conceded_5': away_stats_5.get('passes_conceded_avg', 0),
            'home_tackles_5': home_stats_5.get('tackles_avg', 0),
            'away_tackles_5': away_stats_5.get('tackles_avg', 0),
            'home_tackles_conceded_5': home_stats_5.get('tackles_conceded_avg', 0),
            'away_tackles_conceded_5': away_stats_5.get('tackles_conceded_avg', 0),
            'home_interceptions_5': home_stats_5.get('interceptions_avg', 0),
            'away_interceptions_5': away_stats_5.get('interceptions_avg', 0),
            'home_interceptions_conceded_5': home_stats_5.get('interceptions_conceded_avg', 0),
            'away_interceptions_conceded_5': away_stats_5.get('interceptions_conceded_avg', 0),
            'home_corners_5': home_stats_5.get('corners_avg', 0),
            'away_corners_5': away_stats_5.get('corners_avg', 0),
            'home_corners_conceded_5': home_stats_5.get('corners_conceded_avg', 0),
            'away_corners_conceded_5': away_stats_5.get('corners_conceded_avg', 0),

            # Player stats (approximations)
            'home_successful_passes_pct_conceded_5': home_stats_5.get('passes_pct_conceded', 0),
            'away_successful_passes_pct_conceded_5': away_stats_5.get('passes_pct_conceded', 0),
            'home_big_chances_created_5': home_stats_5.get('big_chances_created_avg', 0),
            'away_big_chances_created_5': away_stats_5.get('big_chances_created_avg', 0),
            'home_big_chances_created_conceded_5': home_stats_5.get('big_chances_created_conceded_avg', 0),
            'away_big_chances_created_conceded_5': away_stats_5.get('big_chances_created_conceded_avg', 0),
            'home_player_clearances_5': home_stats_5.get('interceptions_avg', 0) * 1.5,
            'away_player_clearances_5': away_stats_5.get('interceptions_avg', 0) * 1.5,
            'home_player_rating_5': 6.5 + home_stats_5.get('form', 0) * 0.1,  # Scale down from form
            'away_player_rating_5': 6.5 + away_stats_5.get('form', 0) * 0.1,
            'home_player_touches_5': home_stats_5.get('possession_pct_avg', 0) * 6,  # Fixed: was 'possession_avg'
            'away_player_touches_5': away_stats_5.get('possession_pct_avg', 0) * 6,
            'home_player_duels_won_5': home_stats_5.get('tackles_avg', 0) * 1.2,
            'away_player_duels_won_5': away_stats_5.get('tackles_avg', 0) * 1.2,

            # Rolling stats (10 games)
            'home_goals_10': home_stats_10.get('goals_avg', 0),
            'away_goals_10': away_stats_10.get('goals_avg', 0),
            'home_goals_conceded_10': home_stats_10.get('goals_conceded_avg', 0),
            'away_goals_conceded_10': away_stats_10.get('goals_conceded_avg', 0),
            'home_xg_10': home_stats_10.get('xg_avg', 0),
            'away_xg_10': away_stats_10.get('xg_avg', 0),
            'home_xg_conceded_10': home_stats_10.get('xg_conceded_avg', 0),
            'away_xg_conceded_10': away_stats_10.get('xg_conceded_avg', 0),
            'home_shots_total_10': home_stats_10.get('shots_total_avg', 0),
            'away_shots_total_10': away_stats_10.get('shots_total_avg', 0),
            'home_shots_total_conceded_10': home_stats_10.get('shots_total_conceded_avg', 0),
            'away_shots_total_conceded_10': away_stats_10.get('shots_total_conceded_avg', 0),
            'home_shots_on_target_10': home_stats_10.get('shots_on_target_avg', 0),
            'away_shots_on_target_10': away_stats_10.get('shots_on_target_avg', 0),
            'home_shots_on_target_conceded_10': home_stats_10.get('shots_on_target_conceded_avg', 0),
            'away_shots_on_target_conceded_10': away_stats_10.get('shots_on_target_conceded_avg', 0),
            'home_possession_pct_10': home_stats_10.get('possession_pct_avg', 0),
            'away_possession_pct_10': away_stats_10.get('possession_pct_avg', 0),
            'home_possession_pct_conceded_10': home_stats_10.get('possession_pct_conceded_avg', 0),
            'away_possession_pct_conceded_10': away_stats_10.get('possession_pct_conceded_avg', 0),
            'home_successful_passes_pct_10': home_stats_10.get('passes_pct', 0),
            'away_successful_passes_pct_10': away_stats_10.get('passes_pct', 0),
            'home_dangerous_attacks_10': home_stats_10.get('dangerous_attacks_avg', 0),
            'away_dangerous_attacks_10': away_stats_10.get('dangerous_attacks_avg', 0),
            'home_dangerous_attacks_conceded_10': home_stats_10.get('dangerous_attacks_conceded_avg', 0),
            'away_dangerous_attacks_conceded_10': away_stats_10.get('dangerous_attacks_conceded_avg', 0),
            'home_passes_10': home_stats_10.get('passes_avg', 0),
            'away_passes_10': away_stats_10.get('passes_avg', 0),
            'home_passes_conceded_10': home_stats_10.get('passes_conceded_avg', 0),
            'away_passes_conceded_10': away_stats_10.get('passes_conceded_avg', 0),
            'home_successful_passes_10': home_stats_10.get('successful_passes_avg', 0),
            'away_successful_passes_10': away_stats_10.get('successful_passes_avg', 0),
            'home_successful_passes_conceded_10': home_stats_10.get('successful_passes_conceded_avg', 0),
            'away_successful_passes_conceded_10': away_stats_10.get('successful_passes_conceded_avg', 0),
            'home_tackles_10': home_stats_10.get('tackles_avg', 0),
            'away_tackles_10': away_stats_10.get('tackles_avg', 0),
            'home_tackles_conceded_10': home_stats_10.get('tackles_conceded_avg', 0),
            'away_tackles_conceded_10': away_stats_10.get('tackles_conceded_avg', 0),
            'home_interceptions_10': home_stats_10.get('interceptions_avg', 0),
            'away_interceptions_10': away_stats_10.get('interceptions_avg', 0),
            'home_interceptions_conceded_10': home_stats_10.get('interceptions_conceded_avg', 0),
            'away_interceptions_conceded_10': away_stats_10.get('interceptions_conceded_avg', 0),
            'home_corners_10': home_stats_10.get('corners_avg', 0),
            'away_corners_10': away_stats_10.get('corners_avg', 0),
            'home_corners_conceded_10': home_stats_10.get('corners_conceded_avg', 0),
            'away_corners_conceded_10': away_stats_10.get('corners_conceded_avg', 0),

            # Form tracking (10 games)
            'home_wins_10': home_stats_10.get('wins', 0),
            'away_wins_10': away_stats_10.get('wins', 0),
            'home_draws_10': home_stats_10.get('draws', 0),
            'away_draws_10': away_stats_10.get('draws', 0),
            'home_losses_10': home_stats_10.get('losses', 0),
            'away_losses_10': away_stats_10.get('losses', 0),
            'home_form_10': home_stats_10.get('form', 0),
            'away_form_10': away_stats_10.get('form', 0),
            'form_diff_10': home_stats_10.get('form', 0) - away_stats_10.get('form', 0),

            # Player stats (approximations for window 10)
            'home_successful_passes_pct_conceded_10': home_stats_10.get('passes_pct_conceded', 0),
            'away_successful_passes_pct_conceded_10': away_stats_10.get('passes_pct_conceded', 0),
            'home_big_chances_created_10': home_stats_10.get('big_chances_created_avg', 0),
            'away_big_chances_created_10': away_stats_10.get('big_chances_created_avg', 0),
            'home_big_chances_created_conceded_10': home_stats_10.get('big_chances_created_conceded_avg', 0),
            'away_big_chances_created_conceded_10': away_stats_10.get('big_chances_created_conceded_avg', 0),
            'home_player_clearances_10': home_stats_10.get('interceptions_avg', 0) * 1.5,
            'away_player_clearances_10': away_stats_10.get('interceptions_avg', 0) * 1.5,
            'home_player_rating_10': 6.5 + home_stats_10.get('form', 0) * 0.1,
            'away_player_rating_10': 6.5 + away_stats_10.get('form', 0) * 0.1,
            'home_player_touches_10': home_stats_10.get('possession_pct_avg', 0) * 6,
            'away_player_touches_10': away_stats_10.get('possession_pct_avg', 0) * 6,
            'home_player_duels_won_10': home_stats_10.get('tackles_avg', 0) * 1.2,
            'away_player_duels_won_10': away_stats_10.get('tackles_avg', 0) * 1.2,

            # Attack/defense strength (multiple windows)
            'home_attack_strength_3': home_stats_3.get('goals_avg', 0) / max(home_stats_3.get('xg_avg', 0), 0.1),
            'away_attack_strength_3': away_stats_3.get('goals_avg', 0) / max(away_stats_3.get('xg_avg', 0), 0.1),
            'home_defense_strength_3': home_stats_3.get('goals_conceded_avg', 0) / max(home_stats_3.get('xg_conceded_avg', 0), 0.1),
            'away_defense_strength_3': away_stats_3.get('goals_conceded_avg', 0) / max(away_stats_3.get('xg_conceded_avg', 0), 0.1),
            'home_attack_strength_5': home_stats_5.get('goals_avg', 0) / max(home_stats_5.get('xg_avg', 0), 0.1),
            'away_attack_strength_5': away_stats_5.get('goals_avg', 0) / max(away_stats_5.get('xg_avg', 0), 0.1),
            'home_defense_strength_5': home_stats_5.get('goals_conceded_avg', 0) / max(home_stats_5.get('xg_conceded_avg', 0), 0.1),
            'away_defense_strength_5': away_stats_5.get('goals_conceded_avg', 0) / max(away_stats_5.get('xg_conceded_avg', 0), 0.1),
            'home_attack_strength_10': home_stats_10.get('goals_avg', 0) / max(home_stats_10.get('xg_avg', 0), 0.1),
            'away_attack_strength_10': away_stats_10.get('goals_avg', 0) / max(away_stats_10.get('xg_avg', 0), 0.1),
            'home_defense_strength_10': home_stats_10.get('goals_conceded_avg', 0) / max(home_stats_10.get('xg_conceded_avg', 0), 0.1),
            'away_defense_strength_10': away_stats_10.get('goals_conceded_avg', 0) / max(away_stats_10.get('xg_conceded_avg', 0), 0.1),
        }

        # Add H2H features
        h2h = self.calculate_h2h(home_team_id, away_team_id)
        features.update(h2h)

        # Override player stats with REAL data from lineups if available
        if use_real_player_data and lineups_data:
            logger.info("📊 Replacing approximations with real player statistics from lineup")

            # Get real player stats for home team
            home_player_features = self.player_manager.build_feature_dict_from_lineup(
                lineups_data['home_player_ids'],
                prefix='home'
            )

            # Get real player stats for away team
            away_player_features = self.player_manager.build_feature_dict_from_lineup(
                lineups_data['away_player_ids'],
                prefix='away'
            )

            # Update features with real player data (replaces approximations)
            features.update(home_player_features)
            features.update(away_player_features)

            # Get lineup quality scores for logging
            home_quality = self.player_manager.get_lineup_quality_score(
                lineups_data['home_player_ids']
            )
            away_quality = self.player_manager.get_lineup_quality_score(
                lineups_data['away_player_ids']
            )

            logger.info(f"  Home lineup: {home_quality['players_found']}/{len(lineups_data['home_player_ids'])} "
                       f"players found ({home_quality['coverage_pct']:.1f}% coverage), "
                       f"avg rating: {home_quality['avg_rating']:.2f}")
            logger.info(f"  Away lineup: {away_quality['players_found']}/{len(lineups_data['away_player_ids'])} "
                       f"players found ({away_quality['coverage_pct']:.1f}% coverage), "
                       f"avg rating: {away_quality['avg_rating']:.2f}")

        # Add contextual features
        # Derive season name from date (July-June season)
        year = fixture_date.year
        month = fixture_date.month
        if month >= 7:  # July onwards is new season
            season_name = f"{year}/{year+1}"
        else:  # Jan-June is previous season
            season_name = f"{year-1}/{year}"

        # Try to get REAL position and points from ESPN API
        home_standings = None
        away_standings = None

        if home_team_name and league_name:
            home_standings = get_team_standings(home_team_name, league_name)
        if away_team_name and league_name:
            away_standings = get_team_standings(away_team_name, league_name)

        # Use real standings if available, otherwise estimate from Elo + form
        if home_standings:
            home_est_position = home_standings['position']
            home_est_points = home_standings['points']
            logger.info(f"Using real standings for {home_team_name}: pos={home_est_position}, pts={home_est_points}")
        else:
            # Fallback: estimate from Elo and form
            home_est_position = max(1, min(20, 21 - ((home_elo - 1400) / 20)))
            home_est_points = home_stats_10.get('form', 15) * 1.9
            logger.info(f"Using estimated standings for home team (ID {home_team_id})")

        if away_standings:
            away_est_position = away_standings['position']
            away_est_points = away_standings['points']
            logger.info(f"Using real standings for {away_team_name}: pos={away_est_position}, pts={away_est_points}")
        else:
            # Fallback: estimate from Elo and form
            away_est_position = max(1, min(20, 21 - ((away_elo - 1400) / 20)))
            away_est_points = away_stats_10.get('form', 15) * 1.9
            logger.info(f"Using estimated standings for away team (ID {away_team_id})")

        features.update({
            'season_name': season_name,
            'day_of_week': fixture_date.weekday(),  # 0=Monday, 6=Sunday

            # Fetch real-time betting odds
        })
        
        logger.info("Fetching real-time betting odds...")
        odds_data = self.odds_fetcher.get_odds(fixture_id) if fixture_id else self.odds_fetcher._get_neutral_odds()
        
        features.update({
            # Market features (real-time betting odds)
            'odds_home': odds_data['odds_home'],
            'odds_draw': odds_data['odds_draw'],
            'odds_away': odds_data['odds_away'],
            'odds_total': odds_data['odds_total'],
            'odds_home_draw_ratio': odds_data['odds_home_draw_ratio'],
            'odds_home_away_ratio': odds_data['odds_home_away_ratio'],
            'market_home_away_ratio': odds_data['market_home_away_ratio'],

            # Position and points (estimated from Elo and form)
            'home_position': home_est_position,
            'away_position': away_est_position,
            'position_diff': home_est_position - away_est_position,  # Negative = home better
            'home_points': home_est_points,
            'away_points': away_est_points,
            'points_diff': home_est_points - away_est_points,  # Positive = home better
            
            # League ID (for league-specific patterns)
            'league_id': league_id if league_id else 0,  # Default to 0 if not provided


            'home_injuries': self.get_team_injuries(home_team_id),
            'away_injuries': self.get_team_injuries(away_team_id),
        })
        
        # Calculate injury difference
        features['injury_diff'] = features['home_injuries'] - features['away_injuries']
        
        # Add remaining features
        features.update({
            'round_num': 20,
            'season_progress': 0.5,
            'is_early_season': 0,
            'is_weekend': 1 if fixture_date.weekday() >= 5 else 0
        })


        # Calculate EMA features
        logger.info("Calculating EMA features...")
        home_ema = self.calculate_ema_features(home_matches)
        away_ema = self.calculate_ema_features(away_matches)
        
        # Add EMA features to features dict
        features.update({
            'home_goals_ema': home_ema['goals_ema'],
            'away_goals_ema': away_ema['goals_ema'],
            'home_goals_conceded_ema': home_ema['goals_conceded_ema'],
            'away_goals_conceded_ema': away_ema['goals_conceded_ema'],
            'home_xg_ema': home_ema['xg_ema'],
            'away_xg_ema': away_ema['xg_ema'],
            'home_xg_conceded_ema': home_ema['xg_conceded_ema'],
            'away_xg_conceded_ema': away_ema['xg_conceded_ema'],
            'home_shots_total_ema': home_ema['shots_total_ema'],
            'away_shots_total_ema': away_ema['shots_total_ema'],
            'home_shots_total_conceded_ema': home_ema['shots_total_conceded_ema'],
            'away_shots_total_conceded_ema': away_ema['shots_total_conceded_ema'],
            'home_shots_on_target_ema': home_ema['shots_on_target_ema'],
            'away_shots_on_target_ema': away_ema['shots_on_target_ema'],
            'home_shots_on_target_conceded_ema': home_ema['shots_on_target_conceded_ema'],
            'away_shots_on_target_conceded_ema': away_ema['shots_on_target_conceded_ema'],
            'home_possession_pct_ema': home_ema['possession_pct_ema'],
            'away_possession_pct_ema': away_ema['possession_pct_ema'],
            'home_possession_pct_conceded_ema': home_ema['possession_pct_conceded_ema'],
            'away_possession_pct_conceded_ema': away_ema['possession_pct_conceded_ema'],
        })

        # Calculate rest days features
        logger.info("Calculating rest days features...")
        home_rest = self.calculate_rest_days(home_matches, fixture_date)
        away_rest = self.calculate_rest_days(away_matches, fixture_date)
        
        # Add rest days features
        features.update({
            'days_rest_home': home_rest['days_rest'],
            'days_rest_away': away_rest['days_rest'],
            'home_short_rest': home_rest['short_rest'],
            'away_short_rest': away_rest['short_rest'],
            'rest_diff': home_rest['days_rest'] - away_rest['days_rest'],
        })

        logger.info(f"Built {len(features)} features")
        return features


def get_upcoming_fixtures(target_date: str) -> pd.DataFrame:
    """Fetch upcoming fixtures for a specific date."""
    logger.info(f"Fetching fixtures for {target_date}")

    try:
        url = f"{BASE_URL}/fixtures/between/{target_date}/{target_date}"
        params = {
            'api_token': API_KEY,
            'include': 'participants;league;venue'
        }

        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or 'data' not in data:
            logger.warning(f"No fixtures found for {target_date}")
            return pd.DataFrame()

        fixtures = []
        for fixture in data['data']:
            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            fixtures.append({
                'fixture_id': fixture['id'],
                'date': fixture.get('starting_at'),
                'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                'league_id': fixture.get('league_id', 0),  # Add league_id
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name'],
                'venue': fixture.get('venue', {}).get('name', 'Unknown') if fixture.get('venue') else 'Unknown'
            })

        df = pd.DataFrame(fixtures)
        logger.info(f"Found {len(df)} fixtures")
        return df

    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        return pd.DataFrame()


def load_models(model_name='stacking'):
    """Load trained models."""
    models = {}

    # Load XGBoost (draw-tuned model for better predictions)
    xgb_path = MODELS_DIR / "xgboost_model_draw_tuned.joblib"
    if xgb_path.exists() and model_name in ['xgboost', 'all']:
        try:
            import importlib
            xgb_module = importlib.import_module('06_model_xgboost')
            XGBoostFootballModel = xgb_module.XGBoostFootballModel
            xgb_model = XGBoostFootballModel()
            xgb_model.load(xgb_path)
            models['xgboost'] = xgb_model
            logger.info("Loaded XGBoost model")
        except Exception as e:
            logger.warning(f"Could not load XGBoost: {e}")

    # Load stacking ensemble (requires base models)
    stacking_path = MODELS_DIR / "stacking_ensemble.joblib"
    if stacking_path.exists() and model_name in ['stacking', 'all']:
        try:
            import importlib

            # Load base models first
            elo_module = importlib.import_module('04_model_baseline_elo')
            dc_module = importlib.import_module('05_model_dixon_coles')
            xgb_module = importlib.import_module('06_model_xgboost')
            ensemble_module = importlib.import_module('07_model_ensemble')

            EloProbabilityModel = elo_module.EloProbabilityModel
            DixonColesModel = dc_module.DixonColesModel
            CalibratedDixonColes = dc_module.CalibratedDixonColes
            XGBoostFootballModel = xgb_module.XGBoostFootballModel
            StackingEnsemble = ensemble_module.StackingEnsemble

            # Load each base model
            elo_model = EloProbabilityModel()
            elo_path = MODELS_DIR / "elo_model.joblib"
            if elo_path.exists():
                elo_model.load(elo_path)

            # Load Dixon-Coles (saved as CalibratedDixonColes)
            dc_path = MODELS_DIR / "dixon_coles_model.joblib"
            if dc_path.exists():
                import joblib
                dc_data = joblib.load(dc_path)
                # Reconstruct the calibrated model
                base_dc = DixonColesModel()
                base_dc.attack = dc_data['base_model_data']['attack']
                base_dc.defense = dc_data['base_model_data']['defense']
                base_dc.home_adv = dc_data['base_model_data']['home_adv']
                base_dc.rho = dc_data['base_model_data']['rho']
                base_dc.team_to_idx = dc_data['base_model_data']['team_to_idx']
                base_dc.idx_to_team = dc_data['base_model_data']['idx_to_team']
                base_dc.time_decay = dc_data['base_model_data']['time_decay']
                base_dc.max_goals = dc_data['base_model_data']['max_goals']
                base_dc.is_fitted = dc_data['base_model_data']['is_fitted']

                dc_model = CalibratedDixonColes(base_dc)
                dc_model.calibrators = dc_data['calibrators']
                dc_model.is_calibrated = dc_data['is_calibrated']
            else:
                # Fallback to uncalibrated
                dc_model = DixonColesModel()

            xgb_model = XGBoostFootballModel()
            xgb_path = MODELS_DIR / "xgboost_model_draw_tuned.joblib"
            if xgb_path.exists():
                xgb_model.load(xgb_path)

            # Create stacking ensemble
            stacking_model = StackingEnsemble()
            stacking_model.load(stacking_path)

            # Add base models
            stacking_model.add_model('elo', elo_model)
            stacking_model.add_model('dixon_coles', dc_model)
            stacking_model.add_model('xgboost', xgb_model)

            models['stacking'] = stacking_model
            logger.info("Loaded Stacking Ensemble model with base models")
        except Exception as e:
            logger.warning(f"Could not load Stacking Ensemble: {e}")
            import traceback
            traceback.print_exc()

    return models


def main():
    parser = argparse.ArgumentParser(description="Predict matches using live API data")
    parser.add_argument(
        "--date",
        default="today",
        help="Date to predict (today, tomorrow, or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--home",
        help="Home team name (e.g., 'Man City', 'Liverpool')"
    )
    parser.add_argument(
        "--away",
        help="Away team name (e.g., 'Arsenal', 'Chelsea')"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--model",
        default="stacking",
        choices=["xgboost", "stacking"],
        help="Model to use for predictions (default: stacking)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll for betting strategy (default: 1000)"
    )
    parser.add_argument(
        "--no-betting",
        action="store_true",
        help="Disable betting strategy recommendations"
    )

    args = parser.parse_args()

    # Check if team names provided (--home and --away)
    if args.home and args.away:
        print("=" * 80)
        print(f"SEARCHING FOR: {args.home} vs {args.away}")
        print("Using real-time API data")
        print("=" * 80)

        # Initialize calculator to search for fixture
        calculator = LiveFeatureCalculator()

        # Search for the fixture
        fixture_info = calculator.search_fixture_by_teams(args.home, args.away, days_ahead=14)

        if not fixture_info:
            print(f"\n❌ ERROR: Could not find fixture matching:")
            print(f"   Home: {args.home}")
            print(f"   Away: {args.away}")
            print(f"\nTips:")
            print("- Check team name spelling (e.g., 'Man City' or 'Manchester City')")
            print("- Ensure match is within next 14 days")
            print("- Try with abbreviated names (e.g., 'Liverpool' instead of 'Liverpool FC')")
            return

        # Convert to fixtures_df format
        fixtures_df = pd.DataFrame([fixture_info])

        print(f"\n✅ Found fixture:")
        print(f"   {fixture_info['home_team_name']} vs {fixture_info['away_team_name']}")
        print(f"   Date: {fixture_info['date']}")
        print(f"   League: {fixture_info['league_name']}")
        print(f"   Fixture ID: {fixture_info['fixture_id']}")

    else:
        # Parse date
        if args.date.lower() == "today":
            target_date = datetime.now().date()
        elif args.date.lower() == "tomorrow":
            target_date = (datetime.now() + timedelta(days=1)).date()
        else:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

        date_str = target_date.strftime("%Y-%m-%d")

        print("=" * 80)
        print(f"LIVE MATCH PREDICTIONS FOR {date_str}")
        print("Using real-time API data")
        print("=" * 80)

        # Get fixtures
        fixtures_df = get_upcoming_fixtures(date_str)

        if fixtures_df.empty:
            print(f"\nNo fixtures found for {date_str}")
            return

        print(f"\nFound {len(fixtures_df)} fixtures")

    # Load models
    print("\nLoading models...")
    models = load_models(model_name=args.model)

    if not models:
        print("ERROR: No models loaded")
        return

    model_name = args.model
    model = models.get(model_name)

    # Initialize feature calculator (used for both search and feature building)
    calculator = LiveFeatureCalculator()

    # Initialize betting strategy
    betting_strategy = None
    current_bankroll = args.bankroll
    if not args.no_betting:
        betting_strategy = SmartMultiOutcomeStrategy(bankroll=current_bankroll)
        print(f"\n💰 Betting Strategy Enabled (Bankroll: £{current_bankroll:,.2f})")
        print("   Rules: Away ≥50% | Draw (teams within 5%) | Home ≥51%")

    # Make predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS (using live API data)")
    if betting_strategy:
        print("with BETTING RECOMMENDATIONS")
    print("=" * 80)

    predictions = []
    total_bets = 0
    total_staked = 0.0

    for idx, fixture in fixtures_df.iterrows():
        print(f"\n{fixture['league_name']}")
        print(f"{fixture['date']} | {fixture['venue']}")
        print(f"{fixture['home_team_name']} vs {fixture['away_team_name']}")

        try:
            # Build features from live API data
            features = calculator.build_features_for_match(
                fixture['home_team_id'],
                fixture['away_team_id'],
                pd.to_datetime(fixture['date']),
                home_team_name=fixture['home_team_name'],
                away_team_name=fixture['away_team_name'],
                league_name=fixture['league_name'],
                league_id=fixture.get('league_id', 0),  # Add league_id
                fixture_id=fixture.get('fixture_id')  # For lineup fetching
            )

            if not features:
                print(f"  ERROR: Could not fetch data from API")
                predictions.append({
                    'date': fixture['date'],
                    'league': fixture['league_name'],
                    'home_team': fixture['home_team_name'],
                    'away_team': fixture['away_team_name'],
                    'venue': fixture['venue'],
                    'error': 'Could not fetch API data'
                })
                continue

            # Convert to DataFrame
            feature_df = pd.DataFrame([features])

            # Add team info for Dixon-Coles model (needed by stacking ensemble)
            feature_df['home_team_id'] = fixture['home_team_id']
            feature_df['away_team_id'] = fixture['away_team_id']
            feature_df['home_team_name'] = fixture['home_team_name']
            feature_df['away_team_name'] = fixture['away_team_name']

            # Make prediction
            probs = model.predict_proba(feature_df)[0]

            result = {
                'date': fixture['date'],
                'league': fixture['league_name'],
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'venue': fixture['venue'],
                'model_used': model_name,
                'home_win_prob': float(probs[2]),
                'draw_prob': float(probs[1]),
                'away_win_prob': float(probs[0]),
                'predicted_outcome': ['Away Win', 'Draw', 'Home Win'][np.argmax(probs)],
                'data_source': 'live_api'
            }

            print(f"  Model: {model_name}")
            print(f"  Home Win: {result['home_win_prob']:.1%}")
            print(f"  Draw:     {result['draw_prob']:.1%}")
            print(f"  Away Win: {result['away_win_prob']:.1%}")
            print(f"  → Prediction: {result['predicted_outcome']}")
            print(f"  ✨ Using live API data")

            # Apply betting strategy
            if betting_strategy:
                match_data = {
                    'match_id': f"{fixture['date']}_{fixture['home_team_name']}_{fixture['away_team_name']}",
                    'date': str(fixture['date']),
                    'home_team': fixture['home_team_name'],
                    'away_team': fixture['away_team_name'],
                    'home_prob': result['home_win_prob'],
                    'draw_prob': result['draw_prob'],
                    'away_prob': result['away_win_prob']
                }

                # Update strategy bankroll
                betting_strategy.bankroll = current_bankroll

                # Get betting recommendations
                bet_recommendations = betting_strategy.evaluate_match(match_data)

                if bet_recommendations:
                    for bet in bet_recommendations:
                        print(f"\n  💡 BET RECOMMENDATION:")
                        print(f"     Bet: {bet.bet_outcome}")
                        print(f"     Stake: £{bet.stake:.2f}")
                        print(f"     Fair Odds: {bet.fair_odds:.2f}")
                        print(f"     Expected Value: £{bet.expected_value:+.2f}")
                        print(f"     Rule: {bet.rule_applied}")

                        # Update tracking
                        total_bets += 1
                        total_staked += bet.stake
                        current_bankroll -= bet.stake

                        # Add betting info to result
                        result['bet_placed'] = True
                        result['bet_outcome'] = bet.bet_outcome
                        result['bet_stake'] = bet.stake
                        result['bet_odds'] = bet.fair_odds
                        result['bet_ev'] = bet.expected_value
                        result['bet_rule'] = bet.rule_applied
                        result['bankroll_after'] = current_bankroll
                else:
                    result['bet_placed'] = False

            predictions.append(result)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            print(f"  ERROR: {e}")
            predictions.append({
                'date': fixture['date'],
                'league': fixture['league_name'],
                'home_team': fixture['home_team_name'],
                'away_team': fixture['away_team_name'],
                'venue': fixture['venue'],
                'error': str(e)
            })

    # Save to CSV if requested
    if args.output:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

    print("\n" + "=" * 80)
    print(f"Completed {len(predictions)} predictions using live API data")
    print("=" * 80)

    # Betting strategy summary
    if betting_strategy and total_bets > 0:
        print("\n" + "=" * 80)
        print("BETTING STRATEGY SUMMARY")
        print("=" * 80)
        print(f"Total Bets Placed: {total_bets}")
        print(f"Total Staked: £{total_staked:,.2f}")
        print(f"Remaining Bankroll: £{current_bankroll:,.2f}")
        print(f"Amount at Risk: £{total_staked:,.2f} ({total_staked/args.bankroll*100:.1f}% of initial)")
        print(f"\nNote: These are paper trading recommendations.")
        print(f"Track actual results to calculate real ROI.")
    elif betting_strategy and total_bets == 0:
        print("\n💡 No betting opportunities met strategy criteria.")


if __name__ == "__main__":
    main()
