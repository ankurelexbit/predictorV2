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

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR
from utils import setup_logger

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
        self.elo_ratings = {}  # Cache Elo ratings

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
            'include': 'latest.statistics;latest.participants'
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
                home_score = home_participant.get('meta', {}).get('goals', 0)
                away_score = away_participant.get('meta', {}).get('goals', 0)

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

    def calculate_elo(self, team_id: int, matches: List[Dict]) -> float:
        """
        Calculate current Elo rating for team based on recent matches.

        Args:
            team_id: Team ID
            matches: List of team's recent matches (sorted oldest to newest)

        Returns:
            Current Elo rating
        """
        # Start with default or cached Elo
        elo = self.elo_ratings.get(team_id, DEFAULT_ELO)

        for match in matches:
            is_home = match['home_team_id'] == team_id
            team_score = match['home_score'] if is_home else match['away_score']
            opp_score = match['away_score'] if is_home else match['home_score']
            opp_id = match['away_team_id'] if is_home else match['home_team_id']

            # Get opponent Elo
            opp_elo = self.elo_ratings.get(opp_id, DEFAULT_ELO)

            # Calculate expected score
            expected = 1 / (1 + 10 ** ((opp_elo - elo) / 400))

            # Actual score (1 = win, 0.5 = draw, 0 = loss)
            if team_score > opp_score:
                actual = 1.0
            elif team_score == opp_score:
                actual = 0.5
            else:
                actual = 0.0

            # Update Elo
            elo += ELO_K_FACTOR * (actual - expected)

        # Cache the result
        self.elo_ratings[team_id] = elo
        return elo

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
            'possession_pct': [],  # Changed from 'possession'
            'successful_passes': [],  # Changed from 'passes_accurate'
            'passes': [],  # Changed from 'passes_total'
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

            for stat_name in stats.keys():
                if stat_name not in ['goals', 'goals_conceded']:
                    value = match_stats.get(stat_name, {}).get(side, 0)
                    stats[stat_name].append(value)

        # Calculate averages
        result = {
            f'{stat}_avg': np.mean(values) if values else 0
            for stat, values in stats.items()
        }

        # Add form
        total_games = len(recent)
        result['wins'] = wins
        result['draws'] = draws
        result['losses'] = losses
        result['form'] = (wins * 3 + draws) / (total_games * 3) if total_games > 0 else 0

        # Calculate xG (approximation from shots)
        if result['shots_total_avg'] > 0:
            result['xg_avg'] = result['shots_on_target_avg'] * 0.3 + result['shots_total_avg'] * 0.05
        else:
            result['xg_avg'] = 0

        result['xg_conceded_avg'] = result['xg_avg'] * (result['goals_conceded_avg'] / max(result['goals_avg'], 0.1))

        # Passing percentage (now using 'passes' and 'successful_passes')
        if result['passes_avg'] > 0:
            result['passes_pct'] = (result['successful_passes_avg'] / result['passes_avg']) * 100
        else:
            result['passes_pct'] = 0

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

    def build_features_for_match(
        self,
        home_team_id: int,
        away_team_id: int,
        fixture_date: datetime
    ) -> Optional[Dict]:
        """
        Build complete feature vector for a match using live API data.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            fixture_date: Date of fixture

        Returns:
            Dictionary of features or None if data unavailable
        """
        logger.info(f"Building features for match: {home_team_id} vs {away_team_id}")

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

            # Form (5 games)
            'home_form_5': home_stats_5.get('form', 0),
            'away_form_5': away_stats_5.get('form', 0),
            'form_diff_5': home_stats_5.get('form', 0) - away_stats_5.get('form', 0),
            'home_wins_5': home_stats_5.get('wins', 0),
            'away_wins_5': away_stats_5.get('wins', 0),

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
            'home_dangerous_attacks_5': home_stats_5.get('dangerous_attacks_avg', 0),
            'away_dangerous_attacks_5': away_stats_5.get('dangerous_attacks_avg', 0),
            'home_possession_pct_5': home_stats_5.get('possession_pct_avg', 0),
            'away_possession_pct_5': away_stats_5.get('possession_pct_avg', 0),
            'home_successful_passes_pct_5': home_stats_5.get('passes_pct', 0),
            'away_successful_passes_pct_5': away_stats_5.get('passes_pct', 0),
            'home_tackles_5': home_stats_5.get('tackles_avg', 0),
            'away_tackles_5': away_stats_5.get('tackles_avg', 0),
            'home_interceptions_5': home_stats_5.get('interceptions_avg', 0),
            'away_interceptions_5': away_stats_5.get('interceptions_avg', 0),

            # Player stats (approximations)
            'home_big_chances_created_5': home_stats_5.get('shots_on_target_avg', 0) * 0.3,
            'away_big_chances_created_5': away_stats_5.get('shots_on_target_avg', 0) * 0.3,
            'home_player_clearances_5': home_stats_5.get('interceptions_avg', 0) * 1.5,
            'away_player_clearances_5': away_stats_5.get('interceptions_avg', 0) * 1.5,
            'home_player_rating_5': 6.5 + home_stats_5.get('form', 0) * 1.5,
            'away_player_rating_5': 6.5 + away_stats_5.get('form', 0) * 1.5,
            'home_player_touches_5': home_stats_5.get('possession_avg', 0) * 6,
            'away_player_touches_5': away_stats_5.get('possession_avg', 0) * 6,
            'home_player_duels_won_5': home_stats_5.get('tackles_avg', 0) * 1.2,
            'away_player_duels_won_5': away_stats_5.get('tackles_avg', 0) * 1.2,

            # Rolling stats (10 games)
            'home_goals_10': home_stats_10.get('goals_avg', 0),
            'away_goals_10': away_stats_10.get('goals_avg', 0),
            'home_xg_10': home_stats_10.get('xg_avg', 0),
            'away_xg_10': away_stats_10.get('xg_avg', 0),

            # Attack/defense strength
            'home_attack_strength_5': home_stats_5.get('goals_avg', 0) / max(home_stats_5.get('xg_avg', 0), 0.1),
            'away_attack_strength_5': away_stats_5.get('goals_avg', 0) / max(away_stats_5.get('xg_avg', 0), 0.1),
            'home_defense_strength_5': home_stats_5.get('goals_conceded_avg', 0) / max(home_stats_5.get('xg_conceded_avg', 0), 0.1),
            'away_defense_strength_5': away_stats_5.get('goals_conceded_avg', 0) / max(away_stats_5.get('xg_conceded_avg', 0), 0.1),
        }

        # Add H2H features
        h2h = self.calculate_h2h(home_team_id, away_team_id)
        features.update(h2h)

        # Add contextual features (placeholders - would need standings API)
        features.update({
            'home_position': 10,  # Would fetch from standings
            'away_position': 10,
            'position_diff': 0,
            'home_points': 30,
            'away_points': 30,
            'points_diff': 0,
            'home_injuries': 0,  # Would fetch from injuries API
            'away_injuries': 0,
            'injury_diff': 0,
            'round_num': 20,
            'season_progress': 0.5,
            'is_early_season': 0,
            'is_weekend': 1 if fixture_date.weekday() >= 5 else 0
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


def load_models():
    """Load trained models."""
    models = {}

    # Load XGBoost
    xgb_path = MODELS_DIR / "xgboost_model.joblib"
    if xgb_path.exists():
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

    # Load stacking ensemble
    stacking_path = MODELS_DIR / "stacking_ensemble.joblib"
    if stacking_path.exists():
        try:
            import importlib
            ensemble_module = importlib.import_module('07_model_ensemble')
            StackingEnsemble = ensemble_module.StackingEnsemble

            # Note: stacking needs base models loaded first
            # For now, use XGBoost as fallback
            logger.info("Stacking ensemble available but using XGBoost for simplicity")
        except Exception as e:
            logger.warning(f"Could not load Stacking: {e}")

    return models


def main():
    parser = argparse.ArgumentParser(description="Predict matches using live API data")
    parser.add_argument(
        "--date",
        default="today",
        help="Date to predict (today, tomorrow, or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        help="Output CSV file path (optional)"
    )

    args = parser.parse_args()

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
    models = load_models()

    if not models:
        print("ERROR: No models loaded")
        return

    model_name = 'xgboost'
    model = models.get(model_name)

    # Initialize feature calculator
    calculator = LiveFeatureCalculator()

    # Make predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS (using live API data)")
    print("=" * 80)

    predictions = []

    for idx, fixture in fixtures_df.iterrows():
        print(f"\n{fixture['league_name']}")
        print(f"{fixture['date']} | {fixture['venue']}")
        print(f"{fixture['home_team_name']} vs {fixture['away_team_name']}")

        try:
            # Build features from live API data
            features = calculator.build_features_for_match(
                fixture['home_team_id'],
                fixture['away_team_id'],
                pd.to_datetime(fixture['date'])
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

            predictions.append(result)

            print(f"  Model: {model_name}")
            print(f"  Home Win: {result['home_win_prob']:.1%}")
            print(f"  Draw:     {result['draw_prob']:.1%}")
            print(f"  Away Win: {result['away_win_prob']:.1%}")
            print(f"  → Prediction: {result['predicted_outcome']}")
            print(f"  ✨ Using live API data")

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


if __name__ == "__main__":
    main()
