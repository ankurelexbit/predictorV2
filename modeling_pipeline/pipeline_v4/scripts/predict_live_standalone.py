#!/usr/bin/env python3
"""
Standalone Live Prediction Pipeline
====================================

Production-ready live prediction that fetches ALL data on-the-fly from API.
REUSES existing pillar engines from src/features/ - NO code duplication!

Key Features:
- Fetches upcoming fixtures from API
- Downloads recent team matches on-the-fly for rolling stats
- Uses SAME pillar engines as training (feature parity guaranteed)
- Completely independent from training pipeline data
- Suitable for containerized deployment

Usage:
    python3 scripts/predict_live_standalone.py --date today
    python3 scripts/predict_live_standalone.py --date 2026-02-15 --league-id 8
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import argparse
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing pillar engines (NO code duplication!)
from src.features.pillar1_fundamentals import Pillar1FundamentalsEngine
from src.features.pillar2_modern_analytics import Pillar2ModernAnalyticsEngine
from src.features.pillar3_hidden_edges import Pillar3HiddenEdgesEngine
from src.features.elo_calculator import EloCalculator
from src.features.standings_calculator import StandingsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PRODUCTION_MODEL_PATH = 'models/with_draw_features/xgboost_fixed.joblib'  # Fixed model without class weight bias
HISTORICAL_LOOKBACK_DAYS = 180  # How far back to fetch team history (accounts for winter breaks)


class InMemoryDataLoader:
    """
    Lightweight data loader for API-fetched data.
    Implements same interface as JSONDataLoader for pillar engine compatibility.
    """

    def __init__(self):
        self.fixtures_cache = {}  # fixture_id -> fixture dict
        self.statistics_cache = {}  # fixture_id -> statistics list
        self.lineups_cache = {}  # fixture_id -> lineups list
        self._fixtures_df = None  # DataFrame cache

    def add_fixtures(self, fixtures_df: pd.DataFrame):
        """Add fixtures to in-memory cache."""
        for _, row in fixtures_df.iterrows():
            fixture_dict = row.to_dict()
            fixture_id = row['fixture_id'] if 'fixture_id' in row else row.get('id')
            self.fixtures_cache[fixture_id] = fixture_dict

            # Cache statistics and lineups if present
            if 'statistics' in fixture_dict:
                self.statistics_cache[fixture_id] = fixture_dict['statistics']
            if 'lineups' in fixture_dict:
                self.lineups_cache[fixture_id] = fixture_dict['lineups']

        # Update DataFrame cache
        self._fixtures_df = fixtures_df

    def load_all_fixtures(self) -> pd.DataFrame:
        """
        Return all cached fixtures as DataFrame.
        Compatible with JSONDataLoader interface.
        """
        if self._fixtures_df is not None:
            return self._fixtures_df

        # Build DataFrame from cache
        if self.fixtures_cache:
            df = pd.DataFrame(list(self.fixtures_cache.values()))
            # Ensure starting_at is datetime
            if 'starting_at' in df.columns:
                df['starting_at'] = pd.to_datetime(df['starting_at'])
            self._fixtures_df = df
            return df

        return pd.DataFrame()

    def get_fixture(self, fixture_id: int):
        """Get fixture by ID."""
        return self.fixtures_cache.get(fixture_id)

    def get_fixtures_before(
        self,
        as_of_date: datetime,
        league_id: Optional[int] = None,
        season_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get all fixtures before a specific date.
        Compatible with JSONDataLoader interface.
        """
        df = self.load_all_fixtures()

        if df.empty:
            return df

        # Filter by date
        mask = df['starting_at'] < as_of_date

        # Filter by league
        if league_id is not None:
            mask &= (df['league_id'] == league_id)

        # Filter by season
        if season_id is not None:
            mask &= (df['season_id'] == season_id)

        return df[mask].copy()

    def get_team_fixtures(
        self,
        team_id: int,
        before_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get fixtures for a specific team before a date.
        Compatible with JSONDataLoader interface.
        """
        df = self.load_all_fixtures()

        if df.empty:
            return df

        # Filter by team and date
        mask = (
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['starting_at'] < before_date)
        )

        # Only include completed fixtures (with results)
        if 'result' in df.columns:
            mask &= df['result'].notna()

        team_fixtures = df[mask].sort_values('starting_at', ascending=False)

        if limit is not None:
            team_fixtures = team_fixtures.head(limit)

        return team_fixtures

    def get_statistics(self, fixture_id: int):
        """Get statistics for fixture."""
        # Try cache first
        if fixture_id in self.statistics_cache:
            return self.statistics_cache[fixture_id]

        # Fallback to fixture cache
        fixture = self.fixtures_cache.get(fixture_id, {})
        return fixture.get('statistics', [])

    def get_lineups(self, fixture_id: int):
        """Get lineups for fixture."""
        # Try cache first
        if fixture_id in self.lineups_cache:
            return self.lineups_cache[fixture_id]

        # Fallback to fixture cache
        fixture = self.fixtures_cache.get(fixture_id, {})
        return fixture.get('lineups', [])


class APIStandingsCalculator(StandingsCalculator):
    """
    Standings calculator that uses API data instead of calculating from fixtures.

    CRITICAL: Standalone pipeline doesn't have all league fixtures, so we fetch
    standings directly from SportMonks API instead of calculating.
    """

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.api_standings_cache = {}  # season_id -> DataFrame

    def calculate_standings_at_date(
        self,
        fixtures_df: pd.DataFrame,
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> pd.DataFrame:
        """
        Override to fetch standings from API instead of calculating from fixtures.

        This is necessary because fixtures_df only contains ~40 fixtures (2 teams),
        but accurate standings require ALL fixtures in the league (~250+).
        """
        # Check cache
        if season_id in self.api_standings_cache:
            return self.api_standings_cache[season_id].copy()

        # Fetch from API
        standings_df = self.api_client.fetch_season_standings(season_id)

        if standings_df is not None:
            # Cache it
            self.api_standings_cache[season_id] = standings_df.copy()
            return standings_df

        # Fallback to parent method (will be inaccurate but better than nothing)
        logger.warning(f"  ⚠️ Failed to fetch API standings, using calculated (may be inaccurate)")
        return super().calculate_standings_at_date(fixtures_df, season_id, league_id, as_of_date)


class StandaloneLivePipeline:
    """Standalone live prediction pipeline using existing pillar engines."""

    def __init__(self, api_key: str):
        """Initialize pipeline with API key."""
        self.api_key = api_key
        self.base_url = "https://api.sportmonks.com/v3/football"
        self.model = None

        # Create in-memory data loader
        self.data_loader = InMemoryDataLoader()

        # Initialize calculators (use API-based standings!)
        self.elo_calculator = EloCalculator()
        self.standings_calculator = APIStandingsCalculator(self)  # Pass self for API access

        # Initialize pillar engines (reuse existing code!)
        self.pillar1 = Pillar1FundamentalsEngine(self.data_loader, self.standings_calculator, self.elo_calculator)
        self.pillar2 = Pillar2ModernAnalyticsEngine(self.data_loader)
        self.pillar3 = Pillar3HiddenEdgesEngine(self.data_loader, self.standings_calculator, self.elo_calculator)

        logger.info("✅ Standalone pipeline initialized (using API-based standings)")

    def _api_call(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API call with error handling."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params['api_token'] = self.api_key

        try:
            response = requests.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API call failed for {endpoint}: {e}")
            return None

    def load_model(self, model_path: str = PRODUCTION_MODEL_PATH):
        """Load production model."""
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_file)
        logger.info(f"✅ Loaded model from {model_path}")

    def fetch_upcoming_fixtures(
        self,
        start_date: str,
        end_date: str,
        league_id: Optional[int] = None
    ) -> List[Dict]:
        """Fetch upcoming fixtures for date range."""
        logger.info(f"Fetching fixtures: {start_date} to {end_date}")

        endpoint = f"fixtures/between/{start_date}/{end_date}"
        params = {'include': 'participants;league;state'}

        if league_id:
            params['filters'] = f'fixtureLeagues:{league_id}'

        data = self._api_call(endpoint, params)

        if not data or 'data' not in data:
            logger.warning("No fixtures found")
            return []

        fixtures = []
        for fixture in data['data']:
            # Only include upcoming fixtures
            state_id = fixture.get('state_id')
            if state_id not in [1, 18]:  # 1=NS, 18=TBD
                continue

            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            fixtures.append({
                'fixture_id': fixture['id'],
                'starting_at': fixture.get('starting_at'),
                'league_id': fixture.get('league_id'),
                'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                'season_id': fixture.get('season_id'),
                'home_team_id': home_team['id'],
                'home_team_name': home_team['name'],
                'away_team_id': away_team['id'],
                'away_team_name': away_team['name']
            })

        logger.info(f"✅ Found {len(fixtures)} upcoming fixtures")
        return fixtures

    def fetch_team_recent_matches(
        self,
        team_id: int,
        end_date: str,
        num_matches: int = 20,
        days_back: int = 180
    ) -> List[Dict]:
        """
        Fetch team's recent completed matches.

        Uses /fixtures/between/{startDate}/{endDate}/{teamID} endpoint.

        Args:
            team_id: Team ID
            end_date: End date for search
            num_matches: Maximum number of matches to return
            days_back: How many days back to search (default 180 for winter breaks)
        """
        # Go back N days to ensure we get enough matches
        # 180 days is safer than 120 to account for winter breaks in some leagues
        end_dt = datetime.strptime(end_date[:10], '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days_back)
        start_date = start_dt.strftime('%Y-%m-%d')

        logger.debug(f"  Fetching matches for team {team_id}")

        endpoint = f"fixtures/between/{start_date}/{end_date}/{team_id}"
        params = {
            'include': 'participants;scores;statistics;state',
            'filters': 'fixtureStates:5'  # Only finished matches
        }

        data = self._api_call(endpoint, params)

        if not data or 'data' not in data:
            logger.warning(f"  No matches found for team {team_id}")
            return []

        matches = []
        for fixture in data['data']:
            if fixture.get('state_id') != 5:  # Double-check FT
                continue

            participants = fixture.get('participants', [])
            if len(participants) < 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            # Get scores
            scores = fixture.get('scores', [])
            home_score = 0
            away_score = 0

            for score in scores:
                if score.get('description') == 'CURRENT':
                    score_data = score.get('score', {})
                    if score_data.get('participant') == 'home':
                        home_score = score_data.get('goals', 0)
                    elif score_data.get('participant') == 'away':
                        away_score = score_data.get('goals', 0)

            # Convert to format expected by pillar engines
            match = {
                'fixture_id': fixture['id'],
                'starting_at': fixture.get('starting_at'),
                'home_team_id': home_team['id'],
                'away_team_id': away_team['id'],
                'home_score': home_score,
                'away_score': away_score,
                'result': 'H' if home_score > away_score else ('A' if away_score > home_score else 'D'),
                'state_id': 5,
                'league_id': fixture.get('league_id'),
                'season_id': fixture.get('season_id'),
                'statistics': fixture.get('statistics', [])
            }

            matches.append(match)

        # Sort by date (most recent first) and limit
        matches.sort(key=lambda x: x['starting_at'], reverse=True)
        return matches[:num_matches]

    def fetch_season_standings(self, season_id: int) -> Optional[pd.DataFrame]:
        """
        Fetch current standings for a season from SportMonks API.

        CRITICAL: This fetches ACTUAL standings from API, not calculated from fixtures.
        This is necessary because standalone pipeline doesn't have all league fixtures.

        Args:
            season_id: Season ID

        Returns:
            DataFrame with standings or None if error
        """
        logger.debug(f"  Fetching standings for season {season_id}")

        endpoint = f"standings/seasons/{season_id}"
        # Include details to get played, wins, draws, losses, etc.
        params = {'include': 'details'}
        data = self._api_call(endpoint, params)

        if not data or 'data' not in data:
            logger.warning(f"  ⚠️ No standings found for season {season_id}")
            return None

        standings_list = []
        for entry in data['data']:
            # Parse details array to get stats (API returns them with type_ids)
            details = entry.get('details', [])
            details_dict = {d['type_id']: d['value'] for d in details} if details else {}

            # Type IDs from SportMonks API:
            # 185 = Matches played, 129 = Wins, 130 = Draws, 132 = Losses
            # 133 = Goals for, 134 = Goals against, 179 = Goal difference
            played = details_dict.get(185, 0)
            wins = details_dict.get(129, 0)
            draws = details_dict.get(130, 0)
            losses = details_dict.get(132, 0)
            goals_for = details_dict.get(133, 0)
            goals_against = details_dict.get(134, 0)
            goal_diff = details_dict.get(179, 0)

            # If played is still 0, try to infer from wins+draws+losses
            if played == 0 and (wins > 0 or draws > 0 or losses > 0):
                played = wins + draws + losses

            standings_list.append({
                'team_id': entry.get('participant_id'),
                'position': entry.get('position'),
                'points': entry.get('points'),
                'played': played,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goal_diff,
            })

        if not standings_list:
            return None

        df = pd.DataFrame(standings_list)
        # Calculate points per game, handle division by zero
        df['points_per_game'] = df.apply(
            lambda row: row['points'] / row['played'] if row['played'] > 0 else 0,
            axis=1
        )

        logger.debug(f"  ✓ Fetched standings with {len(df)} teams")
        return df

    def build_fixtures_dataframe(
        self,
        home_matches: List[Dict],
        away_matches: List[Dict],
        upcoming_fixture: Dict
    ) -> pd.DataFrame:
        """
        Build fixtures DataFrame from API data.
        This DataFrame is what the pillar engines expect.
        """
        # Combine all matches
        all_matches = home_matches + away_matches

        # Remove duplicates (same fixture might appear in both teams' history)
        seen_ids = set()
        unique_matches = []
        for match in all_matches:
            if match['fixture_id'] not in seen_ids:
                unique_matches.append(match)
                seen_ids.add(match['fixture_id'])

        # Add upcoming fixture (without result) for context
        unique_matches.append({
            'fixture_id': upcoming_fixture['fixture_id'],
            'starting_at': upcoming_fixture['starting_at'],
            'home_team_id': upcoming_fixture['home_team_id'],
            'away_team_id': upcoming_fixture['away_team_id'],
            'league_id': upcoming_fixture['league_id'],
            'season_id': upcoming_fixture['season_id'],
            'home_score': None,
            'away_score': None,
            'result': None,
            'state_id': 1,
            'statistics': []
        })

        # Convert to DataFrame
        df = pd.DataFrame(unique_matches)
        df['starting_at'] = pd.to_datetime(df['starting_at'])

        return df

    def generate_features(self, fixture: Dict) -> Optional[Dict]:
        """
        Generate all 162 features using existing pillar engines.
        NO code duplication - reuses src/features/ code!
        """
        logger.info(f"Generating features: {fixture['home_team_name']} vs {fixture['away_team_name']}")

        # Fetch recent matches for both teams
        end_date = fixture['starting_at'][:10]

        home_matches = self.fetch_team_recent_matches(
            fixture['home_team_id'],
            end_date,
            num_matches=20,
            days_back=HISTORICAL_LOOKBACK_DAYS
        )

        away_matches = self.fetch_team_recent_matches(
            fixture['away_team_id'],
            end_date,
            num_matches=20,
            days_back=HISTORICAL_LOOKBACK_DAYS
        )

        # Validate we have enough matches for reliable rolling stats
        if not home_matches or not away_matches:
            logger.warning("  ✗ No match data found")
            return None

        if len(home_matches) < 10:
            logger.warning(f"  ⚠️  Home team ({fixture['home_team_name']}) only has {len(home_matches)} matches (need 10+)")
        if len(away_matches) < 10:
            logger.warning(f"  ⚠️  Away team ({fixture['away_team_name']}) only has {len(away_matches)} matches (need 10+)")

        logger.debug(f"  Fetched {len(home_matches)} home matches, {len(away_matches)} away matches")

        # Build fixtures DataFrame (format expected by pillar engines)
        fixtures_df = self.build_fixtures_dataframe(home_matches, away_matches, fixture)

        # Add fixtures to data loader cache
        self.data_loader.add_fixtures(fixtures_df)

        # Parse as_of_date
        as_of_date = pd.to_datetime(fixture['starting_at'])

        try:
            # Generate features using EXISTING pillar engines (no duplication!)
            features = {}

            # Pillar 1: Fundamentals (50 features)
            logger.debug("  Generating Pillar 1 features...")
            features.update(self.pillar1.generate_features(
                fixture['home_team_id'],
                fixture['away_team_id'],
                fixture['season_id'],
                fixture['league_id'],
                as_of_date,
                fixtures_df
            ))

            # Pillar 2: Modern Analytics (60 features)
            logger.debug("  Generating Pillar 2 features...")
            features.update(self.pillar2.generate_features(
                fixture['home_team_id'],
                fixture['away_team_id'],
                as_of_date
            ))

            # Pillar 3: Hidden Edges (52 features including 12 draw features)
            logger.debug("  Generating Pillar 3 features...")
            features.update(self.pillar3.generate_features(
                fixture['home_team_id'],
                fixture['away_team_id'],
                fixture['season_id'],
                fixture['league_id'],
                as_of_date,
                fixtures_df
            ))

            logger.info(f"  ✓ Generated {len(features)} features")
            return features

        except Exception as e:
            logger.error(f"  ✗ Error generating features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def make_predictions(self, fixtures: List[Dict]) -> pd.DataFrame:
        """Generate features and make predictions for all fixtures."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("="*80)
        logger.info(f"GENERATING FEATURES & PREDICTIONS FOR {len(fixtures)} FIXTURES")
        logger.info("="*80)

        results = []

        for fixture in fixtures:
            try:
                # Generate features using existing pillar engines
                features = self.generate_features(fixture)

                if not features:
                    continue

                # Convert to DataFrame
                feature_df = pd.DataFrame([features])

                # Get feature columns (exclude metadata)
                feature_cols = [c for c in feature_df.columns if c not in [
                    'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
                    'match_date', 'home_score', 'away_score', 'result', 'target',
                    'home_team_name', 'away_team_name', 'state_id', 'league_name'
                ]]

                X = feature_df[feature_cols]

                # Make prediction
                probabilities = self.model.predict_proba(X)[0]
                prediction = np.argmax(probabilities)

                # Store result
                results.append({
                    'fixture_id': fixture['fixture_id'],
                    'match_date': fixture['starting_at'],
                    'league_name': fixture['league_name'],
                    'home_team_name': fixture['home_team_name'],
                    'away_team_name': fixture['away_team_name'],
                    'away_win_prob': probabilities[0],
                    'draw_prob': probabilities[1],
                    'home_win_prob': probabilities[2],
                    'predicted_outcome': prediction,
                    'predicted_outcome_label': ['Away Win', 'Draw', 'Home Win'][prediction],
                    'confidence': probabilities.max()
                })

                logger.info(f"  ✓ {fixture['home_team_name']} vs {fixture['away_team_name']}")

            except Exception as e:
                logger.error(f"  ✗ Error for {fixture.get('home_team_name')} vs {fixture.get('away_team_name')}: {e}")

        return pd.DataFrame(results)

    def display_results(self, predictions_df: pd.DataFrame):
        """Display predictions in readable format."""
        logger.info("\n" + "="*80)
        logger.info("PREDICTIONS")
        logger.info("="*80)

        for idx, row in predictions_df.iterrows():
            print(f"\n{row['league_name']}")
            print(f"{row['match_date']}")
            print(f"{row['home_team_name']} vs {row['away_team_name']}")
            print(f"  Home Win: {row['home_win_prob']:.1%}")
            print(f"  Draw:     {row['draw_prob']:.1%}")
            print(f"  Away Win: {row['away_win_prob']:.1%}")
            print(f"  → Prediction: {row['predicted_outcome_label']} (confidence: {row['confidence']:.1%})")


def parse_date(date_str: str) -> str:
    """Parse date string to YYYY-MM-DD format."""
    if date_str.lower() == 'today':
        return datetime.now().strftime('%Y-%m-%d')
    elif date_str.lower() == 'tomorrow':
        return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")


def main():
    import os

    parser = argparse.ArgumentParser(
        description='Standalone live prediction pipeline (API-only, reuses existing pillar engines)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--date', help='Single date (YYYY-MM-DD, today, tomorrow)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--league-id', type=int, help='League ID filter')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--model', default=PRODUCTION_MODEL_PATH, help='Model path')

    args = parser.parse_args()

    # Validate arguments
    if args.date and (args.start_date or args.end_date):
        parser.error("Cannot use --date with --start-date/--end-date")

    if not args.date and not (args.start_date and args.end_date):
        parser.error("Must provide either --date or both --start-date and --end-date")

    # Check API key
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        logger.error("SPORTMONKS_API_KEY environment variable not set")
        return 1

    # Parse dates
    if args.date:
        start_date = parse_date(args.date)
        end_date = start_date
    else:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)

    # Initialize pipeline
    logger.info("="*80)
    logger.info("STANDALONE LIVE PREDICTION PIPELINE")
    logger.info("(Using existing pillar engines - NO code duplication)")
    logger.info("="*80)
    logger.info(f"Date range: {start_date} to {end_date}")
    if args.league_id:
        logger.info(f"League filter: {args.league_id}")

    pipeline = StandaloneLivePipeline(api_key)

    # Load model
    pipeline.load_model(args.model)

    # Fetch upcoming fixtures
    fixtures = pipeline.fetch_upcoming_fixtures(start_date, end_date, args.league_id)

    if not fixtures:
        logger.warning("No fixtures found")
        return 1

    # Make predictions
    predictions_df = pipeline.make_predictions(fixtures)

    if len(predictions_df) == 0:
        logger.warning("No predictions generated")
        return 1

    # Display results
    pipeline.display_results(predictions_df)

    # Save to CSV
    if args.output:
        predictions_df.to_csv(args.output, index=False)
        logger.info(f"\n✅ Predictions saved to {args.output}")

    logger.info("\n" + "="*80)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
