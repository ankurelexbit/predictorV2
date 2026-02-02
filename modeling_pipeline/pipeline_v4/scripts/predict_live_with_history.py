#!/usr/bin/env python3
"""
Production Live Prediction Pipeline with Historical Context
===========================================================

Loads 1 year of historical data on startup to properly initialize:
- Elo ratings for all teams
- Statistics for derived xG calculation
- Form and standings data

Uses the SAME methods as training (SportMonksClient + statistics parsing).

Uses conservative model with draw weights (1.2/1.5/1.0).

Usage:
    export SPORTMONKS_API_KEY="your_key"

    # Initialize and verify
    python3 scripts/predict_live_with_history.py --verify

    # Predict upcoming fixtures
    python3 scripts/predict_live_with_history.py --predict-upcoming --days-ahead 7
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
from typing import List, Dict, Optional
import joblib
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sportmonks_client import SportMonksClient
from src.features import (
    EloCalculator,
    Pillar1FundamentalsEngine,
    Pillar2ModernAnalyticsEngine,
    Pillar3HiddenEdgesEngine,
)
from src.features.standings_calculator import StandingsCalculator

try:
    from config.model_config import EloConfig
except ImportError:
    # Fallback to defaults if config not available
    class EloConfig:
        K_FACTOR = 32
        HOME_ADVANTAGE = 35
        INITIAL_ELO = 1500

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PRODUCTION_MODEL_PATH = 'models/with_draw_features/conservative_with_draw_features.joblib'


def extract_statistics_from_fixture(fixture: dict) -> dict:
    """
    Extract all statistics from a fixture into flat dictionary.

    This is the SAME function used in convert_json_to_csv.py for training data.
    """
    stats_dict = {}

    # Get statistics array
    statistics = fixture.get('statistics', [])

    # Map of type_id to stat name (from SportMonks API)
    stat_type_map = {
        42: 'shots_total',
        86: 'shots_on_target',
        41: 'shots_off_target',
        58: 'shots_blocked',
        49: 'shots_inside_box',
        50: 'shots_outside_box',
        56: 'fouls',
        34: 'corners',
        51: 'offsides',
        45: 'ball_possession',
        84: 'yellow_cards',
        83: 'red_cards',
        78: 'tackles',
        100: 'interceptions',
        101: 'clearances',
        80: 'passes_total',
        81: 'passes_accurate',
        82: 'passes_percentage',
        43: 'attacks',
        44: 'dangerous_attacks',
    }

    # Initialize all stats to None
    for side in ['home', 'away']:
        for stat_name in stat_type_map.values():
            stats_dict[f'{side}_{stat_name}'] = None

    # Extract statistics
    for stat in statistics:
        type_id = stat.get('type_id')
        location = stat.get('location', '').lower()  # 'home' or 'away'
        value = stat.get('data', {}).get('value')

        if type_id in stat_type_map and location in ['home', 'away']:
            stat_name = stat_type_map[type_id]
            stats_dict[f'{location}_{stat_name}'] = value

    return stats_dict


class InMemoryDataLoader:
    """Lightweight in-memory data loader matching JSONDataLoader interface."""

    def __init__(self):
        self._fixtures_df = None

    def add_fixtures(self, fixtures_df: pd.DataFrame):
        """Add fixtures to cache."""
        if self._fixtures_df is None:
            self._fixtures_df = fixtures_df.copy()
        else:
            self._fixtures_df = pd.concat([self._fixtures_df, fixtures_df], ignore_index=True)

        # Sort by date for chronological processing
        self._fixtures_df = self._fixtures_df.sort_values('starting_at').reset_index(drop=True)

    @property
    def fixtures_df(self):
        return self._fixtures_df if self._fixtures_df is not None else pd.DataFrame()

    def get_fixtures_before_date(self, before_date: datetime) -> pd.DataFrame:
        """Get fixtures before a specific date."""
        if self._fixtures_df is None or len(self._fixtures_df) == 0:
            return pd.DataFrame()

        mask = pd.to_datetime(self._fixtures_df['starting_at']) < before_date
        return self._fixtures_df[mask].copy()

    def get_fixtures_before(self, before_date: datetime, league_id: int = None,
                           season_id: int = None) -> pd.DataFrame:
        """Get fixtures before a date with optional filters."""
        if self._fixtures_df is None or len(self._fixtures_df) == 0:
            return pd.DataFrame()

        mask = pd.to_datetime(self._fixtures_df['starting_at']) < before_date

        if league_id is not None:
            mask = mask & (self._fixtures_df['league_id'] == league_id)

        if season_id is not None:
            mask = mask & (self._fixtures_df['season_id'] == season_id)

        return self._fixtures_df[mask].copy()

    def get_team_fixtures(self, team_id: int, before_date: datetime,
                         limit: int = None) -> pd.DataFrame:
        """Get team's fixtures before a date."""
        if self._fixtures_df is None or len(self._fixtures_df) == 0:
            return pd.DataFrame()

        mask = (
            (pd.to_datetime(self._fixtures_df['starting_at']) < before_date) &
            ((self._fixtures_df['home_team_id'] == team_id) |
             (self._fixtures_df['away_team_id'] == team_id))
        )

        team_fixtures = self._fixtures_df[mask].sort_values('starting_at', ascending=False)

        if limit:
            team_fixtures = team_fixtures.head(limit)

        return team_fixtures.copy()

    def get_fixture(self, fixture_id: int) -> Optional[Dict]:
        """Get a single fixture by ID."""
        if self._fixtures_df is None or len(self._fixtures_df) == 0:
            return None

        matches = self._fixtures_df[self._fixtures_df['id'] == fixture_id]
        if len(matches) == 0:
            return None

        return matches.iloc[0].to_dict()

    def get_statistics(self, fixture_id: int) -> List:
        """Get statistics for a fixture (not used as stats are in DataFrame)."""
        return []

    def get_lineups(self, fixture_id: int) -> List:
        """Get lineups for a fixture (not implemented)."""
        return []


class HistoricalDataFetcher:
    """Fetches historical data using the SAME methods as training."""

    def __init__(self, api_key: str):
        self.client = SportMonksClient(api_key=api_key)

    def fetch_historical_fixtures(
        self,
        days_back: int = 365,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical fixtures with statistics.

        Uses SportMonksClient.get_fixtures_between() which includes statistics
        embedded in the response - exactly like training data download!

        Args:
            days_back: Days of history to fetch (default: 365, ignored if start_date provided)
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format, defaults to today)

        Returns:
            DataFrame with fixtures and statistics
        """
        # Use provided dates if available, otherwise calculate from days_back
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = datetime.now() - timedelta(days=days_back)

        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = datetime.now()

        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')

        logger.info(f"Fetching historical fixtures from {start_str} to {end_str}")

        # Fetch in 30-day chunks (smaller chunks for faster API responses)
        all_fixtures = []
        current_start = start_dt
        chunk_num = 1

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=30), end_dt)
            chunk_start_str = current_start.strftime('%Y-%m-%d')
            chunk_end_str = current_end.strftime('%Y-%m-%d')

            logger.info(f"   Chunk {chunk_num}: {chunk_start_str} to {chunk_end_str}...")
            logger.info(f"      Making API call...")

            try:
                import time
                start_time = time.time()

                # Uses SAME method as backfill_historical_data.py!
                # Statistics are embedded in the response
                # Filter at API level for finished fixtures only (state_id=5)
                fixtures = self.client.get_fixtures_between(
                    chunk_start_str,
                    chunk_end_str,
                    include_details=True,  # Includes statistics, lineups, etc.
                    finished_only=True  # Only finished fixtures
                )

                elapsed = time.time() - start_time
                logger.info(f"      API call completed in {elapsed:.1f}s")

                logger.info(f"      Got {len(fixtures)} finished fixtures")
                all_fixtures.extend(fixtures)

            except Exception as e:
                logger.error(f"      API call failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Continue with next chunk

            current_start = current_end + timedelta(days=1)
            chunk_num += 1

        if not all_fixtures:
            logger.warning("   ⚠️  No historical fixtures found")
            return pd.DataFrame()

        logger.info(f"   ✅ Fetched {len(all_fixtures)} fixtures total")

        # Convert to DataFrame using SAME logic as convert_json_to_csv.py
        return self._fixtures_to_dataframe(all_fixtures)

    def _fixtures_to_dataframe(self, fixtures: List[Dict]) -> pd.DataFrame:
        """
        Convert fixtures to DataFrame with statistics.

        Uses SAME logic as convert_json_to_csv.py for training data.
        """
        logger.info("   Converting fixtures to DataFrame...")

        fixture_records = []

        for fixture in fixtures:
            # Extract basic fixture info
            participants = fixture.get('participants', [])
            if len(participants) != 2:
                continue

            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if not home_team or not away_team:
                continue

            # Get scores
            scores = fixture.get('scores', [])
            home_score = None
            away_score = None

            for score in scores:
                if score.get('description') == 'CURRENT':
                    participant = score.get('score', {}).get('participant')
                    goals = score.get('score', {}).get('goals')

                    if participant == 'home':
                        home_score = goals
                    elif participant == 'away':
                        away_score = goals

            # Determine result
            result = None
            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    result = 'H'
                elif home_score < away_score:
                    result = 'A'
                else:
                    result = 'D'

            # Extract statistics using SAME function as training!
            stats = extract_statistics_from_fixture(fixture)

            # Combine all data
            fixture_data = {
                'id': fixture.get('id'),
                'league_id': fixture.get('league_id'),
                'season_id': fixture.get('season_id'),
                'starting_at': fixture.get('starting_at'),
                'home_team_id': home_team.get('id'),
                'home_team_name': home_team.get('name'),
                'away_team_id': away_team.get('id'),
                'away_team_name': away_team.get('name'),
                'home_score': home_score,
                'away_score': away_score,
                'result': result,
                'state_id': fixture.get('state_id'),
                **stats,  # Add all statistics columns (home_shots_total, etc.)
            }

            fixture_records.append(fixture_data)

        df = pd.DataFrame(fixture_records)

        # Deduplicate by ID
        df = df.drop_duplicates(subset=['id'])

        # Parse dates
        df['starting_at'] = pd.to_datetime(df['starting_at'])

        # Sort by date
        df = df.sort_values('starting_at').reset_index(drop=True)

        logger.info(f"   ✅ Converted to DataFrame: {len(df)} fixtures")
        logger.info(f"      Columns: {len(df.columns)} (including statistics)")

        return df


class ProductionLivePipeline:
    """Production pipeline with historical context."""

    def __init__(
        self,
        api_key: str,
        load_history_days: int = 365,
        history_start_date: str = None,
        history_end_date: str = None
    ):
        """
        Initialize production pipeline.

        Args:
            api_key: SportMonks API key
            load_history_days: Days of history to load (default: 365, ignored if history_start_date provided)
            history_start_date: Optional start date for historical data (YYYY-MM-DD)
            history_end_date: Optional end date for historical data (YYYY-MM-DD, defaults to today)
        """
        self.api_key = api_key
        self.base_url = "https://api.sportmonks.com/v3/football"

        logger.info("=" * 80)
        logger.info("INITIALIZING PRODUCTION PIPELINE")
        logger.info("=" * 80)

        # Fetch historical data using SAME methods as training
        fetcher = HistoricalDataFetcher(api_key)
        historical_df = fetcher.fetch_historical_fixtures(
            days_back=load_history_days,
            start_date=history_start_date,
            end_date=history_end_date
        )

        # Initialize data loader
        self.data_loader = InMemoryDataLoader()
        if len(historical_df) > 0:
            self.data_loader.add_fixtures(historical_df)
            logger.info(f"   ✅ Loaded {len(historical_df)} historical fixtures")
        else:
            logger.warning("   ⚠️  No historical data loaded")

        logger.info("\n📊 Initializing calculators...")

        # Initialize Elo calculator
        self.elo_calculator = EloCalculator(
            k_factor=EloConfig.K_FACTOR,
            home_advantage=EloConfig.HOME_ADVANTAGE,
            initial_elo=EloConfig.INITIAL_ELO
        )
        if len(historical_df) > 0:
            self.elo_calculator.calculate_elo_history(historical_df)
            logger.info(f"   ✅ Calculated Elo for {len(self.elo_calculator.elo_history)} teams")
        else:
            logger.warning("   ⚠️  No historical data - Elo will use defaults")

        # Initialize standings calculator
        self.standings_calculator = StandingsCalculator()

        # Initialize feature engines
        logger.info("\n🏗️  Initializing feature engines...")
        self.pillar1_engine = Pillar1FundamentalsEngine(
            self.data_loader,
            self.standings_calculator,
            self.elo_calculator
        )
        self.pillar2_engine = Pillar2ModernAnalyticsEngine(self.data_loader)
        self.pillar3_engine = Pillar3HiddenEdgesEngine(
            self.data_loader,
            self.standings_calculator,
            self.elo_calculator
        )

        # Model will be loaded separately
        self.model = None

        logger.info("=" * 80)
        logger.info("✅ PIPELINE INITIALIZED")
        logger.info("=" * 80)

    def load_model(self, model_path: str = PRODUCTION_MODEL_PATH):
        """Load trained model."""
        logger.info(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        logger.info(f"✅ Loaded model from {model_path}")

    def predict(self, fixture: Dict) -> Optional[Dict]:
        """
        Generate prediction for a fixture.

        Args:
            fixture: Fixture dict from API

        Returns:
            Prediction dict with probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Extract fixture info
        participants = fixture.get('participants', [])
        if len(participants) != 2:
            return None

        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

        if not home_team or not away_team:
            return None

        # Create fixture row for feature generation
        fixture_row = pd.Series({
            'id': fixture.get('id'),
            'home_team_id': home_team.get('id'),
            'away_team_id': away_team.get('id'),
            'starting_at': pd.to_datetime(fixture.get('starting_at')),
            'league_id': fixture.get('league_id'),
            'season_id': fixture.get('season_id'),
        })

        # Generate features
        try:
            features = {}

            # Pillar 1: Fundamentals
            p1_features = self.pillar1_engine.generate_features(
                home_team_id=fixture_row['home_team_id'],
                away_team_id=fixture_row['away_team_id'],
                season_id=fixture_row['season_id'],
                league_id=fixture_row['league_id'],
                as_of_date=fixture_row['starting_at'],
                fixtures_df=self.data_loader.fixtures_df
            )
            features.update(p1_features)

            # Pillar 2: Modern Analytics
            p2_features = self.pillar2_engine.generate_features(
                home_team_id=fixture_row['home_team_id'],
                away_team_id=fixture_row['away_team_id'],
                as_of_date=fixture_row['starting_at']
            )
            features.update(p2_features)

            # Pillar 3: Hidden Edges
            p3_features = self.pillar3_engine.generate_features(
                home_team_id=fixture_row['home_team_id'],
                away_team_id=fixture_row['away_team_id'],
                season_id=fixture_row['season_id'],
                league_id=fixture_row['league_id'],
                as_of_date=fixture_row['starting_at'],
                fixtures_df=self.data_loader.fixtures_df
            )
            features.update(p3_features)

            # Convert to DataFrame
            X = pd.DataFrame([features])

            # Predict probabilities
            probs = self.model.predict_proba(X)[0]

            # Model outputs: [Away, Draw, Home]
            return {
                'home_prob': float(probs[2]),
                'draw_prob': float(probs[1]),
                'away_prob': float(probs[0]),
                'features': features,  # Include all features used
            }

        except Exception as e:
            logger.error(f"Error generating features for fixture {fixture.get('id')}: {e}")
            return None


def main():
    """Test the pipeline."""
    parser = argparse.ArgumentParser(description='Production prediction with historical context')
    parser.add_argument('--verify', action='store_true', help='Verify pipeline initialization')
    parser.add_argument('--history-days', type=int, default=365, help='Days of history to load (default: 365)')

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        logger.error("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    # Initialize pipeline
    pipeline = ProductionLivePipeline(api_key, load_history_days=args.history_days)

    # Load model
    pipeline.load_model()

    if args.verify:
        # Verification mode
        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION")
        logger.info("=" * 80 + "\n")

        # Check Elo ratings
        logger.info("Elo Ratings for Sample Teams:")
        sample_team_ids = [2, 8, 83, 85]  # Sample team IDs
        for team_id in sample_team_ids:
            elo = pipeline.elo_calculator.get_elo_at_date(team_id, datetime.now())
            if elo:
                logger.info(f"   Team {team_id}: {elo:.1f}")
            else:
                logger.info(f"   Team {team_id}: Not found (will use default 1500)")

        logger.info("\n✅ Pipeline ready for production\n")


if __name__ == '__main__':
    main()
