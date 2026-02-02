#!/usr/bin/env python3
"""
Live Prediction Pipeline for V4
================================

Downloads live fixtures from SportMonks API and makes predictions using
the V4 production model with 162-feature engineering framework.

Usage:
    # Predict for a specific date
    python3 scripts/predict_live_v4.py --date 2026-02-15

    # Predict for a date range
    python3 scripts/predict_live_v4.py --start-date 2026-02-15 --end-date 2026-02-17

    # Predict for today
    python3 scripts/predict_live_v4.py --date today

    # Predict for tomorrow
    python3 scripts/predict_live_v4.py --date tomorrow

    # Save predictions to CSV
    python3 scripts/predict_live_v4.py --date today --output predictions.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import argparse
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sportmonks_client import SportMonksClient
from src.features.feature_orchestrator import FeatureOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SPORTMONKS_API_KEY = os.environ.get('SPORTMONKS_API_KEY')
PRODUCTION_MODEL_PATH = 'models/with_draw_features/conservative_with_draw_features.joblib'


class LivePredictionPipeline:
    """Live prediction pipeline using V4 feature orchestrator."""

    def __init__(self, api_key: str, data_dir: str = 'data/historical'):
        """
        Initialize live prediction pipeline.

        Args:
            api_key: SportMonks API key
            data_dir: Directory containing historical data for feature calculation
        """
        self.api_key = api_key
        self.client = SportMonksClient(api_key)
        self.orchestrator = FeatureOrchestrator(data_dir=data_dir)
        self.model = None

        logger.info("✅ Live prediction pipeline initialized")

    def load_model(self, model_path: str = PRODUCTION_MODEL_PATH):
        """Load production model."""
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_file)
        logger.info(f"✅ Loaded model from {model_path}")

    def fetch_fixtures(
        self,
        start_date: str,
        end_date: str,
        league_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch fixtures from SportMonks API.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_id: Optional league filter

        Returns:
            List of fixture dictionaries
        """
        logger.info(f"Fetching fixtures from {start_date} to {end_date}")

        try:
            url = f"https://api.sportmonks.com/v3/football/fixtures/between/{start_date}/{end_date}"
            params = {
                'api_token': self.api_key,
                'include': 'participants;league;scores;statistics;lineups'
            }

            if league_id:
                params['filters'] = f'fixtureLeagues:{league_id}'

            response = requests.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data or 'data' not in data:
                logger.warning(f"No fixtures found for {start_date} to {end_date}")
                return []

            fixtures = []
            for fixture in data['data']:
                # Only include scheduled/not started fixtures
                state_id = fixture.get('state_id')
                if state_id not in [1, 2, 3, 9]:  # NS, LIVE, HT, ABAN (not started/upcoming)
                    continue

                participants = fixture.get('participants', [])
                if len(participants) < 2:
                    continue

                home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                away_team = next((p for p in participants if p.get('meta', {}).get('location') != 'home'), None)

                if not home_team or not away_team:
                    continue

                fixture_dict = {
                    'fixture_id': fixture['id'],
                    'starting_at': fixture.get('starting_at'),
                    'league_id': fixture.get('league_id'),
                    'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                    'season_id': fixture.get('season_id'),
                    'home_team_id': home_team['id'],
                    'home_team_name': home_team['name'],
                    'away_team_id': away_team['id'],
                    'away_team_name': away_team['name'],
                    'venue': fixture.get('venue', {}).get('name', 'Unknown') if fixture.get('venue') else 'Unknown',
                    'state_name': fixture.get('state', {}).get('name', 'Unknown')
                }

                fixtures.append(fixture_dict)

            logger.info(f"✅ Found {len(fixtures)} upcoming fixtures")
            return fixtures

        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            return []

    def download_fixture_data(
        self,
        fixtures: List[Dict],
        output_dir: str = 'data/historical/fixtures'
    ) -> Path:
        """
        Download detailed fixture data including statistics and lineups.
        Saves to historical directory so FeatureOrchestrator can access them.

        Args:
            fixtures: List of fixture dictionaries
            output_dir: Directory to save downloaded data

        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading detailed data for {len(fixtures)} fixtures...")

        downloaded_ids = []

        for fixture in fixtures:
            fixture_id = fixture['fixture_id']

            try:
                # Download fixture with full details
                url = f"https://api.sportmonks.com/v3/football/fixtures/{fixture_id}"
                params = {
                    'api_token': self.api_key,
                    'include': 'participants;statistics;lineups;scores;league;season;state'
                }

                response = requests.get(url, params=params, verify=False, timeout=30)
                response.raise_for_status()
                data = response.json()

                if data and 'data' in data:
                    # Save to file (use date-based naming like backfill script)
                    fixture_date = fixture['starting_at'][:10]  # YYYY-MM-DD
                    filename = output_path / f"{fixture_date}_{fixture_id}.json"
                    with open(filename, 'w') as f:
                        json.dump(data['data'], f, indent=2)

                    downloaded_ids.append(fixture_id)
                    logger.info(f"  ✓ Downloaded fixture {fixture_id}")
                else:
                    logger.warning(f"  ✗ No data for fixture {fixture_id}")

            except Exception as e:
                logger.error(f"  ✗ Error downloading fixture {fixture_id}: {e}")

        logger.info(f"✅ Downloaded {len(downloaded_ids)} fixtures to {output_path}")
        return output_path

    def generate_features(
        self,
        fixtures: List[Dict],
        reload_orchestrator: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Generate 162 features for fixtures using FeatureOrchestrator.

        Args:
            fixtures: List of fixture dictionaries
            reload_orchestrator: Whether to reload orchestrator to pick up new data

        Returns:
            DataFrame with features or None if failed
        """
        if not fixtures:
            logger.warning("No fixtures to generate features for")
            return None

        # Reload orchestrator to pick up newly downloaded fixtures
        if reload_orchestrator:
            logger.info("Reloading FeatureOrchestrator with new fixture data...")
            self.orchestrator = FeatureOrchestrator(data_dir='data/historical')

        logger.info(f"Generating 162 features for {len(fixtures)} fixtures...")

        fixture_features = []

        for fixture in fixtures:
            try:
                # Generate features for this fixture
                features = self.orchestrator.generate_features_for_fixture(
                    fixture_id=fixture['fixture_id'],
                    as_of_date=None  # Use fixture date
                )

                if features:
                    # Add team names and league name for display
                    features['home_team_name'] = fixture['home_team_name']
                    features['away_team_name'] = fixture['away_team_name']
                    features['league_name'] = fixture['league_name']

                    fixture_features.append(features)
                    logger.info(f"  ✓ {fixture['home_team_name']} vs {fixture['away_team_name']}")
                else:
                    logger.warning(f"  ✗ No features for {fixture['home_team_name']} vs {fixture['away_team_name']}")

            except Exception as e:
                logger.error(f"  ✗ Error for fixture {fixture['fixture_id']}: {e}")
                import traceback
                traceback.print_exc()

        if not fixture_features:
            logger.error("Failed to generate features for any fixtures")
            return None

        df = pd.DataFrame(fixture_features)
        logger.info(f"✅ Generated features for {len(df)} fixtures")

        return df

    def make_predictions(
        self,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions using the production model.

        Args:
            features_df: DataFrame with features

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Making predictions for {len(features_df)} fixtures...")

        # Get feature columns (exclude metadata)
        feature_cols = [c for c in features_df.columns if c not in [
            'fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
            'match_date', 'home_score', 'away_score', 'result', 'target',
            'home_team_name', 'away_team_name', 'state_id', 'league_name'
        ]]

        X = features_df[feature_cols]

        # Make predictions
        probabilities = self.model.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)

        # Create results dataframe
        results = features_df[['fixture_id', 'match_date', 'league_name',
                               'home_team_name', 'away_team_name']].copy()

        results['away_win_prob'] = probabilities[:, 0]
        results['draw_prob'] = probabilities[:, 1]
        results['home_win_prob'] = probabilities[:, 2]
        results['predicted_outcome'] = predictions
        results['predicted_outcome_label'] = results['predicted_outcome'].map({
            0: 'Away Win',
            1: 'Draw',
            2: 'Home Win'
        })

        # Add confidence score (max probability)
        results['confidence'] = probabilities.max(axis=1)

        logger.info(f"✅ Predictions complete")

        return results

    def update_historical_data(self, days_back: int = 30):
        """
        Download recent historical data to ensure rolling stats are up-to-date.

        Args:
            days_back: Number of days of historical data to download
        """
        logger.info("="*80)
        logger.info(f"UPDATING HISTORICAL DATA (last {days_back} days)")
        logger.info("="*80)

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        logger.info(f"Downloading: {start_date} to {end_date}")

        try:
            # Use backfill script to download historical data
            import subprocess
            result = subprocess.run(
                [
                    'python3',
                    'scripts/backfill_historical_data.py',
                    '--start-date', start_date,
                    '--end-date', end_date,
                    '--output-dir', 'data/historical'
                ],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                logger.info("✅ Historical data updated successfully")
            else:
                logger.warning(f"⚠️ Historical update had issues: {result.stderr}")

        except Exception as e:
            logger.error(f"❌ Error updating historical data: {e}")
            logger.warning("Continuing with existing historical data...")

    def run(
        self,
        start_date: str,
        end_date: str,
        league_id: Optional[int] = None,
        output_file: Optional[str] = None,
        download_data: bool = True,
        update_historical: bool = False,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Run complete live prediction pipeline.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_id: Optional league filter
            output_file: Optional CSV file to save predictions
            download_data: Whether to download and save raw fixture data
            update_historical: Whether to download recent historical data first
            days_back: Days of historical data to download (if update_historical=True)

        Returns:
            DataFrame with predictions
        """
        logger.info("="*80)
        logger.info("LIVE PREDICTION PIPELINE - V4")
        logger.info("="*80)
        logger.info(f"Date range: {start_date} to {end_date}")
        if league_id:
            logger.info(f"League filter: {league_id}")

        # Update historical data first if requested
        if update_historical:
            self.update_historical_data(days_back)

        # Step 1: Fetch fixtures
        fixtures = self.fetch_fixtures(start_date, end_date, league_id)

        if not fixtures:
            logger.warning("No fixtures found")
            return pd.DataFrame()

        # Step 2: Download detailed data (optional)
        if download_data:
            self.download_fixture_data(fixtures)

        # Step 3: Generate features
        features_df = self.generate_features(fixtures)

        if features_df is None or len(features_df) == 0:
            logger.error("No features generated")
            return pd.DataFrame()

        # Step 4: Load model
        if self.model is None:
            self.load_model()

        # Step 5: Make predictions
        predictions_df = self.make_predictions(features_df)

        # Step 6: Save results
        if output_file:
            predictions_df.to_csv(output_file, index=False)
            logger.info(f"✅ Predictions saved to {output_file}")

        # Display results
        self.display_results(predictions_df)

        logger.info("="*80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("="*80)

        return predictions_df

    def display_results(self, predictions_df: pd.DataFrame):
        """Display prediction results in a readable format."""
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
        # Validate format
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD, 'today', or 'tomorrow'")


def main():
    parser = argparse.ArgumentParser(
        description='Live prediction pipeline for V4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/predict_live_v4.py --date today
  python3 scripts/predict_live_v4.py --date 2026-02-15
  python3 scripts/predict_live_v4.py --start-date 2026-02-15 --end-date 2026-02-17
  python3 scripts/predict_live_v4.py --date today --league-id 8 --output predictions.csv
        """
    )

    parser.add_argument('--date', help='Single date to predict (YYYY-MM-DD, today, or tomorrow)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--league-id', type=int, help='League ID filter (e.g., 8 for Premier League)')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--no-download', action='store_true', help='Skip downloading raw fixture data')
    parser.add_argument('--model', default=PRODUCTION_MODEL_PATH, help=f'Model path (default: {PRODUCTION_MODEL_PATH})')
    parser.add_argument('--update-historical', action='store_true', help='Download recent historical data first (recommended)')
    parser.add_argument('--days-back', type=int, default=30, help='Days of historical data to download (default: 30)')

    args = parser.parse_args()

    # Validate date arguments
    if args.date and (args.start_date or args.end_date):
        parser.error("Cannot use --date with --start-date/--end-date")

    if not args.date and not (args.start_date and args.end_date):
        parser.error("Must provide either --date or both --start-date and --end-date")

    # Check API key
    if not SPORTMONKS_API_KEY:
        logger.error("SPORTMONKS_API_KEY environment variable not set")
        logger.info("Set it with: export SPORTMONKS_API_KEY='your_api_key'")
        return 1

    # Parse dates
    if args.date:
        start_date = parse_date(args.date)
        end_date = start_date
    else:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)

    # Initialize pipeline
    pipeline = LivePredictionPipeline(
        api_key=SPORTMONKS_API_KEY,
        data_dir='data/historical'
    )

    # Override model path if specified
    if args.model != PRODUCTION_MODEL_PATH:
        pipeline.load_model(args.model)

    # Run pipeline
    try:
        predictions_df = pipeline.run(
            start_date=start_date,
            end_date=end_date,
            league_id=args.league_id,
            output_file=args.output,
            download_data=not args.no_download,
            update_historical=args.update_historical,
            days_back=args.days_back
        )

        if len(predictions_df) == 0:
            logger.warning("No predictions generated")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import os
    sys.exit(main())
