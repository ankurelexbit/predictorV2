#!/usr/bin/env python3
"""
Update Match Results and Calculate PnL
=======================================

Fetches actual match results from SportMonks API and updates predictions
in the PostgreSQL database with:
- Actual scores
- Actual result (H/D/A)
- Whether bet won
- Profit/loss using market odds

Usage:
    export SPORTMONKS_API_KEY="your_key"

    # Update results for last 2 days
    python3 scripts/update_results.py --days-back 2

    # Update specific date range
    python3 scripts/update_results.py --start-date 2026-01-30 --end-date 2026-02-01
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config.production_config import DATABASE_URL
from src.database import DatabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SportMonksClient:
    """Simple SportMonks API client."""

    BASE_URL = "https://api.sportmonks.com/v3/football"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, endpoint: str, params: dict = None):
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params['api_token'] = self.api_key
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_fixtures_between(self, start_date: str, end_date: str, finished_only: bool = True):
        """Get fixtures between dates."""
        params = {
            'include': 'participants;scores',
            'per_page': 100
        }
        if finished_only:
            params['filters'] = 'fixtureStateIds:5'

        all_fixtures = []
        page = 1

        while True:
            params['page'] = page
            data = self._request(f'fixtures/between/{start_date}/{end_date}', params)
            fixtures = data.get('data', [])
            if not fixtures:
                break
            all_fixtures.extend(fixtures)
            if not data.get('pagination', {}).get('has_more', False):
                break
            page += 1

        return all_fixtures


class ResultUpdater:
    """Updates predictions with actual match results."""

    def __init__(self, api_key: str, database_url: str = None):
        self.api_key = api_key
        self.client = SportMonksClient(api_key)
        self.db = DatabaseClient(database_url or DATABASE_URL)

    def get_pending_fixtures(self, start_date: str, end_date: str) -> List[Dict]:
        """Get predictions that need results."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT fixture_id, match_date, home_team_name, away_team_name
                FROM predictions
                WHERE actual_result IS NULL
                  AND match_date >= %s
                  AND match_date <= %s
                ORDER BY match_date
            """, (start_date, end_date))

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def fetch_fixture_results(self, fixture_ids: List[int], start_date: str, end_date: str) -> Dict[int, Dict]:
        """Fetch actual results for fixtures from API."""
        results = {}
        target_ids = set(fixture_ids)

        logger.info(f"Fetching finished fixtures between {start_date} and {end_date}...")

        try:
            fixtures = self.client.get_fixtures_between(start_date, end_date, finished_only=True)
            logger.info(f"Retrieved {len(fixtures)} finished fixtures from API")

            for fixture in fixtures:
                fixture_id = fixture.get('id')
                if fixture_id not in target_ids:
                    continue

                # Extract scores
                home_score, away_score = None, None
                for score in fixture.get('scores', []):
                    if score.get('description') == 'CURRENT':
                        score_data = score.get('score', {})
                        if score_data.get('participant') == 'home':
                            home_score = score_data.get('goals')
                        elif score_data.get('participant') == 'away':
                            away_score = score_data.get('goals')

                if home_score is not None and away_score is not None:
                    results[fixture_id] = {
                        'home_score': home_score,
                        'away_score': away_score
                    }

        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")

        return results

    def update_results(self, start_date: str, end_date: str, model_version: str = 'v5'):
        """Update results for date range."""
        logger.info("=" * 60)
        logger.info("UPDATING MATCH RESULTS")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")

        # Get pending fixtures
        logger.info("\nFetching pending predictions from database...")
        pending = self.get_pending_fixtures(start_date, end_date)

        if not pending:
            logger.info("No pending results to update")
            return

        logger.info(f"Found {len(pending)} predictions awaiting results")

        # Show sample
        logger.info("\nPending matches:")
        for p in pending[:5]:
            logger.info(f"  {str(p['match_date'])[:10]}: {p['home_team_name']} vs {p['away_team_name']}")
        if len(pending) > 5:
            logger.info(f"  ... and {len(pending) - 5} more")

        # Fetch results from API
        logger.info("\nFetching actual results from API...")
        fixture_ids = [p['fixture_id'] for p in pending]
        results = self.fetch_fixture_results(fixture_ids, start_date, end_date)

        logger.info(f"Retrieved {len(results)} finished matches")

        if not results:
            logger.info("No finished matches found")
            return

        # Update database
        logger.info("\nUpdating database...")
        try:
            updated_count = self.db.update_actual_results_batch(results)
        except Exception as e:
            logger.error(f"Error updating results: {e}")
            updated_count = 0

        # Update market predictions table (if it exists)
        try:
            market_updated = self.db.update_market_actuals_batch(results)
            if market_updated > 0:
                logger.info(f"Updated {market_updated} market prediction actuals")

            # Calculate bet results for goals market bets
            bet_updated = self.db.update_market_bet_results_batch(results)
            if bet_updated > 0:
                logger.info(f"Updated {bet_updated} market bet results (O/U 2.5, BTTS)")
        except Exception as e:
            logger.debug(f"Market actuals/bet update skipped: {e}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("UPDATE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successfully updated: {updated_count}")
        logger.info(f"Still pending: {len(pending) - len(results)}")

        # Show quick PnL
        if updated_count > 0:
            logger.info("\nQuick PnL check:")
            days = (datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
            perf = self.db.get_betting_performance(days=days, model_version=model_version)

            if perf['total_bets'] > 0:
                logger.info(f"  Bets: {perf['total_bets']}")
                logger.info(f"  Win Rate: {perf['win_rate']:.1%}")
                logger.info(f"  Profit: ${perf['total_profit']:.2f}")
                logger.info(f"  ROI: {perf['roi']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Update match results and calculate PnL')

    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days-back', type=int, help='Update last N days')
    date_group.add_argument('--start-date', help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-version', default='v5', help='Model version')

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        logger.error("SPORTMONKS_API_KEY not set")
        sys.exit(1)

    # Determine date range
    if args.days_back:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days_back)).strftime('%Y-%m-%d')
    else:
        if not args.end_date:
            parser.error("--end-date required with --start-date")
        start_date = args.start_date
        end_date = args.end_date

    # Run update
    updater = ResultUpdater(api_key)
    updater.update_results(start_date, end_date, args.model_version)


if __name__ == '__main__':
    main()
