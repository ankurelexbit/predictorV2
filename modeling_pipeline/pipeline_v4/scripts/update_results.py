#!/usr/bin/env python3
"""
Update Match Results and Calculate PnL
=======================================

Fetches actual match results and updates predictions table with:
- Actual scores
- Actual result (H/D/A)
- Whether bet won
- Profit/loss using real market odds

Usage:
    export SPORTMONKS_API_KEY="your_key"
    export DATABASE_URL="postgresql://..."

    # Update results for last 2 days
    python3 scripts/update_results.py --days-back 2

    # Update specific date range
    python3 scripts/update_results.py \
      --start-date 2026-01-30 \
      --end-date 2026-02-01
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SupabaseClient
from src.data.sportmonks_client import SportMonksClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultUpdater:
    """Updates predictions with actual match results."""

    def __init__(self, api_key: str, database_url: str):
        self.api_key = api_key
        self.client = SportMonksClient(api_key)
        self.db = SupabaseClient(database_url)

    def get_pending_fixtures(self, start_date: str, end_date: str, model_version: str = 'v4_conservative') -> List[Dict]:
        """
        Get predictions that need results.
        Returns unique fixtures (across all model versions).
        """
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
        """
        Fetch actual results for fixtures from API.

        Args:
            fixture_ids: List of fixture IDs to get results for
            start_date: Start date to search (YYYY-MM-DD)
            end_date: End date to search (YYYY-MM-DD)

        Returns:
            Dict mapping fixture_id to {home_score, away_score}
        """
        results = {}

        # Convert fixture_ids to set for fast lookup
        target_ids = set(fixture_ids)

        logger.info(f"Fetching finished fixtures between {start_date} and {end_date}...")

        try:
            # Use SportMonksClient to fetch all finished fixtures in date range
            # This is more reliable than /fixtures/multi/ endpoint
            fixtures = self.client.get_fixtures_between(
                start_date=start_date,
                end_date=end_date,
                include_details=True,
                finished_only=True  # Only get state_id = 5
            )

            logger.info(f"Retrieved {len(fixtures)} finished fixtures from API")

            # Parse fixtures we care about
            for fixture in fixtures:
                fixture_id = fixture.get('id')

                # Only process fixtures we're looking for
                if fixture_id not in target_ids:
                    continue

                # Extract scores from scores array
                home_score = None
                away_score = None

                scores = fixture.get('scores', [])
                for score in scores:
                    if score.get('description') == 'CURRENT':
                        score_data = score.get('score', {})
                        participant = score_data.get('participant')
                        goals = score_data.get('goals')

                        if participant == 'home':
                            home_score = goals
                        elif participant == 'away':
                            away_score = goals

                if home_score is not None and away_score is not None:
                    results[fixture_id] = {
                        'home_score': home_score,
                        'away_score': away_score
                    }
                else:
                    logger.warning(f"Could not extract scores for fixture {fixture_id}")

        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return results

    def update_results(self, start_date: str, end_date: str, model_version: str = 'v4_conservative'):
        """Update results for date range."""
        logger.info("=" * 80)
        logger.info("UPDATING MATCH RESULTS")
        logger.info("=" * 80)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Model version: {model_version}")

        # Get pending fixtures
        logger.info("\n📋 Fetching pending predictions from database...")
        pending = self.get_pending_fixtures(start_date, end_date, model_version)

        if not pending:
            logger.info("✅ No pending results to update")
            return

        logger.info(f"Found {len(pending)} predictions awaiting results")

        # Show pending matches
        logger.info("\nPending matches:")
        for p in pending[:5]:
            logger.info(f"  {p['match_date']}: {p['home_team_name']} vs {p['away_team_name']}")
        if len(pending) > 5:
            logger.info(f"  ... and {len(pending) - 5} more")

        # Fetch results from API
        logger.info("\n🔍 Fetching actual results from API...")
        fixture_ids = [p['fixture_id'] for p in pending]
        results = self.fetch_fixture_results(fixture_ids, start_date, end_date)

        logger.info(f"Retrieved {len(results)} finished matches")

        if not results:
            logger.info("⚠️  No finished matches found")
            return

        # Update database in batch
        logger.info("\n💾 Updating database with results (batch mode)...")
        try:
            updated_count = self.db.update_actual_results_batch(results)
            error_count = 0
        except Exception as e:
            logger.error(f"Error updating results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            updated_count = 0
            error_count = len(results)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("UPDATE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Successfully updated: {updated_count}")
        if error_count > 0:
            logger.info(f"❌ Errors: {error_count}")
        logger.info(f"⏳ Still pending: {len(pending) - updated_count}")

        # Show PnL for updated period
        if updated_count > 0:
            logger.info("\n📊 Performance for updated period:")
            performance = self.db.get_betting_performance(days=7, model_version=model_version)

            if performance['total_bets'] > 0:
                logger.info(f"  Bets: {performance['total_bets']}")
                logger.info(f"  Win Rate: {performance['win_rate']:.1%}")
                logger.info(f"  Total Profit: ${performance['total_profit']:.2f}")
                logger.info(f"  ROI: {performance['roi']:.1f}%")

        logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Update match results and calculate PnL')

    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days-back', type=int, help='Update last N days')
    date_group.add_argument('--start-date', help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-version', default='v4_conservative', help='Model version (default: v4_conservative)')

    args = parser.parse_args()

    # Get credentials
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    database_url = os.environ.get('DATABASE_URL')

    if not api_key:
        logger.error("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    if not database_url:
        logger.error("❌ DATABASE_URL not set")
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
    updater = ResultUpdater(api_key, database_url)
    updater.update_results(start_date, end_date, args.model_version)


if __name__ == '__main__':
    main()
