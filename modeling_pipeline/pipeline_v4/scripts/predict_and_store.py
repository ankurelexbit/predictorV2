#!/usr/bin/env python3
"""
Production Prediction Pipeline with Supabase Storage
====================================================

Fetches upcoming fixtures, generates predictions, applies betting strategy,
and stores results in Supabase database.

Usage:
    # Set environment variables
    export SPORTMONKS_API_KEY="your_api_key"
    export DATABASE_URL="postgresql://..."

    # Run for next 7 days
    python3 scripts/predict_and_store.py --days-ahead 7

    # Run for specific date range
    python3 scripts/predict_and_store.py --start-date 2026-02-01 --end-date 2026-02-07
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline
from src.database import SupabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionPredictionPipeline:
    """Production pipeline for predictions with database storage."""

    def __init__(self, api_key: str, database_url: str):
        """
        Initialize production pipeline.

        Args:
            api_key: SportMonks API key
            database_url: Supabase database URL
        """
        self.pipeline = StandaloneLivePipeline(api_key)
        self.db = SupabaseClient(database_url)

        # Default thresholds (based on V2 strategy)
        self.thresholds = {
            'home': 0.48,
            'draw': 0.35,
            'away': 0.45
        }

    def fetch_upcoming_fixtures(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch upcoming fixtures.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of fixture dictionaries
        """
        logger.info(f"📅 Fetching upcoming fixtures: {start_date} to {end_date}")

        endpoint = f"fixtures/between/{start_date}/{end_date}"
        base_params = {
            'include': 'participants;league;state',
            'filters': 'fixtureStates:1,2,3'  # Not started, Live, Halftime
        }

        all_fixtures = []
        page = 1

        while True:
            params = base_params.copy()
            params['page'] = page

            data = self.pipeline._api_call(endpoint, params)

            if not data or 'data' not in data:
                break

            fixtures_data = data['data']
            if not fixtures_data:
                break

            for fixture in fixtures_data:
                participants = fixture.get('participants', [])
                if len(participants) < 2:
                    continue

                home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
                away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

                if not home_team or not away_team:
                    continue

                all_fixtures.append({
                    'fixture_id': fixture['id'],
                    'starting_at': fixture.get('starting_at'),
                    'league_id': fixture.get('league_id'),
                    'league_name': fixture.get('league', {}).get('name', 'Unknown'),
                    'season_id': fixture.get('season_id'),
                    'home_team_id': home_team['id'],
                    'home_team_name': home_team['name'],
                    'away_team_id': away_team['id'],
                    'away_team_name': away_team['name'],
                    'state_id': fixture.get('state_id')
                })

            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break

            page += 1

            if page > 100:  # Safety limit
                break

        logger.info(f"✅ Found {len(all_fixtures)} upcoming fixtures")
        return all_fixtures

    def generate_predictions(self, fixtures: List[Dict]) -> List[Dict]:
        """
        Generate predictions for fixtures.

        Args:
            fixtures: List of fixture dictionaries

        Returns:
            List of prediction dictionaries
        """
        logger.info(f"🤖 Generating predictions for {len(fixtures)} fixtures...")

        predictions = []

        for fixture in fixtures:
            try:
                features = self.pipeline.generate_features(fixture)

                if not features:
                    logger.warning(f"Failed to generate features for fixture {fixture['fixture_id']}")
                    continue

                features_df = pd.DataFrame([features])
                probas = self.pipeline.model.predict_proba(features_df)[0]

                # probas = [away, draw, home]
                away_prob = float(probas[0])
                draw_prob = float(probas[1])
                home_prob = float(probas[2])

                # Determine predicted outcome
                max_prob = max(home_prob, draw_prob, away_prob)
                if max_prob == home_prob:
                    predicted_outcome = 'H'
                elif max_prob == away_prob:
                    predicted_outcome = 'A'
                else:
                    predicted_outcome = 'D'

                # Apply betting strategy
                bet_decision = self._apply_strategy(home_prob, draw_prob, away_prob)

                prediction = {
                    'fixture_id': fixture['fixture_id'],
                    'match_date': fixture['starting_at'],
                    'league_id': fixture.get('league_id'),
                    'league_name': fixture.get('league_name'),
                    'season_id': fixture.get('season_id'),
                    'home_team_id': fixture['home_team_id'],
                    'home_team_name': fixture['home_team_name'],
                    'away_team_id': fixture['away_team_id'],
                    'away_team_name': fixture['away_team_name'],
                    'pred_home_prob': home_prob,
                    'pred_draw_prob': draw_prob,
                    'pred_away_prob': away_prob,
                    'predicted_outcome': predicted_outcome,
                    **bet_decision
                }

                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Error generating prediction for fixture {fixture['fixture_id']}: {e}")
                continue

        logger.info(f"✅ Generated {len(predictions)} predictions")
        return predictions

    def _apply_strategy(self, home_prob: float, draw_prob: float, away_prob: float) -> Dict:
        """
        Apply threshold-based betting strategy.

        Args:
            home_prob: Home win probability
            draw_prob: Draw probability
            away_prob: Away win probability

        Returns:
            Dictionary with bet decision
        """
        # Calculate implied odds (with 5% margin)
        home_odds = 1 / home_prob * 0.95 if home_prob > 0.01 else 100
        draw_odds = 1 / draw_prob * 0.95 if draw_prob > 0.01 else 100
        away_odds = 1 / away_prob * 0.95 if away_prob > 0.01 else 100

        # Check which outcomes exceed thresholds
        candidates = []

        if home_prob > self.thresholds['home']:
            candidates.append(('Home Win', home_prob, home_odds))

        if draw_prob > self.thresholds['draw']:
            candidates.append(('Draw', draw_prob, draw_odds))

        if away_prob > self.thresholds['away']:
            candidates.append(('Away Win', away_prob, away_odds))

        # Decision
        if not candidates:
            return {
                'bet_outcome': None,
                'bet_probability': None,
                'bet_odds': None,
                'should_bet': False
            }

        # Pick highest probability among candidates
        bet_outcome, bet_prob, bet_odds = max(candidates, key=lambda x: x[1])

        return {
            'bet_outcome': bet_outcome,
            'bet_probability': float(bet_prob),
            'bet_odds': float(bet_odds),
            'should_bet': True
        }

    def run(self, start_date: str, end_date: str):
        """
        Run full prediction pipeline.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info("=" * 80)
        logger.info("PRODUCTION PREDICTION PIPELINE - V4")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Strategy: Home>{self.thresholds['home']:.0%}, "
                   f"Draw>{self.thresholds['draw']:.0%}, "
                   f"Away>{self.thresholds['away']:.0%}")
        logger.info("=" * 80)

        # Initialize database
        logger.info("\n📦 Initializing database...")
        self.db.create_tables()

        # Load model
        logger.info("\n🤖 Loading V4 model...")
        self.pipeline.load_model()

        # Fetch fixtures
        fixtures = self.fetch_upcoming_fixtures(start_date, end_date)

        if not fixtures:
            logger.warning("⚠️  No upcoming fixtures found")
            return

        # Generate predictions
        predictions = self.generate_predictions(fixtures)

        if not predictions:
            logger.warning("⚠️  No predictions generated")
            return

        # Store in database
        logger.info("\n💾 Storing predictions in Supabase...")
        self.db.store_predictions_batch(predictions)

        # Summary
        total_predictions = len(predictions)
        bets_to_place = sum(1 for p in predictions if p['should_bet'])
        bet_pct = bets_to_place / total_predictions * 100 if total_predictions > 0 else 0

        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Predictions: {total_predictions}")
        logger.info(f"Recommended Bets: {bets_to_place} ({bet_pct:.1f}%)")

        # Show recommended bets
        if bets_to_place > 0:
            logger.info("\n📊 RECOMMENDED BETS:")
            logger.info("-" * 80)

            for pred in predictions:
                if pred['should_bet']:
                    logger.info(f"  {pred['home_team_name']} vs {pred['away_team_name']}")
                    logger.info(f"    Bet: {pred['bet_outcome']} @ {pred['bet_odds']:.2f} "
                               f"(confidence: {pred['bet_probability']:.1%})")
                    logger.info(f"    Match: {pred['match_date']}")
                    logger.info("")

        logger.info("=" * 80)
        logger.info("✅ Pipeline complete")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Production prediction pipeline with Supabase storage')

    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days-ahead', type=int, help='Number of days ahead to predict (e.g., 7)')
    date_group.add_argument('--start-date', help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end-date', help='End date (YYYY-MM-DD), required with --start-date')

    # Threshold overrides
    parser.add_argument('--home-threshold', type=float, default=0.48, help='Home threshold (default: 0.48)')
    parser.add_argument('--draw-threshold', type=float, default=0.35, help='Draw threshold (default: 0.35)')
    parser.add_argument('--away-threshold', type=float, default=0.45, help='Away threshold (default: 0.45)')

    args = parser.parse_args()

    # Determine date range
    if args.days_ahead:
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=args.days_ahead)).strftime('%Y-%m-%d')
    else:
        if not args.end_date:
            parser.error("--end-date is required when using --start-date")
        start_date = args.start_date
        end_date = args.end_date

    # Get credentials
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    database_url = os.environ.get('DATABASE_URL')

    if not api_key:
        logger.error("❌ SPORTMONKS_API_KEY not set")
        sys.exit(1)

    if not database_url:
        logger.error("❌ DATABASE_URL not set")
        sys.exit(1)

    # Initialize and run pipeline
    pipeline = ProductionPredictionPipeline(api_key, database_url)

    # Apply threshold overrides
    pipeline.thresholds = {
        'home': args.home_threshold,
        'draw': args.draw_threshold,
        'away': args.away_threshold
    }

    pipeline.run(start_date, end_date)


if __name__ == '__main__':
    main()
