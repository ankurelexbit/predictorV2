#!/usr/bin/env python3
"""
Production Prediction Script with Full Historical Context
==========================================================

Loads 1 year of historical data on startup, then generates predictions
for upcoming matches and stores in Supabase.

Uses conservative model with draw weights (1.2/1.5/1.0).

Usage:
    export SPORTMONKS_API_KEY="your_key"
    export DATABASE_URL="postgresql://..."

    # Predict next 7 days
    python3 scripts/predict_production.py --days-ahead 7

    # Specific date range
    python3 scripts/predict_production.py --start-date 2026-02-01 --end-date 2026-02-07
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

from scripts.predict_live_with_history import ProductionLivePipeline, HistoricalDataFetcher
from src.database import SupabaseClient
from config import production_config as config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionPredictionEngine:
    """Production prediction engine with database storage."""

    def __init__(
        self,
        api_key: str,
        database_url: str,
        history_days: int = 365,
        history_start_date: str = None,
        history_end_date: str = None,
        model_path: str = None
    ):
        """
        Initialize production engine.

        Args:
            api_key: SportMonks API key
            database_url: Supabase database URL
            history_days: Days of history to load (default: 365, ignored if history_start_date provided)
            history_start_date: Optional start date for historical data (YYYY-MM-DD)
            history_end_date: Optional end date for historical data (YYYY-MM-DD)
            model_path: Optional custom model path (default: models/with_draw_features/conservative_with_draw_features.joblib)
        """
        logger.info("=" * 80)
        logger.info("PRODUCTION PREDICTION ENGINE - V4")
        logger.info("=" * 80)

        # Store model path
        self.model_path = model_path

        # Initialize pipeline with historical context
        self.pipeline = ProductionLivePipeline(
            api_key,
            load_history_days=history_days,
            history_start_date=history_start_date,
            history_end_date=history_end_date
        )

        # Load model
        if self.model_path:
            logger.info(f"\n🤖 Loading model from: {self.model_path}...")
            self.pipeline.load_model(self.model_path)
        else:
            logger.info(f"\n🤖 Loading production model: {config.MODEL_INFO['name']}...")
            logger.info(f"   Path: {config.MODEL_PATH}")
            logger.info(f"   Version: {config.MODEL_VERSION_TAG}")
            self.pipeline.load_model(config.MODEL_PATH)

        # Initialize database
        self.db = SupabaseClient(database_url)
        self.db.create_tables()

        # Thresholds from config
        self.thresholds = config.THRESHOLDS
        logger.info(f"\n📊 Betting Thresholds:")
        logger.info(f"   Home: {self.thresholds['home']:.2f}")
        logger.info(f"   Draw: {self.thresholds['draw']:.2f}")
        logger.info(f"   Away: {self.thresholds['away']:.2f}")

        # League filtering
        self.filter_top_5 = config.FILTER_TOP_5_ONLY
        self.top_5_leagues = config.TOP_5_LEAGUES
        if self.filter_top_5:
            logger.info(f"\n🎯 League Filter: TOP 5 ONLY")
            logger.info(f"   Leagues: {self.top_5_leagues}")
        else:
            logger.info(f"\n🌍 League Filter: ALL LEAGUES")

    def _extract_odds(self, odds_data: List[Dict]) -> Dict:
        """
        Extract best and average odds for home/draw/away from odds data.

        Args:
            odds_data: List of odds objects from fixture (flat list from SportMonks API v3)

        Returns:
            Dictionary with odds data
        """
        home_odds_list = []
        draw_odds_list = []
        away_odds_list = []

        # Market ID 1 is "1X2" or "Full Time Result"
        # SportMonks API v3 returns odds as a flat list with market_id directly on each item
        for odds_item in odds_data:
            # Filter for 1X2 market (market_id = 1)
            if odds_item.get('market_id') != 1:
                continue

            label = odds_item.get('label', '').lower()
            value = odds_item.get('value')

            if value is None:
                continue

            try:
                value = float(value)

                # Map labels to home/draw/away
                if label in ['1', 'home', 'home win']:
                    home_odds_list.append(value)
                elif label in ['x', 'draw']:
                    draw_odds_list.append(value)
                elif label in ['2', 'away', 'away win']:
                    away_odds_list.append(value)
            except (ValueError, TypeError):
                continue

        # Calculate best (highest) and average odds
        return {
            'best_home_odds': max(home_odds_list) if home_odds_list else None,
            'best_draw_odds': max(draw_odds_list) if draw_odds_list else None,
            'best_away_odds': max(away_odds_list) if away_odds_list else None,
            'avg_home_odds': sum(home_odds_list) / len(home_odds_list) if home_odds_list else None,
            'avg_draw_odds': sum(draw_odds_list) / len(draw_odds_list) if draw_odds_list else None,
            'avg_away_odds': sum(away_odds_list) / len(away_odds_list) if away_odds_list else None,
            'odds_count': len(home_odds_list),
        }

    def fetch_upcoming_fixtures(self, start_date: str, end_date: str, include_finished: bool = False) -> List[Dict]:
        """
        Fetch fixtures for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_finished: If True, includes finished matches (state 5) for backtesting
        """
        fixture_type = "all fixtures" if include_finished else "upcoming fixtures"
        logger.info(f"\n📅 Fetching {fixture_type}: {start_date} to {end_date}")

        endpoint = f"fixtures/between/{start_date}/{end_date}"

        # State filters:
        # 1 = Not Started, 2 = Live, 3 = Finished with lineup, 5 = Finished
        if include_finished:
            # Include all states for backtesting
            state_filter = 'fixtureStates:1,2,3,5'
        else:
            # Only upcoming/live fixtures
            state_filter = 'fixtureStates:1,2,3'

        base_params = {
            'include': 'participants;league;state;odds',
            'filters': state_filter
        }

        all_fixtures = []
        page = 1

        while True:
            params = base_params.copy()
            params['page'] = page
            params['api_token'] = self.pipeline.api_key

            import requests
            response = requests.get(
                f"{self.pipeline.base_url}/{endpoint}",
                params=params,
                verify=False,
                timeout=30
            )

            data = response.json()

            # Debug logging on first page
            if page == 1:
                logger.info(f"   API Response status: {response.status_code}")
                if 'data' in data:
                    logger.info(f"   First page returned {len(data['data'])} fixtures")
                else:
                    logger.warning(f"   No 'data' key in response. Keys: {list(data.keys())}")
                    if 'message' in data:
                        logger.warning(f"   API message: {data['message']}")
                    if 'errors' in data:
                        logger.error(f"   API errors: {data['errors']}")

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

                # Extract odds (best and average for home/draw/away)
                odds_data = self._extract_odds(fixture.get('odds', []))

                # Store both the original fixture (for predict()) and extracted data (for database)
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
                    **odds_data,  # Add odds data
                    '_original_fixture': fixture,  # Keep original for predict()
                })

            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break

            page += 1

            if page > 100:
                break

        logger.info(f"✅ Found {len(all_fixtures)} upcoming fixtures")

        # Apply league filtering
        if self.filter_top_5 and self.top_5_leagues:
            original_count = len(all_fixtures)
            all_fixtures = [f for f in all_fixtures if f['league_id'] in self.top_5_leagues]
            filtered_count = len(all_fixtures)
            logger.info(f"   Filtered to Top 5 leagues: {filtered_count}/{original_count} fixtures")

        return all_fixtures

    def generate_predictions(self, fixtures: List[Dict]) -> List[Dict]:
        """Generate predictions with betting strategy."""
        logger.info(f"\n🤖 Generating predictions...")

        predictions = []

        for i, fixture in enumerate(fixtures, 1):
            if i % 10 == 1:
                logger.info(f"   Processing fixture {i}/{len(fixtures)}...")

            try:
                # Use original fixture for prediction
                original_fixture = fixture.get('_original_fixture', fixture)
                result = self.pipeline.predict(original_fixture)

                if not result:
                    if i <= 3:  # Log first 3 failures
                        logger.warning(f"   Failed to predict fixture {fixture['fixture_id']}: predict() returned None")
                    continue

                home_prob = result['home_prob']
                draw_prob = result['draw_prob']
                away_prob = result['away_prob']

                # Determine predicted outcome
                max_prob = max(home_prob, draw_prob, away_prob)
                if max_prob == home_prob:
                    predicted_outcome = 'H'
                elif max_prob == away_prob:
                    predicted_outcome = 'A'
                else:
                    predicted_outcome = 'D'

                # Apply betting strategy (use market odds if available)
                bet_decision = self._apply_strategy(
                    home_prob, draw_prob, away_prob,
                    fixture.get('best_home_odds'),
                    fixture.get('best_draw_odds'),
                    fixture.get('best_away_odds')
                )

                # Extract features from result
                features = result.get('features', {})

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
                    'model_version': 'v4_conservative',
                    # Odds data
                    'best_home_odds': fixture.get('best_home_odds'),
                    'best_draw_odds': fixture.get('best_draw_odds'),
                    'best_away_odds': fixture.get('best_away_odds'),
                    'avg_home_odds': fixture.get('avg_home_odds'),
                    'avg_draw_odds': fixture.get('avg_draw_odds'),
                    'avg_away_odds': fixture.get('avg_away_odds'),
                    'odds_count': fixture.get('odds_count', 0),
                    # Features used for prediction (162 features)
                    'features': features,
                    **bet_decision
                }

                predictions.append(prediction)

            except Exception as e:
                if i <= 3:  # Log first 3 errors in detail
                    logger.error(f"Error predicting fixture {fixture.get('fixture_id', 'unknown')}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                continue

        logger.info(f"✅ Generated {len(predictions)} predictions")
        return predictions

    def _apply_strategy(
        self,
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        market_home_odds: float = None,
        market_draw_odds: float = None,
        market_away_odds: float = None
    ) -> Dict:
        """
        Apply threshold betting strategy.

        Uses market odds if available, otherwise calculates implied odds from probabilities.
        """
        # Use market odds if available, otherwise calculate from probabilities
        home_odds = market_home_odds if market_home_odds else (1 / home_prob * 0.95 if home_prob > 0.01 else 100)
        draw_odds = market_draw_odds if market_draw_odds else (1 / draw_prob * 0.95 if draw_prob > 0.01 else 100)
        away_odds = market_away_odds if market_away_odds else (1 / away_prob * 0.95 if away_prob > 0.01 else 100)

        candidates = []

        if home_prob > self.thresholds['home']:
            candidates.append(('Home Win', home_prob, home_odds))

        if draw_prob > self.thresholds['draw']:
            candidates.append(('Draw', draw_prob, draw_odds))

        if away_prob > self.thresholds['away']:
            candidates.append(('Away Win', away_prob, away_odds))

        if not candidates:
            return {
                'bet_outcome': None,
                'bet_probability': None,
                'bet_odds': None,
                'should_bet': False
            }

        bet_outcome, bet_prob, bet_odds = max(candidates, key=lambda x: x[1])

        return {
            'bet_outcome': bet_outcome,
            'bet_probability': float(bet_prob),
            'bet_odds': float(bet_odds),
            'should_bet': True
        }

    def run(self, start_date: str, end_date: str, include_finished: bool = False):
        """
        Run prediction pipeline.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_finished: If True, includes finished fixtures for backtesting
        """
        # Fetch fixtures
        fixtures = self.fetch_upcoming_fixtures(start_date, end_date, include_finished)

        if not fixtures:
            logger.warning("⚠️  No upcoming fixtures")
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
        total = len(predictions)
        bets = sum(1 for p in predictions if p['should_bet'])
        bet_pct = bets / total * 100 if total > 0 else 0

        # Breakdown by outcome
        home_bets = sum(1 for p in predictions if p.get('bet_outcome') == 'Home Win')
        draw_bets = sum(1 for p in predictions if p.get('bet_outcome') == 'Draw')
        away_bets = sum(1 for p in predictions if p.get('bet_outcome') == 'Away Win')

        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Predictions: {total}")
        logger.info(f"Recommended Bets: {bets} ({bet_pct:.1f}%)")
        logger.info(f"  Home Wins: {home_bets}")
        logger.info(f"  Draws: {draw_bets}")
        logger.info(f"  Away Wins: {away_bets}")

        # Show top bets
        if bets > 0:
            logger.info("\n📊 Top 5 Recommended Bets:")
            logger.info("-" * 80)

            sorted_preds = sorted(
                [p for p in predictions if p['should_bet']],
                key=lambda x: x['bet_probability'],
                reverse=True
            )[:5]

            for pred in sorted_preds:
                logger.info(f"  {pred['home_team_name']} vs {pred['away_team_name']}")
                logger.info(f"    Bet: {pred['bet_outcome']} @ {pred['bet_odds']:.2f} "
                           f"(confidence: {pred['bet_probability']:.1%})")

        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Production predictions with historical context')

    # Prediction date range
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days-ahead', type=int, help='Days ahead to predict')
    date_group.add_argument('--start-date', help='Start date for predictions (YYYY-MM-DD)')

    parser.add_argument('--end-date', help='End date for predictions (YYYY-MM-DD)')

    # Historical data options
    parser.add_argument('--history-days', type=int, default=365,
                       help='Days of history to load (default: 365, ignored if --history-start-date provided)')
    parser.add_argument('--history-start-date',
                       help='Optional start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--history-end-date',
                       help='Optional end date for historical data (YYYY-MM-DD, defaults to today)')

    # Model options
    parser.add_argument('--model-path',
                       help='Optional custom model path (default: models/with_draw_features/conservative_with_draw_features.joblib)')

    # Backtesting options
    parser.add_argument('--include-finished', action='store_true',
                       help='Include finished fixtures (for backtesting on past data)')

    args = parser.parse_args()

    # Determine prediction date range
    if args.days_ahead:
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=args.days_ahead)).strftime('%Y-%m-%d')
    else:
        if not args.end_date:
            parser.error("--end-date required with --start-date")
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

    # Run pipeline
    engine = ProductionPredictionEngine(
        api_key,
        database_url,
        history_days=args.history_days,
        history_start_date=args.history_start_date,
        history_end_date=args.history_end_date,
        model_path=args.model_path
    )
    engine.run(start_date, end_date, include_finished=args.include_finished)


if __name__ == '__main__':
    main()
