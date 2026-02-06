#!/usr/bin/env python3
"""
Live Prediction Script for V5 Pipeline
=======================================

Generates predictions for upcoming fixtures and saves to PostgreSQL.

Features:
- Loads historical data for context (Elo, form, standings)
- Uses CatBoost+LightGBM ensemble
- Applies V5 thresholds (H=0.45, A=0.45, D=0.35)
- Filters by odds range (1.3-3.0)
- Saves predictions to PostgreSQL

Usage:
    export SPORTMONKS_API_KEY="your_key"

    # Predict upcoming fixtures
    python3 scripts/predict_live.py --days-ahead 7

    # Predict with specific strategy profile
    python3 scripts/predict_live.py --days-ahead 7 --strategy conservative

    # Dry run (don't save to DB)
    python3 scripts/predict_live.py --days-ahead 7 --dry-run
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import argparse
import json

import pandas as pd
import numpy as np
import joblib
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.production_config import (
    DATABASE_URL, THRESHOLDS, ODDS_FILTER, TOP_5_LEAGUES,
    FILTER_TOP_5_ONLY, STRATEGY_PROFILES, get_active_strategy,
    EloConfig, get_latest_model_path
)
from src.database import DatabaseClient
from src.features import (
    EloCalculator,
    StandingsCalculator,
    Pillar1FundamentalsEngine,
    Pillar2ModernAnalyticsEngine,
    Pillar3HiddenEdgesEngine,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InMemoryDataLoader:
    """Lightweight in-memory data loader."""

    def __init__(self):
        self._fixtures_df = None

    def add_fixtures(self, fixtures_df: pd.DataFrame):
        """Add fixtures to cache."""
        if self._fixtures_df is None:
            self._fixtures_df = fixtures_df.copy()
        else:
            self._fixtures_df = pd.concat([self._fixtures_df, fixtures_df], ignore_index=True)
        self._fixtures_df = self._fixtures_df.sort_values('starting_at').reset_index(drop=True)

    @property
    def fixtures_df(self):
        return self._fixtures_df if self._fixtures_df is not None else pd.DataFrame()

    def get_fixtures_before(self, before_date: datetime, league_id: int = None, season_id: int = None):
        if self._fixtures_df is None or len(self._fixtures_df) == 0:
            return pd.DataFrame()
        mask = pd.to_datetime(self._fixtures_df['starting_at']) < before_date
        if league_id is not None:
            mask = mask & (self._fixtures_df['league_id'] == league_id)
        if season_id is not None:
            mask = mask & (self._fixtures_df['season_id'] == season_id)
        return self._fixtures_df[mask].copy()

    def get_team_fixtures(self, team_id: int, before_date: datetime, limit: int = None):
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

    def get_fixture(self, fixture_id: int):
        if self._fixtures_df is None or len(self._fixtures_df) == 0:
            return None
        matches = self._fixtures_df[self._fixtures_df['id'] == fixture_id]
        return matches.iloc[0].to_dict() if len(matches) > 0 else None

    def get_statistics(self, fixture_id: int):
        return []

    def get_lineups(self, fixture_id: int):
        return []


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

    def get_fixtures_between(self, start_date: str, end_date: str, include_details: bool = True, finished_only: bool = False, league_id: int = None, include_odds: bool = False):
        """Get fixtures between dates using /between endpoint with proper pagination."""
        endpoint = f'fixtures/between/{start_date}/{end_date}'
        all_fixtures = []
        page = 1

        while True:
            params = {'page': page}

            filters = []
            if finished_only:
                filters.append('fixtureStates:5')
            if league_id:
                filters.append(f'fixtureLeagues:{league_id}')
            if filters:
                params['filters'] = ';'.join(filters)

            includes = ['participants', 'scores']
            if include_details:
                includes.append('statistics')
            if include_odds:
                includes.append('odds.bookmaker')
            params['include'] = ';'.join(includes)

            data = self._request(endpoint, params)
            fixtures = data.get('data', [])
            if not fixtures:
                break
            all_fixtures.extend(fixtures)

            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break
            page += 1

            if page > 500:
                break

        return all_fixtures

    def get_upcoming_fixtures(self, days_ahead: int = 7, league_ids: list = None):
        """Get upcoming fixtures with odds."""
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        endpoint = f'fixtures/between/{start_date}/{end_date}'
        all_fixtures = []
        page = 1

        while True:
            params = {'page': page, 'include': 'participants;odds.bookmaker'}
            if league_ids:
                params['filters'] = f'fixtureLeagues:{",".join(map(str, league_ids))}'

            data = self._request(endpoint, params)
            fixtures = data.get('data', [])
            if not fixtures:
                break
            all_fixtures.extend(fixtures)

            pagination = data.get('pagination', {})
            if not pagination.get('has_more', False):
                break
            page += 1

            if page > 500:
                break

        return all_fixtures


def extract_statistics_from_fixture(fixture: dict) -> dict:
    """Extract statistics from fixture into flat dict."""
    stats_dict = {}
    statistics = fixture.get('statistics', [])

    # Aligned with build_fixtures_csv.py and data/reference/sportmonks_types.json
    stat_type_map = {
        42: 'shots_total', 86: 'shots_on_target', 49: 'shots_inside_box',
        50: 'shots_outside_box', 580: 'big_chances', 58: 'shots_blocked',
        45: 'possession', 43: 'attacks', 44: 'dangerous_attacks',
        34: 'corners', 51: 'offsides', 57: 'saves',
        78: 'tackles', 100: 'interceptions', 56: 'fouls',
        80: 'passes', 81: 'passes_accurate',
    }

    for side in ['home', 'away']:
        for stat_name in stat_type_map.values():
            stats_dict[f'{side}_{stat_name}'] = None

    for stat in statistics:
        type_id = stat.get('type_id')
        location = stat.get('location', '').lower()
        value = stat.get('data', {}).get('value')
        if type_id in stat_type_map and location in ['home', 'away']:
            stats_dict[f'{location}_{stat_type_map[type_id]}'] = value

    return stats_dict


def parse_fixtures_to_df(fixtures: list) -> pd.DataFrame:
    """Convert fixtures list to DataFrame."""
    records = []
    for fixture in fixtures:
        participants = fixture.get('participants', [])
        if len(participants) != 2:
            continue

        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
        if not home_team or not away_team:
            continue

        scores = fixture.get('scores', [])
        home_score, away_score = None, None
        for score in scores:
            if score.get('description') == 'CURRENT':
                score_data = score.get('score', {})
                if score_data.get('participant') == 'home':
                    home_score = score_data.get('goals')
                elif score_data.get('participant') == 'away':
                    away_score = score_data.get('goals')

        result = None
        if home_score is not None and away_score is not None:
            result = 'H' if home_score > away_score else ('A' if home_score < away_score else 'D')

        stats = extract_statistics_from_fixture(fixture)

        records.append({
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
            **stats
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=['id'])
        df['starting_at'] = pd.to_datetime(df['starting_at'])
        df = df.sort_values('starting_at').reset_index(drop=True)
    return df


def extract_odds(fixture: dict) -> dict:
    """Extract 1X2 (Fulltime Result) odds from fixture."""
    odds_data = fixture.get('odds', [])
    all_home, all_draw, all_away = [], [], []

    for odd in odds_data:
        # Filter to Fulltime Result market (market_id=1)
        if odd.get('market_id') != 1:
            continue

        label = (odd.get('label') or '').lower()
        value = odd.get('value')
        if value is None:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if label == 'home':
            all_home.append(value)
        elif label == 'draw':
            all_draw.append(value)
        elif label == 'away':
            all_away.append(value)

    return {
        'best_home_odds': max(all_home) if all_home else None,
        'best_draw_odds': max(all_draw) if all_draw else None,
        'best_away_odds': max(all_away) if all_away else None,
        'avg_home_odds': float(np.mean(all_home)) if all_home else None,
        'avg_draw_odds': float(np.mean(all_draw)) if all_draw else None,
        'avg_away_odds': float(np.mean(all_away)) if all_away else None,
        'odds_count': len(all_home)
    }


class LivePipeline:
    """Live prediction pipeline."""

    def __init__(self, api_key: str, history_days: int = 365, strategy: str = None, reference_date: datetime = None):
        self.api_key = api_key
        self.client = SportMonksClient(api_key)
        self.reference_date = reference_date or datetime.now()
        self.is_backtest = reference_date is not None

        # Load strategy config
        self.strategy_config = STRATEGY_PROFILES.get(strategy) if strategy else get_active_strategy()
        self.thresholds = self.strategy_config['thresholds']
        self.odds_filter = self.strategy_config['odds_filter']

        logger.info("=" * 60)
        logger.info("INITIALIZING V5 LIVE PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Strategy: {strategy or 'default'}")
        logger.info(f"Thresholds: H={self.thresholds['home']}, A={self.thresholds['away']}, D={self.thresholds['draw']}")
        if self.is_backtest:
            logger.info(f"Reference date: {reference_date.strftime('%Y-%m-%d')} (backdated mode)")

        # Initialize data loader and fetch historical data via API
        self.data_loader = InMemoryDataLoader()
        self._load_history_from_api(history_days)

        # Initialize calculators
        logger.info("\nInitializing calculators...")
        self.elo_calc = EloCalculator(
            k_factor=EloConfig.K_FACTOR,
            home_advantage=EloConfig.HOME_ADVANTAGE,
            initial_elo=EloConfig.INITIAL_ELO
        )
        fixtures_df = self.data_loader.fixtures_df
        if len(fixtures_df) > 0:
            self.elo_calc.calculate_elo_history(fixtures_df)
            logger.info(f"Calculated Elo for {len(self.elo_calc.elo_history)} teams")

        self.standings_calc = StandingsCalculator()

        # Initialize feature engines
        logger.info("Initializing feature engines...")
        self.pillar1 = Pillar1FundamentalsEngine(self.data_loader, self.standings_calc, self.elo_calc)
        self.pillar2 = Pillar2ModernAnalyticsEngine(self.data_loader)
        self.pillar3 = Pillar3HiddenEdgesEngine(self.data_loader, self.standings_calc, self.elo_calc)

        # Load model
        logger.info("\nLoading model...")
        model_path = get_latest_model_path()
        self.model_data = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE READY")
        logger.info("=" * 60)

    def _load_history_from_api(self, history_days: int):
        """Load historical data from API (with date chunking for API limits)."""
        logger.info(f"\nFetching {history_days} days of historical data from API...")
        chunk_days = 30  # SportMonks API date range limit
        all_fixtures = []

        current_start = self.reference_date - timedelta(days=history_days)
        end = self.reference_date
        chunk_num = 0
        total_chunks = (history_days // chunk_days) + 1

        while current_start < end:
            chunk_end = min(current_start + timedelta(days=chunk_days), end)
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = chunk_end.strftime('%Y-%m-%d')
            chunk_num += 1

            logger.info(f"  Chunk {chunk_num}/{total_chunks}: {start_str} to {end_str}...")

            # Fetch per league to keep responses manageable
            for league_id in TOP_5_LEAGUES:
                fixtures = self.client.get_fixtures_between(
                    start_str, end_str,
                    include_details=True,
                    finished_only=True,
                    league_id=league_id
                )
                all_fixtures.extend(fixtures)

            logger.info(f"    Total so far: {len(all_fixtures)} fixtures")
            current_start = chunk_end + timedelta(days=1)

        historical_df = parse_fixtures_to_df(all_fixtures)
        if len(historical_df) > 0:
            self.data_loader.add_fixtures(historical_df)
        logger.info(f"Loaded {len(historical_df)} historical fixtures from API")

    def generate_features(self, fixture: dict) -> dict:
        """Generate features for a fixture."""
        participants = fixture.get('participants', [])
        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

        if not home_team or not away_team:
            return None

        as_of_date = pd.to_datetime(fixture.get('starting_at'))
        home_team_id = home_team.get('id')
        away_team_id = away_team.get('id')
        season_id = fixture.get('season_id')
        league_id = fixture.get('league_id')

        features = {}

        # Pillar 1
        features.update(self.pillar1.generate_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date, self.data_loader.fixtures_df
        ))

        # Pillar 2
        features.update(self.pillar2.generate_features(
            home_team_id, away_team_id, as_of_date
        ))

        # Pillar 3
        features.update(self.pillar3.generate_features(
            home_team_id, away_team_id, season_id, league_id, as_of_date, self.data_loader.fixtures_df
        ))

        return features

    def predict(self, fixture: dict) -> dict:
        """Generate prediction for a fixture."""
        features = self.generate_features(fixture)
        if features is None:
            return None

        # Get model components
        catboost = self.model_data['catboost']
        lightgbm = self.model_data['lightgbm']
        feature_cols = self.model_data['feature_cols']

        # Prepare feature vector
        X = pd.DataFrame([features])

        # Ensure all columns exist (fill missing with 0)
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Get ensemble probabilities
        probs_cat = catboost.predict_proba(X)[0]
        probs_lgb = lightgbm.predict_proba(X)[0]
        probs = (probs_cat + probs_lgb) / 2  # [Away, Draw, Home]

        p_away, p_draw, p_home = probs

        # Determine predicted outcome and whether to bet
        predicted_outcome = 'Home Win' if p_home >= p_away and p_home >= p_draw else (
            'Away Win' if p_away >= p_draw else 'Draw')

        # Get odds
        odds = extract_odds(fixture)

        # Apply thresholds and odds filter
        bet_outcome = None
        bet_probability = None
        bet_odds = None
        should_bet = False

        candidates = []
        if p_home >= self.thresholds['home']:
            candidates.append(('Home Win', p_home, odds.get('best_home_odds')))
        if p_away >= self.thresholds['away']:
            candidates.append(('Away Win', p_away, odds.get('best_away_odds')))
        if p_draw >= self.thresholds['draw']:
            candidates.append(('Draw', p_draw, odds.get('best_draw_odds')))

        # Filter by odds range (skip if odds not available, e.g. backtest mode)
        has_odds = any(odd is not None for _, _, odd in candidates)
        if has_odds and self.odds_filter.get('enabled', True):
            min_odds = self.odds_filter.get('min', 1.3)
            max_odds = self.odds_filter.get('max', 3.0)
            candidates = [(o, p, odd) for o, p, odd in candidates
                         if odd and min_odds <= odd <= max_odds]

        if candidates:
            # Pick highest probability
            best = max(candidates, key=lambda x: x[1])
            bet_outcome, bet_probability, bet_odds = best
            should_bet = True

        # Extract team info
        participants = fixture.get('participants', [])
        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), {})
        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), {})

        return {
            'fixture_id': fixture.get('id'),
            'match_date': fixture.get('starting_at'),
            'league_id': fixture.get('league_id'),
            'season_id': fixture.get('season_id'),
            'home_team_id': home_team.get('id'),
            'home_team_name': home_team.get('name'),
            'away_team_id': away_team.get('id'),
            'away_team_name': away_team.get('name'),
            'pred_home_prob': float(p_home),
            'pred_draw_prob': float(p_draw),
            'pred_away_prob': float(p_away),
            'predicted_outcome': predicted_outcome,
            'bet_outcome': bet_outcome,
            'bet_probability': float(bet_probability) if bet_probability is not None else None,
            'bet_odds': float(bet_odds) if bet_odds is not None else None,
            'should_bet': should_bet,
            **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in odds.items()},
            'features': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in features.items()},
            'model_version': 'v5'
        }


def print_backtest_summary(predictions: list, start_date: str, end_date: str):
    """Print detailed backtest results comparing predictions vs actual."""
    bets = [p for p in predictions if p['should_bet']]
    total_matches = len(predictions)
    total_bets = len(bets)

    if total_bets == 0:
        logger.info("No bets recommended in this period.")
        return

    # Calculate weeks
    from datetime import datetime as dt
    d1 = dt.strptime(start_date, '%Y-%m-%d')
    d2 = dt.strptime(end_date, '%Y-%m-%d')
    weeks = max((d2 - d1).days / 7, 1)

    # Track results per outcome
    outcome_map = {'Home Win': 'H', 'Away Win': 'A', 'Draw': 'D'}
    results = {'Home Win': {'bets': 0, 'correct': 0}, 'Away Win': {'bets': 0, 'correct': 0}, 'Draw': {'bets': 0, 'correct': 0}}

    logger.info("\n" + "=" * 100)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 100)
    logger.info(f"Period: {start_date} to {end_date} ({weeks:.1f} weeks)")
    logger.info(f"Total matches: {total_matches} | Bets recommended: {total_bets}")
    logger.info("")

    # Match-by-match results
    logger.info(f"{'Date':<12} {'Match':<45} {'Bet':<12} {'Prob':>6} {'Actual':>8} {'Result':>8}")
    logger.info("-" * 100)

    for pred in sorted(bets, key=lambda x: x['match_date']):
        bet_outcome = pred['bet_outcome']
        actual_result = pred.get('actual_result')

        if bet_outcome and actual_result:
            expected_code = outcome_map.get(bet_outcome)
            is_correct = expected_code == actual_result
            result_str = 'WIN' if is_correct else 'LOSS'

            results[bet_outcome]['bets'] += 1
            if is_correct:
                results[bet_outcome]['correct'] += 1
        else:
            result_str = 'N/A'

        match_str = f"{pred['home_team_name']} vs {pred['away_team_name']}"
        if len(match_str) > 43:
            match_str = match_str[:43]

        date_str = str(pred['match_date'])[:10]
        actual_str = actual_result or '?'
        prob_str = f"{pred['bet_probability']:.0%}" if pred['bet_probability'] else 'N/A'

        logger.info(f"{date_str:<12} {match_str:<45} {bet_outcome:<12} {prob_str:>6} {actual_str:>8} {result_str:>8}")

    # Summary
    total_correct = sum(r['correct'] for r in results.values())
    total_counted = sum(r['bets'] for r in results.values())
    overall_wr = total_correct / total_counted * 100 if total_counted > 0 else 0

    logger.info("")
    logger.info("=" * 100)
    logger.info("SUMMARY")
    logger.info("=" * 100)
    logger.info(f"Total bets:     {total_counted}")
    logger.info(f"Bets per week:  {total_counted / weeks:.1f}")
    logger.info(f"Overall WR:     {overall_wr:.1f}% ({total_correct}/{total_counted})")
    logger.info("")

    for outcome in ['Home Win', 'Away Win', 'Draw']:
        r = results[outcome]
        if r['bets'] > 0:
            wr = r['correct'] / r['bets'] * 100
            logger.info(f"  {outcome:<10}: {r['bets']:>3} bets, {r['correct']:>3} correct, WR: {wr:.1f}%")
        else:
            logger.info(f"  {outcome:<10}:   0 bets")

    logger.info("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='V5 Live Prediction')
    parser.add_argument('--days-ahead', type=int, default=7, help='Days ahead to predict')
    parser.add_argument('--history-days', type=int, default=365, help='Days of history to load')
    parser.add_argument('--strategy', choices=list(STRATEGY_PROFILES.keys()), help='Betting strategy')
    parser.add_argument('--dry-run', action='store_true', help='Do not save to database')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get('SPORTMONKS_API_KEY')
    if not api_key:
        logger.error("SPORTMONKS_API_KEY not set")
        sys.exit(1)

    # =========================================================================
    # BACKDATED MODE
    # =========================================================================
    if args.start_date and args.end_date:
        logger.info("=" * 80)
        logger.info("BACKDATED PREDICTION MODE")
        logger.info("=" * 80)

        reference_date = datetime.strptime(args.start_date, '%Y-%m-%d')

        # Initialize pipeline with reference_date (loads history before start_date)
        pipeline = LivePipeline(
            api_key,
            history_days=args.history_days,
            strategy=args.strategy,
            reference_date=reference_date
        )

        # Fetch finished fixtures in the backtest window (with odds for PnL)
        logger.info(f"\nFetching finished fixtures from {args.start_date} to {args.end_date}...")
        backtest_fixtures = pipeline.client.get_fixtures_between(
            args.start_date, args.end_date,
            include_details=True,
            finished_only=True,
            include_odds=True
        )

        # Filter to top 5 leagues if configured
        if FILTER_TOP_5_ONLY:
            backtest_fixtures = [f for f in backtest_fixtures if f.get('league_id') in TOP_5_LEAGUES]

        logger.info(f"Found {len(backtest_fixtures)} finished fixtures in top 5 leagues")

        # Generate predictions and compare with actuals
        predictions = []
        for fixture in backtest_fixtures:
            try:
                pred = pipeline.predict(fixture)
                if pred is None:
                    continue

                # Add actual result from finished fixture
                participants = fixture.get('participants', [])
                scores = fixture.get('scores', [])
                home_score, away_score = None, None
                for score in scores:
                    if score.get('description') == 'CURRENT':
                        score_data = score.get('score', {})
                        if score_data.get('participant') == 'home':
                            home_score = score_data.get('goals')
                        elif score_data.get('participant') == 'away':
                            away_score = score_data.get('goals')

                if home_score is not None and away_score is not None:
                    actual = 'H' if home_score > away_score else ('A' if home_score < away_score else 'D')
                    pred['actual_result'] = actual
                    pred['actual_home_score'] = home_score
                    pred['actual_away_score'] = away_score

                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict fixture {fixture.get('id')}: {e}")

        # Print backtest summary
        print_backtest_summary(predictions, args.start_date, args.end_date)

        # Optionally save to database
        if not args.dry_run and predictions:
            logger.info("\nSaving to database...")
            db = DatabaseClient(DATABASE_URL)
            db.create_tables()
            count = db.store_predictions_batch(predictions)
            logger.info(f"Saved {count} predictions")

        return

    # =========================================================================
    # LIVE MODE (existing behavior)
    # =========================================================================
    pipeline = LivePipeline(api_key, history_days=args.history_days, strategy=args.strategy)

    # Fetch upcoming fixtures
    logger.info(f"\nFetching fixtures for next {args.days_ahead} days...")
    league_filter = TOP_5_LEAGUES if FILTER_TOP_5_ONLY else None
    fixtures = pipeline.client.get_upcoming_fixtures(days_ahead=args.days_ahead, league_ids=league_filter)
    logger.info(f"Found {len(fixtures)} upcoming fixtures")

    # Generate predictions
    predictions = []
    for fixture in fixtures:
        try:
            pred = pipeline.predict(fixture)
            if pred:
                predictions.append(pred)
        except Exception as e:
            logger.warning(f"Failed to predict fixture {fixture.get('id')}: {e}")

    logger.info(f"\nGenerated {len(predictions)} predictions")

    # Filter to bets
    bets = [p for p in predictions if p['should_bet']]
    logger.info(f"Recommended bets: {len(bets)}")

    # Display predictions
    if bets:
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDED BETS")
        logger.info("=" * 80)

        for pred in sorted(bets, key=lambda x: x['match_date']):
            logger.info(f"\n{pred['match_date'][:10]}: {pred['home_team_name']} vs {pred['away_team_name']}")
            logger.info(f"  Prediction: {pred['bet_outcome']} ({pred['bet_probability']:.1%})")
            logger.info(f"  Odds: {pred['bet_odds']:.2f}" if pred['bet_odds'] else "  Odds: N/A")
            logger.info(f"  Probs: H={pred['pred_home_prob']:.1%}, D={pred['pred_draw_prob']:.1%}, A={pred['pred_away_prob']:.1%}")

    # Save to database
    if not args.dry_run and predictions:
        logger.info("\nSaving to database...")
        db = DatabaseClient(DATABASE_URL)
        db.create_tables()
        count = db.store_predictions_batch(predictions)
        logger.info(f"Saved {count} predictions")
    elif args.dry_run:
        logger.info("\nDry run - not saving to database")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
