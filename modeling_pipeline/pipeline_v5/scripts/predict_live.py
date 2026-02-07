#!/usr/bin/env python3
"""
Live Prediction Script for V5 Pipeline (Multi-Strategy)
========================================================

Generates predictions for upcoming fixtures using 3 independent strategies
and saves all to PostgreSQL. Each fixture gets up to 3 bet decisions.

Strategies (configured in config/production_config.py → BETTING_STRATEGIES):
  - threshold:  Probability thresholds + odds filter (baseline)
  - hybrid:     Thresholds (no odds filter) + GBM bet selector
  - selector:   Pure GBM bet selector

Usage:
    export SPORTMONKS_API_KEY="your_key"

    # Predict upcoming fixtures (all 3 strategies)
    python3 scripts/predict_live.py --days-ahead 7

    # Dry run (don't save to DB)
    python3 scripts/predict_live.py --days-ahead 7 --dry-run

    # Backtest mode
    python3 scripts/predict_live.py --start-date 2026-01-01 --end-date 2026-01-31
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
    DATABASE_URL, TOP_5_LEAGUES,
    FILTER_TOP_5_ONLY, BETTING_STRATEGIES, GOALS_BETTING_STRATEGIES,
    EloConfig, get_latest_model_path, get_latest_goals_model_path,
)
from src.goals import PoissonGoalsModel
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


def extract_goals_odds(fixture: dict) -> dict:
    """Extract Over/Under 2.5 and BTTS odds from fixture.

    SportMonks market IDs:
        market_id=80: Over/Under (labels: Over/Under, total field = line e.g. 2.5)
        market_id=14: BTTS (labels: Yes/No)
    """
    odds_data = fixture.get('odds', [])
    all_over_2_5, all_under_2_5 = [], []
    all_btts_yes, all_btts_no = [], []

    for odd in odds_data:
        market_id = odd.get('market_id')
        label = (odd.get('label') or '').lower().strip()
        value = odd.get('value')
        if value is None:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if market_id == 80:
            # Over/Under — filter for 2.5 line
            total = odd.get('total')
            if total is not None:
                try:
                    total = float(total)
                except (ValueError, TypeError):
                    continue
                if abs(total - 2.5) < 0.01:
                    if label == 'over':
                        all_over_2_5.append(value)
                    elif label == 'under':
                        all_under_2_5.append(value)

        elif market_id == 14:
            # BTTS
            if label == 'yes':
                all_btts_yes.append(value)
            elif label == 'no':
                all_btts_no.append(value)

    return {
        'ou_2_5_over_odds': max(all_over_2_5) if all_over_2_5 else None,
        'ou_2_5_under_odds': max(all_under_2_5) if all_under_2_5 else None,
        'btts_yes_odds': max(all_btts_yes) if all_btts_yes else None,
        'btts_no_odds': max(all_btts_no) if all_btts_no else None,
    }


def evaluate_goals_bets(market_pred: dict, goals_odds: dict,
                        strategies: dict = None) -> dict:
    """Evaluate value-based betting decisions for goals markets.

    Compares model probabilities to implied probabilities from bookmaker odds.
    Bets when edge (model_prob - implied_prob) exceeds threshold.

    Args:
        market_pred: Market prediction dict with over_2_5_prob, btts_prob, etc.
        goals_odds: Dict from extract_goals_odds() with best odds.
        strategies: GOALS_BETTING_STRATEGIES config dict.

    Returns:
        Dict with bet decisions to merge into market_pred.
    """
    if strategies is None:
        strategies = GOALS_BETTING_STRATEGIES

    result = {}
    result.update(goals_odds)  # Always store odds

    # --- O/U 2.5 ---
    ou_cfg = strategies.get('ou_2_5', {})
    if ou_cfg.get('enabled', False):
        over_prob = market_pred.get('over_2_5_prob')
        over_odds = goals_odds.get('ou_2_5_over_odds')
        under_odds = goals_odds.get('ou_2_5_under_odds')

        if over_prob is not None:
            under_prob = 1.0 - over_prob
            min_edge = ou_cfg.get('min_edge', 0.05)
            min_odds = ou_cfg.get('min_odds', 1.5)
            max_odds = ou_cfg.get('max_odds', 3.0)
            min_prob = ou_cfg.get('min_prob', 0.55)

            # Check Over bet
            if over_odds and over_prob >= min_prob and min_odds <= over_odds <= max_odds:
                over_implied = 1.0 / over_odds
                edge = over_prob - over_implied
                if edge >= min_edge:
                    result['ou_2_5_bet'] = 'over'
                    result['ou_2_5_bet_odds'] = float(over_odds)
                    result['ou_2_5_edge'] = round(float(edge), 4)

            # Check Under bet (only if no Over bet)
            if 'ou_2_5_bet' not in result and under_odds and under_prob >= min_prob and min_odds <= under_odds <= max_odds:
                under_implied = 1.0 / under_odds
                edge = under_prob - under_implied
                if edge >= min_edge:
                    result['ou_2_5_bet'] = 'under'
                    result['ou_2_5_bet_odds'] = float(under_odds)
                    result['ou_2_5_edge'] = round(float(edge), 4)

    # --- BTTS ---
    btts_cfg = strategies.get('btts', {})
    if btts_cfg.get('enabled', False):
        btts_prob = market_pred.get('btts_prob')
        yes_odds = goals_odds.get('btts_yes_odds')
        no_odds = goals_odds.get('btts_no_odds')

        if btts_prob is not None:
            no_prob = 1.0 - btts_prob
            min_edge = btts_cfg.get('min_edge', 0.05)
            min_odds = btts_cfg.get('min_odds', 1.5)
            max_odds = btts_cfg.get('max_odds', 3.0)
            min_prob = btts_cfg.get('min_prob', 0.55)

            # Check Yes bet
            if yes_odds and btts_prob >= min_prob and min_odds <= yes_odds <= max_odds:
                yes_implied = 1.0 / yes_odds
                edge = btts_prob - yes_implied
                if edge >= min_edge:
                    result['btts_bet'] = 'yes'
                    result['btts_bet_odds'] = float(yes_odds)
                    result['btts_edge'] = round(float(edge), 4)

            # Check No bet (only if no Yes bet)
            if 'btts_bet' not in result and no_odds and no_prob >= min_prob and min_odds <= no_odds <= max_odds:
                no_implied = 1.0 / no_odds
                edge = no_prob - no_implied
                if edge >= min_edge:
                    result['btts_bet'] = 'no'
                    result['btts_bet_odds'] = float(no_odds)
                    result['btts_edge'] = round(float(edge), 4)

    return result


class LivePipeline:
    """Live prediction pipeline with multi-strategy support."""

    def __init__(self, api_key: str, history_days: int = 365, reference_date: datetime = None):
        self.api_key = api_key
        self.client = SportMonksClient(api_key)
        self.reference_date = reference_date or datetime.now()
        self.is_backtest = reference_date is not None

        # Load multi-strategy config
        self.strategies = {k: v for k, v in BETTING_STRATEGIES.items() if v.get('enabled', True)}

        logger.info("=" * 60)
        logger.info("INITIALIZING V5 LIVE PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Active strategies: {list(self.strategies.keys())}")
        for name, cfg in self.strategies.items():
            logger.info(f"  {name}: {cfg.get('description', '')}")
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

        # Load goals model (optional — for advanced market predictions)
        self.goals_model = None
        goals_path = get_latest_goals_model_path()
        if goals_path and Path(goals_path).exists():
            self.goals_model = PoissonGoalsModel()
            self.goals_model.load(goals_path)
            logger.info(f"Loaded goals model from {goals_path}")
        else:
            logger.info("Goals model not found — market predictions will be skipped")

        # Load bet selector (needed by hybrid + selector strategies)
        self.selectors = {}  # strategy_name -> BetSelector
        self.market_extractor = None
        needs_selector = any('selector' in cfg for cfg in self.strategies.values())

        if needs_selector:
            from src.market import MarketFeatureExtractor, BetSelector
            self.market_extractor = MarketFeatureExtractor()

            for strat_name, cfg in self.strategies.items():
                sel_cfg = cfg.get('selector')
                if not sel_cfg:
                    continue
                sel_path = Path(__file__).parent.parent / sel_cfg['model_path']
                if sel_path.exists():
                    selector = BetSelector(min_confidence=sel_cfg.get('min_confidence', 0.55))
                    selector.load(str(sel_path))
                    self.selectors[strat_name] = selector
                    logger.info(f"Loaded bet selector for '{strat_name}' (conf={selector.min_confidence})")
                else:
                    logger.warning(f"Bet selector not found at {sel_path} for '{strat_name}'")

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

    def predict(self, fixture: dict) -> tuple:
        """Generate predictions for a fixture (one per enabled strategy).

        Returns:
            (predictions_list, market_pred_or_none) — list of 1X2 prediction dicts
            and optional market prediction dict.
        """
        features = self.generate_features(fixture)
        if features is None:
            return [], None

        # Get model components
        catboost = self.model_data['catboost']
        lightgbm = self.model_data['lightgbm']
        feature_cols = self.model_data['feature_cols']

        # Prepare feature vector
        X = pd.DataFrame([features])
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Get ensemble probabilities (computed ONCE, shared by all strategies)
        probs_cat = catboost.predict_proba(X)[0]
        probs_lgb = lightgbm.predict_proba(X)[0]
        probs = (probs_cat + probs_lgb) / 2  # [Away, Draw, Home]
        p_away, p_draw, p_home = probs

        predicted_outcome = 'Home Win' if p_home >= p_away and p_home >= p_draw else (
            'Away Win' if p_away >= p_draw else 'Draw')

        # Extract odds (computed ONCE)
        odds = extract_odds(fixture)

        # Extract market features ONCE (shared by hybrid + selector strategies)
        market_full_feats = None
        if self.market_extractor:
            raw_market = self.market_extractor.extract_from_api(fixture.get('odds', []))
            if raw_market is not None:
                model_probs = {'home': float(p_home), 'draw': float(p_draw), 'away': float(p_away)}
                market_full_feats = self.market_extractor.build_full_features(raw_market, model_probs)

        # Extract team info
        participants = fixture.get('participants', [])
        home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), {})
        away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), {})

        # Base prediction dict (shared fields)
        base = {
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
            **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in odds.items()},
            'features': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in features.items()},
            'model_version': 'v5',
        }

        # Generate market predictions from goals model (if available)
        market_pred = None
        if self.goals_model is not None:
            try:
                lh, la = self.goals_model.predict_single(X)
                markets = self.goals_model.derive_markets(lh, la)
                market_pred = {
                    'fixture_id': fixture.get('id'),
                    'match_date': fixture.get('starting_at'),
                    'league_id': fixture.get('league_id'),
                    'home_team_name': home_team.get('name'),
                    'away_team_name': away_team.get('name'),
                    **markets,
                }
            except Exception as e:
                logger.warning(f"Goals model failed for fixture {fixture.get('id')}: {e}")

        # Evaluate goals market bets (if market_pred exists and odds available)
        if market_pred is not None:
            goals_odds = extract_goals_odds(fixture)
            bet_decisions = evaluate_goals_bets(market_pred, goals_odds)
            market_pred.update(bet_decisions)

        # Evaluate each strategy independently
        results = []
        for strat_name, strat_cfg in self.strategies.items():
            bet_outcome, bet_probability, bet_odds_val, should_bet = self._evaluate_strategy(
                strat_name, strat_cfg, p_home, p_draw, p_away, odds, market_full_feats
            )

            pred = dict(base)
            pred['strategy'] = strat_name
            pred['bet_outcome'] = bet_outcome
            pred['bet_probability'] = float(bet_probability) if bet_probability is not None else None
            pred['bet_odds'] = float(bet_odds_val) if bet_odds_val is not None else None
            pred['should_bet'] = should_bet
            results.append(pred)

        return results, market_pred

    def _evaluate_strategy(self, strat_name: str, strat_cfg: dict,
                           p_home: float, p_draw: float, p_away: float,
                           odds: dict, market_full_feats: dict):
        """Evaluate a single strategy for one fixture.

        Returns:
            (bet_outcome, bet_probability, bet_odds, should_bet)
        """
        has_thresholds = 'thresholds' in strat_cfg
        has_selector = 'selector' in strat_cfg and strat_name in self.selectors

        # Step 1: Threshold check (if applicable)
        threshold_candidate = None
        if has_thresholds:
            thresholds = strat_cfg['thresholds']
            candidates = []
            if p_home >= thresholds.get('home', 1.0):
                candidates.append(('Home Win', p_home, odds.get('best_home_odds')))
            if p_away >= thresholds.get('away', 1.0):
                candidates.append(('Away Win', p_away, odds.get('best_away_odds')))
            if p_draw >= thresholds.get('draw', 1.0):
                candidates.append(('Draw', p_draw, odds.get('best_draw_odds')))

            # Apply odds filter if configured
            odds_filter = strat_cfg.get('odds_filter', {})
            if odds_filter.get('enabled', False):
                has_any_odds = any(odd is not None for _, _, odd in candidates)
                if has_any_odds:
                    min_odds = odds_filter.get('min', 1.3)
                    max_odds = odds_filter.get('max', 3.5)
                    candidates = [(o, p, odd) for o, p, odd in candidates
                                  if odd and min_odds <= odd <= max_odds]

            if candidates:
                threshold_candidate = max(candidates, key=lambda x: x[1])

        # Step 2: Apply selector if present
        if has_selector and has_thresholds:
            # Hybrid mode: threshold must pass first, then selector decides
            if threshold_candidate is None:
                return None, None, None, False
            if market_full_feats is None:
                return None, None, None, False
            if self.selectors[strat_name].predict(market_full_feats):
                return threshold_candidate[0], threshold_candidate[1], threshold_candidate[2], True
            return None, None, None, False

        elif has_selector and not has_thresholds:
            # Pure selector mode: selector decides, bet on highest-prob outcome
            if market_full_feats is None:
                return None, None, None, False
            if self.selectors[strat_name].predict(market_full_feats):
                prob_map = {'Home Win': (p_home, odds.get('best_home_odds')),
                            'Away Win': (p_away, odds.get('best_away_odds')),
                            'Draw': (p_draw, odds.get('best_draw_odds'))}
                best = max(prob_map, key=lambda k: prob_map[k][0])
                return best, prob_map[best][0], prob_map[best][1], True
            return None, None, None, False

        elif has_thresholds:
            # Pure threshold mode
            if threshold_candidate:
                return threshold_candidate[0], threshold_candidate[1], threshold_candidate[2], True
            return None, None, None, False

        return None, None, None, False


def print_backtest_summary(predictions: list, start_date: str, end_date: str):
    """Print detailed backtest results with per-strategy breakdown."""
    from datetime import datetime as dt
    d1 = dt.strptime(start_date, '%Y-%m-%d')
    d2 = dt.strptime(end_date, '%Y-%m-%d')
    weeks = max((d2 - d1).days / 7, 1)

    # Group by strategy
    by_strategy = {}
    for p in predictions:
        strat = p.get('strategy', 'threshold')
        by_strategy.setdefault(strat, []).append(p)

    logger.info("\n" + "=" * 100)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 100)
    logger.info(f"Period: {start_date} to {end_date} ({weeks:.1f} weeks)")

    outcome_map = {'Home Win': 'H', 'Away Win': 'A', 'Draw': 'D'}

    for strat_name, strat_preds in by_strategy.items():
        bets = [p for p in strat_preds if p['should_bet']]
        total_bets = len(bets)

        logger.info(f"\n--- Strategy: {strat_name} ({total_bets} bets) ---")

        if total_bets == 0:
            logger.info("  No bets recommended.")
            continue

        wins, losses, total_profit = 0, 0, 0.0
        for pred in bets:
            actual = pred.get('actual_result')
            bet_code = outcome_map.get(pred['bet_outcome'])
            if actual and bet_code:
                if bet_code == actual:
                    wins += 1
                    total_profit += (pred['bet_odds'] - 1) if pred['bet_odds'] else 0
                else:
                    losses += 1
                    total_profit -= 1.0

        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        roi = total_profit / (wins + losses) * 100 if (wins + losses) > 0 else 0

        logger.info(f"  Bets: {wins + losses} | Wins: {wins} | WR: {wr:.1f}% | Profit: ${total_profit:.2f} | ROI: {roi:.1f}%")

    logger.info("\n" + "=" * 100)


def print_live_summary(predictions: list):
    """Print multi-strategy bet recommendations."""
    bets = [p for p in predictions if p['should_bet']]
    if not bets:
        logger.info("No bets recommended.")
        return

    # Group by fixture
    by_fixture = {}
    for p in bets:
        fid = p['fixture_id']
        by_fixture.setdefault(fid, []).append(p)

    logger.info("\n" + "=" * 100)
    logger.info("RECOMMENDED BETS (all strategies)")
    logger.info("=" * 100)

    for fid, fixture_bets in sorted(by_fixture.items(), key=lambda x: x[1][0]['match_date']):
        first = fixture_bets[0]
        logger.info(f"\n{str(first['match_date'])[:10]}: {first['home_team_name']} vs {first['away_team_name']}")
        logger.info(f"  Probs: H={first['pred_home_prob']:.1%}, D={first['pred_draw_prob']:.1%}, A={first['pred_away_prob']:.1%}")

        for pred in fixture_bets:
            odds_str = f"{pred['bet_odds']:.2f}" if pred['bet_odds'] else "N/A"
            logger.info(f"  [{pred['strategy']:<10}] {pred['bet_outcome']} ({pred['bet_probability']:.1%}) @ {odds_str}")

    # Summary counts
    logger.info(f"\n--- Summary ---")
    for strat in sorted(set(p['strategy'] for p in predictions)):
        strat_bets = [p for p in predictions if p['strategy'] == strat and p['should_bet']]
        logger.info(f"  {strat}: {len(strat_bets)} bets")
    logger.info("=" * 100)


def print_market_summary(market_predictions: list):
    """Print market predictions (O/U, BTTS, top scoreline) + betting recommendations."""
    logger.info("\n" + "=" * 100)
    logger.info("MARKET PREDICTIONS (Goals Model)")
    logger.info("=" * 100)

    market_bets = []
    for mp in sorted(market_predictions, key=lambda x: str(x.get('match_date', ''))):
        date_str = str(mp.get('match_date', ''))[:10]
        logger.info(f"\n{date_str}: {mp.get('home_team_name')} vs {mp.get('away_team_name')}")
        logger.info(f"  xG: Home={mp['home_goals_lambda']:.2f}, Away={mp['away_goals_lambda']:.2f}")
        logger.info(f"  O/U 2.5: {mp['over_2_5_prob']:.1%} over | O/U 1.5: {mp['over_1_5_prob']:.1%} over | O/U 3.5: {mp['over_3_5_prob']:.1%} over")
        logger.info(f"  BTTS: {mp['btts_prob']:.1%}")

        scorelines = mp.get('top_scorelines', [])
        if scorelines:
            sl_str = ', '.join(f"{s['home']}-{s['away']} ({s['prob']:.1%})" for s in scorelines[:3])
            logger.info(f"  Top scorelines: {sl_str}")

        # Show odds and betting decisions
        odds_parts = []
        if mp.get('ou_2_5_over_odds'):
            odds_parts.append(f"O/U 2.5: {mp['ou_2_5_over_odds']:.2f}/{mp.get('ou_2_5_under_odds', 0):.2f}")
        if mp.get('btts_yes_odds'):
            odds_parts.append(f"BTTS: {mp['btts_yes_odds']:.2f}/{mp.get('btts_no_odds', 0):.2f}")
        if odds_parts:
            logger.info(f"  Odds: {' | '.join(odds_parts)}")

        bets = []
        if mp.get('ou_2_5_bet'):
            bets.append(f"O/U 2.5 {mp['ou_2_5_bet'].upper()} @ {mp['ou_2_5_bet_odds']:.2f} (edge={mp['ou_2_5_edge']:.1%})")
        if mp.get('btts_bet'):
            bets.append(f"BTTS {mp['btts_bet'].upper()} @ {mp['btts_bet_odds']:.2f} (edge={mp['btts_edge']:.1%})")
        if bets:
            logger.info(f"  >>> BET: {' | '.join(bets)}")
            market_bets.append(mp)

    # Summary
    ou_bets = [mp for mp in market_predictions if mp.get('ou_2_5_bet')]
    btts_bets = [mp for mp in market_predictions if mp.get('btts_bet')]
    logger.info(f"\n--- Goals Market Bets Summary ---")
    logger.info(f"  O/U 2.5 bets: {len(ou_bets)} | BTTS bets: {len(btts_bets)}")
    logger.info("=" * 100)


def print_market_backtest_summary(market_predictions: list):
    """Print backtest PnL for goals market bets."""
    ou_bets = [mp for mp in market_predictions if mp.get('ou_2_5_bet')]
    btts_bets = [mp for mp in market_predictions if mp.get('btts_bet')]

    if not ou_bets and not btts_bets:
        return

    logger.info("\n" + "=" * 100)
    logger.info("GOALS MARKET BACKTEST PnL")
    logger.info("=" * 100)

    for label, bets, bet_key, odds_key in [
        ('O/U 2.5', ou_bets, 'ou_2_5_bet', 'ou_2_5_bet_odds'),
        ('BTTS', btts_bets, 'btts_bet', 'btts_bet_odds'),
    ]:
        if not bets:
            logger.info(f"\n{label}: No bets")
            continue

        wins, losses, profit = 0, 0, 0.0
        for mp in bets:
            actual_h = mp.get('actual_home_score')
            actual_a = mp.get('actual_away_score')
            if actual_h is None or actual_a is None:
                continue

            bet_dir = mp[bet_key]
            bet_odds = mp[odds_key]

            if label == 'O/U 2.5':
                actual_over = (actual_h + actual_a) > 2.5
                won = (bet_dir == 'over' and actual_over) or (bet_dir == 'under' and not actual_over)
            else:  # BTTS
                actual_btts = actual_h >= 1 and actual_a >= 1
                won = (bet_dir == 'yes' and actual_btts) or (bet_dir == 'no' and not actual_btts)

            if won:
                wins += 1
                profit += (bet_odds - 1)
            else:
                losses += 1
                profit -= 1.0

        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        roi = profit / total * 100 if total > 0 else 0
        logger.info(f"\n{label}:")
        logger.info(f"  Bets: {total} | Wins: {wins} | WR: {wr:.1f}% | Profit: ${profit:.2f} | ROI: {roi:.1f}%")

    logger.info("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='V5 Live Prediction (multi-strategy)')
    parser.add_argument('--days-ahead', type=int, default=7, help='Days ahead to predict')
    parser.add_argument('--history-days', type=int, default=365, help='Days of history to load')
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
        logger.info("BACKDATED PREDICTION MODE (multi-strategy)")
        logger.info("=" * 80)

        reference_date = datetime.strptime(args.start_date, '%Y-%m-%d')

        pipeline = LivePipeline(
            api_key,
            history_days=args.history_days,
            reference_date=reference_date,
        )

        logger.info(f"\nFetching finished fixtures from {args.start_date} to {args.end_date}...")
        backtest_fixtures = pipeline.client.get_fixtures_between(
            args.start_date, args.end_date,
            include_details=True,
            finished_only=True,
            include_odds=True
        )

        if FILTER_TOP_5_ONLY:
            backtest_fixtures = [f for f in backtest_fixtures if f.get('league_id') in TOP_5_LEAGUES]

        logger.info(f"Found {len(backtest_fixtures)} finished fixtures in top 5 leagues")

        # Generate predictions (returns list of lists → flatten)
        predictions = []
        market_predictions = []
        for fixture in backtest_fixtures:
            try:
                preds, market_pred = pipeline.predict(fixture)
                if not preds:
                    continue

                # Add actual result from finished fixture
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
                    for p in preds:
                        p['actual_result'] = actual
                        p['actual_home_score'] = home_score
                        p['actual_away_score'] = away_score
                    if market_pred:
                        market_pred['actual_home_score'] = home_score
                        market_pred['actual_away_score'] = away_score

                predictions.extend(preds)
                if market_pred:
                    market_predictions.append(market_pred)
            except Exception as e:
                logger.warning(f"Failed to predict fixture {fixture.get('id')}: {e}")

        print_backtest_summary(predictions, args.start_date, args.end_date)
        if market_predictions:
            print_market_summary(market_predictions)
            print_market_backtest_summary(market_predictions)

        if not args.dry_run and predictions:
            logger.info("\nSaving to database...")
            db = DatabaseClient(DATABASE_URL)
            db.create_tables()
            count = db.store_predictions_batch(predictions)
            logger.info(f"Saved {count} predictions")
            if market_predictions:
                mkt_count = db.store_market_predictions_batch(market_predictions)
                logger.info(f"Saved {mkt_count} market predictions")

        return

    # =========================================================================
    # LIVE MODE
    # =========================================================================
    pipeline = LivePipeline(api_key, history_days=args.history_days)

    logger.info(f"\nFetching fixtures for next {args.days_ahead} days...")
    league_filter = TOP_5_LEAGUES if FILTER_TOP_5_ONLY else None
    fixtures = pipeline.client.get_upcoming_fixtures(days_ahead=args.days_ahead, league_ids=league_filter)
    logger.info(f"Found {len(fixtures)} upcoming fixtures")

    # Generate predictions (list of lists → flatten)
    predictions = []
    market_predictions = []
    for fixture in fixtures:
        try:
            preds, market_pred = pipeline.predict(fixture)
            predictions.extend(preds)
            if market_pred:
                market_predictions.append(market_pred)
        except Exception as e:
            logger.warning(f"Failed to predict fixture {fixture.get('id')}: {e}")

    n_fixtures = len(set(p['fixture_id'] for p in predictions))
    logger.info(f"\nGenerated {len(predictions)} predictions across {n_fixtures} fixtures")

    bets = [p for p in predictions if p['should_bet']]
    logger.info(f"Recommended bets: {len(bets)}")
    if market_predictions:
        logger.info(f"Market predictions: {len(market_predictions)} fixtures")

    print_live_summary(predictions)
    if market_predictions:
        print_market_summary(market_predictions)

    # Save to database
    if not args.dry_run and predictions:
        logger.info("\nSaving to database...")
        db = DatabaseClient(DATABASE_URL)
        db.create_tables()
        count = db.store_predictions_batch(predictions)
        logger.info(f"Saved {count} predictions")
        if market_predictions:
            mkt_count = db.store_market_predictions_batch(market_predictions)
            logger.info(f"Saved {mkt_count} market predictions")
    elif args.dry_run:
        logger.info("\nDry run - not saving to database")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
