#!/usr/bin/env python3
"""
PostgreSQL Database Client for V5 Pipeline
==========================================

Handles storing and retrieving predictions from local PostgreSQL database.
Uses the existing public.predictions table schema.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import execute_values
from contextlib import contextmanager

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Client for storing predictions in PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database client.

        Args:
            database_url: PostgreSQL connection string. If None, reads from DATABASE_URL env var.
        """
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        if not self.database_url:
            # Try loading from config
            try:
                from config.production_config import DATABASE_URL
                self.database_url = DATABASE_URL
            except ImportError:
                raise ValueError("DATABASE_URL not provided")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = psycopg2.connect(self.database_url)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    fixture_id INTEGER NOT NULL,
                    match_date TIMESTAMP NOT NULL,
                    league_id INTEGER,
                    league_name VARCHAR(255),
                    season_id INTEGER,
                    home_team_id INTEGER NOT NULL,
                    home_team_name VARCHAR(255) NOT NULL,
                    away_team_id INTEGER NOT NULL,
                    away_team_name VARCHAR(255) NOT NULL,

                    -- Predictions
                    pred_home_prob FLOAT NOT NULL,
                    pred_draw_prob FLOAT NOT NULL,
                    pred_away_prob FLOAT NOT NULL,
                    predicted_outcome VARCHAR(10) NOT NULL,

                    -- Betting decision
                    strategy VARCHAR(50) DEFAULT 'threshold',
                    bet_outcome VARCHAR(20),
                    bet_probability FLOAT,
                    bet_odds FLOAT,
                    should_bet BOOLEAN DEFAULT FALSE,

                    -- Market odds (best available)
                    best_home_odds FLOAT,
                    best_draw_odds FLOAT,
                    best_away_odds FLOAT,
                    avg_home_odds FLOAT,
                    avg_draw_odds FLOAT,
                    avg_away_odds FLOAT,
                    odds_count INTEGER DEFAULT 0,

                    -- Features used for prediction (stored as JSONB)
                    features JSONB,

                    -- Actual results (filled in after match)
                    actual_home_score INTEGER,
                    actual_away_score INTEGER,
                    actual_result VARCHAR(10),
                    bet_won BOOLEAN,
                    bet_profit FLOAT,

                    -- Metadata
                    model_version VARCHAR(50) DEFAULT 'v5',
                    prediction_timestamp TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Migration: add strategy column to existing tables (must run BEFORE strategy indexes)
            cursor.execute("""
                DO $$ BEGIN
                    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS strategy VARCHAR(50) DEFAULT 'threshold';
                EXCEPTION WHEN others THEN NULL;
                END $$;
            """)

            # Create indexes (safe to run now that strategy column exists)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_fixture_id ON predictions(fixture_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_match_date ON predictions(match_date);
                CREATE INDEX IF NOT EXISTS idx_predictions_should_bet ON predictions(should_bet);
                CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league_id, match_date);
                CREATE INDEX IF NOT EXISTS idx_predictions_fixture_model ON predictions(fixture_id, model_version, prediction_timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_strategy ON predictions(strategy);
            """)

            # Unique constraint: one prediction per fixture+strategy+model
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_fixture_strategy
                    ON predictions(fixture_id, strategy, model_version);
            """)

            # Market predictions table (Poisson goals model)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_predictions (
                    id SERIAL PRIMARY KEY,
                    fixture_id INTEGER NOT NULL,
                    match_date TIMESTAMP NOT NULL,
                    league_id INTEGER,
                    home_team_name VARCHAR(255),
                    away_team_name VARCHAR(255),

                    -- Poisson parameters
                    home_goals_lambda FLOAT NOT NULL,
                    away_goals_lambda FLOAT NOT NULL,

                    -- Over/Under probabilities
                    over_0_5_prob FLOAT,
                    over_1_5_prob FLOAT,
                    over_2_5_prob FLOAT,
                    over_3_5_prob FLOAT,

                    -- BTTS
                    btts_prob FLOAT,

                    -- Asian Handicap (home perspective)
                    handicap_minus_2_5_prob FLOAT,
                    handicap_minus_1_5_prob FLOAT,
                    handicap_minus_0_5_prob FLOAT,
                    handicap_plus_0_5_prob FLOAT,
                    handicap_plus_1_5_prob FLOAT,
                    handicap_plus_2_5_prob FLOAT,

                    -- Most likely scorelines (top 5 as JSONB)
                    top_scorelines JSONB,

                    -- Actuals (filled after match)
                    actual_home_score INTEGER,
                    actual_away_score INTEGER,

                    -- Metadata
                    model_version VARCHAR(50) DEFAULT 'goals_v1',
                    prediction_timestamp TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE(fixture_id, model_version)
                );
            """)

            # Migration: add betting columns to existing market_predictions table
            cursor.execute("""
                DO $$ BEGIN
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS ou_2_5_over_odds FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS ou_2_5_under_odds FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS btts_yes_odds FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS btts_no_odds FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS ou_2_5_bet VARCHAR(10);
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS ou_2_5_bet_odds FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS ou_2_5_edge FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS ou_2_5_bet_won BOOLEAN;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS ou_2_5_profit FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS btts_bet VARCHAR(10);
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS btts_bet_odds FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS btts_edge FLOAT;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS btts_bet_won BOOLEAN;
                    ALTER TABLE market_predictions ADD COLUMN IF NOT EXISTS btts_profit FLOAT;
                EXCEPTION WHEN others THEN NULL;
                END $$;
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_predictions_fixture_id ON market_predictions(fixture_id);
                CREATE INDEX IF NOT EXISTS idx_market_predictions_match_date ON market_predictions(match_date);
            """)

            logger.info("Database tables created successfully")

    def store_prediction(self, prediction: Dict) -> int:
        """
        Store a single prediction in the database.

        Args:
            prediction: Dictionary containing prediction data

        Returns:
            ID of inserted record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO predictions (
                    fixture_id, match_date, league_id, league_name, season_id,
                    home_team_id, home_team_name, away_team_id, away_team_name,
                    pred_home_prob, pred_draw_prob, pred_away_prob, predicted_outcome,
                    strategy, bet_outcome, bet_probability, bet_odds, should_bet,
                    best_home_odds, best_draw_odds, best_away_odds,
                    avg_home_odds, avg_draw_odds, avg_away_odds, odds_count,
                    features,
                    model_version
                ) VALUES (
                    %(fixture_id)s, %(match_date)s, %(league_id)s, %(league_name)s, %(season_id)s,
                    %(home_team_id)s, %(home_team_name)s, %(away_team_id)s, %(away_team_name)s,
                    %(pred_home_prob)s, %(pred_draw_prob)s, %(pred_away_prob)s, %(predicted_outcome)s,
                    %(strategy)s, %(bet_outcome)s, %(bet_probability)s, %(bet_odds)s, %(should_bet)s,
                    %(best_home_odds)s, %(best_draw_odds)s, %(best_away_odds)s,
                    %(avg_home_odds)s, %(avg_draw_odds)s, %(avg_away_odds)s, %(odds_count)s,
                    %(features)s,
                    %(model_version)s
                )
                ON CONFLICT (fixture_id, strategy, model_version) DO UPDATE SET
                    bet_outcome = EXCLUDED.bet_outcome,
                    bet_probability = EXCLUDED.bet_probability,
                    bet_odds = EXCLUDED.bet_odds,
                    should_bet = EXCLUDED.should_bet,
                    prediction_timestamp = NOW()
                RETURNING id
            """, prediction)

            result = cursor.fetchone()
            return result[0] if result else None

    def _clean_nan_values(self, obj):
        """Recursively convert NaN values to None for JSON serialization."""
        import math

        if isinstance(obj, dict):
            return {k: self._clean_nan_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan_values(item) for item in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        else:
            return obj

    def store_predictions_batch(self, predictions: List[Dict]) -> int:
        """
        Store multiple predictions in a batch.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Number of predictions stored
        """
        if not predictions:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            import json
            values = [
                (
                    p['fixture_id'], p['match_date'], p.get('league_id'), p.get('league_name'),
                    p.get('season_id'), p['home_team_id'], p['home_team_name'],
                    p['away_team_id'], p['away_team_name'],
                    p['pred_home_prob'], p['pred_draw_prob'], p['pred_away_prob'],
                    p['predicted_outcome'],
                    p.get('strategy', 'threshold'),
                    p.get('bet_outcome'), p.get('bet_probability'), p.get('bet_odds'),
                    p.get('should_bet', False),
                    p.get('best_home_odds'), p.get('best_draw_odds'), p.get('best_away_odds'),
                    p.get('avg_home_odds'), p.get('avg_draw_odds'), p.get('avg_away_odds'),
                    p.get('odds_count', 0),
                    json.dumps(self._clean_nan_values(p.get('features', {}))),
                    p.get('model_version', 'v5')
                )
                for p in predictions
            ]

            execute_values(cursor, """
                INSERT INTO predictions (
                    fixture_id, match_date, league_id, league_name, season_id,
                    home_team_id, home_team_name, away_team_id, away_team_name,
                    pred_home_prob, pred_draw_prob, pred_away_prob, predicted_outcome,
                    strategy, bet_outcome, bet_probability, bet_odds, should_bet,
                    best_home_odds, best_draw_odds, best_away_odds,
                    avg_home_odds, avg_draw_odds, avg_away_odds, odds_count,
                    features,
                    model_version
                ) VALUES %s
                ON CONFLICT (fixture_id, strategy, model_version) DO UPDATE SET
                    bet_outcome = EXCLUDED.bet_outcome,
                    bet_probability = EXCLUDED.bet_probability,
                    bet_odds = EXCLUDED.bet_odds,
                    should_bet = EXCLUDED.should_bet,
                    prediction_timestamp = NOW()
            """, values)

            logger.info(f"Stored {len(predictions)} predictions in database")
            return len(predictions)

    def update_actual_result(self, fixture_id: int, home_score: int, away_score: int,
                            model_version: str = 'v5'):
        """
        Update prediction with actual match result.

        Args:
            fixture_id: Fixture ID
            home_score: Actual home team score
            away_score: Actual away team score
            model_version: Model version
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Determine actual result
            if home_score > away_score:
                actual_result = 'H'
            elif away_score > home_score:
                actual_result = 'A'
            else:
                actual_result = 'D'

            # Get ALL predictions for this fixture (may be multiple strategy rows)
            cursor.execute("""
                SELECT model_version, strategy, bet_outcome, should_bet, best_home_odds, best_draw_odds, best_away_odds
                FROM predictions
                WHERE fixture_id = %s
            """, (fixture_id,))

            results = cursor.fetchall()
            if not results:
                logger.warning(f"No predictions found for fixture {fixture_id}")
                return

            # Update each prediction (each strategy row independently)
            for row in results:
                model_ver, strat, bet_outcome, should_bet, best_home_odds, best_draw_odds, best_away_odds = row

                bet_won = None
                bet_profit = None

                if should_bet and bet_outcome:
                    bet_code = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}.get(bet_outcome)

                    if bet_outcome == 'Home Win':
                        market_odds = best_home_odds
                    elif bet_outcome == 'Draw':
                        market_odds = best_draw_odds
                    elif bet_outcome == 'Away Win':
                        market_odds = best_away_odds
                    else:
                        market_odds = None

                    bet_won = (bet_code == actual_result)

                    if market_odds and bet_won:
                        bet_profit = (market_odds - 1) * 1.0  # $1 stake
                    elif market_odds:
                        bet_profit = -1.0
                    else:
                        bet_profit = None

                cursor.execute("""
                    UPDATE predictions
                    SET actual_home_score = %s,
                        actual_away_score = %s,
                        actual_result = %s,
                        bet_won = %s,
                        bet_profit = %s,
                        updated_at = NOW()
                    WHERE fixture_id = %s AND model_version = %s AND strategy = %s
                """, (home_score, away_score, actual_result, bet_won, bet_profit,
                      fixture_id, model_ver, strat))

            logger.info(f"Updated result for fixture {fixture_id}")

    def update_actual_results_batch(self, results: Dict[int, Dict]) -> int:
        """
        Update multiple fixtures with actual results in batch.

        Args:
            results: Dict mapping fixture_id -> {home_score, away_score}

        Returns:
            Number of predictions updated
        """
        if not results:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            fixture_ids = list(results.keys())
            cursor.execute("""
                SELECT fixture_id, model_version, strategy, bet_outcome, should_bet,
                       best_home_odds, best_draw_odds, best_away_odds
                FROM predictions
                WHERE fixture_id = ANY(%s)
            """, (fixture_ids,))

            predictions = cursor.fetchall()
            if not predictions:
                return 0

            update_values = []
            for row in predictions:
                fixture_id, model_ver, strat, bet_outcome, should_bet, best_home_odds, best_draw_odds, best_away_odds = row

                if fixture_id not in results:
                    continue

                home_score = results[fixture_id]['home_score']
                away_score = results[fixture_id]['away_score']

                if home_score > away_score:
                    actual_result = 'H'
                elif away_score > home_score:
                    actual_result = 'A'
                else:
                    actual_result = 'D'

                bet_won = None
                bet_profit = None

                if should_bet and bet_outcome:
                    bet_code = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}.get(bet_outcome)

                    if bet_outcome == 'Home Win':
                        market_odds = best_home_odds
                    elif bet_outcome == 'Draw':
                        market_odds = best_draw_odds
                    elif bet_outcome == 'Away Win':
                        market_odds = best_away_odds
                    else:
                        market_odds = None

                    bet_won = (bet_code == actual_result)

                    if market_odds and bet_won:
                        bet_profit = (market_odds - 1) * 1.0
                    elif market_odds:
                        bet_profit = -1.0

                update_values.append((
                    home_score, away_score, actual_result, bet_won, bet_profit,
                    fixture_id, model_ver, strat or 'threshold'
                ))

            if update_values:
                execute_values(cursor, """
                    UPDATE predictions AS p
                    SET actual_home_score = v.home_score::integer,
                        actual_away_score = v.away_score::integer,
                        actual_result = v.actual_result,
                        bet_won = v.bet_won::boolean,
                        bet_profit = v.bet_profit::double precision,
                        updated_at = NOW()
                    FROM (VALUES %s) AS v(home_score, away_score, actual_result, bet_won, bet_profit, fixture_id, model_version, strategy)
                    WHERE p.fixture_id = v.fixture_id::integer AND p.model_version = v.model_version AND p.strategy = v.strategy
                """, update_values)

                logger.info(f"Batch updated {len(update_values)} predictions")
                return len(update_values)

            return 0

    def get_pending_results(self, model_version: str = 'v5') -> List[Dict]:
        """Get predictions that don't have actual results yet."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT fixture_id, match_date, home_team_name, away_team_name,
                       pred_home_prob, pred_draw_prob, pred_away_prob,
                       bet_outcome, should_bet
                FROM predictions
                WHERE actual_result IS NULL
                  AND model_version = %s
                  AND match_date < NOW()
                ORDER BY match_date DESC
            """, (model_version,))

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_betting_performance(self, days: int = 30, model_version: str = 'v5',
                               strategy: str = None) -> Dict:
        """Get betting performance metrics for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN bet_won THEN 1 ELSE 0 END) as wins,
                    SUM(bet_profit) as total_profit,
                    AVG(bet_probability) as avg_confidence
                FROM predictions
                WHERE should_bet = TRUE
                  AND actual_result IS NOT NULL
                  AND model_version = %s
                  AND match_date >= NOW() - INTERVAL '%s days'
            """
            params = [model_version, days]
            if strategy:
                query += " AND strategy = %s"
                params.append(strategy)
            cursor.execute(query, params)

            row = cursor.fetchone()

            total_bets = row[0] or 0
            wins = row[1] or 0
            total_profit = row[2] or 0.0
            avg_confidence = row[3] or 0.0

            win_rate = wins / total_bets if total_bets > 0 else 0
            roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

            return {
                'total_bets': total_bets,
                'wins': wins,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'roi': roi,
                'avg_confidence': avg_confidence,
                'period_days': days
            }

    def get_detailed_pnl(self, start_date: str = None, end_date: str = None,
                         model_version: str = 'v5', strategy: str = None) -> Dict:
        """Get detailed PnL breakdown."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            date_filter = ""
            params = [model_version]

            if start_date:
                date_filter += " AND match_date >= %s"
                params.append(start_date)
            if end_date:
                date_filter += " AND match_date <= %s"
                params.append(end_date)
            if strategy:
                date_filter += " AND strategy = %s"
                params.append(strategy)

            cursor.execute(f"""
                SELECT
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN bet_won = TRUE THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN bet_won = FALSE THEN 1 ELSE 0 END) as losses,
                    SUM(bet_profit) as total_profit,
                    AVG(bet_probability) as avg_confidence,
                    AVG(CASE WHEN bet_outcome = 'Home Win' THEN best_home_odds
                            WHEN bet_outcome = 'Draw' THEN best_draw_odds
                            WHEN bet_outcome = 'Away Win' THEN best_away_odds
                        END) as avg_odds,
                    SUM(CASE WHEN bet_outcome = 'Home Win' THEN 1 ELSE 0 END) as home_bets,
                    SUM(CASE WHEN bet_outcome = 'Draw' THEN 1 ELSE 0 END) as draw_bets,
                    SUM(CASE WHEN bet_outcome = 'Away Win' THEN 1 ELSE 0 END) as away_bets,
                    SUM(CASE WHEN bet_outcome = 'Home Win' AND bet_won = TRUE THEN 1 ELSE 0 END) as home_wins,
                    SUM(CASE WHEN bet_outcome = 'Draw' AND bet_won = TRUE THEN 1 ELSE 0 END) as draw_wins,
                    SUM(CASE WHEN bet_outcome = 'Away Win' AND bet_won = TRUE THEN 1 ELSE 0 END) as away_wins,
                    SUM(CASE WHEN bet_outcome = 'Home Win' THEN bet_profit ELSE 0 END) as home_profit,
                    SUM(CASE WHEN bet_outcome = 'Draw' THEN bet_profit ELSE 0 END) as draw_profit,
                    SUM(CASE WHEN bet_outcome = 'Away Win' THEN bet_profit ELSE 0 END) as away_profit
                FROM predictions
                WHERE should_bet = TRUE
                  AND actual_result IS NOT NULL
                  AND model_version = %s
                  {date_filter}
            """, tuple(params))

            row = cursor.fetchone()

            total_bets = row[0] or 0
            wins = row[1] or 0
            losses = row[2] or 0
            total_profit = row[3] or 0.0
            avg_confidence = row[4] or 0.0
            avg_odds = row[5] or 0.0
            home_bets = row[6] or 0
            draw_bets = row[7] or 0
            away_bets = row[8] or 0
            home_wins = row[9] or 0
            draw_wins = row[10] or 0
            away_wins = row[11] or 0
            home_profit = row[12] or 0.0
            draw_profit = row[13] or 0.0
            away_profit = row[14] or 0.0

            win_rate = wins / total_bets if total_bets > 0 else 0
            roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

            # Monthly breakdown
            cursor.execute(f"""
                SELECT
                    DATE_TRUNC('month', match_date) as month,
                    COUNT(*) as bets,
                    SUM(CASE WHEN bet_won = TRUE THEN 1 ELSE 0 END) as wins,
                    SUM(bet_profit) as profit
                FROM predictions
                WHERE should_bet = TRUE
                  AND actual_result IS NOT NULL
                  AND model_version = %s
                  {date_filter}
                GROUP BY month
                ORDER BY month DESC
                LIMIT 12
            """, tuple(params))

            monthly_stats = [
                {
                    'month': row[0].strftime('%Y-%m') if row[0] else None,
                    'bets': row[1],
                    'wins': row[2],
                    'profit': float(row[3]) if row[3] else 0.0,
                    'win_rate': row[2] / row[1] if row[1] > 0 else 0
                }
                for row in cursor.fetchall()
            ]

            return {
                'summary': {
                    'total_bets': total_bets,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'roi': roi,
                    'avg_confidence': avg_confidence,
                    'avg_odds': avg_odds
                },
                'by_outcome': {
                    'home': {
                        'bets': home_bets,
                        'wins': home_wins,
                        'win_rate': home_wins / home_bets if home_bets > 0 else 0,
                        'profit': home_profit
                    },
                    'draw': {
                        'bets': draw_bets,
                        'wins': draw_wins,
                        'win_rate': draw_wins / draw_bets if draw_bets > 0 else 0,
                        'profit': draw_profit
                    },
                    'away': {
                        'bets': away_bets,
                        'wins': away_wins,
                        'win_rate': away_wins / away_bets if away_bets > 0 else 0,
                        'profit': away_profit
                    }
                },
                'monthly': monthly_stats,
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }

    def store_market_predictions_batch(self, predictions: List[Dict]) -> int:
        """Store market predictions (Poisson goals model) in batch.

        Args:
            predictions: List of dicts with keys matching market_predictions columns.

        Returns:
            Number of predictions stored.
        """
        if not predictions:
            return 0

        import json

        with self.get_connection() as conn:
            cursor = conn.cursor()

            values = [
                (
                    p['fixture_id'], p['match_date'], p.get('league_id'),
                    p.get('home_team_name'), p.get('away_team_name'),
                    p['home_goals_lambda'], p['away_goals_lambda'],
                    p.get('over_0_5_prob'), p.get('over_1_5_prob'),
                    p.get('over_2_5_prob'), p.get('over_3_5_prob'),
                    p.get('btts_prob'),
                    p.get('handicap_minus_2_5_prob'), p.get('handicap_minus_1_5_prob'),
                    p.get('handicap_minus_0_5_prob'), p.get('handicap_plus_0_5_prob'),
                    p.get('handicap_plus_1_5_prob'), p.get('handicap_plus_2_5_prob'),
                    json.dumps(p.get('top_scorelines', [])),
                    p.get('ou_2_5_over_odds'), p.get('ou_2_5_under_odds'),
                    p.get('btts_yes_odds'), p.get('btts_no_odds'),
                    p.get('ou_2_5_bet'), p.get('ou_2_5_bet_odds'), p.get('ou_2_5_edge'),
                    p.get('btts_bet'), p.get('btts_bet_odds'), p.get('btts_edge'),
                    p.get('model_version', 'goals_v1'),
                )
                for p in predictions
            ]

            execute_values(cursor, """
                INSERT INTO market_predictions (
                    fixture_id, match_date, league_id,
                    home_team_name, away_team_name,
                    home_goals_lambda, away_goals_lambda,
                    over_0_5_prob, over_1_5_prob, over_2_5_prob, over_3_5_prob,
                    btts_prob,
                    handicap_minus_2_5_prob, handicap_minus_1_5_prob,
                    handicap_minus_0_5_prob, handicap_plus_0_5_prob,
                    handicap_plus_1_5_prob, handicap_plus_2_5_prob,
                    top_scorelines,
                    ou_2_5_over_odds, ou_2_5_under_odds,
                    btts_yes_odds, btts_no_odds,
                    ou_2_5_bet, ou_2_5_bet_odds, ou_2_5_edge,
                    btts_bet, btts_bet_odds, btts_edge,
                    model_version
                ) VALUES %s
                ON CONFLICT (fixture_id, model_version) DO UPDATE SET
                    home_goals_lambda = EXCLUDED.home_goals_lambda,
                    away_goals_lambda = EXCLUDED.away_goals_lambda,
                    over_0_5_prob = EXCLUDED.over_0_5_prob,
                    over_1_5_prob = EXCLUDED.over_1_5_prob,
                    over_2_5_prob = EXCLUDED.over_2_5_prob,
                    over_3_5_prob = EXCLUDED.over_3_5_prob,
                    btts_prob = EXCLUDED.btts_prob,
                    handicap_minus_2_5_prob = EXCLUDED.handicap_minus_2_5_prob,
                    handicap_minus_1_5_prob = EXCLUDED.handicap_minus_1_5_prob,
                    handicap_minus_0_5_prob = EXCLUDED.handicap_minus_0_5_prob,
                    handicap_plus_0_5_prob = EXCLUDED.handicap_plus_0_5_prob,
                    handicap_plus_1_5_prob = EXCLUDED.handicap_plus_1_5_prob,
                    handicap_plus_2_5_prob = EXCLUDED.handicap_plus_2_5_prob,
                    top_scorelines = EXCLUDED.top_scorelines,
                    ou_2_5_over_odds = EXCLUDED.ou_2_5_over_odds,
                    ou_2_5_under_odds = EXCLUDED.ou_2_5_under_odds,
                    btts_yes_odds = EXCLUDED.btts_yes_odds,
                    btts_no_odds = EXCLUDED.btts_no_odds,
                    ou_2_5_bet = EXCLUDED.ou_2_5_bet,
                    ou_2_5_bet_odds = EXCLUDED.ou_2_5_bet_odds,
                    ou_2_5_edge = EXCLUDED.ou_2_5_edge,
                    btts_bet = EXCLUDED.btts_bet,
                    btts_bet_odds = EXCLUDED.btts_bet_odds,
                    btts_edge = EXCLUDED.btts_edge,
                    prediction_timestamp = NOW()
            """, values)

            logger.info(f"Stored {len(predictions)} market predictions")
            return len(predictions)

    def update_market_actuals_batch(self, results: Dict[int, Dict]) -> int:
        """Update market_predictions with actual scores.

        Args:
            results: Dict mapping fixture_id -> {home_score, away_score}

        Returns:
            Number of rows updated.
        """
        if not results:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            values = [
                (fixture_id, r['home_score'], r['away_score'])
                for fixture_id, r in results.items()
            ]

            execute_values(cursor, """
                UPDATE market_predictions AS mp
                SET actual_home_score = v.home_score::integer,
                    actual_away_score = v.away_score::integer,
                    updated_at = NOW()
                FROM (VALUES %s) AS v(fixture_id, home_score, away_score)
                WHERE mp.fixture_id = v.fixture_id::integer
                  AND mp.actual_home_score IS NULL
            """, values)

            updated = cursor.rowcount
            if updated > 0:
                logger.info(f"Updated {updated} market prediction actuals")
            return updated

    def get_market_accuracy(self, days: int = 30) -> Dict:
        """Get accuracy report for market predictions.

        Returns dict with accuracy for O/U 2.5, BTTS, and calibration buckets.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    fixture_id, home_goals_lambda, away_goals_lambda,
                    over_0_5_prob, over_1_5_prob, over_2_5_prob, over_3_5_prob,
                    btts_prob,
                    actual_home_score, actual_away_score
                FROM market_predictions
                WHERE actual_home_score IS NOT NULL
                  AND match_date >= NOW() - INTERVAL '%s days'
            """, (days,))

            rows = cursor.fetchall()

            if not rows:
                return {'total': 0, 'message': 'No market predictions with results found'}

            # Calculate accuracy
            ou_results = {0.5: {'correct': 0, 'total': 0}, 1.5: {'correct': 0, 'total': 0},
                          2.5: {'correct': 0, 'total': 0}, 3.5: {'correct': 0, 'total': 0}}
            btts_correct = 0
            btts_total = 0

            for row in rows:
                (fid, lh, la, o05, o15, o25, o35, btts,
                 actual_h, actual_a) = row

                actual_total = actual_h + actual_a
                actual_btts = actual_h >= 1 and actual_a >= 1

                for line, prob in [(0.5, o05), (1.5, o15), (2.5, o25), (3.5, o35)]:
                    if prob is not None:
                        pred_over = prob > 0.5
                        actual_over = actual_total > line
                        ou_results[line]['total'] += 1
                        if pred_over == actual_over:
                            ou_results[line]['correct'] += 1

                if btts is not None:
                    btts_total += 1
                    if (btts > 0.5) == actual_btts:
                        btts_correct += 1

            return {
                'total': len(rows),
                'over_under': {
                    f'over_{str(line).replace(".", "_")}': {
                        'accuracy': r['correct'] / r['total'] if r['total'] > 0 else 0,
                        'total': r['total'],
                    }
                    for line, r in ou_results.items()
                },
                'btts': {
                    'accuracy': btts_correct / btts_total if btts_total > 0 else 0,
                    'total': btts_total,
                },
            }

    def update_market_bet_results_batch(self, results: Dict[int, Dict]) -> int:
        """Calculate bet_won and profit for market bets given actual scores.

        Args:
            results: Dict mapping fixture_id -> {home_score, away_score}

        Returns:
            Number of rows updated.
        """
        if not results:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()

            fixture_ids = list(results.keys())
            cursor.execute("""
                SELECT fixture_id, ou_2_5_bet, ou_2_5_bet_odds, btts_bet, btts_bet_odds
                FROM market_predictions
                WHERE fixture_id = ANY(%s)
                  AND actual_home_score IS NOT NULL
                  AND (ou_2_5_bet IS NOT NULL OR btts_bet IS NOT NULL)
                  AND (ou_2_5_bet_won IS NULL AND btts_bet_won IS NULL)
            """, (fixture_ids,))

            rows = cursor.fetchall()
            if not rows:
                return 0

            updated = 0
            for row in rows:
                fid, ou_bet, ou_odds, btts_bet_val, btts_odds = row
                if fid not in results:
                    continue

                hs = results[fid]['home_score']
                aws = results[fid]['away_score']
                actual_total = hs + aws
                actual_btts = hs >= 1 and aws >= 1

                # O/U 2.5 result
                ou_won = None
                ou_profit = None
                if ou_bet and ou_odds:
                    if ou_bet == 'over':
                        ou_won = actual_total > 2.5
                    elif ou_bet == 'under':
                        ou_won = actual_total < 2.5
                    ou_profit = float(ou_odds - 1) if ou_won else -1.0

                # BTTS result
                btts_won = None
                btts_profit = None
                if btts_bet_val and btts_odds:
                    if btts_bet_val == 'yes':
                        btts_won = actual_btts
                    elif btts_bet_val == 'no':
                        btts_won = not actual_btts
                    btts_profit = float(btts_odds - 1) if btts_won else -1.0

                cursor.execute("""
                    UPDATE market_predictions
                    SET ou_2_5_bet_won = %s, ou_2_5_profit = %s,
                        btts_bet_won = %s, btts_profit = %s,
                        updated_at = NOW()
                    WHERE fixture_id = %s
                """, (ou_won, ou_profit, btts_won, btts_profit, fid))
                updated += 1

            if updated > 0:
                logger.info(f"Updated {updated} market bet results")
            return updated

    def get_market_pnl_report(self, days: int = None, start_date: str = None,
                              end_date: str = None) -> Dict:
        """Get PnL report for goals market bets (O/U 2.5, BTTS).

        Returns dict with per-market stats: bets, wins, win_rate, profit, ROI.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            where = "actual_home_score IS NOT NULL"
            params = []
            if days:
                where += " AND match_date >= NOW() - INTERVAL '%s days'"
                params.append(days)
            if start_date:
                where += " AND match_date >= %s"
                params.append(start_date)
            if end_date:
                where += " AND match_date <= %s"
                params.append(end_date)

            cursor.execute(f"""
                SELECT
                    -- O/U 2.5
                    COUNT(CASE WHEN ou_2_5_bet IS NOT NULL THEN 1 END) as ou_bets,
                    COUNT(CASE WHEN ou_2_5_bet_won = TRUE THEN 1 END) as ou_wins,
                    COALESCE(SUM(CASE WHEN ou_2_5_bet IS NOT NULL THEN ou_2_5_profit ELSE 0 END), 0) as ou_profit,
                    AVG(CASE WHEN ou_2_5_bet IS NOT NULL THEN ou_2_5_edge END) as ou_avg_edge,
                    AVG(CASE WHEN ou_2_5_bet IS NOT NULL THEN ou_2_5_bet_odds END) as ou_avg_odds,
                    -- BTTS
                    COUNT(CASE WHEN btts_bet IS NOT NULL THEN 1 END) as btts_bets,
                    COUNT(CASE WHEN btts_bet_won = TRUE THEN 1 END) as btts_wins,
                    COALESCE(SUM(CASE WHEN btts_bet IS NOT NULL THEN btts_profit ELSE 0 END), 0) as btts_profit_total,
                    AVG(CASE WHEN btts_bet IS NOT NULL THEN btts_edge END) as btts_avg_edge,
                    AVG(CASE WHEN btts_bet IS NOT NULL THEN btts_bet_odds END) as btts_avg_odds
                FROM market_predictions
                WHERE {where}
            """, tuple(params))

            row = cursor.fetchone()

            ou_bets = row[0] or 0
            ou_wins = row[1] or 0
            ou_profit = float(row[2] or 0)
            ou_avg_edge = float(row[3]) if row[3] else 0
            ou_avg_odds = float(row[4]) if row[4] else 0
            btts_bets = row[5] or 0
            btts_wins = row[6] or 0
            btts_profit_total = float(row[7] or 0)
            btts_avg_edge = float(row[8]) if row[8] else 0
            btts_avg_odds = float(row[9]) if row[9] else 0

            # Monthly breakdown
            cursor.execute(f"""
                SELECT
                    DATE_TRUNC('month', match_date) as month,
                    COUNT(CASE WHEN ou_2_5_bet IS NOT NULL THEN 1 END) as ou_bets,
                    COALESCE(SUM(CASE WHEN ou_2_5_bet IS NOT NULL THEN ou_2_5_profit ELSE 0 END), 0) as ou_profit,
                    COUNT(CASE WHEN btts_bet IS NOT NULL THEN 1 END) as btts_bets,
                    COALESCE(SUM(CASE WHEN btts_bet IS NOT NULL THEN btts_profit ELSE 0 END), 0) as btts_profit
                FROM market_predictions
                WHERE {where}
                GROUP BY month
                ORDER BY month DESC
                LIMIT 12
            """, tuple(params))

            monthly = [
                {
                    'month': r[0].strftime('%Y-%m') if r[0] else None,
                    'ou_bets': r[1], 'ou_profit': float(r[2]),
                    'btts_bets': r[3], 'btts_profit': float(r[4]),
                }
                for r in cursor.fetchall()
            ]

            return {
                'ou_2_5': {
                    'bets': ou_bets,
                    'wins': ou_wins,
                    'win_rate': ou_wins / ou_bets if ou_bets > 0 else 0,
                    'profit': ou_profit,
                    'roi': ou_profit / ou_bets * 100 if ou_bets > 0 else 0,
                    'avg_edge': ou_avg_edge,
                    'avg_odds': ou_avg_odds,
                },
                'btts': {
                    'bets': btts_bets,
                    'wins': btts_wins,
                    'win_rate': btts_wins / btts_bets if btts_bets > 0 else 0,
                    'profit': btts_profit_total,
                    'roi': btts_profit_total / btts_bets * 100 if btts_bets > 0 else 0,
                    'avg_edge': btts_avg_edge,
                    'avg_odds': btts_avg_odds,
                },
                'monthly': monthly,
            }

    def get_pnl_report(self, start_date: str = None, end_date: str = None,
                       model_version: str = 'v5', strategy: str = None) -> str:
        """Generate a formatted PnL report string."""
        pnl = self.get_detailed_pnl(start_date, end_date, model_version, strategy=strategy)
        summary = pnl['summary']
        by_outcome = pnl['by_outcome']

        report = []
        report.append("=" * 60)
        report.append("PnL REPORT")
        report.append("=" * 60)

        if pnl['period']['start_date'] or pnl['period']['end_date']:
            period_str = f"{pnl['period']['start_date'] or 'Start'} to {pnl['period']['end_date'] or 'Now'}"
            report.append(f"Period: {period_str}")
        else:
            report.append("Period: All Time")

        report.append(f"Model: {model_version}")
        report.append("")
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Bets: {summary['total_bets']}")
        report.append(f"Wins: {summary['wins']} | Losses: {summary['losses']}")
        report.append(f"Win Rate: {summary['win_rate']:.1%}")
        report.append(f"Total Profit: ${summary['total_profit']:.2f}")
        report.append(f"ROI: {summary['roi']:.1f}%")
        report.append(f"Avg Confidence: {summary['avg_confidence']:.1%}")
        report.append(f"Avg Odds: {summary['avg_odds']:.2f}")
        report.append("")
        report.append("BY OUTCOME")
        report.append("-" * 40)

        for outcome, stats in by_outcome.items():
            if stats['bets'] > 0:
                report.append(f"{outcome.upper()}:")
                report.append(f"  Bets: {stats['bets']} | Wins: {stats['wins']} | WR: {stats['win_rate']:.1%}")
                report.append(f"  Profit: ${stats['profit']:.2f}")

        if pnl['monthly']:
            report.append("")
            report.append("MONTHLY BREAKDOWN")
            report.append("-" * 40)
            report.append(f"{'Month':<10} {'Bets':<6} {'Wins':<6} {'WR':<8} {'Profit':<10}")

            for m in pnl['monthly']:
                report.append(f"{m['month']:<10} {m['bets']:<6} {m['wins']:<6} "
                            f"{m['win_rate']:.1%}   ${m['profit']:<9.2f}")

        report.append("=" * 60)

        return "\n".join(report)
