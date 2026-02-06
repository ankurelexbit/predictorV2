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

                -- Create indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_predictions_fixture_id ON predictions(fixture_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_match_date ON predictions(match_date);
                CREATE INDEX IF NOT EXISTS idx_predictions_should_bet ON predictions(should_bet);
                CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league_id, match_date);
                CREATE INDEX IF NOT EXISTS idx_predictions_fixture_model ON predictions(fixture_id, model_version, prediction_timestamp DESC);
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
                    bet_outcome, bet_probability, bet_odds, should_bet,
                    best_home_odds, best_draw_odds, best_away_odds,
                    avg_home_odds, avg_draw_odds, avg_away_odds, odds_count,
                    features,
                    model_version
                ) VALUES (
                    %(fixture_id)s, %(match_date)s, %(league_id)s, %(league_name)s, %(season_id)s,
                    %(home_team_id)s, %(home_team_name)s, %(away_team_id)s, %(away_team_name)s,
                    %(pred_home_prob)s, %(pred_draw_prob)s, %(pred_away_prob)s, %(predicted_outcome)s,
                    %(bet_outcome)s, %(bet_probability)s, %(bet_odds)s, %(should_bet)s,
                    %(best_home_odds)s, %(best_draw_odds)s, %(best_away_odds)s,
                    %(avg_home_odds)s, %(avg_draw_odds)s, %(avg_away_odds)s, %(odds_count)s,
                    %(features)s,
                    %(model_version)s
                )
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
                    bet_outcome, bet_probability, bet_odds, should_bet,
                    best_home_odds, best_draw_odds, best_away_odds,
                    avg_home_odds, avg_draw_odds, avg_away_odds, odds_count,
                    features,
                    model_version
                ) VALUES %s
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

            # Get ALL predictions for this fixture
            cursor.execute("""
                SELECT model_version, bet_outcome, should_bet, best_home_odds, best_draw_odds, best_away_odds
                FROM predictions
                WHERE fixture_id = %s
            """, (fixture_id,))

            results = cursor.fetchall()
            if not results:
                logger.warning(f"No predictions found for fixture {fixture_id}")
                return

            # Update each prediction
            for row in results:
                model_ver, bet_outcome, should_bet, best_home_odds, best_draw_odds, best_away_odds = row

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
                    WHERE fixture_id = %s AND model_version = %s
                """, (home_score, away_score, actual_result, bet_won, bet_profit,
                      fixture_id, model_ver))

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
                SELECT fixture_id, model_version, bet_outcome, should_bet,
                       best_home_odds, best_draw_odds, best_away_odds
                FROM predictions
                WHERE fixture_id = ANY(%s)
            """, (fixture_ids,))

            predictions = cursor.fetchall()
            if not predictions:
                return 0

            update_values = []
            for row in predictions:
                fixture_id, model_ver, bet_outcome, should_bet, best_home_odds, best_draw_odds, best_away_odds = row

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
                    fixture_id, model_ver
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
                    FROM (VALUES %s) AS v(home_score, away_score, actual_result, bet_won, bet_profit, fixture_id, model_version)
                    WHERE p.fixture_id = v.fixture_id::integer AND p.model_version = v.model_version
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

    def get_betting_performance(self, days: int = 30, model_version: str = 'v5') -> Dict:
        """Get betting performance metrics for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
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
            """, (model_version, days))

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
                         model_version: str = 'v5') -> Dict:
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

    def get_pnl_report(self, start_date: str = None, end_date: str = None,
                       model_version: str = 'v5') -> str:
        """Generate a formatted PnL report string."""
        pnl = self.get_detailed_pnl(start_date, end_date, model_version)
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
