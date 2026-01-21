#!/usr/bin/env python3
"""
Hourly Prediction Pipeline
===========================

Runs every hour to:
1. Fetch upcoming fixtures (next 48 hours)
2. Generate predictions with betting recommendations
3. Store to database
4. Check for lineup updates (if match < 2 hours away)
5. Re-predict when lineups available
6. Track results after matches finish

Usage:
    python hourly_predictions.py [--force] [--lookback-hours 48]
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, DATA_DIR
from utils import setup_logger

logger = setup_logger("hourly_predictions")


class PredictionDatabase:
    """Manages prediction storage in SQLite."""

    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = Path(__file__).parent / db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER NOT NULL,
                prediction_time TIMESTAMP NOT NULL,
                match_date TIMESTAMP NOT NULL,
                home_team VARCHAR(255) NOT NULL,
                away_team VARCHAR(255) NOT NULL,
                league VARCHAR(255),
                venue VARCHAR(255),
                home_win_prob REAL,
                draw_prob REAL,
                away_win_prob REAL,
                predicted_outcome VARCHAR(50),
                lineup_available BOOLEAN DEFAULT 0,
                lineup_coverage_home REAL,
                lineup_coverage_away REAL,
                model_version VARCHAR(50),
                UNIQUE(fixture_id, prediction_time)
            )
        """)

        # Create bets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER NOT NULL,
                prediction_id INTEGER,
                bet_time TIMESTAMP NOT NULL,
                bet_outcome VARCHAR(50) NOT NULL,
                stake REAL NOT NULL,
                odds REAL NOT NULL,
                expected_value REAL,
                rule_applied VARCHAR(255),
                status VARCHAR(50) DEFAULT 'pending',
                actual_outcome VARCHAR(50),
                result VARCHAR(50),
                profit_loss REAL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)

        # Create results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER UNIQUE NOT NULL,
                match_date TIMESTAMP NOT NULL,
                home_team VARCHAR(255) NOT NULL,
                away_team VARCHAR(255) NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                actual_outcome VARCHAR(50),
                result_time TIMESTAMP NOT NULL
            )
        """)

        # Create lineup_updates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lineup_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id INTEGER NOT NULL,
                update_time TIMESTAMP NOT NULL,
                lineup_available BOOLEAN,
                home_players_found INTEGER,
                away_players_found INTEGER,
                UNIQUE(fixture_id, update_time)
            )
        """)

        conn.commit()
        conn.close()

    def save_prediction(self, prediction: Dict) -> int:
        """Save prediction to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO predictions
            (fixture_id, prediction_time, match_date, home_team, away_team, league, venue,
             home_win_prob, draw_prob, away_win_prob, predicted_outcome,
             lineup_available, lineup_coverage_home, lineup_coverage_away, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction['fixture_id'],
            prediction['prediction_time'],
            prediction['match_date'],
            prediction['home_team'],
            prediction['away_team'],
            prediction.get('league'),
            prediction.get('venue'),
            prediction['home_win_prob'],
            prediction['draw_prob'],
            prediction['away_win_prob'],
            prediction['predicted_outcome'],
            prediction.get('lineup_available', False),
            prediction.get('lineup_coverage_home', 0.0),
            prediction.get('lineup_coverage_away', 0.0),
            prediction.get('model_version', 'stacking')
        ))

        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return prediction_id

    def save_bet(self, bet: Dict, prediction_id: int):
        """Save betting recommendation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO bets
            (fixture_id, prediction_id, bet_time, bet_outcome, stake, odds,
             expected_value, rule_applied, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bet['fixture_id'],
            prediction_id,
            bet['bet_time'],
            bet['bet_outcome'],
            bet['stake'],
            bet['odds'],
            bet['expected_value'],
            bet['rule_applied'],
            'pending'
        ))

        conn.commit()
        conn.close()

    def get_pending_bets(self) -> pd.DataFrame:
        """Get all pending bets."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT b.*, p.match_date, p.home_team, p.away_team
            FROM bets b
            JOIN predictions p ON b.fixture_id = p.fixture_id
            WHERE b.status = 'pending'
            AND p.match_date < datetime('now')
        """, conn)
        conn.close()
        return df

    def update_bet_result(self, bet_id: int, actual_outcome: str, profit_loss: float):
        """Update bet with actual result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        result = 'win' if profit_loss > 0 else 'loss'

        cursor.execute("""
            UPDATE bets
            SET actual_outcome = ?, result = ?, profit_loss = ?, status = 'settled'
            WHERE id = ?
        """, (actual_outcome, result, profit_loss, bet_id))

        conn.commit()
        conn.close()


class HourlyPredictor:
    """Manages hourly prediction pipeline."""

    def __init__(self, lookback_hours: int = 48, force: bool = False):
        self.lookback_hours = lookback_hours
        self.force = force
        self.db = PredictionDatabase()
        self.current_time = datetime.now()

    def step_1_fetch_upcoming_fixtures(self) -> pd.DataFrame:
        """Fetch fixtures in the next N hours."""
        logger.info("=" * 80)
        logger.info(f"STEP 1: Fetching fixtures (next {self.lookback_hours} hours)")
        logger.info("=" * 80)

        start_date = self.current_time.date()
        end_date = (self.current_time + timedelta(hours=self.lookback_hours)).date()

        logger.info(f"  Date range: {start_date} to {end_date}")

        # Import here to avoid circular dependencies
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from predict_live import get_upcoming_fixtures

        all_fixtures = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            fixtures = get_upcoming_fixtures(date_str)

            if not fixtures.empty:
                all_fixtures.append(fixtures)
                logger.info(f"  {date_str}: {len(fixtures)} fixtures found")

            current_date += timedelta(days=1)

        if all_fixtures:
            combined = pd.concat(all_fixtures, ignore_index=True)
            logger.info(f"\n✅ Total fixtures found: {len(combined)}")
            return combined
        else:
            logger.info("\n⚠️  No upcoming fixtures found")
            return pd.DataFrame()

    def step_2_generate_predictions(self, fixtures_df: pd.DataFrame) -> List[Dict]:
        """Generate predictions for all fixtures."""
        logger.info("=" * 80)
        logger.info("STEP 2: Generating predictions")
        logger.info("=" * 80)

        if fixtures_df.empty:
            logger.info("  No fixtures to predict")
            return []

        # Run predict_live.py programmatically
        import subprocess

        predictions = []

        # Save fixtures to temp file
        temp_file = Path(__file__).parent / "temp_fixtures.csv"
        fixtures_df.to_csv(temp_file, index=False)

        for _, fixture in fixtures_df.iterrows():
            try:
                logger.info(f"\n  Predicting: {fixture['home_team_name']} vs {fixture['away_team_name']}")

                # Run prediction (you'd integrate this properly)
                # For now, this is a placeholder showing the structure
                prediction = {
                    'fixture_id': fixture['fixture_id'],
                    'prediction_time': self.current_time,
                    'match_date': fixture['date'],
                    'home_team': fixture['home_team_name'],
                    'away_team': fixture['away_team_name'],
                    'league': fixture.get('league_name'),
                    'venue': fixture.get('venue'),
                    'home_win_prob': 0.45,  # TODO: Get from actual model
                    'draw_prob': 0.30,
                    'away_win_prob': 0.25,
                    'predicted_outcome': 'Home Win',
                    'lineup_available': False,
                    'model_version': 'stacking_v1'
                }

                predictions.append(prediction)

            except Exception as e:
                logger.error(f"    ❌ Error predicting fixture {fixture['fixture_id']}: {e}")

        temp_file.unlink(missing_ok=True)

        logger.info(f"\n✅ Generated {len(predictions)} predictions")
        return predictions

    def step_3_save_to_database(self, predictions: List[Dict], bets: List[Dict]):
        """Save predictions and bets to database."""
        logger.info("=" * 80)
        logger.info("STEP 3: Saving to database")
        logger.info("=" * 80)

        for prediction in predictions:
            try:
                prediction_id = self.db.save_prediction(prediction)
                logger.info(f"  Saved prediction {prediction_id}: {prediction['home_team']} vs {prediction['away_team']}")

                # Save associated bets
                fixture_bets = [b for b in bets if b['fixture_id'] == prediction['fixture_id']]
                for bet in fixture_bets:
                    self.db.save_bet(bet, prediction_id)
                    logger.info(f"    └─ Bet: {bet['bet_outcome']} (£{bet['stake']:.2f})")

            except Exception as e:
                logger.error(f"  ❌ Error saving prediction: {e}")

        logger.info(f"\n✅ Saved {len(predictions)} predictions and {len(bets)} bets")

    def step_4_check_lineup_updates(self, fixtures_df: pd.DataFrame):
        """Check if lineups are available for matches < 2 hours away."""
        logger.info("=" * 80)
        logger.info("STEP 4: Checking for lineup updates")
        logger.info("=" * 80)

        cutoff_time = self.current_time + timedelta(hours=2)

        # Filter matches starting soon
        soon_matches = fixtures_df[
            pd.to_datetime(fixtures_df['date']) <= cutoff_time
        ]

        if soon_matches.empty:
            logger.info("  No matches starting within 2 hours")
            return

        logger.info(f"  Found {len(soon_matches)} matches starting within 2 hours")

        for _, match in soon_matches.iterrows():
            # TODO: Check if lineups are available via API
            # If lineups found and not in database, re-predict
            pass

    def step_5_settle_finished_bets(self):
        """Settle bets for finished matches."""
        logger.info("=" * 80)
        logger.info("STEP 5: Settling finished bets")
        logger.info("=" * 80)

        pending_bets = self.db.get_pending_bets()

        if pending_bets.empty:
            logger.info("  No pending bets to settle")
            return

        logger.info(f"  Found {len(pending_bets)} pending bets for finished matches")

        # TODO: Fetch actual results and settle bets
        # For now, this is a placeholder

    def run(self):
        """Run the complete hourly prediction pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("HOURLY PREDICTION PIPELINE")
        logger.info(f"Started: {self.current_time}")
        logger.info("=" * 80 + "\n")

        # Step 1: Fetch upcoming fixtures
        fixtures_df = self.step_1_fetch_upcoming_fixtures()

        if fixtures_df.empty:
            logger.info("\n✅ No fixtures to process - exiting")
            return True

        # Step 2: Generate predictions
        predictions = self.step_2_generate_predictions(fixtures_df)

        # Step 3: Save to database
        bets = []  # TODO: Extract from betting strategy
        self.step_3_save_to_database(predictions, bets)

        # Step 4: Check lineup updates
        self.step_4_check_lineup_updates(fixtures_df)

        # Step 5: Settle finished bets
        self.step_5_settle_finished_bets()

        logger.info("\n" + "=" * 80)
        logger.info("✅ HOURLY PREDICTION PIPELINE COMPLETED")
        logger.info(f"Finished: {datetime.now()}")
        logger.info("=" * 80)

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hourly prediction pipeline")
    parser.add_argument("--lookback-hours", type=int, default=48,
                       help="Hours ahead to look for fixtures (default: 48)")
    parser.add_argument("--force", action="store_true",
                       help="Force prediction even if recently generated")

    args = parser.parse_args()

    predictor = HourlyPredictor(
        lookback_hours=args.lookback_hours,
        force=args.force
    )
    success = predictor.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
