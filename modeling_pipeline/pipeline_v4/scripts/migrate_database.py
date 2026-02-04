#!/usr/bin/env python3
"""
Database Migration Script
=========================

Adds new columns for PnL tracking and feature storage to existing database.

This migration adds:
- Odds columns (best_home_odds, best_draw_odds, best_away_odds, avg_*_odds, odds_count)
- Features column (JSONB) for storing all 162 features
- Updates existing predictions table structure

Usage:
    export DATABASE_URL="postgresql://..."
    python3 scripts/migrate_database.py
"""

import sys
import os
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if variables already exported

from src.database import SupabaseClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_database(database_url: str):
    """Run database migration."""
    logger.info("=" * 80)
    logger.info("DATABASE MIGRATION - PNL TRACKING")
    logger.info("=" * 80)

    db = SupabaseClient(database_url)

    with db.get_connection() as conn:
        cursor = conn.cursor()

        logger.info("\n📊 Checking current schema...")

        # Check if predictions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'predictions'
            )
        """)

        if not cursor.fetchone()[0]:
            logger.info("⚠️  Table 'predictions' does not exist. Creating...")
            db.create_tables()
            logger.info("✅ Tables created successfully")
            return

        # Check which columns are missing
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'predictions'
        """)

        existing_columns = {row[0] for row in cursor.fetchall()}

        # Define new columns to add
        new_columns = {
            'best_home_odds': 'FLOAT',
            'best_draw_odds': 'FLOAT',
            'best_away_odds': 'FLOAT',
            'avg_home_odds': 'FLOAT',
            'avg_draw_odds': 'FLOAT',
            'avg_away_odds': 'FLOAT',
            'odds_count': 'INTEGER DEFAULT 0',
            'features': 'JSONB'
        }

        columns_to_add = {
            col: col_type
            for col, col_type in new_columns.items()
            if col not in existing_columns
        }

        if not columns_to_add:
            logger.info("✅ All columns already exist. No migration needed.")
            return

        logger.info(f"\n🔧 Adding {len(columns_to_add)} new columns...")

        # Add each missing column
        for col_name, col_type in columns_to_add.items():
            logger.info(f"   Adding column: {col_name} ({col_type})")

            cursor.execute(f"""
                ALTER TABLE predictions
                ADD COLUMN IF NOT EXISTS {col_name} {col_type}
            """)

        logger.info("\n✅ Migration completed successfully!")

        # Show summary
        logger.info("\n" + "-" * 80)
        logger.info("MIGRATION SUMMARY")
        logger.info("-" * 80)
        logger.info("Added columns:")
        for col_name in columns_to_add.keys():
            logger.info(f"  ✓ {col_name}")

        # Count existing predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        logger.info(f"\nExisting predictions: {count}")

        if count > 0:
            logger.info("\n⚠️  NOTE: Existing predictions will have NULL values for new columns.")
            logger.info("   Re-run predictions to populate odds and features for upcoming matches.")

        logger.info("\n" + "=" * 80)


def main():
    # Get database URL
    database_url = os.environ.get('DATABASE_URL')

    if not database_url:
        logger.error("❌ DATABASE_URL not set")
        sys.exit(1)

    try:
        migrate_database(database_url)
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
