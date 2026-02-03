#!/usr/bin/env python3
"""
Migrate Database to Support Prediction History
===============================================

This script updates the database schema to support prediction history tracking.

Changes:
1. Removes UNIQUE(fixture_id, model_version) constraint
2. Adds indexes for efficient history queries
3. Enables tracking of how predictions change over time

Run this ONCE on your existing database before using the updated prediction system.

Usage:
    export DATABASE_URL="postgresql://..."
    python3 scripts/migrate_prediction_history.py
"""

import os
import sys
import logging
import psycopg2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_database(database_url: str):
    """
    Migrate database to support prediction history.

    Args:
        database_url: PostgreSQL connection string
    """
    logger.info("="*80)
    logger.info("MIGRATING DATABASE TO SUPPORT PREDICTION HISTORY")
    logger.info("="*80)

    conn = psycopg2.connect(database_url)
    cursor = conn.cursor()

    try:
        # Step 1: Check if unique constraint exists
        logger.info("\n[1/4] Checking for existing UNIQUE constraint...")
        cursor.execute("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'predictions'
              AND constraint_type = 'UNIQUE'
              AND constraint_name LIKE '%fixture_id%'
        """)

        constraints = cursor.fetchall()

        if constraints:
            logger.info(f"   Found {len(constraints)} UNIQUE constraint(s) to remove")

            # Step 2: Drop unique constraint
            logger.info("\n[2/4] Dropping UNIQUE(fixture_id, model_version) constraint...")
            for (constraint_name,) in constraints:
                cursor.execute(f"""
                    ALTER TABLE predictions
                    DROP CONSTRAINT IF EXISTS {constraint_name}
                """)
                logger.info(f"   ✅ Dropped constraint: {constraint_name}")

            conn.commit()
        else:
            logger.info("   No UNIQUE constraints found (already migrated or new database)")

        # Step 3: Add new indexes for history queries
        logger.info("\n[3/4] Adding indexes for prediction history queries...")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_fixture_timestamp
            ON predictions(fixture_id, prediction_timestamp DESC)
        """)
        logger.info("   ✅ Created index: idx_predictions_fixture_timestamp")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_fixture_model
            ON predictions(fixture_id, model_version, prediction_timestamp DESC)
        """)
        logger.info("   ✅ Created index: idx_predictions_fixture_model")

        conn.commit()

        # Step 4: Verify migration
        logger.info("\n[4/4] Verifying migration...")

        # Check table structure
        cursor.execute("""
            SELECT COUNT(*) FROM predictions
        """)
        total_predictions = cursor.fetchone()[0]
        logger.info(f"   Total predictions in database: {total_predictions}")

        # Check for duplicates (fixtures with multiple predictions)
        cursor.execute("""
            SELECT fixture_id, model_version, COUNT(*) as pred_count
            FROM predictions
            GROUP BY fixture_id, model_version
            HAVING COUNT(*) > 1
            ORDER BY pred_count DESC
            LIMIT 5
        """)

        duplicates = cursor.fetchall()
        if duplicates:
            logger.info(f"   ✅ Found {len(duplicates)} fixtures with prediction history")
            logger.info("   Sample fixtures with multiple predictions:")
            for fixture_id, model_version, count in duplicates[:3]:
                logger.info(f"      Fixture {fixture_id} ({model_version}): {count} predictions")
        else:
            logger.info("   No fixtures with multiple predictions yet (fresh migration)")

        logger.info("\n" + "="*80)
        logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("\nNew Features Available:")
        logger.info("  - get_latest_prediction(fixture_id)")
        logger.info("  - get_prediction_timeline(fixture_id)")
        logger.info("  - get_prediction_changes(fixture_id)")
        logger.info("\nYou can now run predictions daily and track changes over time!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


def main():
    """Run migration."""
    database_url = os.environ.get('DATABASE_URL')

    if not database_url:
        logger.error("❌ DATABASE_URL environment variable not set")
        logger.error("\nUsage:")
        logger.error('  export DATABASE_URL="postgresql://user:pass@host:port/database"')
        logger.error("  python3 scripts/migrate_prediction_history.py")
        sys.exit(1)

    logger.info(f"Database: {database_url.split('@')[1] if '@' in database_url else 'localhost'}")

    # Ask for confirmation
    response = input("\n⚠️  This will modify your database schema. Continue? [y/N]: ")
    if response.lower() != 'y':
        logger.info("Migration cancelled")
        sys.exit(0)

    migrate_database(database_url)


if __name__ == '__main__':
    main()
