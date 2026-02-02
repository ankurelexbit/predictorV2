#!/usr/bin/env python3
"""
Test Database Connection
========================

Tests Supabase database connection and creates tables.

Usage:
    export DATABASE_URL="postgresql://..."
    python3 scripts/test_database.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SupabaseClient


def main():
    print("=" * 80)
    print("SUPABASE DATABASE CONNECTION TEST")
    print("=" * 80)

    # Get database URL
    database_url = os.environ.get('DATABASE_URL')

    if not database_url:
        print("❌ DATABASE_URL not set")
        print("\nSet it with:")
        print('export DATABASE_URL="postgresql://user:pass@host:port/database"')
        sys.exit(1)

    print(f"\n📦 Database URL: {database_url[:30]}...")

    # Initialize client
    try:
        print("\n🔌 Connecting to database...")
        db = SupabaseClient(database_url)

        # Test connection
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ Connected successfully!")
            print(f"   PostgreSQL version: {version}")

        # Create tables
        print("\n📊 Creating tables...")
        db.create_tables()

        # Verify tables exist
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'predictions'
            """)
            result = cursor.fetchone()

            if result:
                print("✅ 'predictions' table exists")

                # Get table info
                cursor.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'predictions'
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()

                print(f"\n📋 Table structure ({len(columns)} columns):")
                print("-" * 80)
                for col_name, col_type in columns:
                    print(f"   {col_name:30s} {col_type}")

        print("\n" + "=" * 80)
        print("✅ DATABASE TEST SUCCESSFUL")
        print("=" * 80)
        print("\nYou can now run:")
        print("  python3 scripts/predict_and_store.py --days-ahead 7")

    except Exception as e:
        print(f"\n❌ Database error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
