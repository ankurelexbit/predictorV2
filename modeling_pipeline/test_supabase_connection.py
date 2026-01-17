#!/usr/bin/env python3
"""
Test Supabase Database Connection
==================================

This script tests the connection to your Supabase PostgreSQL database.

Before running:
1. Get your Supabase connection details from: https://supabase.com/dashboard/project/YOUR_PROJECT/settings/database
2. Set the environment variables or update config.py with your credentials:
   - SUPABASE_DB_HOST (e.g., db.xxxxxxxxxxxxxxxxxxxx.supabase.co)
   - SUPABASE_DB_PASSWORD (your database password)
   
Or set the full DATABASE_URL environment variable.
"""

import sys
from pathlib import Path
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATABASE_URL
from utils import setup_logger

# Setup logger
logger = setup_logger("test_connection")


def test_connection():
    """Test the database connection."""
    from sqlalchemy import create_engine
    
    print(f"\nTesting connection to Supabase...")
    print(f"Connection URL: {DATABASE_URL.replace(DATABASE_URL.split('@')[0].split('://')[1].split(':')[1], '****')}")
    
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"\n✅ Successfully connected to PostgreSQL!")
            print(f"Database version: {version}")
            
            # Test current database
            result = conn.execute(text("SELECT current_database()"))
            db_name = result.fetchone()[0]
            print(f"Current database: {db_name}")
            
            # List existing tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            
            if tables:
                print(f"\nExisting tables in database:")
                for table in tables:
                    print(f"  - {table}")
            else:
                print(f"\nNo tables found in database (this is normal for a new database)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed!")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Supabase credentials in config.py")
        print("2. Ensure your IP is allowed in Supabase (Settings > Database > Connection Pooling)")
        print("3. Verify the database URL format")
        return False


def test_create_tables():
    """Test creating the database tables."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_storage", "02_data_storage.py")
    data_storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_storage)
    DatabaseManager = data_storage.DatabaseManager
    
    print("\n\nTesting table creation...")
    
    try:
        db_manager = DatabaseManager()
        
        # Create tables
        db_manager.create_tables()
        print("✅ Tables created successfully!")
        
        # Verify tables were created
        with db_manager.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('teams', 'leagues', 'matches', 'odds_snapshots', 
                                   'prediction_snapshots', 'team_elo_ratings', 'feature_store')
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            
            print(f"\nCreated tables:")
            for table in tables:
                print(f"  - {table}")
        
        return True
        
    except Exception as e:
        print(f"❌ Table creation failed!")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Supabase Database Connection")
    parser.add_argument("--create-tables", action="store_true", help="Also create the database tables")
    args = parser.parse_args()
    
    # Test connection
    if test_connection():
        if args.create_tables:
            test_create_tables()
    else:
        print("\nPlease fix the connection issues before proceeding.")