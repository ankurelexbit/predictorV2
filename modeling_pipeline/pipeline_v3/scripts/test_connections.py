"""
Test SportMonks API and Supabase connections.

This script verifies that your credentials are working correctly.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sportmonks_client import SportMonksClient
from src.utils.database import SupabaseDB
from config.api_config import SportMonksConfig
from config.database_config import DatabaseConfig


def test_sportmonks_api():
    """Test SportMonks API connection."""
    print("=" * 80)
    print("TESTING SPORTMONKS API CONNECTION")
    print("=" * 80)
    
    try:
        # Validate config
        SportMonksConfig.validate()
        print("✅ Configuration validated")
        print(f"   Base URL: {SportMonksConfig.BASE_URL}")
        print(f"   API Key: {SportMonksConfig.API_KEY[:10]}..." if SportMonksConfig.API_KEY else "   API Key: NOT SET")
        
        # Initialize client
        client = SportMonksClient()
        print("✅ Client initialized")
        
        # Test a simple API call - get leagues
        print("\nTesting API call (fetching leagues)...")
        response = client._make_request('/leagues', {'per_page': 5})
        
        if response and 'data' in response:
            leagues = response['data']
            print(f"✅ API call successful!")
            print(f"   Retrieved {len(leagues)} leagues")
            
            if leagues:
                print(f"\n   Sample league:")
                league = leagues[0]
                print(f"   - ID: {league.get('id')}")
                print(f"   - Name: {league.get('name')}")
                print(f"   - Country: {league.get('country', {}).get('name', 'N/A')}")
            
            return True
        else:
            print("❌ API call failed - no data returned")
            return False
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False
    finally:
        try:
            client.close()
        except:
            pass


def test_supabase_connection():
    """Test Supabase database connection."""
    print("\n" + "=" * 80)
    print("TESTING SUPABASE DATABASE CONNECTION")
    print("=" * 80)
    
    try:
        # Validate config
        DatabaseConfig.validate()
        print("✅ Configuration validated")
        print(f"   Supabase URL: {DatabaseConfig.SUPABASE_URL}")
        print(f"   Supabase Key: {DatabaseConfig.SUPABASE_KEY[:10]}..." if DatabaseConfig.SUPABASE_KEY else "   Key: NOT SET")
        
        # Initialize database
        db = SupabaseDB()
        print("✅ Database client initialized")
        
        # Test connection
        print("\nTesting database connection...")
        if db.test_connection():
            print("✅ Database connection successful!")
            
            # Try to count records in tables
            print("\nChecking tables:")
            
            tables = ['matches', 'match_statistics', 'elo_history', 'training_features']
            for table in tables:
                try:
                    result = db.client.table(table).select('id', count='exact').limit(0).execute()
                    count = result.count if hasattr(result, 'count') else 0
                    print(f"   ✅ {table}: {count} records")
                except Exception as e:
                    print(f"   ⚠️  {table}: Table exists but error counting - {str(e)[:50]}")
            
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nMake sure you've set SUPABASE_URL and SUPABASE_KEY in .env file")
        return False
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🔧 CONNECTION TESTS" + "\n")
    
    # Test SportMonks API
    api_success = test_sportmonks_api()
    
    # Test Supabase
    db_success = test_supabase_connection()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"SportMonks API: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"Supabase DB:    {'✅ PASS' if db_success else '❌ FAIL'}")
    print("=" * 80)
    
    if api_success and db_success:
        print("\n🎉 All connections working! You're ready to start downloading data.")
        print("\nNext steps:")
        print("1. Download historical data:")
        print("   python scripts/backfill_historical_data.py --start-date 2024-08-01 --end-date 2025-05-31")
        print("\n2. Generate features:")
        print("   python scripts/generate_training_features.py")
        return 0
    else:
        print("\n⚠️  Some connections failed. Please check your .env file and try again.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
