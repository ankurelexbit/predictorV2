"""
Database configuration.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig:
    """Database configuration."""
    
    # Database Connection
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    # Supabase (if using)
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    # Connection Pool
    POOL_SIZE = 10
    MAX_OVERFLOW = 20
    POOL_TIMEOUT = 30
    
    # Table Names
    TABLES = {
        'matches': 'matches',
        'match_statistics': 'match_statistics',
        'elo_history': 'elo_history',
        'standings_history': 'standings_history',
        'xg_history': 'xg_history',
        'training_features': 'training_features',
        'player_availability': 'player_availability',
        'key_players': 'key_players',
    }
    
    # Batch Insert Size
    BATCH_SIZE = 1000
    
    @classmethod
    def validate(cls):
        """Validate database configuration."""
        if not cls.DATABASE_URL and not (cls.SUPABASE_URL and cls.SUPABASE_KEY):
            raise ValueError("Either DATABASE_URL or SUPABASE credentials must be set")
        return True
    
    @classmethod
    def get_connection_string(cls):
        """Get database connection string."""
        if cls.DATABASE_URL:
            return cls.DATABASE_URL
        # Construct from Supabase if needed
        return None
