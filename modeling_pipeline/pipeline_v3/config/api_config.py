"""
Configuration for SportMonks API.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class SportMonksConfig:
    """SportMonks API configuration."""
    
    # API Credentials
    API_KEY = os.getenv('SPORTMONKS_API_KEY')
    BASE_URL = os.getenv('SPORTMONKS_BASE_URL', 'https://api.sportmonks.com/v3/football')
    
    # Rate Limiting (3000 requests per hour = 50 per minute)
    REQUESTS_PER_MINUTE = 50  # 3000 / 60
    REQUESTS_PER_HOUR = 3000
    
    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    BACKOFF_FACTOR = 2  # exponential backoff
    
    # Timeout
    REQUEST_TIMEOUT = 120  # seconds (increased for requests with many includes)
    
    # Caching
    CACHE_ENABLED = True
    CACHE_DIR = 'data/cache'
    CACHE_EXPIRY_DAYS = 30  # Historical data doesn't change
    
    # Leagues to track
    LEAGUES = {
        'Premier League': 8,
        'La Liga': 564,
        'Bundesliga': 82,
        'Serie A': 384,
        'Ligue 1': 301,
    }
    
    # Seasons
    SEASONS = {
        '2022-2023': 19735,
        '2023-2024': 21646,
        '2024-2025': 23389,
    }
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not cls.API_KEY:
            raise ValueError("SPORTMONKS_API_KEY not set in environment")
        if not cls.BASE_URL:
            raise ValueError("SPORTMONKS_BASE_URL not set in environment")
        return True
