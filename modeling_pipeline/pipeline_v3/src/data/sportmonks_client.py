"""
SportMonks API Client - Base wrapper for all API interactions.
"""
import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import threading

from config.api_config import SportMonksConfig


logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token bucket rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute  # Start with full bucket
        self.max_tokens = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Minimal delay to prevent connection resets."""
        time.sleep(0.2)  # Small delay to avoid being blocked by API


class SportMonksClient:
    """
    Client for SportMonks API v3.0.
    
    Handles authentication, rate limiting, caching, and error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True):
        """
        Initialize SportMonks API client.
        
        Args:
            api_key: API key (defaults to config)
            cache_enabled: Enable response caching
        """
        self.api_key = api_key or SportMonksConfig.API_KEY
        self.base_url = SportMonksConfig.BASE_URL
        self.cache_enabled = cache_enabled and SportMonksConfig.CACHE_ENABLED
        self.cache_dir = Path(SportMonksConfig.CACHE_DIR)
        
        if not self.api_key:
            raise ValueError("API key is required")
        
        # Set up caching
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(SportMonksConfig.REQUESTS_PER_MINUTE)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': self.api_key,
            'Accept': 'application/json'
        })
    
    def _get_cache_path(self, endpoint: str, params: Dict) -> Path:
        """Generate cache file path for request."""
        # Create hash of endpoint + params
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        cache_hash = hash(cache_key)
        return self.cache_dir / f"{cache_hash}.json"
    
    def _get_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Get response from cache if valid."""
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(days=SportMonksConfig.CACHE_EXPIRY_DAYS):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save response to cache."""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request with rate limiting and retries.
        
        Args:
            endpoint: API endpoint (e.g., '/fixtures')
            params: Query parameters
        
        Returns:
            API response data
        """
        params = params or {}
        url = f"{self.base_url}{endpoint}"
        
        # Check cache first
        if self.cache_enabled:
            cache_path = self._get_cache_path(endpoint, params)
            cached_data = self._get_from_cache(cache_path)
            if cached_data:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_data
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make request with retries
        for attempt in range(SportMonksConfig.MAX_RETRIES):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=SportMonksConfig.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Save to cache
                if self.cache_enabled:
                    self._save_to_cache(cache_path, data)
                
                return data
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = SportMonksConfig.RETRY_DELAY * (SportMonksConfig.BACKOFF_FACTOR ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error
                    if attempt < SportMonksConfig.MAX_RETRIES - 1:
                        wait_time = SportMonksConfig.RETRY_DELAY * (SportMonksConfig.BACKOFF_FACTOR ** attempt)
                        logger.warning(f"Server error. Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < SportMonksConfig.MAX_RETRIES - 1:
                    wait_time = SportMonksConfig.RETRY_DELAY * (SportMonksConfig.BACKOFF_FACTOR ** attempt)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise Exception(f"Failed to fetch {endpoint} after {SportMonksConfig.MAX_RETRIES} attempts")
    
    def get_fixtures_between(
        self, 
        start_date: str, 
        end_date: str, 
        league_id: Optional[int] = None,
        include_details: bool = True
    ) -> List[Dict]:
        """
        Get fixtures between two dates.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_id: Optional league ID to filter
            include_details: Include statistics, lineups, etc. (default: True)
        
        Returns:
            List of fixtures
        """
        # Use the /between endpoint which works in API v3
        endpoint = f'/fixtures/between/{start_date}/{end_date}'
        
        # Handle pagination - API returns max 25 per page
        all_fixtures = []
        page = 1
        
        while True:
            params = {'page': page}
            
            # CRITICAL: Include all data in initial call to avoid thousands of additional API calls!
            # This matches the old successful script pattern
            if include_details:
                includes = [
                    "participants",
                    "scores", 
                    "statistics",
                    "lineups.details",  # Player-level statistics
                    "events",
                    "formations",
                    "sidelined",  # Injuries/suspensions
                    "odds",  # Betting odds
                    "state"
                ]
                params['include'] = ';'.join(includes)
            
            response = self._make_request(endpoint, params)
            
            data = response.get('data', [])
            if not data:
                break
            
            all_fixtures.extend(data)
            
            # Check if there are more pages
            pagination = response.get('pagination', {})
            if not pagination.get('has_more', False):
                break
            
            page += 1
            
            # Safety limit to avoid infinite loops
            if page > 1000:
                logger.warning(f"Reached page limit of 1000")
                break
        
        # Filter by league if specified (do it client-side since API filters don't work)
        if league_id:
            all_fixtures = [f for f in all_fixtures if f.get('league_id') == league_id]
        
        return all_fixtures
    
    def get_fixture_by_id(self, fixture_id: int, includes: Optional[List[str]] = None) -> Dict:
        """
        Get fixture by ID with optional includes.
        
        Args:
            fixture_id: Fixture ID
            includes: List of includes (e.g., ['lineups', 'statistics'])
        
        Returns:
            Fixture data
        """
        params = {}
        if includes:
            params['include'] = ';'.join(includes)
        
        response = self._make_request(f'/fixtures/{fixture_id}', params)
        return response.get('data', {})
    
    def get_fixture_statistics(self, fixture_id: int) -> List[Dict]:
        """
        Get match statistics for a fixture.
        
        Args:
            fixture_id: Fixture ID
        
        Returns:
            List of statistics
        """
        # Use include parameter instead of separate endpoint
        fixture = self.get_fixture_by_id(fixture_id, includes=['statistics'])
        return fixture.get('statistics', [])
    
    def get_fixture_lineups(self, fixture_id: int) -> List[Dict]:
        """
        Get lineups for a fixture.
        
        Args:
            fixture_id: Fixture ID
        
        Returns:
            List of lineup data
        """
        # Use include parameter instead of separate endpoint
        fixture = self.get_fixture_by_id(fixture_id, includes=['lineups'])
        return fixture.get('lineups', [])
    
    def get_team_sidelined(self, team_id: int) -> List[Dict]:
        """
        Get sidelined players (injuries/suspensions) for a team.
        
        Args:
            team_id: Team ID
        
        Returns:
            List of sidelined players
        """
        # Use include parameter instead of separate endpoint
        params = {'include': 'sidelined'}
        response = self._make_request(f'/teams/{team_id}', params)
        team_data = response.get('data', {})
        return team_data.get('sidelined', [])
    
    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """
        Get head-to-head fixtures between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
        
        Returns:
            List of H2H fixtures
        """
        response = self._make_request(f'/fixtures/head-to-head/{team1_id}/{team2_id}')
        return response.get('data', [])
    
    def close(self):
        """Close the session."""
        self.session.close()
