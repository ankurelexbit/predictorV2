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
    """No-op rate limiter - let API handle rate limits naturally."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.last_remaining = None
        logger.info(f"RateLimiter disabled - API will handle rate limits naturally")
    
    def wait_if_needed(self):
        """No rate limiting - API returns 429 if limit exceeded."""
        pass
    
    def update_from_headers(self, headers: dict):
        """Track API rate limit headers for monitoring only."""
        remaining = headers.get('X-RateLimit-Remaining')
        if remaining:
            self.last_remaining = int(remaining)
            # Only log if critically low
            if self.last_remaining < 10:
                logger.warning(f"API rate limit very low: {self.last_remaining} remaining")


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
                
                # Update rate limiter with API headers
                self.rate_limiter.update_from_headers(response.headers)
                
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
        include_details: bool = True,
        finished_only: bool = False
    ) -> List[Dict]:
        """
        Get fixtures between two dates.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league_id: Optional league ID to filter
            include_details: Include statistics, lineups, etc. (default: True)
            finished_only: Only include finished fixtures (state_id=5) (default: False)

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

            # Add filters
            filters = []
            if league_id:
                filters.append(f'fixtureLeagues:{league_id}')
            if finished_only:
                filters.append('fixtureStates:5')  # 5 = finished

            if filters:
                params['filters'] = ';'.join(filters)
            
            # Include necessary data
            if include_details:
                # Core includes for feature generation
                # Note: lineups.details, events, formations are very heavy and slow
                # Only include what's actually needed for features
                includes = [
                    "participants",  # Team info
                    "scores",  # Match result
                    "statistics",  # Match statistics (shots, possession, etc.)
                    "state"  # Match state
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

    def get_standings(self, season_id: int, includes: Optional[List[str]] = None) -> Dict:
        """
        Get current league standings for a season.

        This fetches official current standings from SportMonks API, avoiding
        the need to recalculate standings from fixtures for live predictions.

        Args:
            season_id: Season ID
            includes: Optional includes (e.g., ['team', 'details'])

        Returns:
            Dict with standings data including:
            - data: List of standing groups (one per league/stage)
                - Each group has 'details' with team standings:
                    - team_id: Team ID
                    - position: League position
                    - points: Total points
                    - played: Games played
                    - won: Games won
                    - draw: Games drawn
                    - lost: Games lost
                    - goals_for: Goals scored
                    - goals_against: Goals conceded

        Example:
            >>> standings = client.get_standings(season_id=23810)
            >>> for group in standings['data']:
            ...     for detail in group['details']:
            ...         print(f"Position {detail['position']}: Team {detail['team_id']} - {detail['points']} pts")
        """
        params = {}
        if includes:
            params['include'] = ';'.join(includes)

        response = self._make_request(f'/standings/seasons/{season_id}', params)
        return response

    def get_standings_as_dataframe(self, season_id: int, include_details: bool = True) -> 'pd.DataFrame':
        """
        Get standings as a pandas DataFrame (convenience method).

        Args:
            season_id: Season ID
            include_details: If True, fetch additional details (form, stats, etc.)

        Returns:
            DataFrame with columns: team_id, position, points, played, won, draw, lost,
                                   goals_for, goals_against, goal_difference

        Note: Requires pandas to be imported
        """
        import pandas as pd

        # Fetch standings with details if requested
        includes = ['details'] if include_details else None
        standings_data = self.get_standings(season_id, includes=includes)

        # Parse standings data
        rows = []
        for standing in standings_data.get('data', []):
            # Check if this is a flat structure (just standing data) or nested
            if 'participant_id' in standing:
                # Flat structure - standings are direct
                detail = standing.get('details', [{}])[0] if 'details' in standing else {}

                rows.append({
                    'team_id': standing.get('participant_id'),
                    'position': standing.get('position'),
                    'points': standing.get('points', 0),
                    'played': detail.get('games_played', detail.get('played', 0)),
                    'won': detail.get('won', 0),
                    'draw': detail.get('draw', 0),
                    'lost': detail.get('lost', 0),
                    'goals_for': detail.get('goals_for', 0),
                    'goals_against': detail.get('goals_against', 0),
                    'goal_difference': detail.get('goal_difference', 0)
                })
            elif 'details' in standing:
                # Nested structure - standings in details array
                for detail in standing.get('details', []):
                    rows.append({
                        'team_id': detail.get('team_id', detail.get('participant_id')),
                        'position': detail.get('position'),
                        'points': detail.get('points', 0),
                        'played': detail.get('games_played', detail.get('played', 0)),
                        'won': detail.get('won', 0),
                        'draw': detail.get('draw', 0),
                        'lost': detail.get('lost', 0),
                        'goals_for': detail.get('goals_for', 0),
                        'goals_against': detail.get('goals_against', 0),
                        'goal_difference': detail.get('goal_difference', 0)
                    })

        df = pd.DataFrame(rows)

        # Calculate points_per_game if we have played
        if not df.empty and 'played' in df.columns and 'points' in df.columns:
            df['points_per_game'] = df.apply(
                lambda row: row['points'] / row['played'] if row['played'] > 0 else 0.0,
                axis=1
            )

        # Sort by position
        if not df.empty and 'position' in df.columns:
            df = df.sort_values('position').reset_index(drop=True)

        return df

    def close(self):
        """Close the session."""
        self.session.close()
