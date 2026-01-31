"""
Optimized JSON Data Loader for V4 Pipeline - Uses embedded statistics.

Uses ijson for efficient streaming parsing of large JSON files.
Statistics and lineups are embedded in fixture JSON, no need for separate files.
"""
import ijson
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class JSONDataLoader:
    """Load and parse raw JSON fixture data using streaming parser."""
    
    def __init__(self, data_dir: str = 'data/historical'):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing JSON files
        """
        self.data_dir = Path(data_dir)
        self.fixtures_dir = self.data_dir / 'fixtures'
        
        # Cache
        self._fixtures_cache = None
        self._fixtures_dict = {}  # fixture_id -> full fixture data
        
        logger.info(f"Initialized JSONDataLoader (streaming) with data_dir: {self.data_dir}")
    
    def load_all_fixtures(self, cache_full_data: bool = True, use_cache: bool = True, use_csv: bool = True) -> pd.DataFrame:
        """
        Load all fixtures from JSON files using streaming parser.
        
        Args:
            cache_full_data: If True, cache full fixture data for statistics access
            use_cache: If True, try to load from cache first
            use_csv: If True, try to load from CSV first (fastest!)
        
        Returns:
            DataFrame with all fixtures
        """
        if self._fixtures_cache is not None:
            return self._fixtures_cache
        
        # Try to load from CSV (fastest!)
        if use_csv:
            csv_file = Path('data/processed/fixtures_with_stats.csv')
            if csv_file.exists():
                logger.info("Loading fixtures from comprehensive CSV...")
                self._fixtures_cache = pd.read_csv(csv_file)
                self._fixtures_cache['starting_at'] = pd.to_datetime(self._fixtures_cache['starting_at'])
                logger.info(f"✅ Loaded {len(self._fixtures_cache)} fixtures with statistics from CSV (instant!)")
                return self._fixtures_cache
            else:
                logger.info("Comprehensive CSV not found.")
                logger.info("Run 'python3 scripts/convert_json_to_csv.py' to create it")
        
        # Try to load from cache
        if use_cache:
            cache_dir = Path('data/cache')
            fixtures_cache_file = cache_dir / 'fixtures_df.pkl'
            fixtures_dict_cache_file = cache_dir / 'fixtures_dict.pkl'
            
            if fixtures_cache_file.exists() and fixtures_dict_cache_file.exists():
                logger.info("Loading fixtures from cache...")
                import pickle
                
                self._fixtures_cache = pd.read_pickle(fixtures_cache_file)
                with open(fixtures_dict_cache_file, 'rb') as f:
                    self._fixtures_dict = pickle.load(f)
                
                logger.info(f"✅ Loaded {len(self._fixtures_cache)} fixtures from cache (instant!)")
                return self._fixtures_cache
            else:
                logger.info("Cache not found. Loading from JSON files...")
                logger.info("Tip: Run 'python3 scripts/build_cache.py' to build cache for instant loading")
        
        logger.info("Loading all fixtures from JSON (streaming mode)...")
        
        # Find all fixture JSON files
        fixture_files = sorted(self.fixtures_dir.glob('*.json'))
        logger.info(f"Found {len(fixture_files)} JSON files")
        
        all_fixtures = []
        
        for file_path in fixture_files:
            logger.info(f"  Processing {file_path.name}...")
            
            try:
                with open(file_path, 'rb') as f:
                    # Stream parse the JSON array
                    parser = ijson.items(f, 'item')
                    
                    for fixture in parser:
                        # Extract essential fields
                        essential_data = self._extract_essential_fields(fixture)
                        if essential_data:
                            all_fixtures.append(essential_data)
                            
                            # Cache full fixture data if requested
                            if cache_full_data:
                                self._fixtures_dict[essential_data['id']] = fixture
                
                logger.info(f"    ✓ Loaded {len(all_fixtures)} total fixtures so far")
                
            except Exception as e:
                logger.warning(f"    ✗ Failed to parse {file_path.name}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_fixtures)
        
        # Parse dates
        if 'starting_at' in df.columns:
            df['starting_at'] = pd.to_datetime(df['starting_at'])
        
        # Sort by date
        df = df.sort_values('starting_at').reset_index(drop=True)
        
        logger.info(f"✅ Loaded {len(df)} fixtures total")
        logger.info(f"   Date range: {df['starting_at'].min()} to {df['starting_at'].max()}")
        
        self._fixtures_cache = df
        return df
    
    def _extract_essential_fields(self, fixture: Dict) -> Optional[Dict]:
        """
        Extract only essential fields from fixture to reduce memory.
        
        Args:
            fixture: Full fixture dictionary
            
        Returns:
            Dict with essential fields only
        """
        try:
            # Get participants
            participants = fixture.get('participants', [])
            if len(participants) != 2:
                return None
            
            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
            
            if not home_team or not away_team:
                return None
            
            # Get scores
            scores = fixture.get('scores', [])
            home_score = None
            away_score = None
            
            for score in scores:
                if score.get('description') == 'CURRENT':
                    participant = score.get('score', {}).get('participant')
                    goals = score.get('score', {}).get('goals')
                    
                    if participant == 'home':
                        home_score = goals
                    elif participant == 'away':
                        away_score = goals
            
            # Determine result
            result = None
            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    result = 'H'
                elif home_score < away_score:
                    result = 'A'
                else:
                    result = 'D'
            
            return {
                'id': fixture.get('id'),
                'league_id': fixture.get('league_id'),
                'season_id': fixture.get('season_id'),
                'starting_at': fixture.get('starting_at'),
                'home_team_id': home_team.get('id'),
                'away_team_id': away_team.get('id'),
                'home_score': home_score,
                'away_score': away_score,
                'result': result,
                'state_id': fixture.get('state_id'),
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract fields from fixture: {e}")
            return None
    
    def get_fixture(self, fixture_id: int) -> Dict:
        """
        Get a single fixture by ID.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Fixture dictionary
        """
        df = self.load_all_fixtures()
        fixture = df[df['id'] == fixture_id]
        
        if len(fixture) == 0:
            raise ValueError(f"Fixture {fixture_id} not found")
        
        return fixture.iloc[0].to_dict()
    
    def get_fixtures_before(
        self,
        as_of_date: datetime,
        league_id: Optional[int] = None,
        season_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get all fixtures before a specific date.
        
        Args:
            as_of_date: Cutoff date
            league_id: Optional league filter
            season_id: Optional season filter
            
        Returns:
            DataFrame with filtered fixtures
        """
        df = self.load_all_fixtures()
        
        # Filter by date
        mask = df['starting_at'] < as_of_date
        
        # Filter by league
        if league_id is not None:
            mask &= (df['league_id'] == league_id)
        
        # Filter by season
        if season_id is not None:
            mask &= (df['season_id'] == season_id)
        
        return df[mask].copy()
    
    def get_team_fixtures(
        self,
        team_id: int,
        before_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get fixtures for a specific team before a date.
        
        Args:
            team_id: Team ID
            before_date: Cutoff date
            limit: Optional limit on number of fixtures
            
        Returns:
            DataFrame with team's fixtures
        """
        df = self.load_all_fixtures()
        
        # Filter by team and date
        mask = (
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['starting_at'] < before_date) &
            (df['result'].notna())  # Only completed fixtures
        )
        
        team_fixtures = df[mask].sort_values('starting_at', ascending=False)
        
        if limit is not None:
            team_fixtures = team_fixtures.head(limit)
        
        return team_fixtures
    
    def get_statistics(self, fixture_id: int) -> Optional[Dict]:
        """
        Get statistics for a specific fixture from cached data.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Statistics dictionary or None if not found
        """
        # Get from cache
        if fixture_id in self._fixtures_dict:
            fixture_data = self._fixtures_dict[fixture_id]
            return fixture_data.get('statistics', [])
        
        return None
    
    def get_lineups(self, fixture_id: int) -> Optional[Dict]:
        """
        Get lineups for a specific fixture from cached data.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Lineups dictionary or None if not found
        """
        # Get from cache
        if fixture_id in self._fixtures_dict:
            fixture_data = self._fixtures_dict[fixture_id]
            return fixture_data.get('lineups', [])
        
        return None
