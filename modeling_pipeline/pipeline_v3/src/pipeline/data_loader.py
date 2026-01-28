"""
Historical Data Loader

Loads and organizes all historical data from CSV or JSON files.
Provides fast lookups and chronological ordering.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Load and organize historical data for feature generation."""
    
    def __init__(self, data_dir: str = 'data/csv'):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV files (or 'data/historical' for JSON)
        """
        self.data_dir = Path(data_dir)
        self.fixtures_df = None
        self.statistics_df = None
        self.lineups_df = None
        self.sidelined_df = None
        
        logger.info(f"Initializing HistoricalDataLoader with data_dir: {data_dir}")
        
    def load_all(self):
        """Load all data files."""
        logger.info("Loading all data files...")
        self.load_fixtures()
        self.load_statistics()
        self.load_lineups()
        self.load_sidelined()
        logger.info("All data loaded successfully")
        
    def load_fixtures(self) -> pd.DataFrame:
        """
        Load fixtures data.
        
        Returns:
            DataFrame with fixtures sorted chronologically
        """
        fixtures_path = self.data_dir / 'fixtures.csv'
        
        if not fixtures_path.exists():
            raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")
        
        logger.info(f"Loading fixtures from {fixtures_path}")
        self.fixtures_df = pd.read_csv(fixtures_path)
        
        # Convert date columns
        self.fixtures_df['starting_at'] = pd.to_datetime(self.fixtures_df['starting_at'])
        
        # Sort chronologically
        self.fixtures_df = self.fixtures_df.sort_values('starting_at').reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.fixtures_df)} fixtures")
        logger.info(f"Date range: {self.fixtures_df['starting_at'].min()} to {self.fixtures_df['starting_at'].max()}")
        
        return self.fixtures_df
    
    def load_statistics(self) -> pd.DataFrame:
        """
        Load match statistics.
        
        Returns:
            DataFrame with statistics indexed by fixture_id and team_id
        """
        stats_path = self.data_dir / 'statistics.csv'
        
        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        
        logger.info(f"Loading statistics from {stats_path}")
        self.statistics_df = pd.read_csv(stats_path)
        
        logger.info(f"Loaded {len(self.statistics_df)} statistics rows")
        
        return self.statistics_df
    
    def load_lineups(self) -> pd.DataFrame:
        """
        Load player lineups.
        
        Returns:
            DataFrame with lineups
        """
        lineups_path = self.data_dir / 'lineups.csv'
        
        if not lineups_path.exists():
            logger.warning(f"Lineups file not found: {lineups_path}")
            self.lineups_df = pd.DataFrame()
            return self.lineups_df
        
        logger.info(f"Loading lineups from {lineups_path}")
        self.lineups_df = pd.read_csv(lineups_path)
        
        logger.info(f"Loaded {len(self.lineups_df)} lineup rows")
        
        return self.lineups_df
    
    def load_sidelined(self) -> pd.DataFrame:
        """
        Load sidelined (injuries/suspensions) data.
        
        Returns:
            DataFrame with sidelined players
        """
        sidelined_path = self.data_dir / 'sidelined.csv'
        
        if not sidelined_path.exists():
            logger.warning(f"Sidelined file not found: {sidelined_path}")
            self.sidelined_df = pd.DataFrame()
            return self.sidelined_df
        
        logger.info(f"Loading sidelined from {sidelined_path}")
        self.sidelined_df = pd.read_csv(sidelined_path)
        
        logger.info(f"Loaded {len(self.sidelined_df)} sidelined rows")
        
        return self.sidelined_df
    
    def get_fixtures_before_date(
        self, 
        team_id: int, 
        as_of_date: str, 
        n: Optional[int] = None,
        league_id: Optional[int] = None,
        season_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get fixtures for a team before a specific date.
        
        Args:
            team_id: Team ID
            as_of_date: Date cutoff (ISO format)
            n: Number of most recent matches (None = all)
            league_id: Filter by league (optional)
            season_id: Filter by season (optional)
            
        Returns:
            DataFrame with fixtures sorted chronologically
        """
        if self.fixtures_df is None:
            raise ValueError("Fixtures not loaded. Call load_fixtures() first.")
        
        # Convert as_of_date to datetime
        cutoff_date = pd.to_datetime(as_of_date)
        
        # Filter fixtures
        mask = (
            ((self.fixtures_df['home_team_id'] == team_id) | 
             (self.fixtures_df['away_team_id'] == team_id)) &
            (self.fixtures_df['starting_at'] < cutoff_date) &
            (self.fixtures_df['state'] == 'FT')  # Only finished matches
        )
        
        if league_id is not None:
            mask &= (self.fixtures_df['league_id'] == league_id)
        
        if season_id is not None:
            mask &= (self.fixtures_df['season_id'] == season_id)
        
        fixtures = self.fixtures_df[mask].copy()
        
        # Sort chronologically
        fixtures = fixtures.sort_values('starting_at')
        
        # Take last n if specified
        if n is not None:
            fixtures = fixtures.tail(n)
        
        return fixtures
    
    def get_statistics_for_fixture(self, fixture_id: int) -> Dict[str, pd.Series]:
        """
        Get statistics for a specific fixture.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Dict with 'home' and 'away' statistics
        """
        if self.statistics_df is None:
            raise ValueError("Statistics not loaded. Call load_statistics() first.")
        
        stats = self.statistics_df[self.statistics_df['fixture_id'] == fixture_id]
        
        if len(stats) == 0:
            return {'home': pd.Series(), 'away': pd.Series()}
        
        home_stats = stats[stats['is_home'] == True].iloc[0] if len(stats[stats['is_home'] == True]) > 0 else pd.Series()
        away_stats = stats[stats['is_home'] == False].iloc[0] if len(stats[stats['is_home'] == False]) > 0 else pd.Series()
        
        return {'home': home_stats, 'away': away_stats}
    
    def get_h2h_fixtures(
        self, 
        team1_id: int, 
        team2_id: int, 
        as_of_date: str,
        n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get head-to-head fixtures between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            as_of_date: Date cutoff
            n: Number of most recent H2H matches
            
        Returns:
            DataFrame with H2H fixtures
        """
        if self.fixtures_df is None:
            raise ValueError("Fixtures not loaded. Call load_fixtures() first.")
        
        cutoff_date = pd.to_datetime(as_of_date)
        
        # Find all matches between these teams
        mask = (
            (((self.fixtures_df['home_team_id'] == team1_id) & (self.fixtures_df['away_team_id'] == team2_id)) |
             ((self.fixtures_df['home_team_id'] == team2_id) & (self.fixtures_df['away_team_id'] == team1_id))) &
            (self.fixtures_df['starting_at'] < cutoff_date) &
            (self.fixtures_df['state'] == 'FT')
        )
        
        h2h = self.fixtures_df[mask].copy()
        h2h = h2h.sort_values('starting_at')
        
        if n is not None:
            h2h = h2h.tail(n)
        
        return h2h
    
    def validate_data_completeness(self) -> Dict:
        """
        Validate data completeness and quality.
        
        Returns:
            Dict with validation results
        """
        if self.fixtures_df is None:
            raise ValueError("Data not loaded. Call load_all() first.")
        
        results = {
            'total_fixtures': len(self.fixtures_df),
            'fixtures_with_stats': 0,
            'fixtures_with_lineups': 0,
            'date_range': {
                'start': str(self.fixtures_df['starting_at'].min()),
                'end': str(self.fixtures_df['starting_at'].max())
            },
            'leagues': self.fixtures_df['league_id'].nunique(),
            'teams': len(set(self.fixtures_df['home_team_id'].unique()) | 
                         set(self.fixtures_df['away_team_id'].unique())),
            'seasons': self.fixtures_df['season_id'].nunique()
        }
        
        if self.statistics_df is not None and len(self.statistics_df) > 0:
            results['fixtures_with_stats'] = self.statistics_df['fixture_id'].nunique()
        
        if self.lineups_df is not None and len(self.lineups_df) > 0:
            results['fixtures_with_lineups'] = self.lineups_df['fixture_id'].nunique()
        
        logger.info("Data validation results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        
        return results
    
    def get_fixture_info(self, fixture_id: int) -> Optional[pd.Series]:
        """
        Get fixture information.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Series with fixture info or None
        """
        if self.fixtures_df is None:
            raise ValueError("Fixtures not loaded. Call load_fixtures() first.")
        
        fixture = self.fixtures_df[self.fixtures_df['fixture_id'] == fixture_id]
        
        if len(fixture) == 0:
            return None
        
        return fixture.iloc[0]
