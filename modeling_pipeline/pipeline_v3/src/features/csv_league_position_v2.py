"""
CSV-based league position calculator using API standings data.

This version uses the standings data extracted from SportMonks API
participants.meta instead of calculating from match results.
"""

import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CSVLeaguePositionCalculatorV2:
    """
    Calculate league position features from API standings CSV.
    
    Uses official standings from SportMonks API (participants.meta)
    instead of calculating from match results.
    """
    
    def __init__(self, fixtures_df: pd.DataFrame, standings_df: pd.DataFrame = None):
        """
        Initialize with fixtures and standings data.
        
        Args:
            fixtures_df: Fixtures DataFrame
            standings_df: Standings DataFrame (optional, will try to load if None)
        """
        self.fixtures = fixtures_df
        
        # Load or use provided standings
        if standings_df is None:
            try:
                self.standings = pd.read_csv('data/csv/standings.csv')
                logger.info(f"Loaded {len(self.standings)} standings records")
            except FileNotFoundError:
                logger.warning("standings.csv not found - will fall back to calculation")
                self.standings = None
        else:
            self.standings = standings_df
    
    def get_position_features(
        self,
        fixture_id: int,
        as_of_date: str
    ) -> Dict:
        """
        Get league position features for both teams.
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Date to calculate features as of (not used with API data)
        
        Returns:
            Dictionary with position features
        """
        if self.standings is None:
            logger.warning("No standings data available - returning defaults")
            return self._get_default_features()
        
        # Get standings for this fixture
        fixture_standings = self.standings[
            self.standings['fixture_id'] == fixture_id
        ]
        
        if len(fixture_standings) == 0:
            logger.warning(f"No standings found for fixture {fixture_id}")
            return self._get_default_features()
        
        # Get home and away standings
        home_standing = fixture_standings[
            fixture_standings['location'] == 'home'
        ]
        away_standing = fixture_standings[
            fixture_standings['location'] == 'away'
        ]
        
        if len(home_standing) == 0 or len(away_standing) == 0:
            logger.warning(f"Incomplete standings for fixture {fixture_id}")
            return self._get_default_features()
        
        # Extract data
        home = home_standing.iloc[0]
        away = away_standing.iloc[0]
        
        # Calculate features
        home_pos = home['position'] if pd.notna(home['position']) else 20
        away_pos = away['position'] if pd.notna(away['position']) else 20
        home_pts = home['points'] if pd.notna(home['points']) else 0
        away_pts = away['points'] if pd.notna(away['points']) else 0
        home_played = home['played'] if pd.notna(home['played']) else 1
        away_played = away['played'] if pd.notna(away['played']) else 1
        
        return {
            # Parent pipeline naming
            'home_position': int(home_pos),
            'away_position': int(away_pos),
            'position_diff': int(away_pos - home_pos),  # Positive = home ranked higher
            'home_points': int(home_pts),
            'away_points': int(away_pts),
            'points_diff': int(home_pts - away_pts),
            
            # Additional V3 features
            'home_league_position': int(home_pos),
            'away_league_position': int(away_pos),
            'home_points_per_game': float(home_pts / max(home_played, 1)),
            'away_points_per_game': float(away_pts / max(away_played, 1)),
            'home_in_top_6': bool(home_pos <= 6),
            'away_in_top_6': bool(away_pos <= 6),
            'home_in_bottom_3': bool(home_pos >= 18),
            'away_in_bottom_3': bool(away_pos >= 18),
            
            # Bonus features from API data
            'home_wins': int(home['wins']) if pd.notna(home['wins']) else 0,
            'away_wins': int(away['wins']) if pd.notna(away['wins']) else 0,
            'home_draws': int(home['draws']) if pd.notna(home['draws']) else 0,
            'away_draws': int(away['draws']) if pd.notna(away['draws']) else 0,
            'home_losses': int(home['losses']) if pd.notna(home['losses']) else 0,
            'away_losses': int(away['losses']) if pd.notna(away['losses']) else 0,
            'home_gf': int(home['goals_for']) if pd.notna(home['goals_for']) else 0,
            'away_gf': int(away['goals_for']) if pd.notna(away['goals_for']) else 0,
            'home_ga': int(home['goals_against']) if pd.notna(home['goals_against']) else 0,
            'away_ga': int(away['goals_against']) if pd.notna(away['goals_against']) else 0,
            'home_gd': int(home['goal_difference']) if pd.notna(home['goal_difference']) else 0,
            'away_gd': int(away['goal_difference']) if pd.notna(away['goal_difference']) else 0,
            
            # Note: home/away venue splits not available in API data
            # These would need to be calculated separately
            'home_points_at_home': 0,  # TODO: Calculate if needed
            'away_points_away': 0,  # TODO: Calculate if needed
        }
    
    def _get_default_features(self) -> Dict:
        """Return default features when standings not available."""
        return {
            'home_position': 20,
            'away_position': 20,
            'position_diff': 0,
            'home_points': 0,
            'away_points': 0,
            'points_diff': 0,
            'home_league_position': 20,
            'away_league_position': 20,
            'home_points_per_game': 0.0,
            'away_points_per_game': 0.0,
            'home_in_top_6': False,
            'away_in_top_6': False,
            'home_in_bottom_3': False,
            'away_in_bottom_3': False,
            'home_wins': 0,
            'away_wins': 0,
            'home_draws': 0,
            'away_draws': 0,
            'home_losses': 0,
            'away_losses': 0,
            'home_gf': 0,
            'away_gf': 0,
            'home_ga': 0,
            'away_ga': 0,
            'home_gd': 0,
            'away_gd': 0,
            'home_points_at_home': 0,
            'away_points_away': 0,
        }


# For backward compatibility, keep the old calculator available
from .csv_league_position import CSVLeaguePositionCalculator as CSVLeaguePositionCalculatorV1
