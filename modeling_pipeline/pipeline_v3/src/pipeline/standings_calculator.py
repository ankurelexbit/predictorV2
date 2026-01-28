"""
Season-Aware Standings Calculator

Calculates league standings at any point in time, properly separated by
league_id and season_id with point-in-time correctness.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SeasonAwareStandingsCalculator:
    """Calculate league standings at any point in time."""
    
    def __init__(self, fixtures_df: pd.DataFrame):
        """
        Initialize standings calculator.
        
        Args:
            fixtures_df: DataFrame with all fixtures (must be sorted chronologically)
        """
        self.fixtures_df = fixtures_df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(self.fixtures_df['starting_at']):
            self.fixtures_df['starting_at'] = pd.to_datetime(self.fixtures_df['starting_at'])
        
        # Cache for performance
        self._standings_cache = {}
        
        logger.info(f"Initialized SeasonAwareStandingsCalculator with {len(self.fixtures_df)} fixtures")
    
    def calculate_standings_at_date(
        self, 
        league_id: int, 
        season_id: int, 
        as_of_date: str
    ) -> pd.DataFrame:
        """
        Calculate league standings as of a specific date.
        
        CRITICAL: Only uses matches BEFORE as_of_date for point-in-time correctness.
        
        Args:
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff (ISO format)
            
        Returns:
            DataFrame with standings (sorted by position)
            Columns: team_id, position, points, matches_played, wins, draws, losses,
                     goals_for, goals_against, goal_difference, points_per_game
        """
        # Check cache
        cache_key = (league_id, season_id, as_of_date)
        if cache_key in self._standings_cache:
            return self._standings_cache[cache_key].copy()
        
        cutoff_date = pd.to_datetime(as_of_date)
        
        # Filter fixtures: same league, same season, before cutoff date, finished
        mask = (
            (self.fixtures_df['league_id'] == league_id) &
            (self.fixtures_df['season_id'] == season_id) &
            (self.fixtures_df['starting_at'] < cutoff_date) &
            (self.fixtures_df['state'] == 'FT')
        )
        
        relevant_fixtures = self.fixtures_df[mask].copy()
        
        if len(relevant_fixtures) == 0:
            logger.warning(f"No fixtures found for league {league_id}, season {season_id} before {as_of_date}")
            return pd.DataFrame()
        
        # Calculate standings
        standings = self._calculate_standings_from_fixtures(relevant_fixtures)
        
        # Cache result
        self._standings_cache[cache_key] = standings.copy()
        
        return standings
    
    def _calculate_standings_from_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate standings from a set of fixtures.
        
        Args:
            fixtures: DataFrame with fixtures
            
        Returns:
            DataFrame with standings
        """
        # Get all teams
        teams = set(fixtures['home_team_id'].unique()) | set(fixtures['away_team_id'].unique())
        
        standings_data = []
        
        for team_id in teams:
            # Home matches
            home_matches = fixtures[fixtures['home_team_id'] == team_id]
            # Away matches
            away_matches = fixtures[fixtures['away_team_id'] == team_id]
            
            # Initialize stats
            matches_played = len(home_matches) + len(away_matches)
            wins = 0
            draws = 0
            losses = 0
            goals_for = 0
            goals_against = 0
            
            # Home stats
            for _, match in home_matches.iterrows():
                home_score = match.get('home_score', 0)
                away_score = match.get('away_score', 0)
                
                # Skip if scores are missing
                if pd.isna(home_score) or pd.isna(away_score):
                    matches_played -= 1
                    continue
                
                goals_for += home_score
                goals_against += away_score
                
                if home_score > away_score:
                    wins += 1
                elif home_score == away_score:
                    draws += 1
                else:
                    losses += 1
            
            # Away stats
            for _, match in away_matches.iterrows():
                home_score = match.get('home_score', 0)
                away_score = match.get('away_score', 0)
                
                # Skip if scores are missing
                if pd.isna(home_score) or pd.isna(away_score):
                    matches_played -= 1
                    continue
                
                goals_for += away_score
                goals_against += home_score
                
                if away_score > home_score:
                    wins += 1
                elif away_score == home_score:
                    draws += 1
                else:
                    losses += 1
            
            # Calculate points and other metrics
            points = wins * 3 + draws * 1
            goal_difference = goals_for - goals_against
            points_per_game = points / matches_played if matches_played > 0 else 0.0
            
            standings_data.append({
                'team_id': team_id,
                'points': points,
                'matches_played': matches_played,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goal_difference,
                'points_per_game': points_per_game
            })
        
        # Create DataFrame
        standings = pd.DataFrame(standings_data)
        
        # Sort by points (DESC), goal difference (DESC), goals for (DESC)
        standings = standings.sort_values(
            by=['points', 'goal_difference', 'goals_for'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        # Add position
        standings['position'] = range(1, len(standings) + 1)
        
        return standings
    
    def get_team_position(
        self, 
        team_id: int, 
        league_id: int, 
        season_id: int, 
        as_of_date: str
    ) -> Optional[int]:
        """
        Get team's league position at a specific date.
        
        Args:
            team_id: Team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff
            
        Returns:
            Position (1-20) or None if not found
        """
        standings = self.calculate_standings_at_date(league_id, season_id, as_of_date)
        
        if len(standings) == 0:
            return None
        
        team_row = standings[standings['team_id'] == team_id]
        
        if len(team_row) == 0:
            return None
        
        return int(team_row.iloc[0]['position'])
    
    def get_team_points(
        self, 
        team_id: int, 
        league_id: int, 
        season_id: int, 
        as_of_date: str
    ) -> Optional[int]:
        """
        Get team's total points at a specific date.
        
        Args:
            team_id: Team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff
            
        Returns:
            Points or None if not found
        """
        standings = self.calculate_standings_at_date(league_id, season_id, as_of_date)
        
        if len(standings) == 0:
            return None
        
        team_row = standings[standings['team_id'] == team_id]
        
        if len(team_row) == 0:
            return None
        
        return int(team_row.iloc[0]['points'])
    
    def get_team_ppg(
        self, 
        team_id: int, 
        league_id: int, 
        season_id: int, 
        as_of_date: str
    ) -> Optional[float]:
        """
        Get team's points per game at a specific date.
        
        Args:
            team_id: Team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff
            
        Returns:
            Points per game or None if not found
        """
        standings = self.calculate_standings_at_date(league_id, season_id, as_of_date)
        
        if len(standings) == 0:
            return None
        
        team_row = standings[standings['team_id'] == team_id]
        
        if len(team_row) == 0:
            return None
        
        return float(team_row.iloc[0]['points_per_game'])
    
    def is_in_top_6(
        self, 
        team_id: int, 
        league_id: int, 
        season_id: int, 
        as_of_date: str
    ) -> bool:
        """
        Check if team is in top 6 (title contention).
        
        Args:
            team_id: Team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff
            
        Returns:
            True if in top 6, False otherwise
        """
        position = self.get_team_position(team_id, league_id, season_id, as_of_date)
        
        if position is None:
            return False
        
        return position <= 6
    
    def is_in_bottom_3(
        self, 
        team_id: int, 
        league_id: int, 
        season_id: int, 
        as_of_date: str
    ) -> bool:
        """
        Check if team is in bottom 3 (relegation zone).
        
        Args:
            team_id: Team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff
            
        Returns:
            True if in bottom 3, False otherwise
        """
        standings = self.calculate_standings_at_date(league_id, season_id, as_of_date)
        
        if len(standings) == 0:
            return False
        
        position = self.get_team_position(team_id, league_id, season_id, as_of_date)
        
        if position is None:
            return False
        
        # Bottom 3 = positions 18, 19, 20 (for 20-team league)
        total_teams = len(standings)
        return position >= (total_teams - 2)
    
    def get_team_stats(
        self, 
        team_id: int, 
        league_id: int, 
        season_id: int, 
        as_of_date: str
    ) -> Optional[Dict]:
        """
        Get all stats for a team at a specific date.
        
        Args:
            team_id: Team ID
            league_id: League ID
            season_id: Season ID
            as_of_date: Date cutoff
            
        Returns:
            Dict with all stats or None if not found
        """
        standings = self.calculate_standings_at_date(league_id, season_id, as_of_date)
        
        if len(standings) == 0:
            return None
        
        team_row = standings[standings['team_id'] == team_id]
        
        if len(team_row) == 0:
            return None
        
        return team_row.iloc[0].to_dict()
    
    def clear_cache(self):
        """Clear the standings cache."""
        self._standings_cache = {}
        logger.info("Standings cache cleared")
