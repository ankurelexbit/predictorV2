"""
Point-in-Time Standings Calculator for V4 Pipeline.

Calculates league standings at any point in time from fixture results.
Adapted from V3 to work with JSON data from JSONDataLoader.
"""
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StandingsCalculator:
    """
    Calculate point-in-time league standings from fixture results.
    
    Key features:
    - Season-aware: Only uses fixtures from the same season
    - Point-in-time: Only uses fixtures before the target date
    - Accurate: Handles draws, wins, losses correctly
    """
    
    def __init__(self):
        """Initialize standings calculator."""
        self.standings_cache = {}
    
    def calculate_standings_at_date(
        self,
        fixtures_df: pd.DataFrame,
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> pd.DataFrame:
        """
        Calculate league standings as of a specific date.
        
        CRITICAL: Only uses fixtures BEFORE as_of_date for point-in-time correctness.
        
        Args:
            fixtures_df: DataFrame with all fixtures
            season_id: Season ID
            league_id: League ID
            as_of_date: Calculate standings as of this date
        
        Returns:
            DataFrame with standings (sorted by position)
            Columns: team_id, position, points, played, wins, draws, losses,
                     goals_for, goals_against, goal_difference, points_per_game
        """
        # Check cache
        cache_key = (league_id, season_id, as_of_date.isoformat())
        if cache_key in self.standings_cache:
            return self.standings_cache[cache_key].copy()
        
        # Filter fixtures: same season, same league, before date, finished
        mask = (
            (fixtures_df['season_id'] == season_id) &
            (fixtures_df['league_id'] == league_id) &
            (fixtures_df['starting_at'] < as_of_date) &
            (fixtures_df['result'].notna())  # Only completed fixtures
        )
        
        relevant_fixtures = fixtures_df[mask].copy()
        
        if len(relevant_fixtures) == 0:
            # This is normal for first fixtures of a season
            logger.debug(f"No fixtures found for season {season_id}, league {league_id} before {as_of_date} (likely season opener)")
            return pd.DataFrame()
        
        # Calculate standings
        standings = self._calculate_standings_from_fixtures(relevant_fixtures)
        
        # Cache result
        self.standings_cache[cache_key] = standings.copy()
        
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
            matches_played = 0
            wins = 0
            draws = 0
            losses = 0
            goals_for = 0
            goals_against = 0
            
            # Home stats
            for _, match in home_matches.iterrows():
                home_score = match.get('home_score')
                away_score = match.get('away_score')
                
                # Skip if scores are missing
                if pd.isna(home_score) or pd.isna(away_score):
                    continue
                
                matches_played += 1
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
                home_score = match.get('home_score')
                away_score = match.get('away_score')
                
                # Skip if scores are missing
                if pd.isna(home_score) or pd.isna(away_score):
                    continue
                
                matches_played += 1
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
    
    def get_team_standing(
        self,
        team_id: int,
        fixtures_df: pd.DataFrame,
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> Optional[Dict]:
        """
        Get a specific team's standing at a date.
        
        Args:
            team_id: Team ID
            fixtures_df: DataFrame with all fixtures
            season_id: Season ID
            league_id: League ID
            as_of_date: Date cutoff
            
        Returns:
            Dict with team's standing or None if not found
        """
        standings = self.calculate_standings_at_date(
            fixtures_df, season_id, league_id, as_of_date
        )
        
        if len(standings) == 0:
            return None
        
        team_row = standings[standings['team_id'] == team_id]
        
        if len(team_row) == 0:
            return None
        
        return team_row.iloc[0].to_dict()
    
    def get_standing_features(
        self,
        home_team_id: int,
        away_team_id: int,
        fixtures_df: pd.DataFrame,
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> Dict:
        """
        Get all standing-related features for a match.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            fixtures_df: DataFrame with all fixtures
            season_id: Season ID
            league_id: League ID
            as_of_date: Date cutoff
            
        Returns:
            Dict with all standing features
        """
        standings = self.calculate_standings_at_date(
            fixtures_df, season_id, league_id, as_of_date
        )
        
        if len(standings) == 0:
            return self._get_empty_features()
        
        home_standing = standings[standings['team_id'] == home_team_id]
        away_standing = standings[standings['team_id'] == away_team_id]
        
        if len(home_standing) == 0 or len(away_standing) == 0:
            return self._get_empty_features()
        
        home = home_standing.iloc[0]
        away = away_standing.iloc[0]
        
        total_teams = len(standings)
        
        return {
            # Positions
            'home_league_position': int(home['position']),
            'away_league_position': int(away['position']),
            'position_diff': int(home['position'] - away['position']),
            
            # Points
            'home_points': int(home['points']),
            'away_points': int(away['points']),
            'points_diff': int(home['points'] - away['points']),
            'home_points_per_game': float(home['points_per_game']),
            'away_points_per_game': float(away['points_per_game']),
            
            # Position context
            'home_in_top_6': bool(home['position'] <= 6),
            'away_in_top_6': bool(away['position'] <= 6),
            'home_in_bottom_3': bool(home['position'] >= (total_teams - 2)),
            'away_in_bottom_3': bool(away['position'] >= (total_teams - 2)),
            
            # Goal difference
            'home_goal_difference': int(home['goal_difference']),
            'away_goal_difference': int(away['goal_difference']),
        }
    
    def _get_empty_features(self) -> Dict:
        """Return empty features when no data available."""
        return {
            'home_league_position': None,
            'away_league_position': None,
            'position_diff': None,
            'home_points': 0,
            'away_points': 0,
            'points_diff': 0,
            'home_points_per_game': 0.0,
            'away_points_per_game': 0.0,
            'home_in_top_6': False,
            'away_in_top_6': False,
            'home_in_bottom_3': False,
            'away_in_bottom_3': False,
            'home_goal_difference': 0,
            'away_goal_difference': 0,
        }
    
    def clear_cache(self):
        """Clear the standings cache."""
        self.standings_cache = {}
        logger.info("Standings cache cleared")
