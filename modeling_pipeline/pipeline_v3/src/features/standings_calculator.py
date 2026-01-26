"""
Standings Calculator - Calculate point-in-time league standings.

Calculates team standings as of a specific date within a season.
Handles season boundaries correctly (points reset at season end).
"""
from typing import Dict, List, Optional, Tuple
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
        fixtures: List[Dict],
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> Dict[int, Dict]:
        """
        Calculate league standings as of a specific date.
        
        Args:
            fixtures: All fixtures (will be filtered)
            season_id: Season ID (to handle season boundaries)
            league_id: League ID
            as_of_date: Calculate standings as of this date
        
        Returns:
            Dict mapping team_id -> {position, points, played, wins, draws, losses, gf, ga, gd}
        """
        # Filter fixtures: same season, same league, before date, finished
        relevant_fixtures = [
            f for f in fixtures
            if f.get('season_id') == season_id
            and f.get('league_id') == league_id
            and datetime.fromisoformat(f.get('starting_at', '').replace('Z', '+00:00')) < as_of_date
            and f.get('state', {}).get('state') in ['FT', 'AET', 'FT_PEN']  # Finished matches only
        ]
        
        if not relevant_fixtures:
            logger.warning(f"No fixtures found for season {season_id}, league {league_id} before {as_of_date}")
            return {}
        
        # Initialize standings for all teams
        standings = {}
        
        for fixture in relevant_fixtures:
            participants = fixture.get('participants', [])
            if len(participants) != 2:
                continue
            
            # Get teams
            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)
            
            if not home_team or not away_team:
                continue
            
            home_id = home_team.get('id')
            away_id = away_team.get('id')
            
            # Initialize teams if not seen
            for team_id in [home_id, away_id]:
                if team_id not in standings:
                    standings[team_id] = {
                        'team_id': team_id,
                        'points': 0,
                        'played': 0,
                        'wins': 0,
                        'draws': 0,
                        'losses': 0,
                        'goals_for': 0,
                        'goals_against': 0,
                        'goal_difference': 0
                    }
            
            # Get scores
            scores = fixture.get('scores', [])
            home_goals = None
            away_goals = None
            
            for score in scores:
                if score.get('description') == 'CURRENT':
                    participant = score.get('score', {}).get('participant')
                    goals = score.get('score', {}).get('goals', 0)
                    
                    if participant == 'home':
                        home_goals = goals
                    elif participant == 'away':
                        away_goals = goals
            
            if home_goals is None or away_goals is None:
                continue
            
            # Update standings
            standings[home_id]['played'] += 1
            standings[away_id]['played'] += 1
            standings[home_id]['goals_for'] += home_goals
            standings[home_id]['goals_against'] += away_goals
            standings[away_id]['goals_for'] += away_goals
            standings[away_id]['goals_against'] += home_goals
            
            # Determine result and award points
            if home_goals > away_goals:
                # Home win
                standings[home_id]['points'] += 3
                standings[home_id]['wins'] += 1
                standings[away_id]['losses'] += 1
            elif home_goals < away_goals:
                # Away win
                standings[away_id]['points'] += 3
                standings[away_id]['wins'] += 1
                standings[home_id]['losses'] += 1
            else:
                # Draw
                standings[home_id]['points'] += 1
                standings[away_id]['points'] += 1
                standings[home_id]['draws'] += 1
                standings[away_id]['draws'] += 1
        
        # Calculate goal difference
        for team_id in standings:
            standings[team_id]['goal_difference'] = (
                standings[team_id]['goals_for'] - standings[team_id]['goals_against']
            )
        
        # Sort by points, then goal difference, then goals scored
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (
                x[1]['points'],
                x[1]['goal_difference'],
                x[1]['goals_for']
            ),
            reverse=True
        )
        
        # Add positions
        for position, (team_id, team_data) in enumerate(sorted_teams, start=1):
            standings[team_id]['position'] = position
        
        logger.info(f"Calculated standings for {len(standings)} teams in season {season_id}")
        
        return standings
    
    def get_team_standing_at_date(
        self,
        team_id: int,
        fixtures: List[Dict],
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> Dict:
        """
        Get a specific team's standing as of a date.
        
        Args:
            team_id: Team ID
            fixtures: All fixtures
            season_id: Season ID
            league_id: League ID
            as_of_date: Date to calculate standing
        
        Returns:
            Dict with position, points, played, etc.
        """
        standings = self.calculate_standings_at_date(
            fixtures, season_id, league_id, as_of_date
        )
        
        return standings.get(team_id, {
            'position': None,
            'points': 0,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_difference': 0
        })
    
    def calculate_standing_features(
        self,
        home_team_id: int,
        away_team_id: int,
        fixtures: List[Dict],
        season_id: int,
        league_id: int,
        as_of_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate all standing-related features for a match.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            fixtures: All fixtures
            season_id: Season ID
            league_id: League ID
            as_of_date: Date to calculate features
        
        Returns:
            Dict with all 12 standing features
        """
        standings = self.calculate_standings_at_date(
            fixtures, season_id, league_id, as_of_date
        )
        
        home_standing = standings.get(home_team_id, {})
        away_standing = standings.get(away_team_id, {})
        
        home_pos = home_standing.get('position', 20)  # Default to bottom
        away_pos = away_standing.get('position', 20)
        home_pts = home_standing.get('points', 0)
        away_pts = away_standing.get('points', 0)
        home_played = home_standing.get('played', 1)  # Avoid division by zero
        away_played = away_standing.get('played', 1)
        
        features = {
            # Positions
            'home_league_position': home_pos,
            'away_league_position': away_pos,
            'position_diff': home_pos - away_pos,
            
            # Points
            'home_points': home_pts,
            'away_points': away_pts,
            'points_diff': home_pts - away_pts,
            
            # Points per game
            'home_points_per_game': home_pts / max(home_played, 1),
            'away_points_per_game': away_pts / max(away_played, 1),
            
            # Position indicators
            'home_in_top_6': 1 if home_pos <= 6 else 0,
            'away_in_top_6': 1 if away_pos <= 6 else 0,
            'home_in_bottom_3': 1 if home_pos >= 18 else 0,
            'away_in_bottom_3': 1 if away_pos >= 18 else 0,
        }
        
        return features
