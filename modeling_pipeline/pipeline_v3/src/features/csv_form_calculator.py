"""
CSV-based form calculator.

Calculates team form metrics (points, goals, etc.) from fixture data.
"""

import pandas as pd
import numpy as np
from typing import Optional


class CSVFormCalculator:
    """Calculate form metrics from CSV fixture data."""
    
    @staticmethod
    def calculate_team_form(
        fixtures_df: pd.DataFrame,
        team_id: int,
        as_of_date: str,
        n_matches: int = 5
    ) -> dict:
        """
        Calculate form metrics for a team as of a specific date.
        
        Args:
            fixtures_df: DataFrame with fixture data
            team_id: Team ID to calculate form for
            as_of_date: Calculate form using matches BEFORE this date
            n_matches: Number of recent matches to consider
        
        Returns:
            Dictionary with form metrics
        """
        # Get team's matches before the date
        team_matches = fixtures_df[
            (fixtures_df['starting_at'] < as_of_date) &
            ((fixtures_df['home_team_id'] == team_id) | 
             (fixtures_df['away_team_id'] == team_id))
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return {
                'matches_played': 0,
                'points': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_scored': 0,
                'goals_conceded': 0,
                'goal_difference': 0,
                'points_per_game': 0.0,
                'goals_per_game': 0.0,
                'win_rate': 0.0,
            }
        
        # Calculate metrics
        points = 0
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        
        for _, match in team_matches.iterrows():
            is_home = match['home_team_id'] == team_id
            
            # Get scores
            if is_home:
                team_score = match.get('home_score', 0)
                opp_score = match.get('away_score', 0)
            else:
                team_score = match.get('away_score', 0)
                opp_score = match.get('home_score', 0)
            
            # Skip if no scores
            if pd.isna(team_score) or pd.isna(opp_score):
                continue
            
            goals_scored += team_score
            goals_conceded += opp_score
            
            # Calculate points
            if team_score > opp_score:
                points += 3
                wins += 1
            elif team_score == opp_score:
                points += 1
                draws += 1
            else:
                losses += 1
        
        matches_played = len(team_matches)
        
        return {
            'matches_played': matches_played,
            'points': points,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'goal_difference': goals_scored - goals_conceded,
            'points_per_game': points / matches_played if matches_played > 0 else 0.0,
            'goals_per_game': goals_scored / matches_played if matches_played > 0 else 0.0,
            'win_rate': wins / matches_played if matches_played > 0 else 0.0,
        }
    
    @staticmethod
    def calculate_weighted_form(
        fixtures_df: pd.DataFrame,
        team_id: int,
        as_of_date: str,
        n_matches: int = 10,
        decay_factor: float = 0.9
    ) -> float:
        """
        Calculate weighted form score (more recent matches weighted higher).
        
        Args:
            fixtures_df: DataFrame with fixture data
            team_id: Team ID
            as_of_date: Calculate form using matches BEFORE this date
            n_matches: Number of matches to consider
            decay_factor: Weight decay for older matches (0-1)
        
        Returns:
            Weighted form score
        """
        # Get team's recent matches
        team_matches = fixtures_df[
            (fixtures_df['starting_at'] < as_of_date) &
            ((fixtures_df['home_team_id'] == team_id) | 
             (fixtures_df['away_team_id'] == team_id))
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, (_, match) in enumerate(team_matches.iterrows()):
            is_home = match['home_team_id'] == team_id
            
            # Get result
            if is_home:
                team_score = match.get('home_score', 0)
                opp_score = match.get('away_score', 0)
            else:
                team_score = match.get('away_score', 0)
                opp_score = match.get('home_score', 0)
            
            if pd.isna(team_score) or pd.isna(opp_score):
                continue
            
            # Calculate points
            if team_score > opp_score:
                points = 3
            elif team_score == opp_score:
                points = 1
            else:
                points = 0
            
            # Apply weight (more recent = higher weight)
            weight = decay_factor ** i
            weighted_score += points * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0


def main():
    """Test form calculator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load fixtures
    fixtures_df = pd.read_csv('data/csv/fixtures.csv')
    
    # Test with a team
    team_id = fixtures_df['home_team_id'].iloc[0]
    test_date = fixtures_df['starting_at'].iloc[100]
    
    calculator = CSVFormCalculator()
    
    # Calculate form
    form = calculator.calculate_team_form(fixtures_df, team_id, test_date, n_matches=5)
    weighted_form = calculator.calculate_weighted_form(fixtures_df, team_id, test_date)
    
    print(f"Team {team_id} form as of {test_date}:")
    print(f"  Last 5 matches:")
    for key, value in form.items():
        print(f"    {key}: {value}")
    print(f"  Weighted form score: {weighted_form:.2f}")


if __name__ == "__main__":
    main()
