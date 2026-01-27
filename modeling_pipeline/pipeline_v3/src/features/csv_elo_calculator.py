"""
CSV-based Elo rating calculator.

Calculates Elo ratings for all teams over time using fixture data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


class CSVEloCalculator:
    """Calculate Elo ratings from CSV fixture data."""
    
    def __init__(self, k_factor: float = 32, initial_elo: float = 1500, home_advantage: float = 35):
        """
        Initialize Elo calculator.
        
        Args:
            k_factor: K-factor for Elo updates (higher = more volatile)
            initial_elo: Starting Elo rating for all teams
            home_advantage: Home advantage bonus (35 calibrated for modern football)
        """
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.home_advantage = home_advantage
        self.elo_history = {}  # {team_id: [(date, elo), ...]}
    
    def expected_score(self, elo_a: float, elo_b: float, is_home: bool = False) -> float:
        """
        Calculate expected score for team A.
        
        Args:
            elo_a: Elo rating of team A
            elo_b: Elo rating of team B
            is_home: Whether team A is playing at home
        
        Returns:
            Expected score (0-1) for team A
        """
        advantage = self.home_advantage if is_home else 0
        return 1 / (1 + 10 ** ((elo_b - elo_a - advantage) / 400))
    
    def update_elo(self, elo: float, actual_score: float, expected_score: float) -> float:
        """
        Update Elo rating after a match.
        
        Args:
            elo: Current Elo rating
            actual_score: Actual match result (1=win, 0.5=draw, 0=loss)
            expected_score: Expected score from Elo difference
        
        Returns:
            Updated Elo rating
        """
        return elo + self.k_factor * (actual_score - expected_score)
    
    def calculate_elo_history(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Elo ratings for all teams over time.
        
        Args:
            fixtures_df: DataFrame with fixture data
        
        Returns:
            DataFrame with columns: team_id, date, elo_rating
        """
        # Initialize Elo ratings for all teams
        team_elos = {}
        elo_records = []
        
        # Sort fixtures chronologically
        fixtures_df = fixtures_df.sort_values('starting_at').copy()
        
        # Process each fixture
        for _, fixture in fixtures_df.iterrows():
            home_team = fixture['home_team_id']
            away_team = fixture['away_team_id']
            date = fixture['starting_at']
            
            # Skip if missing data
            if pd.isna(home_team) or pd.isna(away_team):
                continue
            
            # Initialize teams if first appearance
            if home_team not in team_elos:
                team_elos[home_team] = self.initial_elo
            if away_team not in team_elos:
                team_elos[away_team] = self.initial_elo
            
            # Get current Elos
            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]
            
            # Record Elos before match
            elo_records.append({
                'team_id': home_team,
                'date': date,
                'elo_rating': home_elo,
                'fixture_id': fixture['fixture_id']
            })
            elo_records.append({
                'team_id': away_team,
                'date': date,
                'elo_rating': away_elo,
                'fixture_id': fixture['fixture_id']
            })
            
            # Calculate expected scores
            home_expected = self.expected_score(home_elo, away_elo)
            away_expected = 1 - home_expected
            
            # Get actual scores
            result = fixture.get('result')
            if pd.isna(result):
                continue  # Skip if no result
            
            if result == 'H':
                home_actual, away_actual = 1.0, 0.0
            elif result == 'A':
                home_actual, away_actual = 0.0, 1.0
            else:  # Draw
                home_actual, away_actual = 0.5, 0.5
            
            # Update Elos
            team_elos[home_team] = self.update_elo(home_elo, home_actual, home_expected)
            team_elos[away_team] = self.update_elo(away_elo, away_actual, away_expected)
        
        # Create DataFrame
        elo_df = pd.DataFrame(elo_records)
        return elo_df
    
    def get_elo_at_date(self, elo_df: pd.DataFrame, team_id: int, date: str) -> float:
        """
        Get Elo rating for a team at a specific date.
        
        Args:
            elo_df: Elo history DataFrame
            team_id: Team ID
            date: Date to get Elo for
        
        Returns:
            Elo rating at that date (or initial if no history)
        """
        team_history = elo_df[
            (elo_df['team_id'] == team_id) &
            (elo_df['date'] < date)
        ].sort_values('date')
        
        if len(team_history) == 0:
            return self.initial_elo
        
        return team_history.iloc[-1]['elo_rating']
    
    def get_elo_change(self, elo_df: pd.DataFrame, team_id: int, date: str, n_matches: int = 5) -> float:
        """
        Get Elo change over last N matches.
        
        Args:
            elo_df: Elo history DataFrame
            team_id: Team ID
            date: Current date
            n_matches: Number of matches to look back
        
        Returns:
            Elo change (current - N matches ago)
        """
        team_history = elo_df[
            (elo_df['team_id'] == team_id) &
            (elo_df['date'] < date)
        ].sort_values('date').tail(n_matches + 1)
        
        if len(team_history) < 2:
            return 0.0
        
        current_elo = team_history.iloc[-1]['elo_rating']
        past_elo = team_history.iloc[0]['elo_rating']
        
        return current_elo - past_elo
    
    def get_league_average_elo(self, elo_df: pd.DataFrame, league_id: int, date: str) -> float:
        """
        Get average Elo for a league at a specific date.
        
        Args:
            elo_df: Elo history DataFrame
            league_id: League ID
            date: Date to calculate average for
        
        Returns:
            Average Elo rating for the league
        """
        # Get most recent Elo for each team before date
        recent_elos = []
        teams = elo_df[elo_df['date'] < date]['team_id'].unique()
        
        for team_id in teams:
            elo = self.get_elo_at_date(elo_df, team_id, date)
            if elo != self.initial_elo:  # Only include teams with history
                recent_elos.append(elo)
        
        if len(recent_elos) == 0:
            return self.initial_elo
        
        return np.mean(recent_elos)


def main():
    """Test Elo calculator."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load fixtures
    fixtures_df = pd.read_csv('data/csv/fixtures.csv')
    
    # Calculate Elo history
    calculator = CSVEloCalculator()
    elo_df = calculator.calculate_elo_history(fixtures_df)
    
    # Save to CSV
    output_file = 'data/csv/elo_history.csv'
    elo_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved Elo history to {output_file}")
    print(f"   Total records: {len(elo_df):,}")
    print(f"   Teams: {elo_df['team_id'].nunique()}")
    print(f"   Date range: {elo_df['date'].min()} to {elo_df['date'].max()}")
    
    # Show sample
    print("\nSample Elo ratings:")
    print(elo_df.head(10))


if __name__ == "__main__":
    main()
