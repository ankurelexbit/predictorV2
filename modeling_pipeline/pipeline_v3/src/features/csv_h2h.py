"""
CSV-based head-to-head calculator.

Calculates H2H history and matchup patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict


class CSVH2HCalculator:
    """Calculate head-to-head features from CSV data."""
    
    def __init__(self, fixtures_df: pd.DataFrame):
        """Initialize with fixtures data."""
        self.fixtures = fixtures_df
    
    def get_h2h_features(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str,
        n_matches: int = 5
    ) -> Dict:
        """
        Calculate H2H features from historical matchups.
        
        Args:
            fixtures_df: DataFrame with fixture data
            home_team_id: Home team ID
            away_team_id: Away team ID
            as_of_date: Calculate using matches BEFORE this date
            n_matches: Number of recent H2H matches to consider
        
        Returns:
            Dictionary with H2H features
        """
        # Get all H2H matches before the date
        h2h_matches = self.fixtures[
            (self.fixtures['starting_at'] < as_of_date) &
            (self.fixtures['result'].notna()) &
            (
                ((self.fixtures['home_team_id'] == home_team_id) & (self.fixtures['away_team_id'] == away_team_id)) |
                ((self.fixtures['home_team_id'] == away_team_id) & (self.fixtures['away_team_id'] == home_team_id))
            )
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(h2h_matches) == 0:
            return {
                'h2h_home_wins_last_5': 0,
                'h2h_draws_last_5': 0,
                'h2h_away_wins_last_5': 0,
                'h2h_home_goals_avg': 0.0,
                'h2h_away_goals_avg': 0.0,
                'h2h_home_win_pct': 0.0,
                'h2h_btts_pct': 0.0,
                'h2h_over_2_5_pct': 0.0,
            }
        
        # Count results from home team's perspective
        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals = []
        away_goals = []
        btts_count = 0
        over_2_5_count = 0
        
        for _, match in h2h_matches.iterrows():
            # Determine if home_team_id was home or away in this H2H match
            if match['home_team_id'] == home_team_id:
                # home_team_id was home
                team_score = match.get('home_score', 0)
                opp_score = match.get('away_score', 0)
            else:
                # home_team_id was away
                team_score = match.get('away_score', 0)
                opp_score = match.get('home_score', 0)
            
            home_goals.append(team_score)
            away_goals.append(opp_score)
            
            # Result
            if team_score > opp_score:
                home_wins += 1
            elif team_score == opp_score:
                draws += 1
            else:
                away_wins += 1
            
            # BTTS (both teams to score)
            if team_score > 0 and opp_score > 0:
                btts_count += 1
            
            # Over 2.5 goals
            if team_score + opp_score > 2.5:
                over_2_5_count += 1
        
        total_matches = len(h2h_matches)
        
        return {
            'h2h_home_wins_last_5': home_wins,
            'h2h_draws_last_5': draws,
            'h2h_away_wins_last_5': away_wins,
            'h2h_home_goals_avg': float(np.mean(home_goals)) if home_goals else 0.0,
            'h2h_away_goals_avg': float(np.mean(away_goals)) if away_goals else 0.0,
            'h2h_home_win_pct': home_wins / total_matches if total_matches > 0 else 0.0,
            'h2h_btts_pct': btts_count / total_matches if total_matches > 0 else 0.0,
            'h2h_over_2_5_pct': over_2_5_count / total_matches if total_matches > 0 else 0.0,
        }


def main():
    """Test H2H calculator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load fixtures
    fixtures_df = pd.read_csv('data/csv/fixtures.csv')
    
    # Test with a specific matchup
    test_match = fixtures_df.iloc[100]
    home_team_id = test_match['home_team_id']
    away_team_id = test_match['away_team_id']
    test_date = test_match['starting_at']
    
    calculator = CSVH2HCalculator()
    
    # Calculate H2H features
    h2h_features = calculator.calculate_h2h_features(
        fixtures_df, home_team_id, away_team_id, test_date
    )
    
    print(f"H2H features for {home_team_id} vs {away_team_id} as of {test_date}:")
    for key, value in h2h_features.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
