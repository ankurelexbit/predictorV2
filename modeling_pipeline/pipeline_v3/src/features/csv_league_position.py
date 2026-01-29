"""
CSV-based league position calculator (API-Link Version).
Directly retrieves position from the API-provided standings.csv instead of recalculating.
"""

import pandas as pd
from typing import Dict, Optional


class CSVLeaguePositionCalculator:
    """Retrieve league position features from API Standings CSV."""
    
    def __init__(self, fixtures_df: pd.DataFrame, standings_df: pd.DataFrame):
        """
        Initialize with fixtures and standings data.
        
        Args:
            fixtures_df: DataFrame with fixture data (used for basic validation if needed)
            standings_df: DataFrame with standings data (fixture_id, team_id, position)
        """
        self.fixtures = fixtures_df
        self.standings = standings_df
        
        # Optimize lookup: Index by fixture_id and team_id for O(1) access
        # We need to look up (fixture_id, team_id) -> position
        # Create a MultiIndex or a dictionary for fast lookup
        # Given 18k fixtures -> 36k standings rows.
        self.position_map = self.standings.set_index(['fixture_id', 'team_id'])['position'].to_dict()

    def get_position_features(
        self,
        fixture_id: int,
        as_of_date: str
    ) -> Dict:
        """
        Get league position features for both teams directly from API data.
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Unused in this version (lookup is by fixture_id)
        
        Returns:
            Dictionary with position features
        """
        # Get fixture details to identify teams
        fixture = self.fixtures[self.fixtures['fixture_id'] == fixture_id].iloc[0]
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        
        # Direct lookup using fixture_id which links to the standing AT THAT TIME
        home_pos = self.position_map.get((fixture_id, home_team_id))
        away_pos = self.position_map.get((fixture_id, away_team_id))
        
        # Handle cases where position is missing (e.g., cup games or data gaps)
        # Convert numpy types to native python if needed
        if hasattr(home_pos, 'item'): home_pos = home_pos.item()
        if hasattr(away_pos, 'item'): away_pos = away_pos.item()

        return {
            # Parent pipeline naming
            'home_position': home_pos,
            'away_position': away_pos,
            'position_diff': (away_pos - home_pos) if (home_pos and away_pos) else None,  # Lower position is better
            
            # Additional V3 features (Points are DEPRECATED as per USER REQUEST)
            # We explicitly return 0 or None for point-derived features to avoid confusion/calculation
            'home_league_position': home_pos,
            'away_league_position': away_pos,
            
            # Boolean flags
            'home_in_top_6': (home_pos <= 6) if home_pos else False,
            'away_in_top_6': (away_pos <= 6) if away_pos else False,
            'home_in_bottom_3': (home_pos >= 18) if home_pos else False, # Approximation (league dependent but valid signal)
            'away_in_bottom_3': (away_pos >= 18) if away_pos else False,
            
             # Zero out calculated features to strictly follow user request
            'home_points': 0, 
            'away_points': 0,
            'points_diff': 0,
            'home_points_per_game': 0,
            'away_points_per_game': 0,
            'home_points_at_home': 0,
            'away_points_away': 0,
        }

def main():
    """Test using loaded CSVs."""
    print("Testing API Position Lookup...")
    # This requires files to exist, skipped for unit test in this context
    pass

if __name__ == "__main__":
    main()
