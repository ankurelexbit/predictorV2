"""
CSV-based league position calculator.
"""

import pandas as pd
from typing import Dict


class CSVLeaguePositionCalculator:
    """Calculate league position features from CSV data."""
    
    def __init__(self, fixtures_df: pd.DataFrame):
        """Initialize with fixtures data."""
        self.fixtures = fixtures_df
    
    @staticmethod
    def calculate_standings(
        fixtures_df: pd.DataFrame,
        league_id: int,
        season_id: int,
        as_of_date: str
    ) -> pd.DataFrame:
        """
        Calculate league table as of a specific date FOR A SPECIFIC SEASON.
        
        Args:
            fixtures_df: DataFrame with fixture data
            league_id: League ID to calculate standings for
            season_id: Season ID (standings reset each season!)
            as_of_date: Calculate standings using matches BEFORE this date
        
        Returns:
            DataFrame with standings (team_id, position, points, GF, GA, GD)
        """
        # Get completed matches for this league AND SEASON before the date
        league_matches = fixtures_df[
            (fixtures_df['league_id'] == league_id) &
            (fixtures_df['season_id'] == season_id) &  # CRITICAL: Filter by season!
            (fixtures_df['starting_at'] < as_of_date) &
            (fixtures_df['result'].notna())
        ].copy()
        
        if len(league_matches) == 0:
            return pd.DataFrame()
        
        # Calculate stats for each team
        team_stats = {}
        
        for _, match in league_matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            # Initialize teams if first appearance
            if home_id not in team_stats:
                team_stats[home_id] = {'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
                                       'gf': 0, 'ga': 0, 'points': 0}
            if away_id not in team_stats:
                team_stats[away_id] = {'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
                                       'gf': 0, 'ga': 0, 'points': 0}
            
            # Update stats
            team_stats[home_id]['played'] += 1
            team_stats[away_id]['played'] += 1
            team_stats[home_id]['gf'] += home_score
            team_stats[home_id]['ga'] += away_score
            team_stats[away_id]['gf'] += away_score
            team_stats[away_id]['ga'] += home_score
            
            # Points
            if home_score > away_score:
                team_stats[home_id]['won'] += 1
                team_stats[home_id]['points'] += 3
                team_stats[away_id]['lost'] += 1
            elif home_score < away_score:
                team_stats[away_id]['won'] += 1
                team_stats[away_id]['points'] += 3
                team_stats[home_id]['lost'] += 1
            else:
                team_stats[home_id]['drawn'] += 1
                team_stats[away_id]['drawn'] += 1
                team_stats[home_id]['points'] += 1
                team_stats[away_id]['points'] += 1
        
        # Create standings DataFrame
        standings_data = []
        for team_id, stats in team_stats.items():
            standings_data.append({
                'team_id': team_id,
                'played': stats['played'],
                'won': stats['won'],
                'drawn': stats['drawn'],
                'lost': stats['lost'],
                'gf': stats['gf'],
                'ga': stats['ga'],
                'gd': stats['gf'] - stats['ga'],
                'points': stats['points'],
                'ppg': stats['points'] / stats['played'] if stats['played'] > 0 else 0,
            })
        
        standings_df = pd.DataFrame(standings_data)
        
        # Sort by points, then GD, then GF
        standings_df = standings_df.sort_values(
            ['points', 'gd', 'gf'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        
        # Add position
        standings_df['position'] = range(1, len(standings_df) + 1)
        
        return standings_df
    
    def get_position_features(
        self,
        fixture_id: int,
        as_of_date: str
    ) -> Dict:
        """
        Get league position features for both teams.
        
        Args:
            fixture_id: Fixture ID
            as_of_date: Date to calculate features as of
        
        Returns:
            Dictionary with position features
        """
        # Get fixture details
        fixture = self.fixtures[self.fixtures['fixture_id'] == fixture_id].iloc[0]
        league_id = fixture['league_id']
        season_id = fixture['season_id']  # CRITICAL: Get season_id
        home_team_id = fixture['home_team_id']
        away_team_id = fixture['away_team_id']
        
        # Calculate standings for this league AND SEASON
        standings = self.calculate_standings(
            self.fixtures,
            league_id,
            season_id,  # CRITICAL: Pass season_id
            as_of_date
        )
        
        if len(standings) == 0:
            # No standings available (early season)
            return {
                'home_league_position': None,
                'away_league_position': None,
                'position_diff': None,
                'home_points': 0,
                'away_points': 0,
                'points_diff': 0,
                'home_points_per_game': 0,
                'away_points_per_game': 0,
                'home_in_top_6': False,
                'away_in_top_6': False,
                'home_in_bottom_3': False,
                'away_in_bottom_3': False,
                'home_points_at_home': 0,
                'away_points_away': 0,
            }
        
        # Get team positions
        home_standing = standings[standings['team_id'] == home_team_id]
        away_standing = standings[standings['team_id'] == away_team_id]
        
        # Extract features
        home_pos = home_standing.iloc[0]['position'] if len(home_standing) > 0 else None
        away_pos = away_standing.iloc[0]['position'] if len(away_standing) > 0 else None
        home_pts = home_standing.iloc[0]['points'] if len(home_standing) > 0 else 0
        away_pts = away_standing.iloc[0]['points'] if len(away_standing) > 0 else 0
        home_ppg = home_standing.iloc[0]['ppg'] if len(home_standing) > 0 else 0
        away_ppg = away_standing.iloc[0]['ppg'] if len(away_standing) > 0 else 0
        
        # Calculate home/away splits
        home_pts_at_home = self._get_points_at_venue(home_team_id, True, league_id, season_id, as_of_date)
        away_pts_away = self._get_points_at_venue(away_team_id, False, league_id, season_id, as_of_date)
        
        return {
            'home_league_position': home_pos,
            'away_league_position': away_pos,
            'position_diff': home_pos - away_pos if home_pos and away_pos else None,
            'home_points': home_pts,
            'away_points': away_pts,
            'points_diff': home_pts - away_pts,
            'home_points_per_game': home_ppg,
            'away_points_per_game': away_ppg,
            'home_in_top_6': home_pos <= 6 if home_pos else False,
            'away_in_top_6': away_pos <= 6 if away_pos else False,
            'home_in_bottom_3': home_pos >= len(standings) - 2 if home_pos else False,
            'away_in_bottom_3': away_pos >= len(standings) - 2 if away_pos else False,
            'home_points_at_home': home_pts_at_home,
            'away_points_away': away_pts_away,
        }
    
    def _get_points_at_venue(
        self,
        team_id: int,
        is_home: bool,
        league_id: int,
        season_id: int,  # CRITICAL: Added season_id parameter
        as_of_date: str
    ) -> int:
        """Get points earned at home or away."""
        if is_home:
            matches = self.fixtures[
                (self.fixtures['home_team_id'] == team_id) &
                (self.fixtures['league_id'] == league_id) &
                (self.fixtures['season_id'] == season_id) &  # CRITICAL: Filter by season
                (self.fixtures['starting_at'] < as_of_date) &
                (self.fixtures['result'].notna())
            ]
        else:
            matches = self.fixtures[
                (self.fixtures['away_team_id'] == team_id) &
                (self.fixtures['league_id'] == league_id) &
                (self.fixtures['season_id'] == season_id) &  # CRITICAL: Filter by season
                (self.fixtures['starting_at'] < as_of_date) &
                (self.fixtures['result'].notna())
            ]
        
        points = 0
        for _, match in matches.iterrows():
            result = match['result']
            if is_home:
                if result == 'H':
                    points += 3
                elif result == 'D':
                    points += 1
            else:
                if result == 'A':
                    points += 3
                elif result == 'D':
                    points += 1
        
        return points


def main():
    """Test league position calculator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load fixtures
    fixtures_df = pd.read_csv('data/csv/fixtures.csv')
    
    # Test with a specific league and date
    league_id = fixtures_df['league_id'].iloc[0]
    season_id = fixtures_df['season_id'].iloc[0] # Added season_id
    test_date = fixtures_df['starting_at'].iloc[100]
    
    calculator = CSVLeaguePositionCalculator(fixtures_df) # Pass fixtures_df to constructor
    
    # Calculate standings
    standings = calculator.calculate_standings(fixtures_df, league_id, season_id, test_date) # Pass season_id
    
    print(f"League {league_id} standings as of {test_date}:")
    print(standings.head(10))
    
    # Get features for a team
    # The new get_position_features takes fixture_id
    # Let's pick a fixture that occurred before test_date
    test_fixture = fixtures_df[
        (fixtures_df['league_id'] == league_id) &
        (fixtures_df['season_id'] == season_id) &
        (fixtures_df['starting_at'] < test_date)
    ].iloc[-1] # Pick the last fixture before test_date
    
    fixture_id = test_fixture['fixture_id']
    home_team_id = test_fixture['home_team_id']
    away_team_id = test_fixture['away_team_id']

    features = calculator.get_position_features(fixture_id, test_date) # Updated call
    
    print(f"\nFixture {fixture_id} ({home_team_id} vs {away_team_id}) position features as of {test_date}:")
    for key, value in features.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
