"""
CSV-based derived xG calculator.

Calculates expected goals (xG) from shot statistics.
"""

import pandas as pd
import numpy as np


class CSVDerivedXGCalculator:
    """Calculate derived xG from statistics CSV."""
    
    @staticmethod
    def calculate_xg(statistics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate xG from shot statistics.
        
        Formula:
            xG = (shots_on_target * 0.32) + (shots_off_target * 0.05) + 
                 (blocked_shots * 0.03)
        
        Args:
            statistics_df: DataFrame with team statistics
        
        Returns:
            DataFrame with xG columns added
        """
        df = statistics_df.copy()
        
        # Get shot columns (handle missing data)
        shots_on_target = df.get('shots_on_target', 0).fillna(0)
        shots_off_target = df.get('shots_off_target', 0).fillna(0)
        blocked_shots = df.get('blocked_shots', 0).fillna(0)
        
        # Calculate xG
        df['xG'] = (
            shots_on_target * 0.32 +
            shots_off_target * 0.05 +
            blocked_shots * 0.03
        )
        
        # Calculate xGA (expected goals against) - defensive metric
        # This will be filled when we have opponent stats
        df['xGA'] = 0.0
        
        return df
    
    @staticmethod
    def calculate_rolling_xg(
        statistics_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        team_id: int,
        as_of_date: str,
        n_matches: int = 5
    ) -> dict:
        """
        Calculate rolling xG averages for a team.
        
        Args:
            statistics_df: DataFrame with team statistics (with xG calculated)
            fixtures_df: DataFrame with fixture data
            team_id: Team ID
            as_of_date: Calculate using matches BEFORE this date
            n_matches: Number of matches for rolling average
        
        Returns:
            Dictionary with xG metrics
        """
        # Get team's recent matches
        team_fixtures = fixtures_df[
            (fixtures_df['starting_at'] < as_of_date) &
            ((fixtures_df['home_team_id'] == team_id) | 
             (fixtures_df['away_team_id'] == team_id))
        ].sort_values('starting_at', ascending=False).head(n_matches)
        
        if len(team_fixtures) == 0:
            return {
                'xG_avg': 0.0,
                'xGA_avg': 0.0,
                'xG_diff_avg': 0.0,
            }
        
        # Get statistics for these fixtures
        fixture_ids = team_fixtures['fixture_id'].tolist()
        team_stats = statistics_df[
            (statistics_df['fixture_id'].isin(fixture_ids)) &
            (statistics_df['team_id'] == team_id)
        ]
        
        if len(team_stats) == 0:
            return {
                'xG_avg': 0.0,
                'xGA_avg': 0.0,
                'xG_diff_avg': 0.0,
            }
        
        # Calculate averages
        xG_avg = team_stats['xG'].mean()
        
        # For xGA, we need opponent stats
        # Get opponent team IDs for each fixture
        opponent_stats_list = []
        for _, fixture in team_fixtures.iterrows():
            opponent_id = (
                fixture['away_team_id'] if fixture['home_team_id'] == team_id
                else fixture['home_team_id']
            )
            
            opp_stats = statistics_df[
                (statistics_df['fixture_id'] == fixture['fixture_id']) &
                (statistics_df['team_id'] == opponent_id)
            ]
            
            if len(opp_stats) > 0:
                opponent_stats_list.append(opp_stats.iloc[0])
        
        if opponent_stats_list:
            opponent_stats_df = pd.DataFrame(opponent_stats_list)
            xGA_avg = opponent_stats_df['xG'].mean()
        else:
            xGA_avg = 0.0
        
        return {
            'xG_avg': float(xG_avg) if not pd.isna(xG_avg) else 0.0,
            'xGA_avg': float(xGA_avg) if not pd.isna(xGA_avg) else 0.0,
            'xG_diff_avg': float(xG_avg - xGA_avg) if not pd.isna(xG_avg) and not pd.isna(xGA_avg) else 0.0,
        }


def main():
    """Test derived xG calculator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load data
    statistics_df = pd.read_csv('data/csv/statistics.csv')
    fixtures_df = pd.read_csv('data/csv/fixtures.csv')
    
    # Calculate xG
    calculator = CSVDerivedXGCalculator()
    statistics_with_xg = calculator.calculate_xg(statistics_df)
    
    # Save updated statistics
    statistics_with_xg.to_csv('data/csv/statistics.csv', index=False)
    
    print("✅ Added xG to statistics.csv")
    print(f"   Average xG: {statistics_with_xg['xG'].mean():.2f}")
    print(f"   Max xG: {statistics_with_xg['xG'].max():.2f}")
    
    # Test rolling xG
    team_id = fixtures_df['home_team_id'].iloc[0]
    test_date = fixtures_df['starting_at'].iloc[100]
    
    rolling_xg = calculator.calculate_rolling_xg(
        statistics_with_xg, fixtures_df, team_id, test_date, n_matches=5
    )
    
    print(f"\nTeam {team_id} rolling xG (last 5 matches):")
    for key, value in rolling_xg.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
