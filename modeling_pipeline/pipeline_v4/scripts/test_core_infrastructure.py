"""
Test script for V4 core infrastructure.

Tests JSON data loader, standings calculator, and Elo calculator.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.json_loader import JSONDataLoader
from src.features.standings_calculator import StandingsCalculator
from src.features.elo_calculator import EloCalculator
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_json_loader():
    """Test JSON data loader."""
    logger.info("=" * 80)
    logger.info("TEST 1: JSON Data Loader")
    logger.info("=" * 80)
    
    loader = JSONDataLoader(data_dir='data/historical')
    
    # Load all fixtures
    fixtures_df = loader.load_all_fixtures()
    logger.info(f"✅ Loaded {len(fixtures_df)} fixtures")
    logger.info(f"   Columns: {list(fixtures_df.columns[:10])}...")
    logger.info(f"   Date range: {fixtures_df['starting_at'].min()} to {fixtures_df['starting_at'].max()}")
    
    # Test get_fixtures_before
    test_date = datetime(2020, 1, 1)
    before_2020 = loader.get_fixtures_before(test_date)
    logger.info(f"✅ Found {len(before_2020)} fixtures before {test_date}")
    
    # Test get_team_fixtures
    team_id = 8  # Liverpool
    team_fixtures = loader.get_team_fixtures(team_id, test_date, limit=10)
    logger.info(f"✅ Found {len(team_fixtures)} fixtures for team {team_id}")
    
    return loader, fixtures_df


def test_standings_calculator(fixtures_df):
    """Test standings calculator."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Standings Calculator")
    logger.info("=" * 80)
    
    calc = StandingsCalculator()
    
    # Calculate standings at a specific date
    test_date = datetime(2020, 1, 1)
    season_id = 10  # 2015/16 season
    league_id = 8   # Premier League
    
    standings = calc.calculate_standings_at_date(
        fixtures_df, season_id, league_id, test_date
    )
    
    logger.info(f"✅ Calculated standings for season {season_id}, league {league_id}")
    logger.info(f"   Total teams: {len(standings)}")
    logger.info(f"\n   Top 5:")
    for _, row in standings.head(5).iterrows():
        logger.info(f"   {row['position']}. Team {row['team_id']}: {row['points']} pts, {row['goal_difference']} GD")
    
    # Test get_standing_features
    home_team_id = 8  # Liverpool
    away_team_id = 1  # West Ham
    
    features = calc.get_standing_features(
        home_team_id, away_team_id, fixtures_df, season_id, league_id, test_date
    )
    
    logger.info(f"\n✅ Standing features for match:")
    logger.info(f"   Home position: {features['home_league_position']}")
    logger.info(f"   Away position: {features['away_league_position']}")
    logger.info(f"   Home points: {features['home_points']}")
    logger.info(f"   Away points: {features['away_points']}")
    
    return calc


def test_elo_calculator(fixtures_df):
    """Test Elo calculator."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Elo Calculator")
    logger.info("=" * 80)
    
    calc = EloCalculator()
    
    # Calculate Elo history
    current_elos = calc.calculate_elo_history(fixtures_df)
    logger.info(f"✅ Calculated Elo for {len(current_elos)} teams")
    
    # Show top 5 teams by Elo
    top_teams = sorted(current_elos.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"\n   Top 5 teams by Elo:")
    for team_id, elo in top_teams:
        logger.info(f"   Team {team_id}: {elo:.0f}")
    
    # Test get_elo_at_date
    test_date = datetime(2020, 1, 1)
    team_id = 8  # Liverpool
    elo_at_date = calc.get_elo_at_date(team_id, test_date)
    logger.info(f"\n✅ Team {team_id} Elo at {test_date}: {elo_at_date:.0f}")
    
    # Test get_elo_change
    elo_change = calc.get_elo_change(team_id, test_date, num_matches=5)
    logger.info(f"✅ Team {team_id} Elo change (last 5): {elo_change:.0f}")
    
    return calc


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("V4 CORE INFRASTRUCTURE TESTS")
    logger.info("=" * 80 + "\n")
    
    try:
        # Test 1: JSON Loader
        loader, fixtures_df = test_json_loader()
        
        # Test 2: Standings Calculator
        standings_calc = test_standings_calculator(fixtures_df)
        
        # Test 3: Elo Calculator
        elo_calc = test_elo_calculator(fixtures_df)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("\nCore infrastructure is working correctly:")
        logger.info("  ✅ JSON Data Loader")
        logger.info("  ✅ Point-in-Time Standings Calculator")
        logger.info("  ✅ Elo Calculator")
        logger.info("\nReady to build Pillar 1 (Fundamentals)!")
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
