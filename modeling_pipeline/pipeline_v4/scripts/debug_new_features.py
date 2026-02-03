"""
Debug script for new features implementation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.data.json_loader import JSONDataLoader
from src.features.standings_calculator import StandingsCalculator
from src.features.pillar3_hidden_edges import Pillar3HiddenEdgesEngine
from src.features.elo_calculator import EloCalculator

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def debug_xg_vs_groups():
    """Debug xG vs top/bottom calculation."""

    print("=" * 80)
    print("Debugging xG vs Top/Bottom Implementation")
    print("=" * 80)

    # Load data
    data_loader = JSONDataLoader()
    fixtures_df = data_loader.load_all_fixtures(use_csv=True)

    # Initialize calculators
    standings_calc = StandingsCalculator()
    elo_calc = EloCalculator(k_factor=32, home_advantage=35, initial_elo=1500)
    elo_calc.calculate_elo_history(fixtures_df)

    # Initialize pillar3
    pillar3 = Pillar3HiddenEdgesEngine(data_loader, standings_calc, elo_calc)

    # Get a sample fixture
    sample = fixtures_df[
        (fixtures_df['league_id'] == 8) &  # Premier League
        (pd.to_datetime(fixtures_df['starting_at']).dt.year == 2023)
    ].iloc[0]

    fixture_id = sample['id']
    home_team_id = sample['home_team_id']
    away_team_id = sample['away_team_id']
    league_id = sample['league_id']
    as_of_date = pd.to_datetime(sample['starting_at'])

    print(f"\nTest Fixture:")
    print(f"  ID: {fixture_id}")
    print(f"  Home: {home_team_id}, Away: {away_team_id}")
    print(f"  League: {league_id}")
    print(f"  Date: {as_of_date}")

    # Test standings calculation
    print("\n1. Testing standings calculation...")
    try:
        standings = standings_calc.calculate_standings_at_date(
            fixtures_df=fixtures_df,
            season_id=None,
            league_id=league_id,
            as_of_date=as_of_date
        )
        print(f"   ✓ Standings calculated: {len(standings)} teams")
        print(f"   Top 3 teams: {standings.iloc[:3]['team_id'].tolist()}")
        print(f"   Bottom 3 teams: {standings.iloc[-3:]['team_id'].tolist()}")
    except Exception as e:
        print(f"   ✗ Standings calculation failed: {e}")
        import traceback
        traceback.print_exc()
        standings = None

    if standings is not None and len(standings) >= 4:
        # Test top/bottom split
        print("\n2. Testing top/bottom split...")
        mid_point = len(standings) // 2
        top_half = set(standings.iloc[:mid_point]['team_id'])
        bottom_half = set(standings.iloc[mid_point:]['team_id'])

        print(f"   Top half ({len(top_half)} teams): {list(top_half)[:5]}...")
        print(f"   Bottom half ({len(bottom_half)} teams): {list(bottom_half)[:5]}...")

        # Test fixture retrieval for home team
        print(f"\n3. Testing fixture retrieval for home team ({home_team_id})...")
        try:
            recent_matches = data_loader.get_team_fixtures(
                team_id=home_team_id,
                before_date=as_of_date,
                limit=20
            )
            print(f"   ✓ Found {len(recent_matches)} recent matches")

            # Count matches vs top/bottom
            vs_top = 0
            vs_bottom = 0
            for _, match in recent_matches.iterrows():
                opp_id = match['away_team_id'] if match['home_team_id'] == home_team_id else match['home_team_id']
                if opp_id in top_half:
                    vs_top += 1
                elif opp_id in bottom_half:
                    vs_bottom += 1

            print(f"   Matches vs top half: {vs_top}")
            print(f"   Matches vs bottom half: {vs_bottom}")

            if vs_top > 0:
                print("\n4. Testing xG calculation vs top half...")
                # Try calculating for one match
                for _, match in recent_matches.iterrows():
                    opp_id = match['away_team_id'] if match['home_team_id'] == home_team_id else match['home_team_id']
                    if opp_id in top_half:
                        is_home = match['home_team_id'] == home_team_id
                        prefix = 'home_' if is_home else 'away_'

                        shots_on_target = match.get(f'{prefix}shots_on_target', 0) or 0
                        shots_inside_box = match.get(f'{prefix}shots_inside_box', 0) or 0

                        xg = (shots_on_target * 0.35) + (shots_inside_box * 0.15)

                        print(f"   Sample match: shots_on_target={shots_on_target}, shots_inside_box={shots_inside_box}, xG={xg:.2f}")
                        break

        except Exception as e:
            print(f"   ✗ Error retrieving fixtures: {e}")
            import traceback
            traceback.print_exc()

    # Test actual method call
    print("\n5. Testing actual _calculate_xg_vs_opposition_groups method...")
    try:
        result = pillar3._calculate_xg_vs_opposition_groups(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id,
            as_of_date=as_of_date,
            fixtures_df=fixtures_df
        )
        print(f"   Result:")
        for key, value in result.items():
            print(f"     {key}: {value}")
    except Exception as e:
        print(f"   ✗ Method failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)


if __name__ == '__main__':
    debug_xg_vs_groups()
