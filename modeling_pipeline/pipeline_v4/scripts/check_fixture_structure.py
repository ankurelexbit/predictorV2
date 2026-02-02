#!/usr/bin/env python3
"""Check fixture structure in production vs training."""

import sys
import os
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline

print("=" * 80)
print("FIXTURE STRUCTURE COMPARISON")
print("=" * 80)

# 1. Check training data fixture structure
print("\n1. TRAINING DATA FIXTURE STRUCTURE:")
print("   " + "-" * 76)

train_df = pd.read_csv('data/training_data_with_draw_features.csv', nrows=1)
print(f"   Total columns: {len(train_df.columns)}")
print("\n   Stat-related columns:")

stat_keywords = ['shots', 'xg', 'attack', 'possession', 'tackle', 'corner']
for col in sorted(train_df.columns):
    if any(keyword in col.lower() for keyword in stat_keywords):
        print(f"      {col}")

# 2. Check production data fixture structure
print("\n2. PRODUCTION DATA FIXTURE STRUCTURE:")
print("   " + "-" * 76)

api_key = os.environ.get('SPORTMONKS_API_KEY')
if not api_key:
    print("   ❌ SPORTMONKS_API_KEY not set")
    sys.exit(1)

pipeline = StandaloneLivePipeline(api_key)

# Fetch team fixtures
print("\n   Fetching team fixtures via API...")
endpoint = f"teams/{2}/fixtures"  # Team ID 2
params = {
    'include': 'statistics;scores;participants',
    'page': 1
}

data = pipeline._api_call(endpoint, params)

if data and 'data' in data:
    fixtures = data['data'][:5]
    print(f"   Fetched {len(fixtures)} fixtures")

    if fixtures:
        print("\n   API Fixture Structure:")
        print(f"      Top-level keys: {list(fixtures[0].keys())}")

        # Check for statistics
        if 'statistics' in fixtures[0]:
            stats = fixtures[0]['statistics']
            print(f"\n      Statistics available: YES")
            print(f"      Statistics structure: {type(stats).__name__}")
            if isinstance(stats, list) and len(stats) > 0:
                print(f"      Number of stat entries: {len(stats)}")
                print(f"      Sample stat entry keys: {list(stats[0].keys())}")
                if 'data' in stats[0]:
                    print(f"      Stat data keys: {list(stats[0]['data'][0].keys()) if stats[0]['data'] else 'empty'}")
        else:
            print(f"\n      Statistics available: NO")
            print(f"      ⚠️  PROBLEM: Statistics not included in API response")

        # Check what data loader sees
        print("\n3. DATA LOADER FIXTURE STRUCTURE:")
        print("   " + "-" * 76)

        # Add fixtures to data loader
        fixtures_list = []
        for f in fixtures:
            participants = f.get('participants', [])
            home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
            away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

            if home_team and away_team:
                scores = f.get('scores', [])
                home_score = None
                away_score = None

                for score in scores:
                    if score.get('description') == 'CURRENT':
                        score_data = score.get('score', {})
                        if score_data.get('participant') == 'home':
                            home_score = score_data.get('goals', 0)
                        elif score_data.get('participant') == 'away':
                            away_score = score_data.get('goals', 0)

                fixture_row = {
                    'fixture_id': f['id'],
                    'home_team_id': home_team['id'],
                    'away_team_id': away_team['id'],
                    'home_score': home_score or 0,
                    'away_score': away_score or 0,
                    'starting_at': f.get('starting_at'),
                    'league_id': f.get('league_id'),
                    'season_id': f.get('season_id'),
                }

                # Add to list
                fixtures_list.append(fixture_row)

        # Convert to DataFrame and add to data loader
        fixtures_df = pd.DataFrame(fixtures_list)
        pipeline.data_loader.add_fixtures(fixtures_df)

        print(f"   Fixtures added to loader: {len(fixtures_df)}")
        print(f"   DataFrame columns: {list(fixtures_df.columns)}")

        # Try to get statistics for a fixture
        if len(fixtures_list) > 0:
            test_fixture_id = fixtures_list[0]['fixture_id']
            stats = pipeline.data_loader.get_statistics(test_fixture_id)

            print(f"\n   get_statistics({test_fixture_id}):")
            if stats:
                print(f"      Result: {type(stats).__name__} with {len(stats)} entries")
            else:
                print(f"      Result: None")
                print(f"      ⚠️  PROBLEM: Statistics not accessible via data loader")

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("\n🔴 ISSUE 1: ELO RATINGS = 1500")
print("   " + "-" * 76)
print("   WHY: EloCalculator.elo_history is empty")
print("   ")
print("   In training:")
print("      • JSONDataLoader loads ALL historical fixtures from CSV")
print("      • EloCalculator.calculate_elo_history() processes all fixtures")
print("      • Builds up Elo ratings over time (1000s of matches)")
print("      • get_elo_at_date() returns calculated rating")
print("   ")
print("   In production:")
print("      • InMemoryDataLoader starts empty")
print("      • Only upcoming fixtures added (no scores/results)")
print("      • calculate_elo_history() NEVER called")
print("      • get_elo_at_date() returns None → defaults to 1500")

print("\n🔴 ISSUE 2: DERIVED xG = 0")
print("   " + "-" * 76)
print("   WHY: Fixtures don't have statistics columns")
print("   ")
print("   In training:")
print("      • CSV has embedded statistics: home_shots_total, away_shots_on_target, etc.")
print("      • _extract_team_stats() reads these columns")
print("      • _derive_xg_from_stats() calculates xG from shots/accuracy")
print("   ")
print("   In production:")
print("      • API fixtures don't have stat columns in DataFrame")
print("      • _extract_team_stats() gets 0 for all stats")
print("      • _derive_xg_from_stats() calculates 0 xG")
print("      • Even if API returns statistics, they're not in DataFrame columns")

print("\n" + "=" * 80)
