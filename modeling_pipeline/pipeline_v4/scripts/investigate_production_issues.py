#!/usr/bin/env python3
"""
Investigate Production Feature Generation Issues
================================================

Systematically checks each feature generation component to identify failures.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline

print("=" * 80)
print("PRODUCTION FEATURE GENERATION INVESTIGATION")
print("=" * 80)

api_key = os.environ.get('SPORTMONKS_API_KEY')
if not api_key:
    print("❌ SPORTMONKS_API_KEY not set")
    sys.exit(1)

pipeline = StandaloneLivePipeline(api_key)

# Get a sample fixture
print("\n1. Fetching sample fixture...")
endpoint = "fixtures/between/2026-02-01/2026-02-02"
params = {
    'include': 'participants;league;state',
    'filters': 'fixtureStates:1,2,3,5',
    'page': 1
}

data = pipeline._api_call(endpoint, params)
fixture = data['data'][0]

participants = fixture.get('participants', [])
home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

fixture_dict = {
    'fixture_id': fixture['id'],
    'starting_at': fixture.get('starting_at'),
    'league_id': fixture.get('league_id'),
    'league_name': fixture.get('league', {}).get('name', 'Unknown'),
    'season_id': fixture.get('season_id'),
    'home_team_id': home_team['id'],
    'home_team_name': home_team['name'],
    'away_team_id': away_team['id'],
    'away_team_name': away_team['name'],
    'state_id': fixture.get('state_id')
}

print(f"   ✅ {home_team['name']} vs {away_team['name']}")
print(f"   League: {fixture_dict['league_name']}")
print(f"   Date: {fixture_dict['starting_at']}")

# ============================================================================
# ISSUE 1: ELO RATINGS
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE 1: ELO RATINGS")
print("=" * 80)

print("\nChecking Elo Calculator...")
elo_calc = pipeline.elo_calculator

print(f"   Elo Calculator type: {type(elo_calc).__name__}")
print(f"   Number of teams in Elo system: {len(elo_calc.ratings)}")

home_elo = elo_calc.get_rating(fixture_dict['home_team_id'])
away_elo = elo_calc.get_rating(fixture_dict['away_team_id'])

print(f"\n   Home team ({home_team['name']}):")
print(f"      Team ID: {fixture_dict['home_team_id']}")
print(f"      Elo rating: {home_elo}")

print(f"\n   Away team ({away_team['name']}):")
print(f"      Team ID: {fixture_dict['away_team_id']}")
print(f"      Elo rating: {away_elo}")

# Check if Elo is just defaults
if home_elo == 1500 and away_elo == 1500:
    print("\n   ⚠️  PROBLEM: Both teams have default Elo (1500)")
    print("\n   Root Cause Analysis:")
    print("      - EloCalculator starts all teams at 1500")
    print("      - In production, no historical matches are loaded to update ratings")
    print("      - In training, Elo was calculated from full historical dataset")

    print("\n   Why this happens:")
    print("      1. Training pipeline loads ALL fixtures from CSV")
    print("      2. Processes them chronologically to build Elo ratings")
    print("      3. Production pipeline only fetches upcoming fixtures")
    print("      4. No historical context = no Elo updates = all teams 1500")

    print("\n   📋 FIX OPTIONS:")
    print("      A. Pre-calculate Elo ratings offline, save to file, load in production")
    print("         - Pros: Fast, accurate, matches training")
    print("         - Cons: Needs periodic updates")
    print("         - Implementation: Add elo_ratings.json to data/processed/")

    print("\n      B. Fetch recent match history on startup, calculate Elo")
    print("         - Pros: Always current")
    print("         - Cons: Slow startup, API calls")
    print("         - Implementation: Fetch last 180 days of fixtures on init")

    print("\n      C. Use API-based Elo/rankings if available")
    print("         - Pros: Authoritative source")
    print("         - Cons: May not exist or match our calculations")
else:
    print("\n   ✅ Elo ratings look reasonable")

# ============================================================================
# ISSUE 2: DERIVED xG
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE 2: DERIVED xG")
print("=" * 80)

print("\nChecking xG calculation for home team...")

# Fetch team's recent fixtures
home_team_id = fixture_dict['home_team_id']
away_team_id = fixture_dict['away_team_id']

try:
    # Get recent fixtures for home team
    home_fixtures = pipeline.data_loader.get_team_fixtures(
        home_team_id,
        before_date=pd.to_datetime(fixture_dict['starting_at'])
    )

    print(f"   Home team recent fixtures: {len(home_fixtures)}")

    if len(home_fixtures) > 0:
        # Check if fixtures have statistics
        sample_fixture = home_fixtures[0]
        print(f"\n   Sample fixture structure:")
        print(f"      Keys: {list(sample_fixture.keys())[:10]}...")

        # Check for statistics
        stats = pipeline.data_loader.get_statistics(sample_fixture.get('fixture_id'))

        if stats:
            print(f"      Statistics available: YES ({len(stats)} stat entries)")
            # Show sample stats
            if len(stats) > 0:
                print(f"      Sample stat: {stats[0] if isinstance(stats, list) else stats}")
        else:
            print(f"      Statistics available: NO")
            print("\n   ⚠️  PROBLEM: No statistics available in API data")

    else:
        print("   ⚠️  PROBLEM: No recent fixtures found")

except Exception as e:
    print(f"   ❌ Error checking fixtures: {e}")

print("\n   Root Cause Analysis:")
print("      - Derived xG requires match statistics (shots, shots on target, etc.)")
print("      - Production pipeline fetches fixtures via API")
print("      - API may not include statistics, or statistics not being parsed")
print("      - Without stats, xG calculation defaults to 0")

print("\n   Why this happens:")
print("      1. Training uses CSV with embedded statistics from backfill")
print("      2. Production uses live API which may not include stats by default")
print("      3. Even if available, statistics might need special API parameters")
print("      4. Statistics endpoint might be separate from fixtures endpoint")

print("\n   📋 FIX OPTIONS:")
print("      A. Include statistics in API calls")
print("         - Pros: Real-time data")
print("         - Cons: More API calls, may not be available")
print("         - Implementation: Add 'statistics' to include parameter")

print("\n      B. Pre-calculate xG metrics offline, save to file")
print("         - Pros: Fast, matches training data")
print("         - Cons: Needs periodic updates")
print("         - Implementation: Add team_xg_stats.json to data/processed/")

print("\n      C. Use simpler proxy metrics (goals, shots)")
print("         - Pros: More widely available")
print("         - Cons: Less accurate than derived xG")
print("         - Implementation: Modify Pillar2 to use basic stats")

print("\n      D. Fetch statistics separately for each team")
print("         - Pros: Complete data")
print("         - Cons: SLOW (many API calls)")
print("         - Implementation: Add statistics fetching to feature generation")

# ============================================================================
# ISSUE 3: DATA LOADER
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE 3: DATA LOADER")
print("=" * 80)

print("\nChecking data loader state...")
loader = pipeline.data_loader

print(f"   Data loader type: {type(loader).__name__}")
print(f"   Fixtures in cache: {len(loader.fixtures_cache)}")

if hasattr(loader, '_fixtures_df') and loader._fixtures_df is not None:
    print(f"   Fixtures DataFrame: {len(loader._fixtures_df)} rows")
else:
    print(f"   Fixtures DataFrame: None")

print("\n   Root Cause Analysis:")
print("      - InMemoryDataLoader starts empty")
print("      - Only stores fixtures it fetches for predictions")
print("      - Doesn't have historical context")
print("      - Training uses JSONDataLoader with full historical CSV")

print("\n   📋 FIX OPTIONS:")
print("      A. Switch to JSONDataLoader with pre-loaded historical data")
print("         - Pros: Same as training, complete history")
print("         - Cons: Requires maintaining local data files")
print("         - Implementation: Use JSONDataLoader instead of InMemoryDataLoader")

print("\n      B. Hybrid approach: Load recent N days on startup")
print("         - Pros: Balance between speed and context")
print("         - Cons: Still needs API calls")
print("         - Implementation: Fetch last 180 days on init, cache locally")

# ============================================================================
# ISSUE 4: STANDINGS
# ============================================================================
print("\n" + "=" * 80)
print("ISSUE 4: LEAGUE STANDINGS")
print("=" * 80)

print("\nChecking standings calculator...")
standings_calc = pipeline.standings_calculator

try:
    league_id = fixture_dict['league_id']
    season_id = fixture_dict['season_id']

    standings = standings_calc.get_standings(league_id, season_id)

    if standings:
        print(f"   ✅ Standings available for league {league_id}")
        print(f"      Teams in standings: {len(standings)}")

        # Check if our teams are there
        home_standing = standings.get(fixture_dict['home_team_id'])
        away_standing = standings.get(fixture_dict['away_team_id'])

        if home_standing:
            print(f"\n   Home team standing:")
            print(f"      Position: {home_standing.get('position')}")
            print(f"      Points: {home_standing.get('points')}")
        else:
            print(f"\n   ⚠️  Home team not in standings")

        if away_standing:
            print(f"\n   Away team standing:")
            print(f"      Position: {away_standing.get('position')}")
            print(f"      Points: {away_standing.get('points')}")
        else:
            print(f"\n   ⚠️  Away team not in standings")
    else:
        print(f"   ⚠️  No standings available")

except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n   Note: Standings are fetched via API, should work in production")
print("   Points difference may be due to different time periods")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF ISSUES")
print("=" * 80)

print("\n🔴 CRITICAL ISSUES (breaking predictions):")
print("   1. Elo ratings all 1500 (no historical context)")
print("   2. Derived xG all 0 (no statistics available)")

print("\n🟡 MODERATE ISSUES (affecting accuracy):")
print("   3. Limited historical context in data loader")

print("\n🟢 MINOR ISSUES:")
print("   4. Points/positions slightly different (expected variance)")

print("\n" + "=" * 80)
print("RECOMMENDED FIX PRIORITY")
print("=" * 80)

print("\n1️⃣  FIX ELO (EASIEST & HIGHEST IMPACT)")
print("   → Option A: Pre-calculate and load from file")
print("   → Estimated effort: 30 minutes")
print("   → Impact: Restores team strength differentiation")

print("\n2️⃣  FIX DERIVED xG (MEDIUM DIFFICULTY)")
print("   → Option B: Pre-calculate and load from file")
print("   → Estimated effort: 1 hour")
print("   → Impact: Restores attack/defense quality metrics")

print("\n3️⃣  ADD HISTORICAL CONTEXT (BIGGER CHANGE)")
print("   → Option B: Load recent fixtures on startup")
print("   → Estimated effort: 2-3 hours")
print("   → Impact: Enables all time-based features to work properly")

print("\n" + "=" * 80)
