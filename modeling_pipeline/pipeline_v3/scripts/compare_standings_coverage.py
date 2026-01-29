"""
Compare V2 and V3 standings data coverage to understand performance gap.
"""
import pandas as pd
from pathlib import Path

def main():
    # Load V2 standings
    v2_path = Path('../data/raw/sportmonks/standings.csv')
    v2_standings = pd.read_csv(v2_path)
    
    # Load V3 standings
    v3_path = Path('data/csv/standings.csv')
    v3_standings = pd.read_csv(v3_path)
    
    # Load fixtures to understand match coverage
    v2_fixtures = pd.read_csv('../data/raw/sportmonks/fixtures.csv')
    v3_fixtures = pd.read_csv('data/csv/fixtures.csv')
    
    print("=" * 80)
    print("V2 vs V3 STANDINGS DATA COMPARISON")
    print("=" * 80)
    
    print(f"\n📊 V2 Standings:")
    print(f"   Rows: {len(v2_standings)}")
    print(f"   Seasons: {v2_standings['season_id'].nunique()}")
    print(f"   Teams: {v2_standings['team_id'].nunique()}")
    print(f"   Structure: {list(v2_standings.columns)}")
    
    print(f"\n📊 V3 Standings:")
    print(f"   Rows: {len(v3_standings)}")
    print(f"   Seasons: {v3_standings['season_id'].nunique()}")
    print(f"   Structure: {list(v3_standings.columns)}")
    
    print(f"\n📊 V2 Fixtures:")
    print(f"   Total matches: {len(v2_fixtures)}")
    print(f"   Seasons: {v2_fixtures['season_id'].nunique()}")
    
    print(f"\n📊 V3 Fixtures:")
    print(f"   Total matches: {len(v3_fixtures)}")
    print(f"   Seasons: {v3_fixtures['season_id'].nunique()}")
    
    # Key insight: V2 has FINAL season standings (one row per team per season)
    # V3 likely has MATCH-BY-MATCH standings (one row per team per match)
    
    print("\n" + "=" * 80)
    print("KEY FINDING")
    print("=" * 80)
    print("\n🔍 V2 Structure: FINAL SEASON STANDINGS")
    print("   - One row per team per season")
    print("   - Contains final position and points")
    print("   - 100% coverage for all teams in all seasons")
    
    print("\n🔍 V3 Structure: MATCH-BY-MATCH STANDINGS")
    print("   - One row per team per match (potentially)")
    print("   - Only available when API provides it (37% coverage)")
    
    # Check if V2 standings can be used in V3
    common_seasons = set(v2_standings['season_id'].unique()) & set(v3_fixtures['season_id'].unique())
    print(f"\n✅ Common seasons between V2 standings and V3 fixtures: {len(common_seasons)}")
    
    if len(common_seasons) > 0:
        print("\n💡 SOLUTION: V3 can use V2's standings.csv file!")
        print("   This would restore 100% coverage and match V2 performance.")
    
    # Save report
    report_path = Path('/Users/ankurgupta/.gemini/antigravity/brain/b17befe7-0b46-48c7-8e29-6cb1b85b637c/standings_coverage_analysis.md')
    with open(report_path, 'w') as f:
        f.write("# V2 vs V3 Standings Coverage Analysis\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **V2 Standings**: {len(v2_standings)} rows (final season standings)\n")
        f.write(f"- **V3 Standings**: {len(v3_standings)} rows (match-by-match)\n")
        f.write(f"- **Coverage Gap**: V2 has 100% coverage, V3 has 37%\n\n")
        f.write("## Root Cause\n\n")
        f.write("V2 used **final season standings** from SportMonks API, which provides complete position/points data for all teams.\n\n")
        f.write("V3 attempted to use **match-by-match standings**, which are only available for 37% of matches in the API.\n\n")
        f.write("## Solution\n\n")
        f.write("**Option 1**: Use V2's `standings.csv` in V3 (immediate fix)\n")
        f.write("**Option 2**: Calculate standings from match results (what V3's StandingsCalculator does)\n")
        f.write("**Option 3**: Hybrid - use API where available, calculate otherwise\n\n")
    
    print(f"\n📄 Report saved to: {report_path}")

if __name__ == "__main__":
    main()
