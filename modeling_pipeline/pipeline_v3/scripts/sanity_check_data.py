#!/usr/bin/env python3
"""
Sanity Check for Historical Data Download
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def sanity_check():
    """Perform comprehensive sanity check on downloaded data."""
    
    data_dir = Path('data/historical')
    
    print("=" * 80)
    print("HISTORICAL DATA SANITY CHECK")
    print("=" * 80)
    print()
    
    # 1. Check directory structure
    print("📁 Directory Structure:")
    print("-" * 80)
    for subdir in ['fixtures', 'statistics', 'lineups', 'sidelined']:
        path = data_dir / subdir
        if path.exists():
            file_count = len(list(path.glob('*.json')))
            size = sum(f.stat().st_size for f in path.glob('*.json'))
            size_mb = size / (1024 * 1024)
            print(f"  ✅ {subdir:15s}: {file_count:6d} files ({size_mb:8.1f} MB)")
        else:
            print(f"  ❌ {subdir:15s}: MISSING")
    print()
    
    # 2. Check fixture files
    print("🏟️  Fixture Data Analysis:")
    print("-" * 80)
    
    fixture_files = list((data_dir / 'fixtures').glob('all_fixtures_*.json'))
    print(f"  Total fixture files: {len(fixture_files)}")
    
    total_fixtures = 0
    fixtures_by_league = defaultdict(int)
    fixtures_by_year = defaultdict(int)
    
    sample_fixture = None
    
    for fixture_file in sorted(fixture_files)[:5]:  # Check first 5 files
        try:
            with open(fixture_file) as f:
                fixtures = json.load(f)
                total_fixtures += len(fixtures)
                
                for fixture in fixtures:
                    league_id = fixture.get('league_id')
                    fixtures_by_league[league_id] += 1
                    
                    # Extract year from starting_at
                    starting_at = fixture.get('starting_at', '')
                    if starting_at:
                        year = starting_at[:4]
                        fixtures_by_year[year] += 1
                    
                    # Save sample fixture
                    if sample_fixture is None:
                        sample_fixture = fixture
        except Exception as e:
            print(f"  ⚠️  Error reading {fixture_file.name}: {e}")
    
    print(f"  Fixtures analyzed (first 5 files): {total_fixtures}")
    print()
    
    # 3. Check data completeness
    print("📊 Data Completeness Check:")
    print("-" * 80)
    
    if sample_fixture:
        required_fields = ['id', 'league_id', 'starting_at', 'participants', 'scores']
        optional_fields = ['statistics', 'lineups', 'events', 'formations', 'sidelined', 'odds']
        
        print("  Required fields:")
        for field in required_fields:
            has_field = field in sample_fixture
            status = "✅" if has_field else "❌"
            print(f"    {status} {field}")
        
        print("\n  Optional fields (detailed data):")
        for field in optional_fields:
            has_field = field in sample_fixture
            value = sample_fixture.get(field, [])
            count = len(value) if isinstance(value, list) else ('present' if value else 'empty')
            status = "✅" if has_field else "❌"
            print(f"    {status} {field:15s}: {count}")
    print()
    
    # 4. Check statistics files
    print("📈 Statistics Files:")
    print("-" * 80)
    stats_files = list((data_dir / 'statistics').glob('fixture_*.json'))
    print(f"  Total statistics files: {len(stats_files)}")
    
    if stats_files:
        # Check a sample
        sample_stats_file = stats_files[0]
        with open(sample_stats_file) as f:
            stats = json.load(f)
            print(f"  Sample file: {sample_stats_file.name}")
            print(f"  Teams in sample: {len(stats)}")
            if stats:
                print(f"  Sample stats keys: {list(stats[0].keys())[:10]}")
    print()
    
    # 5. Check lineups files
    print("👥 Lineups Files:")
    print("-" * 80)
    lineups_files = list((data_dir / 'lineups').glob('fixture_*.json'))
    print(f"  Total lineups files: {len(lineups_files)}")
    
    if lineups_files:
        # Check a sample
        sample_lineups_file = lineups_files[0]
        with open(sample_lineups_file) as f:
            lineups = json.load(f)
            print(f"  Sample file: {sample_lineups_file.name}")
            print(f"  Teams in sample: {len(lineups)}")
            if lineups and len(lineups) > 0:
                print(f"  Sample lineup keys: {list(lineups[0].keys())[:10]}")
    print()
    
    # 6. Check sidelined files
    print("🏥 Sidelined Players:")
    print("-" * 80)
    sidelined_files = list((data_dir / 'sidelined').glob('team_*.json'))
    print(f"  Total sidelined files: {len(sidelined_files)}")
    
    if sidelined_files:
        total_sidelined = 0
        for sf in sidelined_files[:10]:  # Check first 10
            with open(sf) as f:
                sidelined = json.load(f)
                total_sidelined += len(sidelined)
        print(f"  Total sidelined players (first 10 teams): {total_sidelined}")
    print()
    
    # 7. Coverage by league
    print("🏆 Coverage by League:")
    print("-" * 80)
    league_names = {
        8: "Premier League",
        564: "La Liga",
        82: "Bundesliga",
        384: "Serie A",
        301: "Ligue 1"
    }
    
    for league_id, count in sorted(fixtures_by_league.items()):
        league_name = league_names.get(league_id, f"League {league_id}")
        print(f"  {league_name:20s}: {count:4d} fixtures")
    print()
    
    # 8. Coverage by year
    print("📅 Coverage by Year:")
    print("-" * 80)
    for year, count in sorted(fixtures_by_year.items()):
        print(f"  {year}: {count:4d} fixtures")
    print()
    
    # 9. Data quality checks
    print("✅ Data Quality Checks:")
    print("-" * 80)
    
    # Check if statistics count matches expectations
    stats_count = len(stats_files)
    lineups_count = len(lineups_files)
    
    # Not all fixtures have stats/lineups (future fixtures, cancelled, etc.)
    stats_coverage = (stats_count / total_fixtures * 100) if total_fixtures > 0 else 0
    lineups_coverage = (lineups_count / total_fixtures * 100) if total_fixtures > 0 else 0
    
    print(f"  Statistics coverage: {stats_coverage:.1f}% ({stats_count}/{total_fixtures})")
    print(f"  Lineups coverage: {lineups_coverage:.1f}% ({lineups_count}/{total_fixtures})")
    
    if stats_coverage > 90:
        print("  ✅ Excellent statistics coverage!")
    elif stats_coverage > 70:
        print("  ⚠️  Good statistics coverage (some fixtures may be future/cancelled)")
    else:
        print("  ❌ Low statistics coverage - investigate!")
    
    if lineups_coverage > 90:
        print("  ✅ Excellent lineups coverage!")
    elif lineups_coverage > 70:
        print("  ⚠️  Good lineups coverage (some fixtures may be future/cancelled)")
    else:
        print("  ❌ Low lineups coverage - investigate!")
    
    print()
    
    # 10. Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Total fixtures: {total_fixtures:,} (from first 5 files)")
    print(f"✅ Statistics files: {stats_count:,}")
    print(f"✅ Lineups files: {lineups_count:,}")
    print(f"✅ Sidelined teams: {len(sidelined_files)}")
    print(f"✅ Leagues covered: {len(fixtures_by_league)}")
    print(f"✅ Years covered: {len(fixtures_by_year)}")
    print()
    print("🎉 Data download appears successful!")
    print("=" * 80)

if __name__ == "__main__":
    sanity_check()
