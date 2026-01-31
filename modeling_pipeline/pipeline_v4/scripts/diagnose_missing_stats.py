"""
Diagnose why statistics are missing in feature generation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.json_loader import JSONDataLoader
import pandas as pd

print("=" * 80)
print("DIAGNOSING MISSING STATISTICS IN FEATURE GENERATION")
print("=" * 80)
print()

# Test 1: Check CSV file
print("TEST 1: CSV File Statistics")
print("-" * 80)

csv_file = Path('data/processed/fixtures_with_stats.csv')
if csv_file.exists():
    df_csv = pd.read_csv(csv_file)
    print(f"✅ CSV file exists: {csv_file}")
    print(f"   Rows: {len(df_csv):,}")
    print(f"   Columns: {len(df_csv.columns)}")

    # Check for statistics columns
    stat_cols = [c for c in df_csv.columns if c.startswith(('home_', 'away_'))
                 and c not in ['home_team_id', 'home_team_name', 'home_score',
                              'away_team_id', 'away_team_name', 'away_score']]

    print(f"   Statistics columns: {len(stat_cols)}")

    if 'home_shots_total' in df_csv.columns:
        coverage = df_csv['home_shots_total'].notna().sum() / len(df_csv) * 100
        print(f"   home_shots_total coverage: {coverage:.1f}%")

    if 'home_ball_possession' in df_csv.columns:
        coverage = df_csv['home_ball_possession'].notna().sum() / len(df_csv) * 100
        print(f"   home_ball_possession coverage: {coverage:.1f}%")
else:
    print(f"🔴 CSV file NOT found: {csv_file}")
    print("   Run: python3 scripts/convert_json_to_csv.py")

print()

# Test 2: Check what JSONDataLoader loads
print("TEST 2: JSONDataLoader Loaded Data")
print("-" * 80)

try:
    loader = JSONDataLoader('data/historical')
    df_loaded = loader.load_all_fixtures(use_csv=True)

    print(f"✅ Loaded {len(df_loaded):,} fixtures")
    print(f"   Columns: {len(df_loaded.columns)}")

    # Check if statistics columns are present
    if 'home_shots_total' in df_loaded.columns:
        print(f"   ✅ home_shots_total column FOUND")
        coverage = df_loaded['home_shots_total'].notna().sum() / len(df_loaded) * 100
        print(f"      Coverage: {coverage:.1f}%")
    else:
        print(f"   🔴 home_shots_total column NOT FOUND")

    if 'home_ball_possession' in df_loaded.columns:
        print(f"   ✅ home_ball_possession column FOUND")
        coverage = df_loaded['home_ball_possession'].notna().sum() / len(df_loaded) * 100
        print(f"      Coverage: {coverage:.1f}%")
    else:
        print(f"   🔴 home_ball_possession column NOT FOUND")

    # Show all columns
    print()
    print("   All columns loaded:")
    for i, col in enumerate(df_loaded.columns, 1):
        print(f"   {i:3d}. {col}")

except Exception as e:
    print(f"🔴 Error loading fixtures: {e}")

print()

# Test 3: Check cache
print("TEST 3: Cache Files")
print("-" * 80)

cache_dir = Path('data/cache')
if cache_dir.exists():
    cache_files = list(cache_dir.glob('*.pkl'))
    if cache_files:
        print(f"⚠️  Found {len(cache_files)} cache files:")
        for f in cache_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.1f} MB)")
        print()
        print("   These might contain old data WITHOUT statistics.")
        print("   Solution: rm -rf data/cache/")
    else:
        print("✅ Cache directory exists but is empty")
else:
    print("✅ No cache directory found")

print()

# Diagnosis
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()

if csv_file.exists():
    df_csv = pd.read_csv(csv_file)
    has_stats_in_csv = 'home_shots_total' in df_csv.columns

    loader = JSONDataLoader('data/historical')
    df_loaded = loader.load_all_fixtures(use_csv=True)
    has_stats_in_loaded = 'home_shots_total' in df_loaded.columns

    if has_stats_in_csv and has_stats_in_loaded:
        print("✅ GOOD: Statistics are in both CSV and loaded DataFrame")
        print()
        print("The missing data in training features might be due to:")
        print("  1. Not enough history for rolling averages (early matches)")
        print("  2. Feature calculation logic issues")
        print("  3. Some statistics missing for specific matches")
        print()
        print("Expected behavior: ~10-20% missing due to insufficient history")
        print("Your situation: 60-70% missing → suggests a bigger issue")

    elif has_stats_in_csv and not has_stats_in_loaded:
        print("🔴 PROBLEM FOUND: CSV has stats but loaded DataFrame doesn't!")
        print()
        print("Possible causes:")
        print("  1. Old pickle cache being used instead of CSV")
        print("  2. CSV not being loaded properly")
        print()
        print("Solutions:")
        print("  1. Delete cache: rm -rf data/cache/")
        print("  2. Force regenerate: rm data/cache/*.pkl")
        print("  3. Verify CSV loading in json_loader.py")

    elif not has_stats_in_csv:
        print("🔴 PROBLEM: CSV doesn't have statistics columns")
        print()
        print("Solutions:")
        print("  1. Regenerate CSV: python3 scripts/convert_json_to_csv.py")
        print("  2. Check convert_json_to_csv.py for bugs")
else:
    print("🔴 CRITICAL: CSV file doesn't exist")
    print()
    print("Solution:")
    print("  python3 scripts/convert_json_to_csv.py")

print()
print("=" * 80)
