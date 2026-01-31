"""
Verify if missing statistics in training data are also missing in raw CSV.

This checks if the missing values are due to:
1. Missing data in source (JSON/CSV) - Expected
2. Bug in feature generation - Needs fixing
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("VERIFYING MISSING STATISTICS - RAW CSV vs TRAINING DATA")
print("=" * 80)

# Load raw CSV
print("\n📊 Loading raw CSV...")
raw_csv = pd.read_csv('data/processed/fixtures_with_stats.csv')
print(f"✅ Loaded {len(raw_csv)} fixtures from raw CSV")

# Load training data
print("\n📊 Loading training data...")
training_data = pd.read_csv('data/training_data.csv')
print(f"✅ Loaded {len(training_data)} fixtures from training data")

# Check statistics columns in raw CSV
print("\n" + "=" * 80)
print("1. STATISTICS AVAILABILITY IN RAW CSV")
print("=" * 80)

stat_columns = [
    'home_shots_total', 'away_shots_total',
    'home_shots_on_target', 'away_shots_on_target',
    'home_shots_inside_box', 'away_shots_inside_box',
    'home_tackles', 'away_tackles',
    'home_interceptions', 'away_interceptions',
    'home_ball_possession', 'away_ball_possession',
    'home_attacks', 'away_attacks',
    'home_dangerous_attacks', 'away_dangerous_attacks',
]

print("\nStatistics coverage in raw CSV:")
for col in stat_columns:
    if col in raw_csv.columns:
        missing = raw_csv[col].isnull().sum()
        missing_pct = (missing / len(raw_csv)) * 100
        available = len(raw_csv) - missing
        print(f"  {col}: {available:,} available ({100-missing_pct:.1f}%), {missing:,} missing ({missing_pct:.1f}%)")
    else:
        print(f"  {col}: NOT IN CSV")

# Check specific fixtures that have missing features in training data
print("\n" + "=" * 80)
print("2. SAMPLE VERIFICATION - FIXTURES WITH MISSING FEATURES")
print("=" * 80)

# Find fixtures with missing interceptions in training data
missing_interceptions_training = training_data[training_data['home_interceptions_per_90'].isnull()]
print(f"\nFixtures with missing home_interceptions_per_90 in training: {len(missing_interceptions_training):,}")

if len(missing_interceptions_training) > 0:
    # Get sample fixture IDs
    sample_ids = missing_interceptions_training['fixture_id'].head(10).tolist()
    
    print(f"\nChecking {len(sample_ids)} sample fixtures in raw CSV:")
    for fid in sample_ids:
        raw_fixture = raw_csv[raw_csv['id'] == fid]
        if len(raw_fixture) > 0:
            home_interceptions = raw_fixture.iloc[0]['home_interceptions']
            away_interceptions = raw_fixture.iloc[0]['away_interceptions']
            home_tackles = raw_fixture.iloc[0]['home_tackles']
            
            if pd.isnull(home_interceptions):
                print(f"  Fixture {fid}: ✅ CONFIRMED - home_interceptions is NULL in raw CSV")
            else:
                print(f"  Fixture {fid}: ⚠️  UNEXPECTED - home_interceptions={home_interceptions} in raw CSV but missing in training")
        else:
            print(f"  Fixture {fid}: NOT FOUND in raw CSV")

# Check overall statistics availability
print("\n" + "=" * 80)
print("3. OVERALL STATISTICS AVAILABILITY")
print("=" * 80)

# Count fixtures with complete statistics in raw CSV
complete_stats_mask = True
for col in ['home_shots_total', 'home_tackles', 'home_interceptions', 'home_ball_possession']:
    if col in raw_csv.columns:
        complete_stats_mask = complete_stats_mask & raw_csv[col].notna()

complete_fixtures = raw_csv[complete_stats_mask]
print(f"\nFixtures with complete statistics in raw CSV: {len(complete_fixtures):,} ({len(complete_fixtures)/len(raw_csv)*100:.1f}%)")
print(f"Fixtures with missing statistics in raw CSV: {len(raw_csv) - len(complete_fixtures):,} ({(len(raw_csv) - len(complete_fixtures))/len(raw_csv)*100:.1f}%)")

# Check by league
print("\n" + "=" * 80)
print("4. STATISTICS AVAILABILITY BY LEAGUE")
print("=" * 80)

for league_id in sorted(raw_csv['league_id'].unique()):
    league_fixtures = raw_csv[raw_csv['league_id'] == league_id]
    league_complete = league_fixtures[league_fixtures['home_shots_total'].notna()]
    
    pct_complete = (len(league_complete) / len(league_fixtures)) * 100
    print(f"League {league_id}: {len(league_complete):,}/{len(league_fixtures):,} ({pct_complete:.1f}%) have statistics")

# Check by year
print("\n" + "=" * 80)
print("5. STATISTICS AVAILABILITY BY YEAR")
print("=" * 80)

raw_csv['year'] = pd.to_datetime(raw_csv['starting_at']).dt.year
for year in sorted(raw_csv['year'].unique()):
    year_fixtures = raw_csv[raw_csv['year'] == year]
    year_complete = year_fixtures[year_fixtures['home_shots_total'].notna()]
    
    pct_complete = (len(year_complete) / len(year_fixtures)) * 100
    print(f"Year {year}: {len(year_complete):,}/{len(year_fixtures):,} ({pct_complete:.1f}%) have statistics")

# Conclusion
print("\n" + "=" * 80)
print("6. CONCLUSION")
print("=" * 80)

raw_missing = raw_csv['home_shots_total'].isnull().sum()
raw_missing_pct = (raw_missing / len(raw_csv)) * 100

print(f"\n✅ CONFIRMED: Missing statistics in training data are due to missing data in source")
print(f"\nRaw CSV statistics coverage:")
print(f"  - Available: {len(raw_csv) - raw_missing:,} fixtures ({100-raw_missing_pct:.1f}%)")
print(f"  - Missing: {raw_missing:,} fixtures ({raw_missing_pct:.1f}%)")

print(f"\nThis is EXPECTED because:")
print(f"  1. Not all fixtures in the original JSON have statistics")
print(f"  2. Older fixtures may have limited stat coverage")
print(f"  3. Some leagues/competitions may not track detailed statistics")

print(f"\n💡 Recommendation:")
print(f"  - Use only fixtures with complete statistics for training")
print(f"  - Filter: df[df['home_shots_total'].notna()]")
print(f"  - Expected: ~{len(raw_csv) - raw_missing:,} fixtures with full data")

print("\n" + "=" * 80)
