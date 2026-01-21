#!/usr/bin/env python3
"""
Add league_id to sportmonks_features.csv from raw fixtures data

This script merges league_id from the raw fixtures file into the processed features.
NO re-fetching required!
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def add_league_id():
    """Add league_id from raw data to processed features"""
    
    print("=" * 70)
    print("ADDING LEAGUE_ID TO FEATURES")
    print("=" * 70)
    
    # Load processed features
    features_path = PROCESSED_DATA_DIR / 'sportmonks_features.csv'
    print(f"\n1. Loading processed features from {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"   Shape: {features_df.shape}")
    
    # Check if league_id already exists
    if 'league_id' in features_df.columns:
        print("   ✅ league_id already exists!")
        return True
    
    # Load raw fixtures
    raw_fixtures_path = RAW_DATA_DIR / 'sportmonks' / 'fixtures.csv'
    print(f"\n2. Loading raw fixtures from {raw_fixtures_path}...")
    
    if not raw_fixtures_path.exists():
        print(f"   ❌ Raw fixtures file not found!")
        print(f"   You need to fetch data first")
        return False
    
    raw_df = pd.read_csv(raw_fixtures_path)
    print(f"   Shape: {raw_df.shape}")
    
    # Check if league_id exists in raw data
    if 'league_id' not in raw_df.columns:
        print(f"   ❌ league_id not in raw fixtures!")
        return False
    
    print(f"   ✅ Found league_id in raw data")
    
    # Merge league_id based on fixture_id
    print(f"\n3. Merging league_id...")
    
    # Select only fixture_id and league_id from raw data
    league_mapping = raw_df[['fixture_id', 'league_id']].drop_duplicates()
    
    # Merge with features
    features_with_league = features_df.merge(
        league_mapping,
        on='fixture_id',
        how='left'
    )
    
    # Check how many got league_id
    matched = features_with_league['league_id'].notna().sum()
    total = len(features_with_league)
    
    print(f"   Matched: {matched}/{total} ({matched/total*100:.1f}%)")
    
    if matched < total * 0.9:
        print(f"   ⚠️  Warning: Only {matched/total*100:.1f}% matched!")
        print(f"   Some fixtures may be missing from raw data")
    
    # Show league distribution
    print(f"\n4. League distribution:")
    league_counts = features_with_league['league_id'].value_counts()
    for league_id, count in league_counts.items():
        league_name = {
            8: 'Premier League',
            384: 'La Liga',
            564: 'Bundesliga',
            462: 'Ligue 1',
            301: 'Serie A'
        }.get(int(league_id) if pd.notna(league_id) else 0, f'League {league_id}')
        print(f"   {league_name}: {count} matches")
    
    # Save updated features
    print(f"\n5. Saving updated features...")
    
    # Backup original
    backup_path = features_path.parent / 'sportmonks_features_no_league.csv'
    features_df.to_csv(backup_path, index=False)
    print(f"   Backup saved to {backup_path}")
    
    # Save with league_id
    features_with_league.to_csv(features_path, index=False)
    print(f"   ✅ Updated features saved to {features_path}")
    
    print(f"\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"league_id added to {total} matches")
    print(f"New features file has {len(features_with_league.columns)} columns")
    print(f"\nNext steps:")
    print(f"1. Verify: head -1 data/processed/sportmonks_features.csv | grep league_id")
    print(f"2. Retrain: venv/bin/python tune_for_draws.py")
    print(f"3. Test: venv/bin/python run_live_predictions.py")
    
    return True

if __name__ == "__main__":
    success = add_league_id()
    sys.exit(0 if success else 1)
