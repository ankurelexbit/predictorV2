#!/usr/bin/env python3
"""
Diagnose Feature Mismatch Between Training and Production
==========================================================

Compares features generated during training vs production to identify differences.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_live_standalone import StandaloneLivePipeline

# Load training data
print("=" * 80)
print("FEATURE MISMATCH DIAGNOSIS")
print("=" * 80)

print("\n1. Loading training data...")
train_df = pd.read_csv('data/training_data_with_draw_features.csv')

# Get a sample of matches
print(f"   Total training samples: {len(train_df)}")

# Exclude metadata columns
meta_cols = ['fixture_id', 'home_team_id', 'away_team_id', 'season_id', 'league_id',
             'match_date', 'home_score', 'away_score', 'result', 'target']
feature_cols = [c for c in train_df.columns if c not in meta_cols]

print(f"   Feature columns: {len(feature_cols)}")

# Analyze training data statistics
train_features = train_df[feature_cols]

# Convert boolean to int
bool_cols = train_features.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    train_features[col] = train_features[col].astype(int)

print("\n2. Training Data Feature Statistics:")
print("   " + "-" * 76)

# Key features to analyze
key_features = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_league_position', 'away_league_position',
    'home_derived_xg_per_match_5', 'away_derived_xg_per_match_5',
    'home_points', 'away_points'
]

stats_train = {}
for feat in key_features:
    if feat in train_features.columns:
        mean_val = train_features[feat].mean()
        std_val = train_features[feat].std()
        stats_train[feat] = {'mean': mean_val, 'std': std_val}
        print(f"   {feat:40s}: mean={mean_val:8.2f}, std={std_val:8.2f}")

# Now generate features using production pipeline
print("\n3. Generating features using PRODUCTION pipeline...")

api_key = os.environ.get('SPORTMONKS_API_KEY')
if not api_key:
    print("   ❌ SPORTMONKS_API_KEY not set - skipping production test")
    sys.exit(0)

pipeline = StandaloneLivePipeline(api_key)
pipeline.load_model()

# Fetch some recent fixtures
endpoint = "fixtures/between/2026-02-01/2026-02-02"
params = {
    'include': 'participants;league;state',
    'filters': 'fixtureStates:1,2,3,5',  # Include completed for comparison
    'page': 1
}

data = pipeline._api_call(endpoint, params)
fixtures_data = data['data'][:10]  # Get 10 fixtures

print(f"   Fetched {len(fixtures_data)} fixtures for testing")

production_features = []

for fixture in fixtures_data:
    participants = fixture.get('participants', [])
    home_team = next((p for p in participants if p.get('meta', {}).get('location') == 'home'), None)
    away_team = next((p for p in participants if p.get('meta', {}).get('location') == 'away'), None)

    if not home_team or not away_team:
        continue

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

    features = pipeline.generate_features(fixture_dict)
    if features:
        production_features.append(features)

if not production_features:
    print("   ❌ Failed to generate production features")
    sys.exit(1)

prod_df = pd.DataFrame(production_features)
print(f"   Generated features for {len(prod_df)} matches")

print("\n4. Production Feature Statistics:")
print("   " + "-" * 76)

stats_prod = {}
for feat in key_features:
    if feat in prod_df.columns:
        mean_val = prod_df[feat].mean()
        std_val = prod_df[feat].std()
        stats_prod[feat] = {'mean': mean_val, 'std': std_val}
        print(f"   {feat:40s}: mean={mean_val:8.2f}, std={std_val:8.2f}")

# Compare statistics
print("\n5. COMPARISON (Training vs Production):")
print("   " + "-" * 76)
print(f"   {'Feature':40s}  {'Train Mean':>10s}  {'Prod Mean':>10s}  {'Difference':>12s}")
print("   " + "-" * 76)

large_diffs = []

for feat in key_features:
    if feat in stats_train and feat in stats_prod:
        train_mean = stats_train[feat]['mean']
        prod_mean = stats_prod[feat]['mean']
        diff = prod_mean - train_mean
        pct_diff = (diff / train_mean * 100) if abs(train_mean) > 0.001 else 0

        marker = ""
        if abs(pct_diff) > 20:
            marker = " ⚠️  LARGE"
            large_diffs.append((feat, pct_diff, train_mean, prod_mean))

        print(f"   {feat:40s}  {train_mean:10.2f}  {prod_mean:10.2f}  {pct_diff:+11.1f}%{marker}")

# Check for missing features
print("\n6. Missing Feature Check:")
print("   " + "-" * 76)

train_features_set = set(feature_cols)
prod_features_set = set(prod_df.columns)

missing_in_prod = train_features_set - prod_features_set
missing_in_train = prod_features_set - train_features_set

if missing_in_prod:
    print(f"   ⚠️  Features in TRAINING but missing in PRODUCTION ({len(missing_in_prod)}):")
    for feat in sorted(list(missing_in_prod)[:10]):
        print(f"      - {feat}")
    if len(missing_in_prod) > 10:
        print(f"      ... and {len(missing_in_prod) - 10} more")

if missing_in_train:
    print(f"   ⚠️  Features in PRODUCTION but missing in TRAINING ({len(missing_in_train)}):")
    for feat in sorted(list(missing_in_train)[:10]):
        print(f"      - {feat}")
    if len(missing_in_train) > 10:
        print(f"      ... and {len(missing_in_train) - 10} more")

if not missing_in_prod and not missing_in_train:
    print("   ✅ All features present in both datasets")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

if large_diffs:
    print(f"\n⚠️  Found {len(large_diffs)} features with >20% difference:")
    for feat, pct_diff, train_mean, prod_mean in large_diffs[:5]:
        print(f"   • {feat}: {pct_diff:+.1f}% diff (train={train_mean:.2f}, prod={prod_mean:.2f})")

    print("\nPossible causes:")
    print("   1. Training uses historical data, production uses current/recent data")
    print("   2. Different leagues/teams in training vs production")
    print("   3. API data quality differs from historical CSV data")
    print("   4. Feature calculation logic differs between pipelines")
else:
    print("\n✅ No major statistical differences found")

if missing_in_prod or missing_in_train:
    print(f"\n⚠️  Feature schema mismatch:")
    print(f"   Missing in production: {len(missing_in_prod)}")
    print(f"   Missing in training: {len(missing_in_train)}")

print("\n" + "=" * 80)
