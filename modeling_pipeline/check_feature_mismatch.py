"""Check feature mismatch between training and live."""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from predict_live import LiveFeatureCalculator

# Get training features
training_df = pd.read_csv('data/processed/sportmonks_features.csv')
exclude = ['fixture_id', 'date', 'season_id', 'home_team_id', 'away_team_id',
           'home_team_name', 'away_team_name', 'target', 'home_win', 'draw',
           'away_win', 'home_goals', 'away_goals']
training_features = set([c for c in training_df.columns if c not in exclude])

# Get features from a live calculation
calc = LiveFeatureCalculator()
# Use a sample match
live_features_dict = calc.build_features_for_match(29, 109, pd.to_datetime('2026-01-18'))
live_features = set(live_features_dict.keys()) if live_features_dict else set()

print("=" * 80)
print("FEATURE MISMATCH ANALYSIS")
print("=" * 80)

print(f"\nTraining features: {len(training_features)}")
print(f"Live features: {len(live_features)}")

# Features in training but NOT in live
missing_in_live = training_features - live_features
print(f"\n❌ Features missing in live calculator: {len(missing_in_live)}")

# Group missing features by pattern
patterns = {}
for feat in missing_in_live:
    # Extract base pattern
    if 'player_' in feat:
        key = 'player_*'
    elif 'odds_' in feat or 'market_' in feat:
        key = 'market_*'
    elif 'position' in feat or 'points' in feat:
        key = 'position/points'
    elif '_conceded' in feat:
        key = '*_conceded (opponent stats)'
    elif feat in ['season_name', 'round_num', 'season_progress']:
        key = 'metadata'
    else:
        key = 'other'
    
    if key not in patterns:
        patterns[key] = []
    patterns[key].append(feat)

print("\nMissing features by category:")
for pattern, feats in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {pattern}: {len(feats)} features")
    if len(feats) <= 20:  # Show up to 20 features
        for f in feats:
            print(f"    - {f}")

# Features in live but NOT in training (should be none or minimal)
extra_in_live = live_features - training_features
if extra_in_live:
    print(f"\n⚠️  Features in live but not training: {len(extra_in_live)}")
    for feat in list(extra_in_live)[:10]:
        print(f"    - {feat}")

# Calculate coverage
coverage = (len(training_features) - len(missing_in_live)) / len(training_features) * 100
print(f"\n📊 Feature coverage: {coverage:.1f}%")

