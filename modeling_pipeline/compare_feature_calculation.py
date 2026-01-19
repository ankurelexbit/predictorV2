"""Compare how features are calculated in training vs live."""
import pandas as pd
import numpy as np
from pathlib import Path

# Load training features
print("=" * 80)
print("COMPARING TRAINING VS LIVE FEATURE CALCULATION")
print("=" * 80)

training_features = pd.read_csv('data/processed/sportmonks_features.csv')

# Get a recent match from training data
recent_match = training_features[training_features['date'] >= '2023-01-01'].iloc[0]

print("\n1. SAMPLE TRAINING MATCH:")
print("-" * 80)
print(f"Date: {recent_match['date']}")
print(f"Home: {recent_match['home_team_name']} (ID: {recent_match['home_team_id']})")
print(f"Away: {recent_match['away_team_name']} (ID: {recent_match['away_team_id']})")
print(f"\nKey Features from Training:")
print(f"  home_elo: {recent_match['home_elo']:.1f}")
print(f"  away_elo: {recent_match['away_elo']:.1f}")
print(f"  elo_diff: {recent_match['elo_diff']:.1f}")
print(f"  home_form_5: {recent_match['home_form_5']:.2f}")
print(f"  away_form_5: {recent_match['away_form_5']:.2f}")
print(f"  home_goals_5: {recent_match['home_goals_5']:.2f}")
print(f"  away_goals_5: {recent_match['away_goals_5']:.2f}")

# Now check what the live calculator would produce
print("\n2. CHECKING LIVE FEATURE CALCULATOR:")
print("-" * 80)

import sys
sys.path.insert(0, str(Path.cwd()))
from predict_live import LiveFeatureCalculator

calculator = LiveFeatureCalculator()

# Check if calculator has its own Elo calculation
print("\nInspecting LiveFeatureCalculator methods:")
methods = [m for m in dir(calculator) if not m.startswith('_')]
print(f"  Methods: {methods}")

if 'calculate_elo' in methods or 'update_elo' in methods:
    print("  ⚠️ WARNING: LiveFeatureCalculator has its own Elo calculation!")
    print("  This may not match training Elo values")
else:
    print("  ✓ No Elo calculation in LiveFeatureCalculator")
    print("  Elo should come from API or historical data")

# Check form calculation
print("\n3. FORM CALCULATION:")
print("-" * 80)
print("Training data form range:")
print(f"  Min: {training_features['home_form_5'].min():.2f}")
print(f"  Max: {training_features['home_form_5'].max():.2f}")
print(f"  Mean: {training_features['home_form_5'].mean():.2f}")
print(f"  Expected range: 0-15 (0=5 losses, 15=5 wins)")

# Check feature statistics
print("\n4. FEATURE VALUE RANGES:")
print("-" * 80)

key_features = ['home_elo', 'away_elo', 'home_form_5', 'away_form_5', 
                'home_goals_5', 'away_goals_5', 'home_xg_5', 'away_xg_5']

for feat in key_features:
    if feat in training_features.columns:
        values = training_features[feat].dropna()
        print(f"\n{feat}:")
        print(f"  Range: [{values.min():.2f}, {values.max():.2f}]")
        print(f"  Mean: {values.mean():.2f}, Std: {values.std():.2f}")

