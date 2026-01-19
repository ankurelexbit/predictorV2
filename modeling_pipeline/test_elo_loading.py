"""Test if Elo ratings are being loaded correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from predict_live import LiveFeatureCalculator
import pandas as pd

# Create calculator
calc = LiveFeatureCalculator()

print("=" * 80)
print("TESTING ELO LOADING")
print("=" * 80)

print(f"\nNumber of Elo ratings loaded: {len(calc.elo_ratings)}")
print(f"\nSample Elo ratings:")
for team_id in list(calc.elo_ratings.keys())[:10]:
    print(f"  Team {team_id}: {calc.elo_ratings[team_id]:.1f}")

# Test building features for a match
print("\n" + "=" * 80)
print("TEST: Feyenoord vs Sparta Rotterdam")
print("=" * 80)

features = calc.build_features_for_match(29, 109, pd.to_datetime('2026-01-18'))

if features:
    print(f"\nElo features:")
    print(f"  home_elo: {features.get('home_elo', 'MISSING'):.1f}")
    print(f"  away_elo: {features.get('away_elo', 'MISSING'):.1f}")
    print(f"  elo_diff: {features.get('elo_diff', 'MISSING'):.1f}")
    
    print(f"\nForm features:")
    print(f"  home_form_5: {features.get('home_form_5', 'MISSING'):.1f}")
    print(f"  away_form_5: {features.get('away_form_5', 'MISSING'):.1f}")
    
    print(f"\nGoal features:")
    print(f"  home_goals_5: {features.get('home_goals_5', 'MISSING'):.2f}")
    print(f"  away_goals_5: {features.get('away_goals_5', 'MISSING'):.2f}")
    
    # Compare with training data expectation
    print("\n" + "=" * 80)
    print("EXPECTED FROM EARLIER ANALYSIS:")
    print("=" * 80)
    print("  home_elo (Feyenoord): 1456.0")
    print("  away_elo (Sparta): 1519.0")
    print("  elo_diff: -63.0 (Sparta stronger)")
    print("  away_form: 12.0 points")
    print("  home_form: 2.0 points")

