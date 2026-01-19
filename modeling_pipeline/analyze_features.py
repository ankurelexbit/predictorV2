"""
Analyze feature values for Jan 18 predictions to understand team strength differences.
"""
import pandas as pd
import json

# Read predictions
preds = pd.read_csv('predictions_jan_18_stacking.csv')

print("=" * 80)
print("ANALYZING TEAM STRENGTH DIFFERENCES")
print("=" * 80)

# Look at matches where strong teams played away
interesting_matches = [
    "Nantes vs Paris",  # PSG away
    "Feyenoord vs Sparta Rotterdam",  # Feyenoord is strong Dutch team
    "Getafe vs Valencia",  # Both La Liga
]

for match in interesting_matches:
    for _, row in preds.iterrows():
        if f"{row['home_team']} vs {row['away_team']}" in match:
            print(f"\n{row['home_team']} vs {row['away_team']}")
            print(f"  Predicted: H:{row['home_win_prob']:.1%} D:{row['draw_prob']:.1%} A:{row['away_win_prob']:.1%}")
            print(f"  Model: {row['model_used']}")
            break

# Now let's manually fetch features for one match to see Elo diff
print("\n" + "=" * 80)
print("FETCHING DETAILED FEATURES FOR PSG MATCH")
print("=" * 80)

# Import the feature calculator
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from predict_live import LiveFeatureCalculator
from datetime import datetime

calculator = LiveFeatureCalculator()

# Nantes vs PSG (Paris)
# Need to find team IDs - let me check fixtures
from predict_live import get_upcoming_fixtures

fixtures = get_upcoming_fixtures('2026-01-18')
psg_match = fixtures[fixtures['away_team_name'].str.contains('Paris', case=False, na=False)]

if not psg_match.empty:
    match = psg_match.iloc[0]
    print(f"\nMatch: {match['home_team_name']} vs {match['away_team_name']}")
    
    features = calculator.build_features_for_match(
        match['home_team_id'],
        match['away_team_id'],
        pd.to_datetime(match['date'])
    )
    
    if features:
        print(f"\nKey Features:")
        print(f"  Home Elo: {features.get('home_elo', 'N/A'):.1f}")
        print(f"  Away Elo: {features.get('away_elo', 'N/A'):.1f}")
        print(f"  Elo Diff: {features.get('elo_diff', 'N/A'):.1f} (positive = home stronger)")
        print(f"  Home Form (5): {features.get('home_form_5', 'N/A'):.1f}")
        print(f"  Away Form (5): {features.get('away_form_5', 'N/A'):.1f}")
        print(f"  Home Goals (5): {features.get('home_goals_5', 'N/A'):.2f}")
        print(f"  Away Goals (5): {features.get('away_goals_5', 'N/A'):.2f}")

