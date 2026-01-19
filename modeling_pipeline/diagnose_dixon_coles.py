"""
Diagnose why Dixon-Coles has 100% away win bias.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("=" * 80)
print("DIAGNOSING DIXON-COLES AWAY WIN BIAS")
print("=" * 80)

MODELS_DIR = Path("models")
dc_data = joblib.load(MODELS_DIR / "dixon_coles_model.joblib")

print("\nDixon-Coles Parameters:")
print(f"  Home advantage: {dc_data['base_model_data']['home_adv']:.3f}")
print(f"  Rho: {dc_data['base_model_data']['rho']:.3f}")
print(f"  Number of teams: {len(dc_data['base_model_data']['attack'])}")

# Check team strengths for teams in our test matches
teams_to_check = [
    ('SC Heerenveen', 'FC Groningen'),
    ('Parma', 'Genoa'),
    ('Nantes', 'Paris')
]

attack = dc_data['base_model_data']['attack']
defense = dc_data['base_model_data']['defense']

print("\n" + "=" * 80)
print("TEAM STRENGTH PARAMETERS")
print("=" * 80)

for home_team, away_team in teams_to_check:
    print(f"\n{home_team} vs {away_team}:")
    
    if home_team in attack and away_team in attack:
        home_attack = attack[home_team]
        home_defense = defense[home_team]
        away_attack = attack[away_team]
        away_defense = defense[away_team]
        
        print(f"  {home_team}:")
        print(f"    Attack: {home_attack:.3f}")
        print(f"    Defense: {home_defense:.3f}")
        
        print(f"  {away_team}:")
        print(f"    Attack: {away_attack:.3f}")
        print(f"    Defense: {away_defense:.3f}")
        
        # Calculate expected goals
        home_adv = dc_data['base_model_data']['home_adv']
        lambda_home = np.exp(home_attack + away_defense + home_adv)
        lambda_away = np.exp(away_attack + home_defense)
        
        print(f"\n  Expected goals:")
        print(f"    Home: {lambda_home:.2f}")
        print(f"    Away: {lambda_away:.2f}")
        
        if lambda_away > lambda_home:
            print(f"    ⚠️  Away team expected to score MORE despite home advantage!")
    else:
        missing = []
        if home_team not in attack:
            missing.append(home_team)
        if away_team not in attack:
            missing.append(away_team)
        print(f"  ⚠️  Teams not in model: {', '.join(missing)}")
        print(f"  This will use default parameters (0.0) for missing teams")

# Check calibration
print("\n" + "=" * 80)
print("CALIBRATION ANALYSIS")
print("=" * 80)

if dc_data['is_calibrated']:
    print("\nModel IS calibrated")
    print("Calibrators for each class:")
    for class_idx in range(3):
        if class_idx in dc_data['calibrators']:
            calibrator = dc_data['calibrators'][class_idx]
            print(f"  Class {class_idx} ({'Home/Draw/Away'[class_idx*5:(class_idx+1)*5]}): {type(calibrator).__name__}")
else:
    print("\nModel is NOT calibrated")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("\n🔍 Possible causes of away win bias:")
print("1. Teams missing from training data → use default 0.0 attack/defense")
print("2. Calibration may be inverting probabilities")
print("3. Home advantage parameter may be negative or too small")
print("4. Model may be using wrong team names (ID vs name mismatch)")

# Check if most teams are missing
total_teams = len(attack)
print(f"\nModel trained on {total_teams} teams")
print("If test teams aren't in training data, predictions will be wrong")

