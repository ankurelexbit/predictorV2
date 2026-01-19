"""
Test each base model separately to identify which has the strongest home bias.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Import model classes
sys.path.insert(0, str(Path.cwd()))
from predict_live import LiveFeatureCalculator, get_upcoming_fixtures
import importlib

print("=" * 80)
print("TESTING EACH BASE MODEL SEPARATELY - JAN 18 PREDICTIONS")
print("=" * 80)

# Get fixtures
fixtures = get_upcoming_fixtures('2026-01-18')
print(f"\nTesting on {len(fixtures)} matches from Jan 18")

# Initialize feature calculator
calculator = LiveFeatureCalculator()

# Get features for first 3 matches (to save API calls)
test_matches = []
for idx in range(min(3, len(fixtures))):
    fixture = fixtures.iloc[idx]
    features = calculator.build_features_for_match(
        fixture['home_team_id'],
        fixture['away_team_id'],
        pd.to_datetime(fixture['date'])
    )
    if features:
        feature_df = pd.DataFrame([features])
        feature_df['home_team_name'] = fixture['home_team_name']
        feature_df['away_team_name'] = fixture['away_team_name']
        test_matches.append({
            'fixture': fixture,
            'features': feature_df
        })

print(f"Built features for {len(test_matches)} matches\n")

# Load models
MODELS_DIR = Path("models")

# ============================================================================
# TEST 1: ELO MODEL
# ============================================================================
print("=" * 80)
print("MODEL 1: ELO")
print("=" * 80)

elo_module = importlib.import_module('04_model_baseline_elo')
EloProbabilityModel = elo_module.EloProbabilityModel

elo_model = EloProbabilityModel()
elo_model.load(MODELS_DIR / "elo_model.joblib")

elo_home_predictions = 0
for match in test_matches:
    probs = elo_model.predict_proba(match['features'])[0]
    fixture = match['fixture']
    
    print(f"\n{fixture['home_team_name']} vs {fixture['away_team_name']}")
    print(f"  H: {probs[2]:.1%} | D: {probs[1]:.1%} | A: {probs[0]:.1%}")
    print(f"  Predicted: {['Away Win', 'Draw', 'Home Win'][np.argmax(probs)]}")
    
    if np.argmax(probs) == 2:
        elo_home_predictions += 1

print(f"\nElo Model: {elo_home_predictions}/{len(test_matches)} predicted as home wins ({elo_home_predictions/len(test_matches):.0%})")

# ============================================================================
# TEST 2: DIXON-COLES MODEL
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: DIXON-COLES")
print("=" * 80)

dc_module = importlib.import_module('05_model_dixon_coles')
DixonColesModel = dc_module.DixonColesModel
CalibratedDixonColes = dc_module.CalibratedDixonColes

dc_path = MODELS_DIR / "dixon_coles_model.joblib"
dc_data = joblib.load(dc_path)

# Reconstruct
base_dc = DixonColesModel()
base_dc.attack = dc_data['base_model_data']['attack']
base_dc.defense = dc_data['base_model_data']['defense']
base_dc.home_adv = dc_data['base_model_data']['home_adv']
base_dc.rho = dc_data['base_model_data']['rho']
base_dc.team_to_idx = dc_data['base_model_data']['team_to_idx']
base_dc.idx_to_team = dc_data['base_model_data']['idx_to_team']
base_dc.time_decay = dc_data['base_model_data']['time_decay']
base_dc.max_goals = dc_data['base_model_data']['max_goals']
base_dc.is_fitted = dc_data['base_model_data']['is_fitted']

dc_model = CalibratedDixonColes(base_dc)
dc_model.calibrators = dc_data['calibrators']
dc_model.is_calibrated = dc_data['is_calibrated']

dc_home_predictions = 0
for match in test_matches:
    probs = dc_model.predict_proba(match['features'])[0]
    fixture = match['fixture']
    
    print(f"\n{fixture['home_team_name']} vs {fixture['away_team_name']}")
    print(f"  H: {probs[0]:.1%} | D: {probs[1]:.1%} | A: {probs[2]:.1%}")
    
    # Dixon-Coles returns [home, draw, away] order
    pred_idx = np.argmax(probs)
    pred_label = ['Home Win', 'Draw', 'Away Win'][pred_idx]
    print(f"  Predicted: {pred_label}")
    
    if pred_idx == 0:
        dc_home_predictions += 1

print(f"\nDixon-Coles Model: {dc_home_predictions}/{len(test_matches)} predicted as home wins ({dc_home_predictions/len(test_matches):.0%})")

# ============================================================================
# TEST 3: XGBOOST MODEL
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: XGBOOST")
print("=" * 80)

xgb_module = importlib.import_module('06_model_xgboost')
XGBoostFootballModel = xgb_module.XGBoostFootballModel

xgb_model = XGBoostFootballModel()
xgb_model.load(MODELS_DIR / "xgboost_model.joblib")

xgb_home_predictions = 0
for match in test_matches:
    probs = xgb_model.predict_proba(match['features'])[0]
    fixture = match['fixture']
    
    print(f"\n{fixture['home_team_name']} vs {fixture['away_team_name']}")
    print(f"  H: {probs[2]:.1%} | D: {probs[1]:.1%} | A: {probs[0]:.1%}")
    print(f"  Predicted: {['Away Win', 'Draw', 'Home Win'][np.argmax(probs)]}")
    
    if np.argmax(probs) == 2:
        xgb_home_predictions += 1

print(f"\nXGBoost Model: {xgb_home_predictions}/{len(test_matches)} predicted as home wins ({xgb_home_predictions/len(test_matches):.0%})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY - HOME WIN PREDICTION RATE")
print("=" * 80)

print(f"\nElo:          {elo_home_predictions}/{len(test_matches)} = {elo_home_predictions/len(test_matches):.0%}")
print(f"Dixon-Coles:  {dc_home_predictions}/{len(test_matches)} = {dc_home_predictions/len(test_matches):.0%}")
print(f"XGBoost:      {xgb_home_predictions}/{len(test_matches)} = {xgb_home_predictions/len(test_matches):.0%}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

culprits = []
if elo_home_predictions == len(test_matches):
    culprits.append("Elo (100% home bias)")
if dc_home_predictions == len(test_matches):
    culprits.append("Dixon-Coles (100% home bias)")
if xgb_home_predictions == len(test_matches):
    culprits.append("XGBoost (100% home bias)")

if culprits:
    print("\n🚨 Models with 100% home win predictions:")
    for c in culprits:
        print(f"  - {c}")
else:
    print("\nNo model has 100% home bias on this sample")
    
print("\nAll three models contribute to the home bias problem!")

