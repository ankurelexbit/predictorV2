"""Test the retrained models directly with controlled input."""
import pandas as pd
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

# Load the model
print("=" * 80)
print("TESTING RETRAINED STACKING MODEL DIRECTLY")
print("=" * 80)

# Create test data with clear away team advantage
test_data = pd.DataFrame([{
    # Elo: Away team much stronger
    'home_elo': 1400.0,
    'away_elo': 1600.0,
    'elo_diff': -200.0,  # HUGE advantage for away
    
    # Form: Away team better
    'home_form_5': 3.0,  # 1 win in last 5
    'away_form_5': 15.0,  # 5 wins in last 5
    
    # Team IDs
    'home_team_id': 1,
    'away_team_id': 2,
    'home_team_name': 'Weak Home Team',
    'away_team_name': 'Strong Away Team',
    
    # Fill other features with averages
    'home_goals_5': 0.5,
    'away_goals_5': 2.5,
    'home_xg_5': 0.6,
    'away_xg_5': 2.2,
    'form_diff_5': -12,
    'home_form_3': 0,
    'away_form_3': 9,
    'home_wins_3': 0,
    'away_wins_3': 3,
    'home_wins_5': 1,
    'away_wins_5': 5,
}])

# Load stacking ensemble
import importlib
elo_module = importlib.import_module('04_model_baseline_elo')
dc_module = importlib.import_module('05_model_dixon_coles')
xgb_module = importlib.import_module('06_model_xgboost')
ensemble_module = importlib.import_module('07_model_ensemble')

MODELS_DIR = Path("models")

# Load models
print("\nLoading Elo model...")
EloProbabilityModel = elo_module.EloProbabilityModel
elo_model = EloProbabilityModel(home_advantage=50)  # Should be 50 now
elo_model.load(MODELS_DIR / "elo_model.joblib")

print(f"Elo model home advantage: {elo_model.home_advantage}")

probs_elo = elo_model.predict_proba(test_data)[0]
print(f"\nElo predictions (away team +200 Elo):")
print(f"  Home: {probs_elo[2]:.1%}, Draw: {probs_elo[1]:.1%}, Away: {probs_elo[0]:.1%}")

# Load XGBoost
print("\nLoading XGBoost model...")
XGBoostFootballModel = xgb_module.XGBoostFootballModel
xgb_model = XGBoostFootballModel()
xgb_model.load(MODELS_DIR / "xgboost_model.joblib")

probs_xgb = xgb_model.predict_proba(test_data)[0]
print(f"\nXGBoost predictions (away team dominant):")
print(f"  Home: {probs_xgb[2]:.1%}, Draw: {probs_xgb[1]:.1%}, Away: {probs_xgb[0]:.1%}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if probs_elo[0] > probs_elo[2]:
    print("\n✓ Elo correctly predicts AWAY WIN")
else:
    print("\n✗ Elo still predicts HOME WIN despite -200 Elo diff!")
    print("  This means home advantage is still too high!")

if probs_xgb[0] > probs_xgb[2]:
    print("✓ XGBoost correctly predicts AWAY WIN")
else:
    print("✗ XGBoost still predicts HOME WIN despite dominant away team!")

